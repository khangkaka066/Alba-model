import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineBoundedRouter(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int = 8, top_k: int = 2, gamma: float = 10.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gamma = gamma
        self.gate = nn.Parameter(torch.empty(num_experts, hidden_dim))
        nn.init.orthogonal_(self.gate)

    def forward(self, x: torch.Tensor):
        x_norm = F.normalize(x, p=2, dim=-1, eps=1e-7)
        gate_norm = F.normalize(self.gate, p=2, dim=-1, eps=1e-7)
        logits = self.gamma * F.linear(x_norm, gate_norm)
        
        topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)
        
        gate_sq = torch.matmul(gate_norm, gate_norm.t())
        eye = torch.eye(self.num_experts, device=x.device, dtype=x.dtype)
        ortho_loss = torch.sum((gate_sq - eye) ** 2) / (self.num_experts * (self.num_experts - 1))
        
        return topk_weights, topk_indices, logits, ortho_loss

class QwenSubExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class QwenOSMoEBlock(nn.Module):
    def __init__(self, original_mlp: nn.Module, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        hidden_size = original_mlp.gate_proj.in_features
        intermediate_size = original_mlp.gate_proj.out_features
        sub_intermediate_size = max(64, intermediate_size // 4)
        
        self.shared_expert = original_mlp
        self.router = CosineBoundedRouter(hidden_size, num_experts, top_k)
        
        self.sub_experts = nn.ModuleList([
            QwenSubExpert(hidden_size, sub_intermediate_size) for _ in range(num_experts)
        ])
        
        self._initialize_sub_experts(original_mlp, sub_intermediate_size)

    def _initialize_sub_experts(self, original_mlp: nn.Module, sub_dim: int):
        with torch.no_grad():
            orig_gate = original_mlp.gate_proj.weight.data
            orig_up = original_mlp.up_proj.weight.data
            orig_down = original_mlp.down_proj.weight.data
            
            for i, expert in enumerate(self.sub_experts):
                start = (i * sub_dim) % (orig_gate.size(0) - sub_dim + 1)
                end = start + sub_dim
                
                noise_gate = torch.randn_like(orig_gate[start:end, :]) * 0.02
                noise_up = torch.randn_like(orig_up[start:end, :]) * 0.02
                noise_down = torch.randn_like(orig_down[:, start:end]) * 0.02
                
                expert.gate_proj.weight.copy_(orig_gate[start:end, :] + noise_gate)
                expert.up_proj.weight.copy_(orig_up[start:end, :] + noise_up)
                expert.down_proj.weight.copy_(orig_down[:, start:end] + noise_down)

    def forward(self, x: torch.Tensor):
        shared_out = self.shared_expert(x)
        topk_weights, topk_indices, raw_logits, ortho_loss = self.router(x)
        
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        weights_flat = topk_weights.view(-1, self.top_k)
        indices_flat = topk_indices.view(-1, self.top_k)
        
        out_flat = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            k_indices = indices_flat[:, k]
            k_weights = weights_flat[:, k].unsqueeze(-1)
            for e_idx in range(self.num_experts):
                mask = (k_indices == e_idx)
                if mask.any():
                    expert_in = x_flat[mask]
                    expert_out = self.sub_experts[e_idx](expert_in)
                    out_flat[mask] += expert_out * k_weights[mask]
                    
        routed_out = out_flat.view(B, S, D)
        total_out = shared_out + routed_out
        return total_out, ortho_loss, raw_logits

class GORv2ResidualWrapper(nn.Module):
    def __init__(self, core_layer: nn.Module):
        super().__init__()
        self.layer = core_layer

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        x = hidden_states
        layer_output = self.layer(hidden_states, *args, **kwargs)
        
        if isinstance(layer_output, tuple):
            u_proposal = layer_output[0] - x
            rest_outputs = layer_output[1:]
        else:
            u_proposal = layer_output - x
            rest_outputs = ()
            
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        u_norm = torch.norm(u_proposal, p=2, dim=-1, keepdim=True) + 1e-7
        inner_prod = torch.sum(x * u_proposal, dim=-1, keepdim=True)
        
        rho = inner_prod / (x_norm * u_norm)
        u_parallel = (inner_prod / (x_norm ** 2)) * x
        u_perp = u_proposal - u_parallel
        phi_rho = 2.0 * torch.sigmoid(3.0 * rho)
        
        next_x = x + u_perp + phi_rho * u_parallel
        
        if rest_outputs:
            return (next_x,) + rest_outputs
        return next_x

def upcycle_qwen_to_osmoe(model: nn.Module, num_experts: int = 8, top_k: int = 2, enable_gor: bool = True) -> nn.Module:
    decoder_layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        decoder_layers = model.model.layers
    elif hasattr(model, "layers"):
        decoder_layers = model.layers
        
    if decoder_layers is None:
        raise ValueError("Unsupported model structure: unable to locate transformer layers.")
        
    for l_idx, layer in enumerate(decoder_layers):
        if hasattr(layer, "mlp"):
            orig_mlp = layer.mlp
            osmoe_block = QwenOSMoEBlock(orig_mlp, num_experts=num_experts, top_k=top_k)
            layer.mlp = osmoe_block
            
    if enable_gor:
        for l_idx, layer in enumerate(decoder_layers):
            wrapped_layer = GORv2ResidualWrapper(layer)
            decoder_layers[l_idx] = wrapped_layer
            
    return model
