import math
import torch
from torch.optim.optimizer import Optimizer

def zeroth_power_via_newton_schulz_5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()
    X = X / (X.norm() + eps)
    
    if X.size(0) > X.size(1):
        X = X.t()
        transposed = True
    else:
        transposed = False
        
    for _ in range(steps):
        A = torch.mm(X, X.t())
        B = b * A + c * torch.mm(A, A)
        X = a * X + torch.mm(B, X)
        
    if transposed:
        X = X.t()
        
    return X.to(dtype=G.dtype)

class GEOMuon(Optimizer):
    def __init__(self, params, lr: float = 3e-4, momentum: float = 0.95, weight_decay: float = 0.01, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            is_muon = group.get("is_muon", False)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p)
                    if not is_muon:
                        state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                buf = state["momentum_buffer"]

                if is_muon and p.dim() >= 2:
                    if weight_decay != 0:
                        p.mul_(1.0 - lr * weight_decay)
                        
                    buf.mul_(momentum).add_(grad)
                    
                    g_2d = buf.view(buf.size(0), -1)
                    ortho_update = zeroth_power_via_newton_schulz_5(g_2d, steps=ns_steps)
                    ortho_update = ortho_update.view_as(p)
                    
                    d_out, d_in = p.size(0), p.size(1)
                    alpha = max(1.0, math.sqrt(d_out / d_in))
                    
                    p.add_(ortho_update, alpha=-lr * alpha)
                else:
                    if weight_decay != 0:
                        p.mul_(1.0 - lr * weight_decay)
                        
                    buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(0.999).addcmul_(grad, grad, value=0.001)
                    
                    denom = exp_avg_sq.sqrt().add_(1e-8)
                    p.addcdiv_(buf, denom, value=-lr)

        return loss

def create_qwen_geomuon_optimizer(model: torch.nn.Module, lr: float = 3e-4, weight_decay: float = 0.01) -> GEOMuon:
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    param_groups = [
        {"params": muon_params, "is_muon": True, "lr": lr, "weight_decay": weight_decay},
        {"params": adamw_params, "is_muon": False, "lr": lr * 0.5, "weight_decay": weight_decay * 0.1}
    ]

    return GEOMuon(param_groups, lr=lr, weight_decay=weight_decay)
