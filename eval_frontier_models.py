import os
import torch
import numpy as np

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    dot_X = torch.matmul(X, X.t())
    dot_Y = torch.matmul(Y, Y.t())
    
    hsic_xy = torch.sum(dot_X * dot_Y)
    hsic_xx = torch.sum(dot_X * dot_X)
    hsic_yy = torch.sum(dot_Y * dot_Y)
    
    cka = hsic_xy / (torch.sqrt(hsic_xx) * torch.sqrt(hsic_yy) + 1e-12)
    return float(cka.item())

def compute_effective_depth(hidden_states: tuple) -> tuple:
    n_layers = len(hidden_states) - 1
    renewals = []
    
    for l in range(n_layers):
        h_curr = hidden_states[l].squeeze(0).float()
        h_next = hidden_states[l + 1].squeeze(0).float()
        cka_val = linear_cka(h_curr, h_next)
        renewals.append(1.0 - cka_val)
        
    d_eff = sum(renewals)
    return d_eff, renewals

def compute_routing_entropy(router_logits: torch.Tensor) -> float:
    probs = torch.softmax(router_logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
    return float(entropy.mean().item())

def compute_spectral_condition_number(weight_matrix: torch.Tensor) -> tuple:
    W = weight_matrix.float()
    if W.dim() > 2:
        W = W.flatten(1)
    s = torch.linalg.svdvals(W)
    kappa = float((s[0] / (s[-1] + 1e-12)).item())
    return kappa, s.cpu().numpy()

def evaluate_huggingface_model(model_name: str, sample_text: str = None, max_length: int = 512, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if sample_text is None:
        sample_text = "The quick brown fox jumps over the lazy dog. Scientific reasoning requires rigorous verification and mathematical formalization." * 8
        
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    model.eval()
    
    inputs = tokenizer(sample_text, return_tensors="pt", max_length=max_length, truncation=True)
    if torch.cuda.is_available() and device == "cuda":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    hidden_states = outputs.hidden_states
    d_eff, renewals = compute_effective_depth(hidden_states)
    
    first_weight = None
    for p in model.parameters():
        if p.dim() == 2 and p.shape[0] > 128 and p.shape[1] > 128:
            first_weight = p
            break
            
    kappa, s_vals = (None, None)
    if first_weight is not None:
        kappa, s_vals = compute_spectral_condition_number(first_weight)
        
    return {
        "model_name": model_name,
        "n_layers": len(hidden_states) - 1,
        "effective_depth": d_eff,
        "renewals": renewals,
        "condition_number": kappa
    }

if __name__ == "__main__":
    print("Frontier evaluation engine ready.")
