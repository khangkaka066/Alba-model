import os
import math
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_moefication import upcycle_qwen_to_osmoe
from qwen_geo_muon import create_qwen_geomuon_optimizer

def get_curriculum_batch(tokenizer, batch_size: int = 2, seq_len: int = 256, device: str = "cuda"):
    texts = [
        "Mathematical derivation of orthogonal projections: Let x and u be vectors in Hilbert space. The orthogonal component is u_perp = u - <x,u>/<x,x> * x.",
        "Algorithm for Newton-Schulz iteration: Given matrix X_0 = M / ||M||_F, compute X_{k+1} = X_k (a I + b X_k^T X_k + c (X_k^T X_k)^2) for k in range 5.",
        "Deep learning architectures benefit from Mixture of Experts by partitioning monolithic feed-forward blocks into specialized sub-experts.",
        "Chain of thought reasoning requires explicit intermediate verification steps enclosed within think tokens to ensure mathematical exactness."
    ]
    selected = [texts[i % len(texts)] for i in range(batch_size)]
    inputs = tokenizer(selected, return_tensors="pt", max_length=seq_len, padding="max_length", truncation=True)
    if torch.cuda.is_available() and device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

def train_upcycled_qwen(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    total_steps: int = 100,
    lr: float = 3e-4,
    ortho_weight: float = 0.01,
    batch_size: int = 2,
    seq_len: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "qwen_upcycled_checkpoints"
):
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    model = upcycle_qwen_to_osmoe(model, num_experts=8, top_k=2, enable_gor=True)
    optimizer = create_qwen_geomuon_optimizer(model, lr=lr)
    
    model.train()
    print(f"Starting Qwen Upcycled Training: {total_steps} steps on {device}")
    
    for step in range(1, total_steps + 1):
        optimizer.zero_grad()
        batch = get_curriculum_batch(tokenizer, batch_size=batch_size, seq_len=seq_len, device=device)
        
        outputs = model(**batch)
        lm_loss = outputs.loss
        
        total_ortho_loss = torch.tensor(0.0, device=device)
        n_routers = 0
        for m in model.modules():
            if m.__class__.__name__ == "CosineBoundedRouter":
                gate_norm = F.normalize(m.gate, p=2, dim=-1, eps=1e-7)
                gate_sq = torch.matmul(gate_norm, gate_norm.t())
                eye = torch.eye(m.num_experts, device=device, dtype=gate_norm.dtype)
                total_ortho_loss += torch.sum((gate_sq - eye) ** 2) / (m.num_experts * (m.num_experts - 1))
                n_routers += 1
                
        if n_routers > 0:
            total_ortho_loss = total_ortho_loss / n_routers
            
        loss = lm_loss + ortho_weight * total_ortho_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if step % 10 == 0 or step == total_steps:
            print(f"Step {step:04d}/{total_steps} | LM Loss: {lm_loss.item():.4f} | Ortho Loss: {total_ortho_loss.item():.4f} | Total Loss: {loss.item():.4f}")
            
    checkpoint_path = os.path.join(output_dir, "qwen_os_moe_final.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Upcycled Qwen model saved successfully to {checkpoint_path}")
    return model, tokenizer

if __name__ == "__main__":
    train_upcycled_qwen(total_steps=50)
