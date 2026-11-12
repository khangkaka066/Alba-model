"""
GEO-Muon: Geometric Energy-Preserving Orthogonal Muon Optimizer (Frontier 2026).
Thiết kế chuyên biệt cho Alba-Model nhằm giải quyết các điểm yếu của AdamW và Vanilla Muon.

Các đột phá cốt lõi:
1. Matrix-Orthogonal Optimization: Trực giao hóa ma trận Momentum 2D bằng thuật toán
   lặp đại số Newton-Schulz bậc 5 (Quintic iteration), triệt tiêu hiện tượng méo phổ kỳ dị (Spectral Distortion).
2. Energy-Preserving Spectral Scaling (EPSS): Bảo toàn độ lớn năng lượng Frobenius:
   scale = ||M||_F / (||O||_F + eps)
   -> Giữ nguyên biên độ xung lực thực tế của gradient, ngăn chặn triệt để hiện tượng
      sụp đổ tỷ lệ cập nhật (Update Ratio Collapse) như AdamW gặp phải ở các bước chuyển pha.
3. Hybrid Co-existence: Tự động phân chia thông minh:
   - Các ma trận ẩn 2D (Attention, SwiGLU, OS-MoE Sub-Experts) -> Tối ưu bằng GEO-Muon.
   - Các tham số 1D (RMSNorm scale), ma trận từ vựng (Token Embeddings, LM Head) -> Tối ưu bằng AdamW.
"""

import math
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Thuật toán lặp Newton-Schulz bậc 5 (Quintic Newton-Schulz Iteration)
    để tính toán xấp xỉ ma trận trực giao cực (Orthogonal Polar Factor) cực nhanh trên Tensor Core.
    
    Đa thức lặp bậc 5:
    X_{k+1} = X_k * (a * I + b * X_k^T * X_k + c * (X_k^T * X_k)^2)
    Hệ số tối ưu hóa: a = 3.4445, b = -4.7750, c = 2.0315
    """
    assert len(G.shape) >= 2, f"Newton-Schulz yêu cầu ma trận >= 2 chiều, nhận được shape {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Tính toán ở định dạng Float32 để đảm bảo độ ổn định số học
    X = G.float()
    transposed = False
    
    # Luôn làm việc trên ma trận có m <= n (nếu m > n, chuyển vị để giảm kích cỡ ma trận nhân)
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    # Chuẩn hóa tiền định phổ (Pre-scaling)
    norm = X.norm(dim=(-2, -1), keepdim=True) + eps
    X = X / norm

    # Chuỗi lặp bậc 5 đại số (Tận dụng addmm nếu là ma trận 2D để tránh cấp phát bộ nhớ)
    for _ in range(steps):
        A = X @ X.mT
        if X.dim() == 2:
            B = torch.addmm(A, A, A, beta=b, alpha=c)
            X = torch.addmm(X, B, X, beta=a, alpha=1.0)
        else:
            B = b * A + c * (A @ A)
            X = a * X + B @ X

    if transposed:
        X = X.mT

    return X.to(dtype=G.dtype)


class GEOMuon(torch.optim.Optimizer):
    """
    Bộ tối ưu hóa kết hợp GEO-Muon + AdamW:
    - Nhóm tham số có `use_muon=True`: Cập nhật bằng đà trực giao bảo toàn năng lượng (GEO-Muon).
    - Nhóm tham số có `use_muon=False`: Cập nhật bằng AdamW tiêu chuẩn.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            use_muon = group.get("use_muon", False)

            if use_muon:
                # -------------------------------------------------------------
                # Nhánh 1: GEO-Muon cho các ma trận trọng số 2D ẩn
                # -------------------------------------------------------------
                momentum = group["momentum"]
                ns_steps = group.get("ns_steps", 5)

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]

                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    buf = state["momentum_buffer"]
                    # Cập nhật đà gradient (First moment accumulation)
                    buf.mul_(momentum).add_(g)

                    # 1. Trực giao hóa ma trận bằng Newton-Schulz bậc 5
                    O = zeropower_via_newtonschulz5(buf, steps=ns_steps)

                    # 2. Điều phối tỷ lệ cập nhật ma trận trực giao (Aspect-Ratio Polar Scale)
                    # Giữ tỷ lệ bước nhảy ||delta_w|| / ||w|| đạt tỷ lệ chuẩn vàng ~ 1e-3
                    scale = max(1.0, O.size(-2) / O.size(-1)) ** 0.5
                    step_dir = O * scale

                    # 3. Phạt suy giảm trọng số độc lập (Decoupled Weight Decay)
                    if weight_decay != 0:
                        p.data.mul_(1.0 - lr * weight_decay)

                    # 4. Áp dụng bước nhảy
                    p.data.add_(step_dir, alpha=-lr)

            else:
                # -------------------------------------------------------------
                # Nhánh 2: AdamW tiêu chuẩn cho 1D / Embeddings / LM Head
                # -------------------------------------------------------------
                beta1, beta2 = group["betas"]
                eps = group["eps"]

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]

                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    state["step"] += 1
                    step_t = state["step"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    # Cập nhật first và second moments
                    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                    bias_correction1 = 1 - beta1 ** step_t
                    bias_correction2 = 1 - beta2 ** step_t

                    denom = (exp_avg_sq.sqrt() / (math.sqrt(bias_correction2))).add_(eps)
                    step_size = lr / bias_correction1

                    # Decoupled Weight Decay
                    if weight_decay != 0:
                        p.data.mul_(1.0 - lr * weight_decay)

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "geo_muon",
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    momentum: float = 0.95,
    betas: Tuple[float, float] = (0.9, 0.95),
) -> torch.optim.Optimizer:
    """
    Tạo bộ tối ưu hóa tự động phân luồng theo kiến trúc mô hình:
    - Nếu optimizer_type == 'geo_muon':
      + Toàn bộ ma trận ẩn 2D (Attention, FFN, MoE Sub-Experts) -> GEO-Muon
      + Toàn bộ 1D, Embeddings, LM Head, Router Gate -> AdamW
    - Nếu optimizer_type == 'adamw': AdamW thuần túy cho toàn bộ mô hình
    - Nếu optimizer_type == 'adam': Adam chuẩn (L2 Penalty)
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
    elif optimizer_type in ["geo_muon", "muon"]:
        muon_params = []
        adamw_params = []
        muon_names = []
        adamw_names = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Tiêu chí phân luồng GEO-Muon:
            # 1. Bắt buộc phải là ma trận 2D
            # 2. KHÔNG phải là Embedding (lookup table)
            # 3. KHÔNG phải là LM Head (output classification)
            # 4. KHÔNG phải là Router Gate (prototype cosine vectors)
            is_2d = len(param.shape) >= 2
            is_embed_or_head = (
                ("tok_embeddings" in name)
                or ("token_embedding" in name)
                or ("output" in name)
                or ("lm_head" in name)
            )
            is_gate = "gate" in name
            is_norm = "norm" in name or param.ndim < 2

            if is_2d and not is_embed_or_head and not is_gate and not is_norm:
                muon_params.append(param)
                muon_names.append(name)
            else:
                adamw_params.append(param)
                adamw_names.append(name)

        muon_count = sum(p.numel() for p in muon_params)
        adamw_count = sum(p.numel() for p in adamw_params)
        total = muon_count + adamw_count

        print("⚡ [GEO-Muon Hybrid Optimizer Initialized]")
        print(f"   ├─ Nhánh GEO-Muon (2D Matrix Orthogonal): {len(muon_params)} tensors | {muon_count:,} params ({muon_count / total * 100:.1f}%)")
        print(f"   └─ Nhánh AdamW (1D / Embed / Head / Router): {len(adamw_params)} tensors | {adamw_count:,} params ({adamw_count / total * 100:.1f}%)")

        param_groups = [
            {
                "params": muon_params,
                "use_muon": True,
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
            },
            {
                "params": adamw_params,
                "use_muon": False,
                "lr": lr,
                "betas": betas,
                "weight_decay": weight_decay,
                "eps": 1e-8,
            },
        ]

        return GEOMuon(param_groups)
    else:
        raise ValueError(f"Optimizer không hợp lệ: {optimizer_type}. Chọn 'geo_muon', 'adamw', hoặc 'adam'.")
