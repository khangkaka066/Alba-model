from dataclasses import dataclass

@dataclass
class ModelArgs:
    """
    Siêu tham số cấu hình cho mô hình Alba-Model:
    - Scaled default: d=512, L=8, H=8 (~26.64M tham số).
    - Compact: d=256, L=6, H=4 (~5.29M tham số).
    """
    vocab_size: int = 1837        # Sẽ được cập nhật chính xác theo file tokenizer.json
    dim: int = 512                # Số chiều vector embedding (d_model)
    n_layers: int = 8             # Số tầng Transformer (L)
    n_heads: int = 8              # Số attention heads (H), mỗi head d_h = 512 / 8 = 64
    max_seq_len: int = 256        # Độ dài ngữ cảnh tối đa một lần đưa vào (T)
    dropout: float = 0.0          # Dropout
    residual_type: str = "gor"    # Cơ chế Residual: 'gor' (Geometric Orthogonal Residual) hoặc 'standard' (x + u)
    ffn_type: str = "swiglu"      # Khối FFN: 'swiglu' (Dense LLaMA chuẩn) hoặc 'os_moe' (Orthogonal Sub-Expert MoE)
    n_routed_experts: int = 8     # Tổng số Sub-Experts trong khối MoE
    num_experts_per_tok: int = 2  # Số Sub-Experts kích hoạt cho mỗi token (Top-K)
    moe_ortho_coeff: float = 0.01 # Hệ số phạt trực giao giữa các chuyên gia (chống sụp đổ tầng sâu)
