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
