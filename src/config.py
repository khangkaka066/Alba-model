from dataclasses import dataclass

@dataclass
class ModelArgs:
    """
    Siêu tham số cấu hình cho mô hình Mini-LLM (khoảng 5M - 6M tham số).
    Phù hợp chạy mượt trên Colab GPU (T4) lẫn CPU/MPS cá nhân.
    """
    vocab_size: int = 1024        # Sẽ được cập nhật chính xác theo file tokenizer.json
    dim: int = 256                # Số chiều vector embedding (d_model)
    n_layers: int = 6             # Số tầng Transformer (L)
    n_heads: int = 4              # Số attention heads (H), mỗi head có d_h = 256 / 4 = 64
    max_seq_len: int = 256        # Độ dài ngữ cảnh tối đa một lần đưa vào (T)
    dropout: float = 0.0          # Dropout (đặt 0.0 khi dữ liệu sạch để tránh cản trở suy luận)
