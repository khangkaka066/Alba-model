"""
Kiến trúc Decoder-only Transformer hiện đại:
- RMSNorm
- RoPE (Rotary Position Embedding)
- Multi-Head Causal Self-Attention
- SwiGLU Feed-Forward Network
Bám sát tài liệu hướng dẫn công thức từng bước (LLM_CONG_THUC_TUNG_BUOC.md).
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .config import ModelArgs


# -----------------------------------------------------------------------------
# 1. RMSNorm: Chuẩn hóa vector không cần trừ mean
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS(x) = sqrt( mean(x^2) + eps )
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# -----------------------------------------------------------------------------
# 2. RoPE: Rotary Positional Embedding (Mã hóa vị trí quay)
# -----------------------------------------------------------------------------
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Tiền tính toán ma trận pha số phức e^(i * theta) cho các vị trí từ 0 đến end."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # e^(i * theta)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quay các vector Query và Key trong không gian 2D theo vị trí token."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[None, :xq_.shape[1], None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# -----------------------------------------------------------------------------
# 3. Causal Multi-Head Self-Attention
# -----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        
        # Ma trận chiếu tuyến tính cho Q, K, V và Output
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # Batch, Sequence Length, Channels (Embedding Dim)
        
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        # Phân tách thành các heads: (B, T, n_heads, head_dim)
        xq = xq.view(B, T, self.n_heads, self.head_dim)
        xk = xk.view(B, T, self.n_heads, self.head_dim)
        xv = xv.view(B, T, self.n_heads, self.head_dim)
        
        # Áp dụng RoPE vào Query và Key
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        # Chuyển về shape (B, n_heads, T, head_dim) để tính Attention
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Flash Attention / Scaled Dot-Product Attention có sẵn causal mask (tam giác dưới)
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        # Nối các heads lại: (B, T, C)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(output)


# -----------------------------------------------------------------------------
# 4. SwiGLU Feed-Forward Network: W2 * (SiLU(W1 * x) * (W3 * x))
# -----------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            # Chuẩn LLaMA: hidden_dim xấp xỉ 8/3 * dim và làm tròn bội số của 64
            hidden_dim = int(2 * (4 * dim) / 3)
            hidden_dim = (hidden_dim + 63) // 64 * 64
            
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# -----------------------------------------------------------------------------
# 5. Geometric Orthogonal Residual v2 (GOR-v2 - Đột phá kiến trúc 2026)
# -----------------------------------------------------------------------------
class GeometricOrthogonalResidual(nn.Module):
    """
    Geometric Orthogonal Residual v2 (GOR-v2):
    Cải tiến Residual Connection thuần túy bằng toán học hình học vi phân (0 tham số phụ).
    
    Phân rã bản cập nhật u = f(x) trong không gian Hilbert thành:
    1. u_perp (Thành phần trực giao - Tri thức mới hoàn toàn):
       Vuông góc với dòng chính x -> Luôn được cộng trọn vẹn 100%.
    2. u_parallel (Thành phần song song - Củng cố / Xung đột tri thức cũ):
       Điều biến bằng hàm mượt trơn C^infinity:
       phi(rho) = 2.0 * sigmoid(k * rho)
       - Tại rho = 0 (khởi tạo ngẫu nhiên): phi(0) = 1.0 -> Giữ nguyên 100% Identity Residual,
         loại bỏ hoàn toàn độ trễ khởi động ban đầu.
       - Khi rho > 0 (đồng thuận): Củng cố nhẹ tri thức tích lũy (lên tới ~1.5 - 2.0).
       - Khi rho < 0 (xung đột gradient): Dập tắt êm ả về sát 0, triệt tiêu nhiễu phá hoại.
    """
    def __init__(self, eps: float = 1e-7, k: float = 3.0):
        super().__init__()
        self.eps = eps
        self.k = k

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d), u: (B, T, d)
        dot = (x * u).sum(dim=-1, keepdim=True)
        norm_sq_x = (x * x).sum(dim=-1, keepdim=True)
        norm_sq_u = (u * u).sum(dim=-1, keepdim=True)
        
        norm_x = torch.sqrt(norm_sq_x + self.eps)
        norm_u = torch.sqrt(norm_sq_u + self.eps)
        
        # Góc tương quan cosin rho trong khoảng [-1, 1]
        rho = dot / (norm_x * norm_u + self.eps)
        
        # Phân rã trực giao Gram-Schmidt
        u_parallel = (dot / (norm_sq_x + self.eps)) * x
        u_perp = u - u_parallel
        
        # GOR-v2: Điều biến trơn C^infinity bảo toàn Identity tại rho=0
        phi = 2.0 * torch.sigmoid(self.k * rho)
        
        return x + u_perp + phi * u_parallel


# -----------------------------------------------------------------------------
# 6. Transformer Block (Hỗ trợ GOR và Standard Residual)
# -----------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.residual_type = getattr(args, "residual_type", "gor")
        self.attention_norm = RMSNorm(args.dim)
        self.attention = CausalSelfAttention(args)
        self.ffn_norm = RMSNorm(args.dim)
        self.feed_forward = SwiGLU(args.dim)

        if self.residual_type == "gor":
            self.res_attn = GeometricOrthogonalResidual()
            self.res_ffn = GeometricOrthogonalResidual()

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # Nhánh 1: Multi-Head Attention
        u_attn = self.attention(self.attention_norm(x), freqs_cis)
        if self.residual_type == "gor":
            h = self.res_attn(x, u_attn)
        else:
            h = x + u_attn

        # Nhánh 2: SwiGLU Feed-Forward Network
        u_ffn = self.feed_forward(self.ffn_norm(h))
        if self.residual_type == "gor":
            out = self.res_ffn(h, u_ffn)
        else:
            out = h + u_ffn

        return out


# -----------------------------------------------------------------------------
# 7. MiniLLM: Mô hình ngôn ngữ hoàn chỉnh
# -----------------------------------------------------------------------------
class MiniLLM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        # Weight tying: chia sẻ trọng số giữa bảng embedding và tầng output cuối
        self.output.weight = self.tok_embeddings.weight
        
        # Đăng ký ma trận tần số RoPE dạng buffer không cần học
        freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        h = self.tok_embeddings(idx)
        freqs_cis = self.freqs_cis[:T]
        
        # Đi qua từng tầng Transformer
        for layer in self.layers:
            h = layer(h, freqs_cis)
            
        h = self.norm(h)
        logits = self.output(h)
        
        loss = None
        if targets is not None:
            # Cross-Entropy Loss: dự đoán token tiếp theo
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    def count_parameters(self) -> int:
        """Đếm tổng số tham số có thể huấn luyện trong mô hình."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
