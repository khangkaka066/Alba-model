"""
Script huấn luyện Mini-LLM theo chiến lược Đa Giai đoạn (Two-Stage Curriculum)
kèm hệ thống đo lường chuyên sâu cho Bộ tối ưu hóa (Adam / AdamW Optimizer Metrics):
1. Loss Curves (Train Loss & Val Loss)
2. Perplexity (PPL = exp(loss))
3. Learning Rate Schedule (Warmup, Cosine, Annealing)
4. Gradient Norm (Độ lớn đạo hàm trước khi clip)
5. Weight Update Ratio (Độ lớn bước nhảy của Adam: ||delta_w|| / ||w||)
   - Tỷ lệ chuẩn vàng: ~ 1e-3 (0.001)
   - Báo hiệu bước nhảy quá lớn hoặc quá nhỏ của Adam
6. Throughput & Latency (Tokens/sec và ms/step)
7. Tùy chọn so sánh bộ tối ưu: --optimizer adamw hoặc --optimizer adam
"""

import os
import time
import math
import json
import argparse
from dataclasses import asdict
import torch
from tokenizers import Tokenizer

from src.config import ModelArgs
from src.model import MiniLLM
from src.dataset import BinaryDataset, get_batch

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer", "tokenizer.json")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Mini-LLM & Thu thập Metrics nghiên cứu")
    parser.add_argument("--exp_name", type=str, default="modern_transformer_llama", help="Tên thí nghiệm / kiến trúc")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam"], help="Bộ tối ưu: adamw hoặc adam")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Hệ số phạt suy giảm trọng số (Weight Decay)")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--seq_len", type=int, default=128, help="Độ dài ngữ cảnh token")
    parser.add_argument("--max_iters", type=int, default=2500, help="Tổng số bước huấn luyện")
    parser.add_argument("--eval_interval", type=int, default=250, help="Khoảng cách giữa các lần đánh giá")
    parser.add_argument("--dim", type=int, default=512, help="Số chiều vector embedding (d_model)")
    parser.add_argument("--n_layers", type=int, default=8, help="Số tầng Transformer (L)")
    parser.add_argument("--n_heads", type=int, default=8, help="Số attention heads (H)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate cực đại")
    parser.add_argument("--device", type=str, default=None, help="cuda / mps / cpu")
    return parser.parse_args()


@torch.no_grad()
def estimate_loss(model, train_ds, val_ds, batch_size, seq_len, device, eval_iters=30):
    out = {}
    model.eval()
    for split, ds in [("train", train_ds), ("val", val_ds)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(ds, batch_size, seq_len, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def plot_training_dashboard(metrics_data: dict, save_path: str):
    """Vẽ dashboard 6 biểu đồ trực quan hóa toàn diện quá trình huấn luyện và chỉ số Adam."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"⚠️ Không thể vẽ biểu đồ bằng matplotlib ({e}). File metrics.json vẫn được lưu đầy đủ!")
        return

    history = metrics_data["history"]
    if not history:
        return

    steps = [h["step"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    train_ppls = [h["train_ppl"] for h in history]
    val_ppls = [h["val_ppl"] for h in history]
    lrs = [h["lr"] for h in history]
    grad_norms = [h["grad_norm"] for h in history]
    update_ratios = [h.get("update_ratio", 1e-3) for h in history]
    tokens_per_sec = [h["tokens_per_sec"] for h in history]

    split_step = metrics_data.get("phase1_steps", steps[-1] * 0.8)
    opt_name = metrics_data.get("training_config", {}).get("optimizer", "AdamW").upper()

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"🚀 Training Dashboard [{opt_name}]: {metrics_data['exp_name']} (~{metrics_data['total_params']/1e6:.2f}M params)", fontsize=16, fontweight='bold')

    # 1. Loss Curves
    ax = axs[0, 0]
    ax.plot(steps, train_losses, 'b-o', label='Train Loss', lw=2, ms=4)
    ax.plot(steps, val_losses, 'r-s', label='Val Loss', lw=2, ms=4)
    ax.axvline(x=split_step, color='gray', linestyle='--', alpha=0.7, label='Phase 1 / Phase 2 Split')
    ax.set_title("1. Cross-Entropy Loss")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Perplexity (PPL)
    ax = axs[0, 1]
    ax.plot(steps, train_ppls, 'b-o', label='Train PPL', lw=2, ms=4)
    ax.plot(steps, val_ppls, 'r-s', label='Val PPL', lw=2, ms=4)
    ax.axvline(x=split_step, color='gray', linestyle='--', alpha=0.7)
    ax.set_title("2. Perplexity (PPL = exp(loss)) [Càng thấp càng tốt]")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Perplexity")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Learning Rate Schedule
    ax = axs[0, 2]
    ax.plot(steps, lrs, 'g-^', label='Learning Rate', lw=2, ms=4)
    ax.axvline(x=split_step, color='gray', linestyle='--', alpha=0.7, label='Annealing Start')
    ax.set_title(f"3. {opt_name} Learning Rate (Warmup + Cosine + Annealing)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Learning Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Gradient Norm & Stability
    ax = axs[1, 0]
    ax.plot(steps, grad_norms, 'm-d', label='Gradient Norm', lw=2, ms=4)
    ax.axhline(y=1.0, color='red', linestyle=':', label='Clip Threshold (1.0)')
    ax.set_title("4. Gradient Norm (Đo độ ổn định & Chống nổ gradient)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("L2 Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Adam Weight Update-to-Weight Ratio
    ax = axs[1, 1]
    ax.plot(steps, update_ratios, 'c-p', label=f'{opt_name} Update Ratio', lw=2, ms=4)
    ax.axhline(y=1e-3, color='green', linestyle='--', alpha=0.8, label='Tỷ lệ vàng (~1e-3)')
    ax.set_title(f"5. {opt_name} Update Ratio (||Δw|| / ||w||)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Tỷ lệ cập nhật")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Architecture & Results Summary Box
    ax = axs[1, 2]
    ax.axis('off')
    summary_text = (
        f"MODEL SUMMARY:\n"
        f"-----------------------------------------\n"
        f"• Architecture: {metrics_data['exp_name']}\n"
        f"• Optimizer: {opt_name} (Weight Decay: {metrics_data['training_config']['weight_decay']})\n"
        f"• Total Params: {metrics_data['total_params']:,} (~{metrics_data['total_params']/1e6:.2f}M)\n"
        f"• Embedding Dim (d): {metrics_data['architecture']['dim']}\n"
        f"• Transformer Layers (L): {metrics_data['architecture']['n_layers']}\n"
        f"• Attention Heads (H): {metrics_data['architecture']['n_heads']}\n"
        f"• Context Length (T): {metrics_data['architecture']['max_seq_len']}\n"
        f"• Vocab Size (V): {metrics_data['architecture']['vocab_size']}\n"
        f"-----------------------------------------\n"
        f"BEST METRICS:\n"
        f"• Best Val Loss: {metrics_data['best_val_loss']:.4f}\n"
        f"• Best Val Perplexity: {metrics_data['best_val_ppl']:.2f}\n"
        f"• Avg Throughput: {sum(tokens_per_sec)/len(tokens_per_sec):,.0f} tokens/s\n"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#f0f4f8', edgecolor='#336699', lw=1.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"📊 Đã xuất ảnh Training Dashboard tại: {save_path}")


def main():
    args = parse_args()

    # 1. Xác định thiết bị
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔥 Thiết bị huấn luyện: {device}")
    if device == "cuda":
        print(f"   GPU Model: {torch.cuda.get_device_name(0)}")

    # 2. Đọc Tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"📖 Kích thước từ vựng (Vocab Size): {actual_vocab_size}")

    # 3. Nạp dữ liệu 2 giai đoạn
    p1_train = BinaryDataset(os.path.join(DATA_DIR, "phase1_train.bin"))
    p1_val   = BinaryDataset(os.path.join(DATA_DIR, "phase1_val.bin"))
    p2_train = BinaryDataset(os.path.join(DATA_DIR, "phase2_train.bin"))
    p2_val   = BinaryDataset(os.path.join(DATA_DIR, "phase2_val.bin"))
    
    print(f"📊 Dữ liệu Phase 1: {len(p1_train):,} train | {len(p1_val):,} val tokens")
    print(f"📊 Dữ liệu Phase 2: {len(p2_train):,} train | {len(p2_val):,} val tokens")

    # 4. Khởi tạo mô hình
    model_args = ModelArgs(
        vocab_size=actual_vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=256,
        dropout=0.0
    )
    model = MiniLLM(model_args).to(device)
    param_count = model.count_parameters()
    print(f"🧠 Khởi tạo mô hình thành công: {param_count:,} tham số (~{param_count / 1e6:.2f}M)")

    # 5. Khởi tạo bộ tối ưu hóa (Hỗ trợ AdamW và Adam)
    opt_choice = args.optimizer.lower()
    if opt_choice == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay
        )
        print("⚙️ Bộ tối ưu hóa: Standard Adam (L2 Regularization)")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay
        )
        print("⚙️ Bộ tối ưu hóa: AdamW (Decoupled Weight Decay)")

    # 6. Chia số bước: 80% Phase 1, 20% Phase 2
    phase1_iters = int(args.max_iters * 0.8)
    phase2_iters = args.max_iters - phase1_iters
    warmup_steps = 100

    print("\n" + "=" * 65)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN ĐA GIAI ĐOẠN")
    print(f"   - Thí nghiệm: {args.exp_name}")
    print(f"   - Bộ tối ưu: {opt_choice.upper()}")
    print(f"   - Giai đoạn 1 (Nền tảng): Bước 1 ➔ {phase1_iters} (80%)")
    print(f"   - Giai đoạn 2 (Annealing): Bước {phase1_iters + 1} ➔ {args.max_iters} (20%)")
    print("=" * 65)

    # Dữ liệu lưu trữ metrics
    metrics_history = []
    best_val_loss = float("inf")
    best_val_ppl = float("inf")
    start_time = time.time()
    step_timer = time.time()

    for step in range(1, args.max_iters + 1):
        # Xác định giai đoạn hiện tại
        is_phase2 = step > phase1_iters
        current_train_ds = p2_train if is_phase2 else p1_train
        current_val_ds   = p2_val if is_phase2 else p1_val
        phase_label = "Phase 2 (Annealing)" if is_phase2 else "Phase 1 (Foundation)"

        # Learning Rate Schedule
        if not is_phase2:
            if step < warmup_steps:
                lr = args.lr * step / warmup_steps
            else:
                ratio = (step - warmup_steps) / (phase1_iters - warmup_steps)
                lr = 3e-4 + 0.5 * (args.lr - 3e-4) * (1.0 + math.cos(math.pi * ratio))
        else:
            p2_step = step - phase1_iters
            ratio = p2_step / phase2_iters
            lr = 1e-5 + 0.5 * (3e-4 - 1e-5) * (1.0 + math.cos(math.pi * ratio))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Lấy batch và tính loss
        xb, yb = get_batch(current_train_ds, args.batch_size, args.seq_len, device)
        _, loss = model(xb, yb)

        # Cập nhật trọng số & Đo chỉ số Adam
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # 1. Đo Gradient Norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()

        # 2. Đo tỷ lệ cập nhật của Adam: ||delta_w|| / ||w||
        # Lấy một lớp đại diện (ví dụ wq của tầng đầu tiên) để đo bước nhảy
        sample_param = model.layers[0].attention.wq.weight
        old_param = sample_param.data.clone()

        optimizer.step()

        with torch.no_grad():
            delta_w = (sample_param.data - old_param).norm(2)
            w_norm = old_param.norm(2)
            update_ratio = (delta_w / max(w_norm, 1e-7)).item()

        # Đo tốc độ
        dt = time.time() - step_timer
        step_timer = time.time()
        tokens_processed = args.batch_size * args.seq_len
        tokens_per_sec = tokens_processed / max(dt, 1e-5)
        step_time_ms = dt * 1000

        # Đánh giá loss định kỳ
        if step % args.eval_interval == 0 or step == phase1_iters or step == args.max_iters:
            losses = estimate_loss(model, current_train_ds, current_val_ds, args.batch_size, args.seq_len, device)
            
            # Tính Perplexity: PPL = exp(loss)
            train_ppl = math.exp(min(losses["train"], 20.0))
            val_ppl = math.exp(min(losses["val"], 20.0))
            
            elapsed = time.time() - start_time
            print(
                f"[{phase_label}] Step {step:4d}/{args.max_iters} | "
                f"Train Loss: {losses['train']:.4f} (PPL: {train_ppl:.1f}) | "
                f"Val Loss: {losses['val']:.4f} (PPL: {val_ppl:.1f}) | "
                f"GradNorm: {grad_norm:.2f} | "
                f"AdamRatio: {update_ratio:.2e} | "
                f"LR: {lr:.6f} | "
                f"{tokens_per_sec:,.0f} tok/s"
            )

            # Lưu vào metrics history
            metrics_history.append({
                "step": step,
                "phase": phase_label,
                "train_loss": round(losses["train"], 4),
                "val_loss": round(losses["val"], 4),
                "train_ppl": round(train_ppl, 2),
                "val_ppl": round(val_ppl, 2),
                "lr": round(lr, 7),
                "grad_norm": round(grad_norm, 3),
                "update_ratio": round(update_ratio, 6),
                "tokens_per_sec": round(tokens_per_sec, 1),
                "step_time_ms": round(step_time_ms, 2)
            })

            # Lưu checkpoint mô hình tốt nhất
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                best_val_ppl = val_ppl
                checkpoint_file = os.path.join(CHECKPOINT_DIR, "best_model.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_args": asdict(model_args),
                    "step": step,
                    "phase": phase_label,
                    "optimizer": opt_choice,
                    "val_loss": best_val_loss,
                    "val_ppl": best_val_ppl
                }, checkpoint_file)
                print(f"   ⭐ Lưu checkpoint tốt nhất: {checkpoint_file} (Val Loss: {best_val_loss:.4f}, PPL: {best_val_ppl:.2f})")

    # 7. Đóng gói file metrics JSON & Vẽ Dashboard
    metrics_file = os.path.join(CHECKPOINT_DIR, f"{args.exp_name}_metrics.json")
    default_metrics_file = os.path.join(CHECKPOINT_DIR, "metrics.json")
    dashboard_img = os.path.join(CHECKPOINT_DIR, "training_dashboard.png")
    exp_dashboard_img = os.path.join(CHECKPOINT_DIR, f"{args.exp_name}_dashboard.png")

    full_metrics = {
        "exp_name": args.exp_name,
        "total_params": param_count,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_ppl": round(best_val_ppl, 2),
        "phase1_steps": phase1_iters,
        "architecture": asdict(model_args),
        "training_config": {
            "optimizer": opt_choice,
            "weight_decay": args.weight_decay,
            "max_iters": args.max_iters,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "lr": args.lr
        },
        "history": metrics_history
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2, ensure_ascii=False)
    with open(default_metrics_file, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("🎉 Huấn luyện hoàn tất!")
    print(f"📁 Dữ liệu metrics ({opt_choice.upper()}) đã lưu tại: {metrics_file}")
    print(f"📁 Checkpoint lưu tại: {os.path.join(CHECKPOINT_DIR, 'best_model.pt')}")

    # Vẽ dashboard tổng kết
    plot_training_dashboard(full_metrics, dashboard_img)
    plot_training_dashboard(full_metrics, exp_dashboard_img)
    print("=" * 65)


if __name__ == "__main__":
    main()
