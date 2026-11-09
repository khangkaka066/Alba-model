"""
Script so sánh các kiến trúc mô hình (Model Architecture Comparison Tool):
- Nhận vào danh sách các file metrics.json từ các lần chạy huấn luyện khác nhau
- Vẽ biểu đồ đối chiếu trực tiếp:
  1. Validation Loss
  2. Validation Perplexity (PPL)
  3. Gradient Norm (Độ ổn định)
  4. Tốc độ huấn luyện (Tokens/giây)
  5. Bảng so sánh tổng hợp (Params, Loss, PPL, Tốc độ)
- Xuất file ảnh PNG độ nét cao để dùng trong báo cáo / nghiên cứu
"""

import os
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="So sánh hiệu năng các kiến trúc mô hình LLM")
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Danh sách các file metrics.json cần so sánh (vd: exp1_metrics.json exp2_metrics.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_comparison.png",
        help="Tên file ảnh xuất ra (mặc định: model_comparison.png)"
    )
    return parser.parse_args()


def load_experiments(file_paths: list[str]) -> list[dict]:
    experiments = []
    for fp in file_paths:
        if not os.path.exists(fp):
            print(f"⚠️ Cảnh báo: Không tìm thấy file {fp}, bỏ qua!")
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
                experiments.append(data)
                print(f"✅ Đã nạp metrics: {data.get('exp_name', fp)} ({data.get('total_params', 0):,} params)")
        except Exception as e:
            print(f"❌ Lỗi khi đọc file {fp}: {e}")
    return experiments


def plot_comparison(experiments: list[dict], output_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"❌ Cần cài đặt thư viện matplotlib để vẽ biểu đồ so sánh: {e}")
        return

    if not experiments:
        print("❌ Không có dữ liệu thí nghiệm nào để so sánh!")
        return

    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("🔬 BẢNG SO SÁNH HIỆU NĂNG CÁC KIẾN TRÚC MÔ HÌNH LLM", fontsize=16, fontweight="bold")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']

    # 1. Validation Loss Comparison
    ax1 = axs[0, 0]
    for i, exp in enumerate(experiments):
        name = exp.get("exp_name", f"Model {i+1}")
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        hist = exp.get("history", [])
        steps = [h["step"] for h in hist]
        val_losses = [h["val_loss"] for h in hist]
        ax1.plot(steps, val_losses, color=c, marker=m, label=f"{name} (Best: {exp.get('best_val_loss', '?')})", lw=2, ms=5)
    ax1.set_title("1. So sánh Validation Loss (Càng thấp càng tốt)", fontweight="bold")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Val Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Validation Perplexity (PPL) Comparison
    ax2 = axs[0, 1]
    for i, exp in enumerate(experiments):
        name = exp.get("exp_name", f"Model {i+1}")
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        hist = exp.get("history", [])
        steps = [h["step"] for h in hist]
        val_ppls = [h["val_ppl"] for h in hist]
        ax2.plot(steps, val_ppls, color=c, marker=m, label=f"{name} (Best PPL: {exp.get('best_val_ppl', '?')})", lw=2, ms=5)
    ax2.set_title("2. So sánh Perplexity (Độ bất định từ vựng - Log Scale)", fontweight="bold")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Perplexity")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Gradient Norm Stability
    ax3 = axs[1, 0]
    for i, exp in enumerate(experiments):
        name = exp.get("exp_name", f"Model {i+1}")
        c = colors[i % len(colors)]
        hist = exp.get("history", [])
        steps = [h["step"] for h in hist]
        grad_norms = [h["grad_norm"] for h in hist]
        ax3.plot(steps, grad_norms, color=c, label=name, lw=1.5, alpha=0.85)
    ax3.axhline(y=1.0, color='red', linestyle=':', label='Clip Threshold (1.0)')
    ax3.set_title("3. Độ ổn định Gradient Norm (Đo nổ / biến mất gradient)", fontweight="bold")
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Gradient Norm")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Throughput (Tokens / sec)
    ax4 = axs[1, 1]
    for i, exp in enumerate(experiments):
        name = exp.get("exp_name", f"Model {i+1}")
        c = colors[i % len(colors)]
        hist = exp.get("history", [])
        steps = [h["step"] for h in hist]
        tps = [h["tokens_per_sec"] for h in hist]
        avg_tps = sum(tps) / len(tps) if tps else 0
        ax4.plot(steps, tps, color=c, label=f"{name} (Avg: {avg_tps:,.0f} tok/s)", lw=2)
    ax4.set_title("4. Tốc độ Xử lý (Tokens / Giây - Throughput)", fontweight="bold")
    ax4.set_xlabel("Steps")
    ax4.set_ylabel("Tokens / sec")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"\n🎉 Đã xuất thành công biểu đồ so sánh các kiến trúc tại: {output_path}")

    # In bảng tổng kết ra màn hình terminal
    print("\n" + "=" * 80)
    print(f"{'TÊN KIẾN TRÚC':<26} | {'THAM SỐ':<10} | {'BEST LOSS':<10} | {'BEST PPL':<10} | {'TỐC ĐỘ (tok/s)':<14}")
    print("-" * 80)
    for exp in experiments:
        name = exp.get("exp_name", "Unknown")[:25]
        params = f"{exp.get('total_params', 0)/1e6:.2f}M"
        loss = f"{exp.get('best_val_loss', 0):.4f}"
        ppl = f"{exp.get('best_val_ppl', 0):.2f}"
        hist = exp.get("history", [])
        tps_list = [h["tokens_per_sec"] for h in hist]
        avg_tps = f"{sum(tps_list)/len(tps_list):,.0f}" if tps_list else "N/A"
        print(f"{name:<26} | {params:<10} | {loss:<10} | {ppl:<10} | {avg_tps:<14}")
    print("=" * 80 + "\n")


def main():
    args = parse_args()
    experiments = load_experiments(args.files)
    plot_comparison(experiments, args.output)


if __name__ == "__main__":
    main()
