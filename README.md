# LLM From Scratch — Alba-Model Project

Project này được tổ chức theo pipeline: dữ liệu → tokenizer tự viết (Byte-Level BPE) → decoder-only LLM (RMSNorm, RoPE, Attention, SwiGLU) tự train → scaling law → SFT/DPO/RLVR → structured decision benchmark → agent context dài → report/demo.

---

## 🌟 Alba-Model: Triển khai Hiện tại (Executable Implementation)

Hệ thống mã nguồn hoàn chỉnh đã được xây dựng và kiểm thử thành công:
- **Kiến trúc:** Modern Decoder-Only Autoregressive Transformer (~5.29M parameters)
- **Chuẩn thiết kế:** LLaMA 3 / Mistral (RMSNorm Pre-norm, Rotary Positional Embedding RoPE, Causal Attention với SDPA, SwiGLU FFN, Weight Tying)
- **Tokenizer:** Canonical Byte-Level BPE (BBPE) với 1,837 tokens, bảo vệ thẻ `<think>`, tách riêng chữ số `0-9` để hỗ trợ tính toán và suy luận logic
- **Chiến lược Huấn luyện 2 Giai đoạn (Two-Stage Curriculum):**
  - **Phase 1 (Nền tảng - 80% số bước):** 45% Edu Text, 35% Python Code, 10% Math, 10% General Knowledge
  - **Phase 2 (Annealing - 20% số bước):** 40% Deep Math & Logic, 40% Clean Code (docstring & typing), 20% Synthetic CoT kèm hạ nhiệt Learning Rate về $10^{-5}$
- **Bảng điều khiển Colab:** [`main.ipynb`](main.ipynb) — điều khiển toàn bộ pipeline chỉ với 1 click.
- **Hệ thống Đo lường Thí nghiệm:** Tự động ghi nhận Cross-Entropy Loss, Perplexity (PPL), Gradient Norm, Tốc độ (tokens/s), và Adam Update Ratio (`checkpoints/metrics.json`, `checkpoints/training_dashboard.png`).
- **So sánh Mô hình:** Công cụ [`compare_models.py`](compare_models.py) để đối chiếu nhiều kiến trúc khi nghiên cứu.

Chi tiết xem tại [`TRAINING_METHODOLOGY.md`](TRAINING_METHODOLOGY.md).

---

## 🚀 Hướng dẫn Chạy nhanh

### 1. Chạy trên Google Colab
Mở file [`main.ipynb`](main.ipynb) trên Google Colab, chọn **T4 GPU**, và bấm **Run All**.

### 2. Chạy trên máy nội bộ (Local Mac / Linux / Windows)

```bash
# 1. Sinh dữ liệu 2 giai đoạn theo phương pháp
python data/prepare_data.py

# 2. Huấn luyện Byte-Level BPE Tokenizer
python tokenizer/train_tokenizer.py

# 3. Đóng gói dữ liệu sang tensor nhị phân (uint16)
python data/process_data.py

# 4. Huấn luyện mô hình (hỗ trợ AdamW và Adam)
python train.py --exp_name alba_v1 --max_iters 2500 --batch_size 32 --seq_len 128 --lr 1e-3

# 5. Thử nghiệm suy luận và giải toán
python generate.py --prompt "Instruction: Solve the multi-step word problem with detailed logical chain of thought.\nQuestion: Alice has 35 apples. Alice gives 12 to Bob and buys 8 more. How many apples does Alice have now?\n<think>" --temperature 0.3

# 6. So sánh các kiến trúc mô hình khác nhau
python compare_models.py --files checkpoints/alba_v1_metrics.json checkpoints/other_metrics.json --output comparison.png
```

---

## 🎯 Mục tiêu học phần và mục tiêu nghiên cứu gốc

Mục tiêu bắt buộc là tự tay viết tokenizer và các thành phần LLM/Transformer, sau đó train một model decoder-only LFM-inspired khoảng 0.3B. Model này là trọng tâm để học data pipeline, pretraining, kernel, alignment và agent; không thay bằng checkpoint có sẵn.

Trên model tự xây, dự án có thêm một nhánh nghiên cứu edge AI:

> Can a 300M LFM-inspired model with parallel calibrated decision heads outperform LFM2-350M and Qwen3-0.6B autoregressive JSON generation in latency, structured-decision accuracy, and calibration for on-device tasks?

Benchmark gồm sáu model:

| Vai trò | Model |
| --- | --- |
| Proposed system | LFM-inspired 300M tự train + parallel `action`, `slot`, `confidence`, `abstain` heads |
| AR baseline gần kiến trúc | LFM2-350M |
| AR Transformer baseline | Qwen3-0.6B |
| Small-model reference | SmolLM2-135M |
| Small-model reference | MobileLLM-125M |
| Small-model reference | Gemma 3 270M |

---

## 📁 Cấu trúc thư mục

```text
├── main.ipynb                   # Master Google Colab Dashboard
├── TRAINING_METHODOLOGY.md      # Tài liệu chi tiết phương pháp & metrics
├── train.py                     # Script huấn luyện 2 giai đoạn (AdamW/Adam, Cosine LR)
├── generate.py                  # Script sinh chữ & giải toán CoT
├── compare_models.py            # Công cụ đối chiếu & so sánh các kiến trúc
├── data/
│   ├── prepare_data.py          # Script sinh dữ liệu 2 giai đoạn
│   ├── process_data.py          # Script đóng gói nhị phân uint16
│   ├── raw/                     # File text thô
│   └── processed/               # File nhị phân train.bin, val.bin
├── tokenizer/
│   ├── train_tokenizer.py       # Script huấn luyện Byte-Level BPE
│   └── tokenizer.json           # File từ điển 1,837 tokens
└── src/
    ├── config.py                # Siêu tham số ModelArgs
    ├── model.py                 # PyTorch Transformer (RMSNorm, RoPE, Attention, SwiGLU)
    └── dataset.py               # Memory-mapped DataLoader
```

Xem kế hoạch chi tiết tại [`docs/roadmap.md`](docs/roadmap.md).
