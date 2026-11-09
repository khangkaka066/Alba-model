# 🌟 Alba-Model: Mini-LLM with Two-Stage Pre-training Curriculum

**Alba-Model** is a lightweight, modern **Decoder-Only Transformer** (~5.29M parameters) designed for learning, experimenting, and benchmarking language modeling and mathematical reasoning from scratch.

---

## 🏗️ Architecture Specifications (LLaMA / Mistral Style)

- **Architecture:** Modern Decoder-Only Autoregressive Transformer
- **Parameters:** ~5,290,496 (~5.29M params)
- **Embedding Dimension ($d_{\text{model}}$):** 256
- **Transformer Layers ($L$):** 6
- **Attention Heads ($H$):** 4 ($d_h = 64$)
- **Context Length ($T$):** 256 tokens
- **Normalization:** RMSNorm (Pre-normalization)
- **Position Embedding:** Rotary Positional Embedding (RoPE)
- **Feed-Forward Network:** SwiGLU FFN ($W_2(\text{SiLU}(W_1 x) \odot W_3 x)$)
- **Attention:** Scaled Dot-Product Attention with Causal Masking
- **Weight Tying:** Shared Token Embedding and Output Head projection
- **Tokenizer:** Canonical Byte-Level BPE (BBPE) with individual digit splitting (`1,837` vocab size)

---

## 📚 Two-Stage Pre-training Curriculum

Following modern LLM pre-training methodologies (LLaMA 3, DeepSeek-V3, MiniCPM):

| Phase | Steps Ratio | Objective & Data Composition |
|---|---|---|
| **Phase 1: Foundation** | 80% of steps | **45%** Educational Text & Stories, **35%** General Python Code, **10%** Basic Math, **10%** General Knowledge |
| **Phase 2: Annealing Phase** | 20% of steps | **40%** Deep Math & Logic ($mx + b = c$), **40%** Clean Code (Docstrings & Types), **20%** Synthetic CoT (`<think>`) with Learning Rate cooldown |

For full details, see [`TRAINING_METHODOLOGY.md`](TRAINING_METHODOLOGY.md).

---

## 🚀 Quickstart & Usage

### 1. Run on Google Colab
Open [`main.ipynb`](main.ipynb) in Google Colab, select **T4 GPU**, and click **Run All**.

### 2. Run Locally (Mac / Linux / Windows)

```bash
# 1. Generate synthetic curriculum data
python data/prepare_data.py

# 2. Train Byte-Level BPE Tokenizer
python tokenizer/train_tokenizer.py

# 3. Compile datasets into binary tensors (uint16)
python data/process_data.py

# 4. Train model (supports AdamW and Adam)
python train.py --exp_name alba_v1 --max_iters 2500 --batch_size 32 --seq_len 128 --lr 1e-3

# 5. Test inference and math problem solving
python generate.py --prompt "Instruction: Solve the multi-step word problem with detailed logical chain of thought.\nQuestion: Alice has 35 apples. Alice gives 12 to Bob and buys 8 more. How many apples does Alice have now?\n<think>" --temperature 0.3
```

### 3. Compare Multiple Model Architectures
```bash
python compare_models.py --files checkpoints/exp1_metrics.json checkpoints/exp2_metrics.json --output comparison.png
```

---

## 📁 Repository Structure

```text
LLM-model/
├── main.ipynb                   # Master Google Colab Dashboard
├── TRAINING_METHODOLOGY.md      # Detailed curriculum & metrics specifications
├── train.py                     # Multi-phase training script (AdamW/Adam, Cosine LR, Metrics)
├── generate.py                  # Text generation & chain-of-thought solver
├── compare_models.py            # Comparative evaluation & multi-model benchmarking
├── data/
│   ├── prepare_data.py          # Synthetic data generator
│   ├── process_data.py          # Binary tokenizer processor
│   ├── raw/                     # Raw text data
│   └── processed/               # Binary uint16 tensor files
├── tokenizer/
│   ├── train_tokenizer.py       # Byte-level BPE trainer
│   └── tokenizer.json           # Trained vocabulary
└── src/
    ├── config.py                # Model hyperparameter dataclass
    ├── model.py                 # PyTorch Transformer implementation
    └── dataset.py               # Memory-mapped DataLoader
```
