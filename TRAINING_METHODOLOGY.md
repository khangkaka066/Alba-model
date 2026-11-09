# Two-Stage Pre-training Curriculum, Byte-Level BPE & Evaluation Methodology

> This document defines the exact training strategy, curriculum data distribution, tokenizer architecture, and evaluation metrics for the model. As new methods or pipeline updates are developed, this document is updated/overwritten to serve as the single source of truth.

---

## 1. High-Level Training Roadmap

The pre-training process is structured into **two sequential stages** (Curriculum Learning):

```text
[ STAGE 1: FOUNDATION (80% of total training steps) ]
    │  Proportions: 45% Edu Text | 35% Code | 10% Math | 10% General Knowledge
    │  Objective: Build syntax, linguistic grammar, foundational knowledge, general code logic
    ▼
[ STAGE 2: ANNEALING PHASE (20% of total training steps) ]
       Proportions: 40% Math & Logic | 40% Clean Code | 20% Instruction / Synthetic CoT
       Objective: Unlock deep reasoning capabilities, decay learning rate down to zero
```

---

## 2. Phase-by-Phase Breakdown

| Phase | Data Objective | Ratio / Strategy | Active Corpus |
|---|---|---|---|
| **Phase 1: Foundation**<br>*(80% of steps)* | Build grammar, foundational knowledge, general code logic. | Balanced Mixture:<br>• **45%** Educational Text & Stories<br>• **35%** General Code (Python algorithms)<br>• **10%** Basic Arithmetic Math<br>• **10%** General Knowledge | `data/raw/phase1_foundation.txt`<br>➔ `phase1_train.bin` (246,446 tokens) |
| **Phase 2: Annealing Phase**<br>*(20% of steps)* | Breakthrough in reasoning ability, shape high-quality outputs. | High-Quality Annealing Mixture:<br>• **40%** Deep Math & Logic ($mx + b = c$, ratios, inequalities)<br>• **40%** Clean Code (Docstrings & Typing)<br>• **20%** Instruction / Synthetic CoT (`<think>`) | `data/raw/phase2_annealing.txt`<br>➔ `phase2_train.bin` (168,104 tokens) |

---

## 3. Tokenizer Architecture: Byte-Level BPE (BBPE)

The project utilizes a canonical **Byte-Level Byte-Pair Encoding (BBPE)** tokenizer (standardized by GPT-2, GPT-4, and LLaMA):

1. **256 Base UTF-8 Bytes:**
   - Starts directly with the 256 fundamental byte values ($0\text{x}00 \rightarrow 0\text{xFF}$).
   - **Zero Out-of-Vocabulary (OOV):** Any character, emoji, or unseen foreign word (e.g., Vietnamese with diacritics) is decomposed into raw bytes if not merged, completely eliminating `<unk>` failures.
2. **Individual Digit Tokenization:**
   - Digits (`0` through `9`) are strictly separated into independent tokens (`Digits(individual_digits=True)`).
   - Enables the LLM to learn column-by-column arithmetic algorithms.
3. **Byte-Level Whitespace & Code Preservation:**
   - Preserves indentation, newlines, and spaces via byte representations (`Ġ`), making it ideal for programming languages and mathematical scratchpads.
4. **Special Control Tokens:**
   - `<pad>` (0), `<unk>` (1), `<bos>` (2), `<eos>` (3), `<think>` (4), `</think>` (5).
5. **Vocabulary Size:**
   - Set to `1,837` tokens.

---

## 4. Metrics & Evaluation Dashboard (Experiment Tracking)

To systematically compare different model architectures (e.g., RMSNorm vs. LayerNorm, RoPE vs. ALiBi, SwiGLU vs. GELU), each training run automatically collects and records 5 primary scientific metrics:

| Metric | Formula / Definition | Significance in LLM Research |
|---|---|---|
| **Cross-Entropy Loss** | $\mathcal{L} = -\frac{1}{T}\sum \log p(x_t \mid x_{<t})$ | Evaluates raw next-token prediction error on Train and Validation splits. |
| **Perplexity (PPL)** | $\text{PPL} = \exp(\mathcal{L})$ | **Gold standard metric:** Measures average branching factor / uncertainty. Lower means higher language mastery. |
| **Gradient Norm** | $\|\nabla_\theta \mathcal{L}\|_2$ | Measures training stability; detects gradient explosion ($>10$) or vanishing ($ \to 0$) when altering layer components. |
| **Learning Rate** | $\eta_t$ | Verifies proper warm-up, cosine decay, and annealing schedule down to $10^{-5}$. |
| **Throughput & Latency** | Tokens/sec and ms/step | Benchmarks computational efficiency and wall-clock execution cost across architectures. |

### Output Artifacts:
- **`checkpoints/<exp_name>_metrics.json`**: Complete structured JSON history of all evaluation steps, architecture parameters, and timestamps.
- **`checkpoints/training_dashboard.png`**: High-resolution 6-panel visualization dashboard automatically rendered upon completion.
- **`compare_models.py`**: Comparative analysis CLI tool capable of overlaying metrics from $N$ different models onto a unified benchmarking graph.

---

## 5. Active Data File Inventory

| Phase | Raw Text Corpus | Processed Binary Tensors | Active Status |
|---|---|---|---|
| **Phase 1** | `data/raw/phase1_foundation.txt` | `data/processed/phase1_train.bin`<br>`data/processed/phase1_val.bin` | ✅ Active (Foundation) |
| **Phase 2** | `data/raw/phase2_annealing.txt` | `data/processed/phase2_train.bin`<br>`data/processed/phase2_val.bin` | ✅ Active (Annealing) |
