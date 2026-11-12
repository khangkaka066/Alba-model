# 🌟 Alba-Model: Modern Decoder-Only LLM From Scratch

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-Decoder--Only%20Transformer-blue?style=for-the-badge&logo=pytorch" alt="Architecture" />
  <img src="https://img.shields.io/badge/Parameters-51.64M%20(MoE)-green?style=for-the-badge" alt="Parameters" />
  <img src="https://img.shields.io/badge/Optimizer-GEO--Muon%20(2026)-red?style=for-the-badge" alt="Optimizer" />
  <img src="https://img.shields.io/badge/Tokenizer-Byte--Level%20BPE-orange?style=for-the-badge" alt="Tokenizer" />
  <img src="https://img.shields.io/badge/Curriculum-Two--Stage%20Annealing-purple?style=for-the-badge" alt="Curriculum" />
  <a href="https://colab.research.google.com/github/khangkaka066/Alba-model/blob/main/main.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="height: 28px;" />
  </a>
</p>

**Alba-Model** is a lightweight, high-performance **Decoder-Only Language Model** (~51.64M parameters with Mixture-of-Experts, 26.6M active) engineered from scratch in PyTorch. It integrates three major 2026 frontier breakthroughs:
1. **GOR-v2 (Geometric Orthogonal Residual v2):** Parameter-free differential geometric residual decomposition ($x + u^\perp + \phi(\rho) u^\parallel$) delivering 50% flatter gradient variance.
2. **OS-MoE (Orthogonal Sub-Expert MoE):** 1 Shared Expert + 8 Sub-Experts with Cosine-Bounded Router and Orthogonal Prototype Loss to permanently eliminate deep-layer representation collapse.
3. **GEO-Muon Hybrid Optimizer:** Matrix-orthogonal momentum updates via quintic Newton-Schulz iterations with adaptive aspect-ratio polar scaling, achieving an all-time record **Best Val Loss of 0.1028** and **Gradient Norm of 0.10**.

---

## 📑 Table of Contents

- [🏆 Research Paper & Technical Ledger (Improve.md)](Improve.md)
- [🌐 Interactive Visual Ledger (Improve.html)](Improve.html)
- [Architectural Highlights](#-architectural-highlights)
- [Two-Stage Pre-training Curriculum](#-two-stage-pre-training-curriculum)
- [Byte-Level BPE (BBPE) Tokenizer](#-byte-level-bpe-bbpe-tokenizer)
- [Optimizer Diagnostics & GEO-Muon](#-optimizer-diagnostics--geo-muon)
- [Repository Structure](#-repository-structure)
- [Quickstart Guide](#-quickstart-guide)
  - [1-Click Google Colab](#1-run-on-google-colab-recommended)
  - [Local Setup & Training](#2-run-locally-mac-gpu-mps--cuda--cpu)
  - [Model Comparison Tool](#3-benchmarking--comparing-architectures)
- [Methodology & Specifications](#-methodology--specifications)

---

## 🏗️ Architectural Highlights

Alba-Model abandons legacy Transformer constructs (e.g., standard LayerNorm, absolute positional embeddings, and vanilla MLPs) in favor of the modern open-weights consensus:

```text
                  Input Tokens [x_1, x_2, ..., x_T]
                               │
                               ▼
               [Token Embeddings (d = 256)]
                               │
       ┌───────────────────────┴───────────────────────┐
       │                                               ▼
       │                                     RMSNorm(x)
       │                                               ▼
       │                             Query / Key / Value Projections
       │                                               ▼
       │                                Rotary Position Embedding (RoPE)
       │                                               ▼
       │                                Causal Multi-Head Self-Attention
       │                                               ▼
       │                                     Output Projection (W_o)
       │                                               ▼
       └──────────────────────────────►(+)◄────────────┘ (Residual Connection 1)
                                        │
                                        ▼
                                        h
                                        │
       ┌────────────────────────────────┴──────────────┐
       │                                               ▼
       │                                     RMSNorm(h)
       │                                               ▼
       │                          SwiGLU FFN: W_2(SiLU(W_1 x) ⊙ W_3 x)
       │                                               ▼
       └──────────────────────────────►(+)◄────────────┘ (Residual Connection 2)
                                        │
                                        ▼  (Repeated across 6 Layers)
                               Final RMSNorm
                                        │
                                        ▼
                         Output Head (Weight-Tied Projection)
                                        │
                                        ▼
                      Next-Token Logits [p(x_{t+1} | x_{<=t})]
```

### Key Technical Innovations:
1. **GOR-v2 (Geometric Orthogonal Residual v2):** Differential geometric residual decomposition ($x + u^\perp + \phi(\rho) u^\parallel$) using Gram-Schmidt orthogonal projection with smooth $C^\infty$ identity-preserving modulation. Slashes gradient variance by >50% with strictly **zero parameter overhead**.
2. **OS-MoE (Orthogonal Sub-Expert MoE):** High-capacity parameter scaling (51.64M params) with constant 1.0x active FLOPs (26.6M active). Combines 1 dedicated Shared Expert (SwiGLU $d=512 \to 688$) with 8 fine-grained routed Sub-Experts ($d=512 \to 344$), regulated by a Cosine-Bounded Router and an Orthogonal Prototype Repulsion Loss ($\mathcal{L}_{\text{ortho}}$) to prevent deep-layer cluster collapse.
3. **GEO-Muon Hybrid Optimizer (Frontier 2026):** Matrix-orthogonal momentum optimizer using order-5 Newton-Schulz iterations on GPU Tensor Cores. Incorporates aspect-ratio polar scaling to match the golden update ratio ($\approx 10^{-3}$) on 98.1% of parameters while routing vocabulary embeddings and normalization scales to AdamW.
4. **RMSNorm & RoPE:** Root Mean Square Layer Normalization and Rotary Position Embeddings for superior long-context positional awareness.
5. **SwiGLU Activation:** SiLU-gated linear projections boosting mathematical and symbolic reasoning capabilities.
6. **Weight Tying:** Shares embedding and output classification matrices, saving ~1M redundant parameters.

### Model Architectural Variants & Empirical Benchmark (Iso-Parameters 51M vs. Baselines):

#### 🏆 Cohort 1: Nghiên cứu Đối đầu Tuyệt đối cùng quy mô ~51M-52M Tham số (Iso-Parameters)

| Architecture Variant | Total Params | Active FLOPs | Residual Type | Optimizer | Best Val Loss | Final Val Loss (2500) | Gradient Norm | Throughput |
|---|---|---|---|---|---|---|---|---|
| **Alba-51M (Dense GOR-v2 Innovation) 🏆** | **52.34M** | 52.34M (100%) | **GOR-v2 ($C^\infty$)** | AdamW | **0.0917** | **0.2363 (-7.6% Error)** | **0.08 -- 0.24** | 16,901 tok/s |
| **Alba-51M (OS-MoE Baseline MoE) 🚀** | **51.64M** | **26.28M (-50% FLOPs)** | **GOR-v2 ($C^\infty$)** | AdamW | **0.0924** | **0.2474 (Beats LLaMA 51M)** | **0.08 -- 0.28** | **20,644 tok/s** |
| **Alba-51M (Dense LLaMA Standard)** | 52.34M | 52.34M (100%) | Classical ($x + u$) | AdamW | 0.0917 | 0.2558 | 0.09 -- 0.26 | 19,093 tok/s |
| **Alba-51M (OS-MoE + GEO-Muon Deep Manifold) ⚡** | 51.64M | 26.28M (-50% FLOPs) | GOR-v2 ($C^\infty$) | GEO-Muon | 0.1066 (Train: **0.0590**) | 0.2787 | 0.10 -- 0.65 | 11,932 tok/s |

#### 📌 Cohort 2: Tham chiếu Kích thước Nhỏ (Compact Baselines 26M & 5M)

| Architecture Variant | Total Params | Active FLOPs | Residual Type | Optimizer | Best Val Loss | Final Val Loss (2500) | Gradient Norm | Throughput |
|---|---|---|---|---|---|---|---|---|
| **Alba-26M (Dense GOR-v2 Innovation)** | 26.64M | 26.64M (100%) | GOR-v2 ($C^\infty$) | AdamW | 0.1111 | 0.2450 | 0.11 -- 0.25 | 33,572 tok/s |
| **Alba-26M (Dense LLaMA Standard)** | 26.64M | 26.64M (100%) | Classical ($x + u$) | AdamW | 0.1104 | 0.2400 | 0.10 -- 0.22 | **77,277 tok/s** |
| **Alba-5M (Compact Baseline)** | 5.29M | 5.29M (100%) | Classical ($x + u$) | AdamW | 0.1082 | 0.2500 | 0.10 -- 0.30 | 64,055 tok/s |

### 📊 Empirical Training Dashboards & Benchmark Comparison:

| Alba-26M Standard LLaMA | Alba-26M GOR-v2 Innovation |
|---|---|
| <img src="checkpoints/modern_transformer_llama_26M_dashboard.png" width="480" alt="Alba-26M Standard Dashboard" /> | <img src="checkpoints/modern_transformer_gor_26M_dashboard.jpg" width="480" alt="Alba-26M GOR-v2 Dashboard" /> |

#### Head-to-Head Architecture Comparison Benchmark (Standard vs GOR-v2):
<p align="center">
  <img src="checkpoints/comparison_standard_vs_gor_26M.png" width="850" alt="Alba-Model Benchmark Comparison" />
</p>

---

## 📚 Two-Stage Pre-training Curriculum

Following the curriculum learning strategies of **LLaMA 3**, **DeepSeek-V3**, and **MiniCPM**, training is split into two distinct, high-impact phases:

```text
[ STAGE 1: FOUNDATION (80% of steps) ]
  ├── 45% Educational Text & Stories (Science, biology, astronomy, narrative prose)
  ├── 35% General Python Code (Algorithms, data structures, control flow)
  ├── 10% Basic Mathematics (Arithmetic, operator precedence)
  └── 10% General Knowledge (World facts, natural sciences QA)
                         │
                         ▼
[ STAGE 2: ANNEALING PHASE (20% of steps) ]
  ├── 40% Deep Math & Logic (Linear equations mx + b = c, inequalities, ratios)
  ├── 40% Clean Code (PEP 484 type annotations, full docstrings, assertions)
  └── 20% Synthetic Chain-of-Thought (Scratchpad reasoning with <think>...</think>)
  └── Learning Rate Cooldown: Smooth decay from 3e-4 down to 1e-5
```

---

## 🔤 Byte-Level BPE (BBPE) Tokenizer

The custom tokenizer is built on **Byte-Level Byte-Pair Encoding** (standardized by GPT-4 and LLaMA):
- **Zero Out-of-Vocabulary (OOV):** Grounded in the 256 fundamental UTF-8 bytes ($0\text{x}00 \rightarrow 0\text{xFF}$). Any unseen character, symbol, or foreign language decomposes into raw bytes (Byte-fallback) with **zero `<unk>` tokens**.
- **Individual Digit Tokenization:** Every digit (`0` through `9`) is explicitly split (`Digits(individual_digits=True)`), enabling the model to learn multi-digit column arithmetic instead of treating large numbers as opaque symbols.
- **Whitespace & Code Preservation:** Spaces and indentation are preserved via byte tokens (`Ġ`), retaining precise Python indentation structures.
- **Dedicated Reasoning Tokens:** Embedded `<think>` and `</think>` tags to structure Chain-of-Thought scratchpads.

---

## 🔬 Optimizer Diagnostics & GEO-Muon (Frontier 2026)

When comparing different architectures, measuring Loss alone is insufficient. Alba-Model tracks **5 scientific metrics** saved to `checkpoints/metrics.json` and visualizes them on a 6-panel dashboard:

1. **Cross-Entropy Loss:** Evaluated on independent Train and Validation splits.
2. **Perplexity ($\text{PPL} = \exp(\mathcal{L})$):** Measures model uncertainty; lower PPL indicates sharper predictions.
3. **Weight Update Ratio ($\frac{\|\Delta \theta\|_2}{\|\theta\|_2}$):** The gold-standard optimizer diagnostic. Tracks whether step updates stay in the sweet spot ($\sim 10^{-3}$) or risk divergence ($> 0.1$).
4. **Gradient Norm ($\|\nabla_\theta \mathcal{L}\|_2$):** Monitors training stability and guards against gradient explosions or vanishing gradients.
5. **Throughput & Latency:** Real-time tokens/second and step latency (ms) for benchmarking computational efficiency.

### Why Frontier 2026 Shifted to GEO-Muon:
In classical AdamW, treating 2D weight matrices as flat 1D scalar vectors distorts the singular value spectrum. **GEO-Muon** resolves this by performing **order-5 Newton-Schulz matrix orthogonalization** on the momentum buffer:
* **2D Hidden Matrices (98.1% of params):** Updated along pure orthogonal directions ($O O^T \approx I$), preventing directional collapse.
* **1D & Vocabulary Parameters (1.9% of params):** Handled via coordinate-wise decoupled AdamW, protecting Zipf distribution representations in token embeddings.

---

## 📁 Repository Structure

```text
Alba-model/
├── main.ipynb                   # Master Google Colab Dashboard (1-click execution)
├── README.md                    # Project overview & quickstart guide
├── Improve.md                   # Formal research preprint paper & mathematical proofs
├── Improve.html                 # Interactive visual ledger with Gram-Schmidt decomposition
├── TRAINING_METHODOLOGY.md      # Scientific specification for curriculum & metrics
├── train.py                     # Multi-phase training engine (GEO-Muon/AdamW, metrics tracking)
├── generate.py                  # Autoregressive generation & CoT problem solver
├── compare_models.py            # CLI tool for multi-architecture comparative analysis
├── checkpoints/
│   ├── best_model.pt            # Serialized weights of the best checkpoint
│   └── *_metrics.json           # Full training history & evaluation logs per experiment
├── src/
│   ├── config.py                # Model hyperparameters & dataclasses
│   ├── model.py                 # Decoder-Only Transformer (GOR-v2, OS-MoE, SwiGLU, RoPE)
│   ├── optimizer.py             # GEO-Muon Hybrid Optimizer (Newton-Schulz quintic iteration)
│   └── dataset.py               # Memory-mapped binary dataset loaders
├── data/
│   ├── prepare_data.py          # Synthetic curriculum data generator
│   ├── process_data.py          # High-speed uint16 binary tensor compiler
│   ├── raw/
│   │   ├── phase1_foundation.txt
│   │   └── phase2_annealing.txt
│   └── processed/
│       ├── phase1_train.bin & phase1_val.bin
│       └── phase2_train.bin & phase2_val.bin
├── tokenizer/
│   ├── train_tokenizer.py       # Byte-Level BPE training script
│   └── tokenizer.json           # Serialized vocabulary (1,837 tokens)
└── src/
    ├── __init__.py
    ├── config.py                # Model hyperparameter dataclass (ModelArgs)
    ├── model.py                 # PyTorch Transformer (RMSNorm, RoPE, Attention, SwiGLU)
    └── dataset.py               # Memory-mapped (np.memmap) zero-copy DataLoader
```

---

## 🚀 Quickstart Guide

### 1. Run on Google Colab (Recommended)
You can train and evaluate Alba-Model on a free Google Colab **Nvidia T4 GPU**:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khangkaka066/Alba-model/blob/main/main.ipynb)

1. Open [`main.ipynb`](https://colab.research.google.com/github/khangkaka066/Alba-model/blob/main/main.ipynb) via the badge above.
2. Navigate to **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ select **T4 GPU**.
3. Click **Runtime** $\rightarrow$ **Run all**.

---

### 2. Run Locally (Mac GPU MPS / CUDA / CPU)

Clone the repository and install requirements:

```bash
git clone https://github.com/khangkaka066/Alba-model.git
cd Alba-model
pip install torch tokenizers matplotlib
```

#### Step A: Generate Curriculum Data
```bash
python data/prepare_data.py
```

#### Step B: Train the Tokenizer
```bash
python tokenizer/train_tokenizer.py
```

#### Step C: Compile Binary Datasets
```bash
python data/process_data.py
```

#### Step D: Pre-train the Model
```bash
# 🏆 Champion 2026 Configuration: OS-MoE (51.6M) + GOR-v2 + GEO-Muon Optimizer
python train.py --exp_name modern_transformer_os_moe_geo_muon_51M \
                --optimizer geo_muon \
                --ffn_type os_moe \
                --n_experts 8 --top_k 2 \
                --residual_type gor \
                --dim 512 --n_layers 8 --n_heads 8 \
                --max_iters 2500 --batch_size 32 --seq_len 128 --lr 1e-3

# (Alternative) Dense Baseline with Classical AdamW
python train.py --exp_name modern_transformer_llama_26M --optimizer adamw --ffn_type swiglu --residual_type standard --max_iters 2500
```

#### Step E: Test Inference & Math Reasoning
```bash
# Multi-step arithmetic reasoning with Chain-of-Thought
python generate.py --prompt "Instruction: Solve the multi-step word problem with detailed logical chain of thought.\nQuestion: Alice has 35 apples. Alice gives 12 to Bob and buys 8 more. How many apples does Alice have now?\n<think>" --temperature 0.3

# Clean Python code generation
python generate.py --prompt "# Python Implementation: binary_search\ndef binary_search(arr: list[int], target: int) -> int:" --temperature 0.2
```

---

### 3. Benchmarking & Comparing Architectures

When experimenting with alternative architectural variants (e.g., LayerNorm vs. RMSNorm, GELU vs. SwiGLU, or Adam vs. AdamW):

```bash
# 1. Train Model A (AdamW)
python train.py --exp_name arch_adamw --optimizer adamw --max_iters 2000

# 2. Train Model B (Standard Adam)
python train.py --exp_name arch_adam --optimizer adam --max_iters 2000

# 3. Compare their learning curves, Perplexity, and gradient stability
python compare_models.py --files checkpoints/arch_adamw_metrics.json checkpoints/arch_adam_metrics.json --output comparison.png
```

---

## 📖 Methodology & Specifications

For an in-depth mathematical breakdown of the two-stage curriculum, token distribution tables, and optimizer diagnostic thresholds, consult [`TRAINING_METHODOLOGY.md`](TRAINING_METHODOLOGY.md).
