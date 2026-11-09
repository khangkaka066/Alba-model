# File plan — học xây dựng LLM từ đầu

Không cần tạo toàn bộ file ngay từ đầu. Tạo theo milestone; mỗi file mới phải có test hoặc experiment chứng minh nó hoạt động.

## 0. Project và environment

Tạo ở thư mục gốc:

- `pyproject.toml`: dependency và package metadata.
- `.gitignore`: loại checkpoint, log, dataset lớn.
- `Makefile`: các lệnh test, lint, smoke test.

Tạo trong `configs/`:

- `base.yaml`: cấu hình chung.
- `tiny_debug.yaml`: model nhỏ để debug.
- `lfm_inspired_300m.yaml`: cấu hình LFM-inspired khoảng 0,3B.
- `transformer_control.yaml`: cấu hình Transformer decoder để kiểm tra các thành phần Transformer lõi.
- `pretrain.yaml`: cấu hình train chính.
- `benchmark.yaml`: cấu hình benchmark.
- `structured_decision.yaml`: schema, decoding mode, runtime và metric cho benchmark nghiên cứu.

Tạo trong `scripts/`:

- `check_env.py`: kiểm tra Python, PyTorch, CUDA, GPU.
- `count_parameters.py`: đếm tham số thực tế.
- `inspect_checkpoint.py`: kiểm tra checkpoint.

## 1. Dữ liệu

Tạo trong `data/`:

- `README.md`: nguồn và cách tạo dataset.
- `data_card.md`: license, ngôn ngữ, thống kê và giới hạn.
- `manifests/sources.jsonl`: danh sách nguồn dữ liệu.
- `manifests/splits.jsonl`: train/validation/test split.
- `processed/README.md`: quy ước shard và checksum.

Tạo trong `scripts/`:

- `download_data.py`
- `normalize_text.py`
- `filter_documents.py`
- `deduplicate_data.py`
- `split_data.py`
- `build_shards.py`
- `inspect_dataset.py`

Mục tiêu: làm sạch dữ liệu, deduplicate, tránh leakage và tạo shard tái lập.

## 2. Tokenizer tự viết

Tạo trong `tokenizer/src/`:

- `vocab.py`
- `normalizer.py`
- `pretokenizer.py`
- `bpe.py`
- `tokenizer.py`
- `encode.py`
- `decode.py`
- `train_tokenizer.py`

Tạo trong `tokenizer/tests/`:

- `test_normalizer.py`
- `test_bpe.py`
- `test_encode_decode.py`
- `test_special_tokens.py`

Tạo trong `tokenizer/checkpoints/`:

- `lfm_inspired_300m_tokenizer.json`
- `lfm_inspired_300m_vocab.json`

Mục tiêu: hiểu vocabulary, merge rule, special token và ảnh hưởng của tokenizer lên độ dài chuỗi.

## 3. Kiến trúc LFM-inspired

Tạo trong `model/src/`:

- `config.py`: hyperparameter và layer layout.
- `embeddings.py`: token embedding.
- `rmsnorm.py`: RMSNorm.
- `rope.py`: rotary position embedding.
- `swiglu.py`: gated feed-forward network.
- `short_conv.py`: double-gated short convolution.
- `gqa.py`: Grouped Query Attention.
- `cache.py`: KV cache và convolution cache.
- `block.py`: operator + residual + SwiGLU.
- `model.py`: stack các block.
- `lm_head.py`: dự đoán logits.
- `loss.py`: causal language-modeling loss.
- `decision_heads.py`: action, slot, confidence và abstain heads cho proposed parallel decision system.
- `decision_schema.py`: ánh xạ output head sang schema có kiểm tra dependency/validity.

Tạo trong `model/tests/`:

- `test_shapes.py`
- `test_causal_mask.py`
- `test_rope.py`
- `test_short_conv.py`
- `test_gqa.py`
- `test_cache.py`
- `test_lm_loss.py`
- `test_parameter_count.py`
- `test_decision_heads.py`
- `test_decision_schema.py`

Mục tiêu: hiểu tensor shape, causal masking, residual stream, KV cache và cách ghép một LLM.

## 4. Data loader và pretraining

Tạo trong `training/pretrain/`:

- `dataset.py`: đọc shard và token sequence.
- `sampler.py`: tạo batch và sequence sampling.
- `collator.py`: padding/packing/labels.
- `optimizer.py`: optimizer.
- `scheduler.py`: learning-rate schedule.
- `mixed_precision.py`: bf16/fp16 và gradient scaler nếu cần.
- `checkpointing.py`: lưu và resume checkpoint.
- `train_step.py`: một bước forward/backward/update.
- `train.py`: training loop.
- `evaluate.py`: validation loss và perplexity.

Tạo trong `tests/integration/`:

- `test_tiny_training.py`
- `test_resume_training.py`
- `test_distributed_smoke.py`

Mục tiêu: next-token prediction, gradient accumulation, mixed precision, checkpoint resume và distributed training.

## 5. Triton kernel và benchmark

Tạo trong `kernels/triton/`:

- `gqa_forward.py`
- `gqa_backward.py`
- `causal_mask.py`
- `kernel_utils.py`

Tạo trong `kernels/tests/`:

- `test_gqa_correctness.py`
- `test_gqa_gradients.py`
- `test_edge_shapes.py`

Tạo trong `benchmarks/attention/`:

- `benchmark_pytorch.py`
- `benchmark_triton.py`
- `compare_correctness.py`

Tạo trong `benchmarks/multigpu/`:

- `run_gpu_matrix.py`
- `collect_hardware_info.py`

Mục tiêu: đo correctness, latency, throughput, peak memory, forward/backward và train step.

## 6. Scaling law

Tạo trong `training/scaling_laws/`:

- `run_sweep.py`
- `collect_metrics.py`
- `fit_loss_law.py`
- `predict_scale.py`
- `plot_scaling.py`
- `report.md`

Tạo trong `configs/`: `scaling_tiny.yaml`, `scaling_small.yaml`, `scaling_medium.yaml`.

Mục tiêu: nghiên cứu quan hệ giữa số tham số, số token, compute và validation loss.

## 7. SFT, DPO và RLVR

Tạo trong `alignment/sft/`: `prepare_data.py`, `train_sft.py`, `evaluate_sft.py`.

Tạo trong `alignment/dpo/`: `prepare_preferences.py`, `train_dpo.py`, `evaluate_dpo.py`.

Tạo trong `alignment/rlvr/`: `tasks.py`, `verifier.py`, `reward.py`, `train_rlvr.py`, `evaluate_rlvr.py`.

Tạo trong `alignment/evaluation/`: `common_suite.py`, `compare_methods.py`, `failure_analysis.py`.

Mục tiêu: so sánh công bằng bằng cách bắt đầu từ cùng base checkpoint và cùng evaluation suite.

## 8. Structured-decision benchmark

Tạo trong `research/structured_decision/`:

- `README.md`: câu hỏi nghiên cứu, phạm vi claim và cách tái lập.
- `protocol.md`: hardware, precision, quantization, context length, decoding policy và seed.
- `task_schema.json`: canonical schema của action/slots/confidence/abstain.
- `data_card.md`: nguồn, nhãn, split và quy tắc chống leakage.
- `baseline_registry.yaml`: checkpoint, revision, license, tokenizer, runtime và cách adaptation cho 6 model.
- `results/README.md`: quy ước đặt tên raw outputs, aggregate metrics và failure cases.

Tạo trong `alignment/sft/`:

- `prepare_decision_data.py`: tạo target canonical AR-JSON và label cho parallel heads.
- `train_ar_json.py`: SFT/adaptation cho baseline AR-JSON.
- `train_parallel_heads.py`: train proposed decision heads từ checkpoint LFM-inspired 300M.

Tạo trong `benchmarks/structured_decision/`:

- `run_parallel_heads.py`: inference proposed system và deterministic serialization.
- `run_ar_json.py`: inference AR-JSON thường và constrained decoding.
- `run_model_matrix.py`: chạy sáu model với protocol chung.
- `measure_latency.py`: P50/P95 end-to-end time-to-decision, RAM và throughput.
- `score_decisions.py`: full decision accuracy, slot F1, tool-execution success và schema validity.
- `score_calibration.py`: ECE, Brier, NLL, risk-coverage và abstain metrics.
- `analyze_failures.py`: ambiguity, negation, OOD, missing slot và state-conflict analysis.

Tạo trong `configs/`:

- `decision_parallel_lfm_300m.yaml`
- `decision_ar_lfm2_350m.yaml`
- `decision_ar_qwen3_0_6b.yaml`
- `decision_ar_smollm2_135m.yaml`
- `decision_ar_mobilellm_125m.yaml`
- `decision_ar_gemma3_270m.yaml`

Mục tiêu: chứng minh hoặc bác bỏ trên cùng protocol rằng output head song song có Pareto latency, accuracy và calibration tốt hơn AR-JSON cho structured edge decisions.

## 9. Agent và long-context

Tạo trong `agent/environment/`: `base.py`, `tasks.py`, `tools.py`, `mock_world.py`.

Tạo trong `agent/harness/`: `episode.py`, `runner.py`, `replay.py`, `schema.py`.

Tạo trong `agent/evaluation/`: `agent_metrics.py`, `long_context.py`, `self_evaluation.py`, `independent_judge.py`.

Tạo trong `agent/prompts/`: `system.txt`, `tool_use.txt`, `self_eval.txt`.

Mục tiêu: tool call, trajectory, replay, context dài và kiểm tra độ đáng tin của self-evaluation.

## 10. Demo và report

Tạo trong `demo/`: `README.md`, `run_demo.py`, `sample_inputs.jsonl`.

Tạo trong `reports/neurips/`: `main.tex`, `references.bib`, `appendix.tex`, `reproducibility.md`.

Tạo trong `reports/figures/` và `reports/tables/`: `README.md`.

## Thứ tự tạo file tối thiểu

1. Environment, config debug và `scripts/check_env.py`.
2. Tokenizer cùng test tokenizer.
3. Các module Transformer/LFM-inspired và test shape/causal mask.
4. Overfit một batch và kiểm tra số tham số.
5. Data loader, training loop và validation.
6. Triton kernel sau khi PyTorch baseline đúng.
7. Scaling law sau khi có nhiều run.
8. SFT/DPO/RLVR sau khi base model ổn định.
9. Structured-decision benchmark sau SFT: proposed parallel LFM 300M so với năm AR baselines.
10. Agent harness, demo và report.
