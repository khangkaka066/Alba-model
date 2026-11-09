# Roadmap và pipeline thực hiện

Danh sách file cần tự tạo cho từng folder nằm ở [`file_plan.md`](file_plan.md). Roadmap này tập trung vào mục tiêu học và điều kiện hoàn thành của từng giai đoạn.

## 0. Chuẩn hóa mục tiêu, protocol và tiêu chí đánh giá

### Việc cần làm

- Chốt định nghĩa model LFM-inspired 0.3B: số layer, hidden size, số head, context length, vocabulary size, tỷ lệ short-convolution/GQA và số tham số thực tế.
- Chốt tập dữ liệu, ngôn ngữ, license, train/validation/test split và tiêu chí loại bỏ dữ liệu.
- Chốt GPU, số GPU, precision, batch size, gradient accumulation và thời lượng ngân sách train.
- Chốt baseline kỹ thuật: PyTorch attention chuẩn, optimizer chuẩn và một model nhỏ để smoke test.
- Chốt metric cho từng phần: loss/perplexity, throughput, latency, memory, scaling-law fit, reward, pass rate của agent.
- Chốt research question: “Can a 300M LFM-inspired model with parallel calibrated decision heads outperform LFM2-350M and Qwen3-0.6B autoregressive JSON generation in latency, structured-decision accuracy, and calibration for on-device tasks?”
- Chốt six-model benchmark: LFM-inspired 300M, LFM2-350M, Qwen3-0.6B, SmolLM2-135M, MobileLLM-125M và Gemma 3 270M.
- Định nghĩa rõ hai nhóm kết luận: benchmark chính là proposed 300M với LFM2/Qwen; ba model còn lại là small-model reference, không dùng để suy luận nhân quả về kiến trúc.

### Điều kiện hoàn thành

- Có một tài liệu experiment protocol.
- Mọi thí nghiệm đều có seed, config, hardware và version dữ liệu.
- Có schema nhiệm vụ, train/validation/test split riêng cho structured decision và chính sách không leakage từ teacher/test set.
- Có bảng phân công cho 2–3 thành viên.

## 1. Environment và repository

### Việc cần làm

- Tạo environment cố định cho Python, PyTorch, CUDA, Triton và các thư viện đánh giá.
- Kiểm tra GPU, CUDA visibility, NCCL, mixed precision và distributed launch.
- Chuẩn hóa logging, checkpoint naming, config loading và experiment ID.
- Tạo smoke test chạy trên CPU hoặc một GPU nhỏ trước khi chạy thật.

### Điều kiện hoàn thành

- Chạy được một forward/backward nhỏ.
- Chạy được distributed hello-world trên số GPU mục tiêu.
- Có log môi trường và lệnh tái hiện.

## 2. Dữ liệu thô và data cleaning

### Việc cần làm

- Thu thập dữ liệu hợp pháp, lưu manifest nguồn và license.
- Chuẩn hóa encoding, Unicode, whitespace, ký tự đặc biệt và ngôn ngữ.
- Loại bỏ tài liệu rỗng, quá ngắn, lỗi định dạng và nội dung ngoài phạm vi.
- Deduplicate theo exact match và near-duplicate.
- Tách train/validation/test trước khi tạo shard để tránh leakage.
- Ghi lại thống kê: số tài liệu, số token ước lượng, độ dài, ngôn ngữ, tỷ lệ loại bỏ.

### Điều kiện hoàn thành

- Có raw/interim/processed tách biệt.
- Có data card và cleaning report.
- Có checksum hoặc version ID cho dataset.

## 3. Tokenizer tự viết

### Việc cần làm

- Chọn mục tiêu tokenizer: BPE, unigram hoặc WordPiece; ghi rõ lý do.
- Tự viết training algorithm, vocabulary builder, encode/decode và special tokens.
- Đo coverage, compression ratio, fertility, độ dài chuỗi và tốc độ encode.
- Đóng băng vocabulary sau khi train; không thay đổi tokenizer giữa các thí nghiệm chính.
- Viết test round-trip, test Unicode, test special token và test batch encode.

### Điều kiện hoàn thành

- Tokenizer tái lập từ manifest và seed.
- Có benchmark so với một tokenizer tham khảo nhưng không dùng implementation đó làm lõi.
- Có version tokenizer gắn với checkpoint.

## 4. Decoder LLM LFM-inspired 0.3B từ đầu

### Việc cần làm

- Tự triển khai embedding, positional encoding, causal self-attention/GQA, MLP, normalization, residual và LM head; đây là phần Transformer lõi bắt buộc của học phần.
- Tự triển khai hybrid LFM-inspired: double-gated short convolution xen kẽ với GQA. Ghi rõ layer layout để phân biệt với Transformer thuần.
- Chọn initialization, activation, normalization placement và weight tying.
- Kiểm tra causal mask, shape, dtype, gradient flow và loss shift.
- Viết unit test cho từng module và numerical check với tensor nhỏ.
- Chạy overfit trên một batch, sau đó chạy toy corpus trước khi pretrain.
- Train model khoảng 0.3B bằng data pipeline đã version hóa.

### Điều kiện hoàn thành

- Overfit test đạt ngưỡng đặt trước.
- Loss giảm ổn định trên train và validation.
- Có checkpoint tốt nhất, checkpoint cuối và training curve.
- Có bảng tham số, FLOPs ước tính và cấu hình train.
- Có architecture card: layer layout, số tham số theo module, context length và KV/convolution cache.

## 5. Triton attention kernel và benchmark

### Việc cần làm

- Xác định API và shape contract của kernel.
- Viết kernel forward attention trước; sau đó bổ sung backward nếu cần cho training.
- Xử lý causal mask, sequence length, head dimension, dtype và các trường hợp biên.
- Kiểm tra correctness bằng sai số tuyệt đối/tương đối với PyTorch baseline.
- Benchmark riêng forward, backward, training step và inference latency.
- Đo throughput, peak memory, kernel time, scaling theo batch/sequence/head dimension.
- Chạy trên nhiều GPU và ghi rõ model GPU, driver, CUDA, Triton, clock và power mode.

### Điều kiện hoàn thành

- Kernel đạt ngưỡng sai số đã định.
- Có benchmark tự động và biểu đồ.
- Có phân tích trường hợp kernel nhanh hơn/chậm hơn baseline.
- Có quyết định rõ kernel nào được dùng trong train/inference chính.

## 6. Scaling law và dự đoán khi scale model

### Việc cần làm

- Chọn vài quy mô model, số token và ngân sách compute khác nhau.
- Giữ protocol nhất quán giữa các run.
- Thu thập validation loss, training loss, số token, số tham số, FLOPs và wall-clock.
- Fit các dạng scaling law đã chọn; tách train/fit và hold-out để kiểm tra dự đoán.
- Dự đoán loss hoặc chất lượng ở quy mô lớn hơn mà chưa train đầy đủ.
- Phân tích sai số dự đoán và giới hạn ngoại suy.

### Điều kiện hoàn thành

- Có bảng run đầy đủ và mã hóa experiment ID.
- Có fit curve, confidence interval hoặc bootstrap uncertainty.
- Có ít nhất một dự đoán hold-out và đánh giá độ chính xác.

## 7. Alignment trên cùng một base model

### 7.1 SFT

- Chuẩn bị instruction-response dataset và format thống nhất.
- Định nghĩa loss masking cho prompt/response.
- Fine-tune từ cùng một base checkpoint.
- Đánh giá capability, format following, helpfulness và failure cases.

### 7.2 DPO

- Tạo preference pairs trên cùng task distribution.
- Kiểm tra chất lượng chosen/rejected và tránh leakage.
- Giữ base/reference model và hyperparameter protocol rõ ràng.
- Đánh giá cùng bộ test với SFT.

### 7.3 RLVR

- Chọn task có verifier tự động được.
- Tách generator, reward/verifier và policy update.
- Kiểm tra reward hacking, format hacking và over-optimization.
- So sánh với base, SFT và DPO trên cùng test set.

### Điều kiện hoàn thành

- Mọi phương pháp bắt đầu từ cùng base model.
- Có cùng evaluation suite, seed protocol và budget report.
- Có bảng trade-off chất lượng, chi phí, độ ổn định và lỗi.

## 8. Structured decision: parallel heads so với AR-JSON

### Mục tiêu nghiên cứu

Đánh giá liệu LFM-inspired 300M với các decision head chạy song song có nhanh hơn và đáng tin hơn autoregressive JSON generation trên tác vụ edge có schema cố định.

### Systems cần đánh giá

| Nhóm | System | Output |
| --- | --- | --- |
| Proposed | LFM-inspired 300M tự train | parallel action, slot, confidence, abstain heads; chương trình serialize JSON |
| Primary AR baseline | LFM2-350M | canonical AR-JSON, ưu tiên schema/grammar constraint |
| Primary AR baseline | Qwen3-0.6B | canonical AR-JSON, ưu tiên schema/grammar constraint |
| Reference AR baseline | SmolLM2-135M | canonical AR-JSON |
| Reference AR baseline | MobileLLM-125M | canonical AR-JSON |
| Reference AR baseline | Gemma 3 270M | canonical AR-JSON |

### Việc cần làm

- Chọn một miền edge hẹp, có thể verifier tự động: ví dụ smart-home/stateful automation tiếng Việt.
- Chốt canonical schema: `action`, `slots`, `confidence`, `abstain`, `rationale_optional`; serializer của parallel heads không được che giấu lỗi semantic.
- Tạo test set có paraphrase, phủ định, thiếu slot, state conflict, ambiguity và OOD; test set phải bị khóa trước SFT/DPO/RLVR.
- Fine-tune hoặc adapter-tune các AR baseline trên cùng decision-training set và target JSON; nếu chỉ đánh giá zero-shot, ghi rõ đó là external reference chứ không phải so sánh công bằng.
- Train proposed decision heads từ cùng base LFM-inspired checkpoint; dùng calibration objective và abstention/rejection objective nếu phù hợp.
- So sánh AR-JSON thường với AR-JSON + grammar/schema constraint để tránh baseline yếu do lỗi cú pháp.
- Đo trên cùng laptop/runtime/quantization/context length: end-to-end time-to-decision, P50/P95 latency, peak RAM, throughput và năng lượng nếu đo được.
- Đo chất lượng: full decision accuracy, slot F1, tool-execution success, schema validity, ECE, Brier score, NLL, risk-coverage và abstain precision/recall.
- Phân tích tách riêng: lợi ích của output interface; chênh lệch giữa backbone/training data; và Pareto capability-latency-memory.

### Điều kiện hoàn thành

- Có protocol cố định, public task schema và evaluation script tái lập.
- Có bảng chính cho LFM-inspired 300M, LFM2-350M, Qwen3-0.6B; bảng phụ cho SmolLM2, MobileLLM và Gemma.
- Có ablation AR unconstrained, AR constrained và proposed parallel-head system.
- Kết luận chỉ giới hạn ở domain decision đã chọn; không suy rộng thành năng lực tổng quát hơn mọi model lớn.

## 9. Agent environment và harness

### Việc cần làm

- Định nghĩa task, observation, action, tool interface, termination và timeout.
- Viết environment deterministic trước, sau đó mới thêm task khó hơn.
- Tạo harness để chạy batch episode, retry, timeout, logging và replay.
- Chuẩn hóa trajectory schema: prompt, action, tool call, observation, reward, final answer.
- Tạo evaluator độc lập với agent.
- Thiết kế long-context test: retrieval, memory, multi-step reasoning và distractor injection.
- Thử self-evaluation; kiểm tra mức tương quan giữa tự đánh giá và đánh giá độc lập.

### Điều kiện hoàn thành

- Có thể replay một episode từ log.
- Có test chống reward leakage và prompt leakage.
- Có bảng agent performance theo context length.
- Có phân tích trường hợp agent tự đánh giá sai.

## 10. Report NeurIPS và demo

### Report nên có

- Abstract, introduction, related work, method, experimental setup.
- Data and tokenizer, model, Triton kernel, scaling law, alignment, structured-decision study, agent.
- Results, ablation, limitations, ethics/safety, reproducibility và appendix.
- Tất cả hình/bảng đều truy ngược được về experiment ID.

### Demo trực tiếp

- Một lệnh khởi động demo.
- Một model checkpoint cố định.
- Một bộ input dự phòng nếu GPU/network có vấn đề.
- Hiển thị latency, context length, tool calls và kết quả agent.
- Cho phép chuyển giữa parallel decision output và AR-JSON để minh họa trade-off latency/reliability.
- Có video hoặc log dự phòng cho phần không thể chạy live.

## Gợi ý phân công nhóm 2–3 người

### Nhóm 2 người

- Thành viên A: data, tokenizer, Transformer, pretraining, scaling law.
- Thành viên B: Triton kernel, benchmark, alignment, agent, evaluation.
- Cả hai: protocol, review chéo, report và demo.

### Nhóm 3 người

- Thành viên A: data, tokenizer, Transformer và pretraining.
- Thành viên B: Triton, multi-GPU benchmark và scaling law.
- Thành viên C: SFT/DPO/RLVR, agent, harness và evaluation.
- Cả nhóm: thiết kế thí nghiệm, review chéo, report và demo.

## Quy tắc checkpoint và thí nghiệm

Mỗi run nên lưu tối thiểu: `experiment_id`, git commit, config, dataset version, tokenizer version, seed, hardware, command, metrics, checkpoint path và ghi chú thất bại.
