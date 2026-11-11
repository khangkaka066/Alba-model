"""
Script sinh văn bản và giải toán suy luận (Inference):
- Tải checkpoint trọng số mô hình đã huấn luyện
- Sinh văn bản tự hồi quy (Autoregressive Generation) kèm Temperature và Top-K sampling
- Hỗ trợ chế độ chạy thử prompt đơn lẻ hoặc chế độ tương tác (interactive loop)
"""

import os
import argparse
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from src.config import ModelArgs
from src.model import MiniLLM

BASE_DIR = os.path.dirname(__file__)
DEFAULT_CKPT = os.path.join(BASE_DIR, "checkpoints", "best_model.pt")
DEFAULT_TOKENIZER = os.path.join(BASE_DIR, "tokenizer", "tokenizer.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Sinh văn bản từ Mini-LLM")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT, help="Đường dẫn file checkpoint")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER, help="Đường dẫn file tokenizer")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt đầu vào")
    parser.add_argument("--max_new_tokens", type=int, default=120, help="Số token tối đa sinh thêm")
    parser.add_argument("--temperature", type=float, default=0.5, help="Độ sáng tạo / ngẫu nhiên")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K token có xác suất cao nhất")
    parser.add_argument("--interactive", action="store_true", help="Chế độ nhập prompt liên tục")
    return parser.parse_args()


@torch.no_grad()
def generate_text(model, tokenizer, prompt, device, max_new_tokens=120, temperature=0.5, top_k=40):
    model.eval()
    encoded = tokenizer.encode(prompt).ids
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    eos_id = tokenizer.token_to_id("<eos>")

    for _ in range(max_new_tokens):
        # Giới hạn context length bằng max_seq_len của mô hình
        idx_cond = idx if idx.size(1) <= model.args.max_seq_len else idx[:, -model.args.max_seq_len:]
        logits, _ = model(idx_cond)
        
        # Lấy logits của token cuối cùng và chia cho temperature
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        # Top-K filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        # Softmax để lấy phân phối xác suất
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

        # Dừng sớm nếu sinh ra thẻ kết thúc <eos>
        if next_token.item() == eos_id:
            break

    out_tokens = idx[0].tolist()
    return tokenizer.decode(out_tokens)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    if not os.path.exists(args.checkpoint):
        print(f"❌ Chưa tìm thấy file checkpoint: {args.checkpoint}")
        print("💡 Bạn hãy chạy 'python train.py' trước để huấn luyện mô hình!")
        return

    # Tải tokenizer và checkpoint
    tokenizer = Tokenizer.from_file(args.tokenizer)
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except Exception:
        ckpt = torch.load(args.checkpoint, map_location=device)
    
    raw_args = ckpt["model_args"]
    if isinstance(raw_args, dict) and "residual_type" not in raw_args:
        raw_args["residual_type"] = "standard"
    model_args = ModelArgs(**raw_args) if isinstance(raw_args, dict) else raw_args
    
    model = MiniLLM(model_args).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"✅ Đã nạp thành công checkpoint từ bước {ckpt.get('step', '?')} (Val Loss: {ckpt.get('val_loss', '?'):.4f})")

    if args.interactive:
        print("\n💬 CHẾ ĐỘ TƯƠNG TÁC (Gõ 'exit' để thoát)")
        while True:
            prompt = input("\nNhập Prompt > ")
            if prompt.strip().lower() in ["exit", "quit"]:
                break
            # Nếu là câu hỏi toán, tự động thêm thẻ mở suy luận nếu người dùng chưa gõ
            if "Question:" in prompt and "<think>" not in prompt:
                prompt = prompt.strip() + "\n<think>"
            output = generate_text(
                model, tokenizer, prompt, device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )
            print("\n--- KẾT QUẢ SINH ---")
            print(output)
    else:
        # Chạy prompt mẫu nếu không nhập
        prompt = args.prompt
        if prompt is None:
            prompt = "Question: Alice has 25 apples. Alice gives 7 apples to Bob and buys 5 more apples. How many apples does Alice have now?\n<think>"
        
        print(f"\nPrompt đầu vào:\n{prompt}")
        output = generate_text(
            model, tokenizer, prompt, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        print("\n--- KẾT QUẢ SINH ---")
        print(output)


if __name__ == "__main__":
    main()
