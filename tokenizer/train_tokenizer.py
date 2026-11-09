"""
Huấn luyện Byte-level BPE (BBPE) Tokenizer chuẩn công nghiệp (GPT-2, GPT-4, LLaMA style):
- Hoạt động trực tiếp trên 256 byte UTF-8 cơ sở (raw bytes 0x00 -> 0xFF)
- KHÔNG BAO GIỜ bị lỗi từ ngoài từ điển (<unk>) vì luôn có byte-fallback
- Giữ khoảng trắng dưới dạng ký tự byte (Ġ)
- Tách riêng từng chữ số 0-9 để tối ưu cho tư duy số học và suy luận CoT
- Tích hợp các special tokens: <pad>, <unk>, <bos>, <eos>, <think>, </think>
"""

import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Digits, Sequence
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

TOKENIZER_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(TOKENIZER_DIR, "..", "data", "raw")
PHASE1_FILE = os.path.join(RAW_DIR, "phase1_foundation.txt")
PHASE2_FILE = os.path.join(RAW_DIR, "phase2_annealing.txt")
SAVE_PATH = os.path.join(TOKENIZER_DIR, "tokenizer.json")

# Danh sách special tokens
SPECIAL_TOKENS = [
    "<pad>",     # ID: 0
    "<unk>",     # ID: 1 (rất hiếm khi dùng vì BBPE có byte-fallback)
    "<bos>",     # ID: 2 - Bắt đầu văn bản
    "<eos>",     # ID: 3 - Kết thúc văn bản
    "<think>",   # ID: 4 - Mở đầu suy luận Chain-of-Thought
    "</think>"   # ID: 5 - Kết thúc suy luận
]

VOCAB_SIZE = 2048


def train_byte_level_bpe():
    train_files = [PHASE1_FILE, PHASE2_FILE]
    print("=" * 65)
    print("BẮT ĐẦU HUẤN LUYỆN BYTE-LEVEL BPE (BBPE) TOKENIZER")
    print(f"Nguồn dữ liệu: {train_files}")
    print("=" * 65)

    # 1. Khởi tạo mô hình BPE cơ sở
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # 2. Pre-tokenizer chuẩn Byte-level:
    # - Digits: tách riêng từng chữ số '0'..'9' (hỗ trợ tính nhẩm số học)
    # - ByteLevel: chuyển đổi mọi ký tự Unicode về 256 bytes UTF-8 cơ bản
    tokenizer.pre_tokenizer = Sequence([
        Digits(individual_digits=True),
        ByteLevel(add_prefix_space=False, trim_offsets=True)
    ])

    # 3. Decoder chuẩn Byte-level: dịch các byte ngược lại thành ký tự gốc hoàn hảo
    tokenizer.decoder = ByteLevelDecoder()

    # 4. Trainer bắt đầu từ 256 bytes alphabet
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True
    )

    # Huấn luyện trên toàn bộ dữ liệu 2 giai đoạn
    tokenizer.train(files=train_files, trainer=trainer)

    # Lưu tokenizer ra file json
    tokenizer.save(SAVE_PATH)
    actual_vocab_size = tokenizer.get_vocab_size()

    print(f"\n✅ Đã lưu Byte-level BPE Tokenizer tại: {SAVE_PATH}")
    print(f"📖 Kích thước từ vựng thực tế: {actual_vocab_size}")

    # Kiểm tra thử nghiệm
    test_cases = [
        "def binary_search(arr: list[int], target: int) -> int:",
        "Question: Calculate (45 * 12) + 78.\n<think> Step 1: 45 * 12 = 540. </think> Answer: 618.",
        "Tiếng Việt: Mô hình học sâu giải toán và viết code."  # Test khả năng phân rã byte cho ký tự lạ
    ]

    print("\n--- KIỂM TRA MÃ HÓA VÀ GIẢI MÃ BYTE-LEVEL BPE ---")
    for tc in test_cases:
        enc = tokenizer.encode(tc)
        dec = tokenizer.decode(enc.ids)
        print(f"\n[Văn bản gốc]: {tc}")
        print(f"[Số tokens]:   {len(enc.ids)}")
        print(f"[Mảnh tokens]: {enc.tokens[:12]} ...")
        print(f"[Giải mã lại]: {dec}")
    print("=" * 65)


if __name__ == "__main__":
    train_byte_level_bpe()
