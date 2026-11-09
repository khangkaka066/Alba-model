"""
Chuyển đổi dữ liệu 2 giai đoạn (Phase 1 và Phase 2) thành file nhị phân:
- Phase 1: phase1_train.bin, phase1_val.bin (80% tổng bước train)
- Phase 2: phase2_train.bin, phase2_val.bin (20% Annealing phase)
Dạng dữ liệu: uint16 (2 bytes / token), tối ưu bộ nhớ và tốc độ nạp GPU.
"""

import os
import numpy as np
from tokenizers import Tokenizer

DATA_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
TOKENIZER_FILE = os.path.join(DATA_DIR, "..", "tokenizer", "tokenizer.json")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def process_single_phase(phase_name: str, raw_filename: str, val_ratio: float = 0.1):
    raw_path = os.path.join(RAW_DIR, raw_filename)
    train_bin = os.path.join(PROCESSED_DIR, f"{phase_name}_train.bin")
    val_bin = os.path.join(PROCESSED_DIR, f"{phase_name}_val.bin")
    
    print(f"\n--- Đang xử lý: {phase_name.upper()} ({raw_filename}) ---")
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    
    with open(raw_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    tokens = tokenizer.encode(text).ids
    n = len(tokens)
    split_idx = int(n * (1 - val_ratio))
    
    train_tokens = np.array(tokens[:split_idx], dtype=np.uint16)
    val_tokens = np.array(tokens[split_idx:], dtype=np.uint16)
    
    train_tokens.tofile(train_bin)
    val_tokens.tofile(val_bin)
    
    print(f"Tổng số token: {n:,}")
    print(f"  - Train: {len(train_tokens):,} tokens ➔ {train_bin}")
    print(f"  - Val:   {len(val_tokens):,} tokens ➔ {val_bin}")


def main():
    print("=" * 65)
    print("ĐÓNG GÓI NHỊ PHÂN DỮ LIỆU ĐA GIAI ĐOẠN")
    print("=" * 65)
    process_single_phase("phase1", "phase1_foundation.txt", val_ratio=0.1)
    process_single_phase("phase2", "phase2_annealing.txt", val_ratio=0.1)
    print("=" * 65)
    print("🎉 Hoàn tất đóng gói dữ liệu cả 2 giai đoạn!")


if __name__ == "__main__":
    main()
