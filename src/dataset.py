"""
DataLoader tải dữ liệu từ các file nhị phân (train.bin, val.bin).
Sử dụng kỹ thuật memory-mapping (np.memmap) để đọc dữ liệu tức thì từ ổ cứng,
không làm tốn bộ nhớ RAM và tối ưu hóa chuyển tensor lên GPU.
"""

import os
import numpy as np
import torch

class BinaryDataset:
    def __init__(self, data_path: str):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {data_path}")
        # Đọc dữ liệu bằng memmap (không tải toàn bộ file vào RAM)
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')

    def __len__(self):
        return len(self.data)


def get_batch(
    data: BinaryDataset,
    batch_size: int,
    seq_len: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bốc ngẫu nhiên một batch gồm batch_size chuỗi, mỗi chuỗi dài seq_len token.
    - x: chuỗi token đầu vào
    - y: chuỗi token mục tiêu (dịch phải 1 token so với x)
    """
    max_idx = len(data) - seq_len
    # Bốc ngẫu nhiên các chỉ số bắt đầu
    ix = np.random.randint(0, max_idx, size=batch_size)
    
    x = torch.stack([torch.from_numpy((data.data[i : i + seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data.data[i + 1 : i + 1 + seq_len]).astype(np.int64)) for i in ix])
    
    if "cuda" in device:
        # pin_memory giúp đẩy dữ liệu từ CPU lên GPU cực nhanh
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
        
    return x, y
