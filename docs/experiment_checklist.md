# Experiment checklist

## Trước khi chạy

- [ ] Mục tiêu và giả thuyết đã viết rõ.
- [ ] Base model, dataset, tokenizer và split đã cố định.
- [ ] Config có seed và experiment ID.
- [ ] Baseline đã chạy hoặc có lý do chưa chạy.
- [ ] Có giới hạn GPU time, memory và disk.

## Trong khi chạy

- [ ] Log loss, throughput, memory và learning rate.
- [ ] Lưu checkpoint theo quy tắc thống nhất.
- [ ] Ghi lại lỗi, restart và thay đổi ngoài dự kiến.
- [ ] Không sửa dữ liệu hoặc config giữa run mà không đổi experiment ID.

## Sau khi chạy

- [ ] Kiểm tra checkpoint load lại được.
- [ ] Chạy validation/test độc lập.
- [ ] Lưu bảng kết quả dạng máy đọc được.
- [ ] Vẽ biểu đồ từ log gốc.
- [ ] Ghi kết luận, giới hạn và thí nghiệm tiếp theo.

