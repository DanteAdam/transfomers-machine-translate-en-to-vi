# Dịch máy Anh-Việt với Transformer

Dự án này huấn luyện và đánh giá mô hình Transformer dịch song ngữ Anh-Việt bằng bộ dữ liệu IWSLT15. Toàn bộ quy trình (tiền xử lý, huấn luyện, suy luận, trực quan hóa attention và tính BLEU) nằm trong notebook.

## Nội dung chính trong notebook
- Tiền xử lý: tải dữ liệu IWSLT15, tạo tokenizer BPE cho tiếng Anh và tiếng Việt, thêm token đặc biệt `<pad>`, `<bos>`, `<eos>`, `<unk>`.
- Mô hình: kiến trúc Transformer với positional encoding, multi-head attention, encoder/decoder layers và head tùy chỉnh.
- Huấn luyện: hàm train/evaluate, lịch giảm learning rate, tính loss cho train/validation và vẽ đường cong học.
- Suy luận: dịch câu bằng greedy search và beam search, áp dụng ràng buộc no-repeat n-gram.
- Đánh giá: tính BLEU trên tập test (có giới hạn mẫu), in kết quả và so sánh greedy/beam.
- Trực quan hóa: hiển thị ma trận attention encoder self-attention và decoder cross-attention.

## Yêu cầu môi trường
- Python 3.10+ (đề xuất dùng môi trường ảo).
- GPU CUDA nếu muốn huấn luyện nhanh hơn; CPU vẫn chạy nhưng chậm.
- Thư viện chính: `torch`, `torchvision`, `torchaudio`, `datasets`, `tokenizers`, `sacrebleu`, `matplotlib`, `numpy`, `tqdm`, `torchinfo`.

## Cách chạy nhanh
1) Cài phụ thuộc (có thể mất vài phút để tải dữ liệu/tokenizer):
```bash
pip install datasets sacrebleu matplotlib numpy torch torch-summary torchvision torchaudio torchinfo tokenizers
```
2) Mở notebook `transformer-attention-head-4.ipynb` trong VS Code hoặc Jupyter.
3) Chạy tuần tự các ô từ đầu đến cuối. Sau huấn luyện, xem biểu đồ loss, điểm BLEU và ví dụ dịch mẫu.

## Tùy chỉnh
- Điều chỉnh siêu tham số trong biến `hyperparameters` (d_model, num_heads, batch_size, num_epochs, max_len...).
- Đổi tùy chọn suy luận `option` giữa `greedy` và `beam` trong hàm `translate`.
- Thay đổi `beam_size` hoặc `no_repeat_ngram_size` để kiểm soát đa dạng câu dịch.

## Ghi chú dữ liệu
- Dataset: `Angelectronic/IWSLT15_English_Vietnamese` từ Hugging Face Datasets. Dữ liệu được tải và chia train/valid/test trực tiếp trong notebook.

## Kết quả kỳ vọng
- BLEU phụ thuộc vào seed, số epoch và tài nguyên; với thiết lập mặc định, BLEU thử nghiệm trên 200 mẫu được in ở cuối notebook.
