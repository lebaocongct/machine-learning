# Đáp án Quiz Tuần 4

| Câu | Đáp án | Giải thích ngắn |
|---:|:---:|---|
| 1 | B | DL ⊂ ML ⊂ AI; không phải mọi AI đều học từ dữ liệu. |
| 2 | C | Output là số liên tục có nhãn. |
| 3 | A | Không có nhãn/nhóm định trước. |
| 4 | B | Luật đầy đủ, ổn định và xác định. |
| 5 | C | Prediction chưa dẫn đến decision/action có owner. |
| 6 | B | Một prediction gắn với một unit cụ thể. |
| 7 | C | Chỉ biết sau khi ticket được xử lý. |
| 8 | B | Khóa semantics và maturity của label. |
| 9 | C | Proxy cần limitations và kiểm chứng với outcome thật. |
| 10 | B | Mô phỏng triển khai từ quá khứ sang tương lai. |
| 11 | B | Majority baseline cũng đạt 96% nhưng recall positive bằng 0. |
| 12 | B | Precision điều kiện trên predicted positive. |
| 13 | A | Bỏ sót đắt nên thường bảo vệ recall, trong constraint. |
| 14 | B | Validation dùng cho lựa chọn; test dùng ước lượng cuối. |
| 15 | B | Có cả dummy và quy trình vận hành hiện tại. |
| 16 | B | Cập nhật theo lịch, không cần low latency. |
| 17 | D | Human review không tự tạo owner, sửa data leakage hay loại automation bias/capacity. |
| 18 | B | Slice bắt nguồn từ harm/risk và vận hành. |

**19.** `TotalCost = 8×FP + 300×FN = 8×12 + 300×3 = 996`.

**20.** Test phải đại diện dữ liệu chưa thấy để ước lượng cuối cùng; dùng nó để đổi threshold làm rò rỉ thông tin lựa chọn và khiến kết quả lạc quan. Nếu đã làm vậy, đánh dấu test cũ là validation/exploratory, khóa lại policy và đánh giá trên một holdout mới độc lập (hoặc thu dữ liệu tương lai).

