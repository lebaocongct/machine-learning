# Tuần 4 — Machine Learning Problem Framing

Gói học liệu thực hành 10 giờ giúp chuyển một nhu cầu nghiệp vụ thành bài toán ML có thể đo, kiểm thử và triển khai. Case xuyên suốt là **phân luồng ticket hỗ trợ**; bài kiểm tra thực hành dùng case **duyệt giao dịch nghi ngờ gian lận**.

> Tuần này chưa tối ưu mô hình phức tạp. Trọng tâm là quyết định đúng bài toán, khóa prediction contract, đo baseline, chọn metric/threshold có chi phí và loại bỏ leakage trước khi huấn luyện. Vì vậy gói cố ý chưa cài TensorFlow; TensorFlow/Keras bắt đầu sau khi bài toán qua framing gate.

## Kết quả đầu ra

Sau tuần này, người học có thể:

1. Phân biệt AI, ML, Deep Learning; supervised, unsupervised, generative; classification, regression, clustering, ranking.
2. Quyết định khi nào nên dùng ML, luật xác định hoặc quy trình thủ công.
3. Viết ML Problem Canvas có decision, owner, unit/time, target/window, output, metric, baseline, constraint, slice và leakage risk.
4. Viết prediction contract máy đọc được và audit feature availability tại prediction time.
5. Thiết lập dummy/heuristic baseline; giải thích trade-off precision–recall và chi phí FP/FN.
6. Chia train/validation/test theo thời gian, chọn threshold trên validation và mở test đúng một lần.
7. Bảo vệ framing trong 5 phút bằng bằng chứng thay vì trực giác.

## Cấu trúc gói

| Thành phần | Mục đích |
|---|---|
| `01-knowledge.md` | Giáo trình cốt lõi và ví dụ |
| `02-reading-list.md` | Nguồn đọc chính thức, câu hỏi đọc |
| `03-schedule.md` | Lịch 10 giờ theo phiên, checkpoint |
| `notebooks/` | Hai guided lab có assertion và đầu ra |
| `exercises/` | 10 bài luyện tập và starter code |
| `challenge/` | Coding challenge 12 test tự động |
| `project-1/` | Canvas, contract, feature catalog cho Project 1 |
| `assessment/` | Quiz, practical test, defense và rubric |
| `solutions/` | Đáp án, code chuẩn và kết quả tham chiếu |
| `data/` | Dữ liệu synthetic, metadata và generator |
| `report-template.md` | Mẫu báo cáo nộp cuối tuần |
| `VALIDATION.md` | Nhật ký kiểm thử của gói học liệu |

## Cài đặt nhanh

Python 3.11+ được khuyến nghị.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Chạy challenge:

```bash
python -m pytest tests/test_problem_framing.py -q
```

Tái tạo dữ liệu:

```bash
python -m data.generate_datasets
```

## Trình tự học khuyến nghị

1. Đọc `01-knowledge.md`, làm checkpoint không nhìn đáp án.
2. Làm notebook 01 và hoàn thiện bốn problem frame.
3. Làm bài 1–6 trong `exercises/README.md`.
4. Làm notebook 02, ghi rõ metric trade-off và threshold policy.
5. Hoàn thiện `challenge/submission.py` cho đến khi 12 test pass.
6. Nộp Project 1 Canvas + prediction contract + feature catalog.
7. Làm quiz 20 câu và practical test 75 phút.
8. Tự chấm bằng `assessment/rubric.md`, sau đó mới xem `solutions/`.

## Definition of Done

- [ ] Hai notebook chạy từ đầu đến cuối không lỗi.
- [ ] 10 bài tập có bằng chứng/đầu ra.
- [ ] 12/12 public tests pass; không import `solutions`.
- [ ] Canvas có đủ decision, target, unit/time, metric, baseline, constraint, slices và leakage risk.
- [ ] Prediction contract hợp lệ; feature audit không để lọt post-outcome field.
- [ ] Threshold được chọn chỉ bằng validation; test được đánh giá một lần.
- [ ] Quiz đạt ít nhất 14/20 và tổng rubric đạt ít nhất 70/100.
- [ ] Không có critical fail trong rubric.

## Quy ước học thuật

- Dữ liệu trong gói là synthetic; không dùng để kết luận về hệ thống support/fraud thật.
- Ghi rõ giả định, chi phí lỗi và nguồn của mọi target/proxy.
- Không dùng đáp án trước khi nộp bản đầu tiên.
- Kết quả slice nhỏ chỉ là tín hiệu cần điều tra, không tự động chứng minh fairness.
