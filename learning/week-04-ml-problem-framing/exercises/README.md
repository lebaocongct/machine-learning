# 10 bài thực hành Tuần 4

Tạo thư mục `exercises/submission/`. Mỗi bài nộp đúng artifact được chỉ định. Thời lượng mục tiêu: 3–4 giờ, có thể trải trong lịch 10 giờ.

## Bài 1 — Vẽ bản đồ AI/ML/DL (15 phút)

Vẽ một sơ đồ có AI, ML, DL, supervised, unsupervised, generative và vị trí TensorFlow/Keras. Thêm:

- 2 ví dụ ML không phải DL;
- 2 ví dụ DL;
- 1 bài toán không cần ML.

**Nộp:** `01-map.md`.  
**Đạt:** quan hệ tập hợp đúng; không gọi TensorFlow là một loại mô hình.

## Bài 2 — Phân loại tám tình huống (25 phút)

Không đọc `data/dataset_metadata.json`. Đọc `data/scenario_cards.csv`, điền:

```text
scenario_id,ml_suitable,task_family,supervision,delivery_mode,reason
```

Sau đó cài `map_scenario_to_task` trong `exercises/starter.py` và tự tạo ít nhất 8 assertions.

**Nộp:** `02-scenarios.csv`, code và log assertions.  
**Đạt:** ≥7/8 task family khớp đáp án; S05 là rule engine, S07 chưa sẵn sàng vì không có action.

## Bài 3 — ML hay quy tắc? (20 phút)

Với bốn case sau, chọn ML / rule / human workflow / collect-data-first:

1. Tính VAT theo bảng luật công bố.
2. Dự báo demand theo SKU-tuần với 2 năm lịch sử.
3. Tự động từ chối hồ sơ y tế khi chưa có cơ chế kháng nghị.
4. Dashboard “khách nào buồn” nhưng không ai sở hữu action.

Mỗi case viết decision, lý do, rủi ro và điều kiện làm bạn đổi quyết định.

**Nộp:** `03-ml-suitability.md`.  
**Đạt:** không mặc định ML; có điều kiện thay đổi kết luận.

## Bài 4 — Frame bốn bài toán thật (45 phút)

Chọn bốn lĩnh vực khác nhau. Mỗi frame tối đa 200 từ, gồm:

- non-ML goal;
- decision/action/owner;
- unit/time;
- target/window/output;
- metric/baseline/constraint;
- một leakage risk;
- kết luận Go/No-Go.

**Nộp:** `04-four-frames.md`.  
**Đạt:** đủ 4 frame; ít nhất một frame kết luận non-ML hoặc No-Go có lý do.

## Bài 5 — Prediction contract (25 phút)

Sao chép template Project 1 và khóa contract cho support triage. Trả lời ba edge case:

1. Ticket bị reopen sau 72 giờ có đổi label không?
2. Escalation do outage toàn cục được tính thế nào?
3. Ticket thiếu lịch sử customer có được dự đoán không?

Chạy validator trong `solutions/problem_framing_solution.py` chỉ **sau khi** tự review.

**Nộp:** `05-prediction-contract.json` và `05-edge-cases.md`.  
**Đạt:** validator trả `[]`; cửa sổ và semantics không mơ hồ.

## Bài 6 — Leakage red-team (30 phút)

Đọc `data/support_feature_catalog.csv` và `data/support_triage.csv`:

1. Phân loại feature thành safe / unavailable / post-outcome / baseline-only.
2. Cài `audit_prediction_time`.
3. Giải thích tại sao correlation cao chưa đủ kết luận leakage.
4. Đề xuất cách rebuild một feature bị loại mà không nhìn tương lai.

**Nộp:** `06-feature-audit.csv`, code, `06-notes.md`.  
**Kết quả tối thiểu:** phát hiện `resolution_hours`, `final_satisfaction`, `manager_override`.

## Bài 7 — Baseline ladder (25 phút)

Thiết kế baseline ladder cho:

- Support classification: majority → operational heuristic.
- Delivery ETA regression: training median → last-route median.
- Search ranking: popularity → recency-weighted popularity.

Với mỗi baseline ghi input, metric, split và lý do công bằng khi so sánh.

**Nộp:** `07-baselines.md`.  
**Đạt:** ít nhất một dummy và một operational baseline cho mỗi case; không fit baseline bằng test.

## Bài 8 — Metric trade-off và chi phí (30 phút)

Cho hai operating points trên 1.000 ticket:

| Policy | TP | FP | TN | FN |
|---|---:|---:|---:|---:|
| A | 80 | 220 | 680 | 20 |
| B | 65 | 70 | 830 | 35 |

1. Tính accuracy, precision, recall, F1.
2. Tính cost khi FP=10, FN=200.
3. Chọn policy và viết hai trade-off.
4. Tìm tỷ lệ `FN_cost / FP_cost` làm hai policy hòa vốn.
5. Cài `expected_error_cost` và test input âm.

**Nộp:** `08-metrics.md` và code.  
**Đạt tham chiếu:** cost A = 6.200; cost B = 7.700; hòa vốn tại tỷ lệ 10.

## Bài 9 — Batch, online hay HITL? (20 phút)

Chọn mode và fallback cho:

1. Dự báo tồn kho mỗi đêm.
2. Chặn giao dịch trong 150 ms, analyst review vùng score trung gian.
3. Xếp ưu tiên lead mỗi 6 giờ, sales xác nhận.
4. Gợi ý nội dung nhưng bắt buộc legal approval.

Cài `recommend_delivery_mode`, viết rõ rule của bạn và 6 test cases.

**Nộp:** `09-delivery.md` và code.  
**Đạt:** mode phù hợp latency/human constraint; có timeout/fallback.

## Bài 10 — Project 1 Canvas + red-team defense (45 phút)

Hoàn thiện bộ Project 1. Sau đó tự đóng vai reviewer và đặt ít nhất 10 câu hỏi khó:

- Nếu target/proxy sai thì sao?
- Feature này tồn tại lúc nào?
- Ai hành động và capacity bao nhiêu?
- Model phải thắng baseline nào?
- Metric trung bình che slice nào?
- Nếu prediction service timeout thì sao?

Sửa canvas bằng màu/nhật ký thay đổi và thu defense 5 phút hoặc viết transcript.

**Nộp:** toàn bộ `project-1/submission/` và `10-red-team.md`.  
**Đạt:** không critical fail; đủ 8 trường bắt buộc của learning outcome.

## Tự chấm

Mỗi bài 2 điểm:

- 0: không nộp hoặc sai bản chất.
- 1: có artifact nhưng thiếu bằng chứng/giả định.
- 2: đúng, kiểm chứng được, nêu rủi ro/trade-off.

Tổng bài tập quy đổi 20 điểm trong rubric tuần.

