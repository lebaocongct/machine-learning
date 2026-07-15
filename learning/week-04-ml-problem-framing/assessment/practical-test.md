# Practical Test — Fraud Review Framing

**Thời gian:** 75 phút  
**Điểm:** 20  
**Dữ liệu:** `data/fraud_review.csv`, `data/fraud_feature_catalog.csv`  
**Không dùng:** thư mục `solutions/`, target để tạo feature, test để chọn threshold.

## Bối cảnh

Tại thời điểm authorization, hệ thống phải quyết định gửi giao dịch sang manual review hay cho đi qua. Positive target là `confirmed_fraud`; operational score hiện tại là `heuristic_risk_score`.

Giả định đào tạo:

- false positive cost = 8;
- false negative cost = 300;
- split chronological 70/15/15;
- threshold candidates = 0.05, 0.10, …, 0.95;
- tie-break = cost nhỏ nhất → recall cao hơn → threshold thấp hơn.

## Nhiệm vụ

Tạo `assessment/submission/` với các file sau.

### 1. `fraud-canvas.md` — 4 điểm

Viết ngắn gọn:

- non-ML goal, decision, action và owner;
- unit/time, target/window, output và task type;
- primary metric, baseline, constraint;
- hai slices, hai leakage risks, hai non-goals;
- kết luận Go / Conditional Go / No-Go.

### 2. `fraud-contract.json` — 2 điểm

Dùng schema trong Project 1. Contract phải qua `validate_prediction_contract` nếu chạy bằng code tham chiếu sau bài thi.

### 3. `fraud-feature-audit.csv` — 3 điểm

Audit catalog tại prediction time, cho phép offset tối đa `0.02` giờ. Mỗi risk có `feature_name`, `risk_type`, `reason`. Không dùng post-outcome feature trong bất kỳ phép tính score nào.

### 4. `fraud-evaluation.py` và outputs — 7 điểm

1. Sort theo `event_time`.
2. Chia train 70%, validation 15%, test 15%.
3. Báo prevalence train/validation/test và majority-class test baseline.
4. Trên validation, sweep thresholds; lưu `fraud-threshold-sweep.csv`.
5. Chọn threshold theo tie-break đã cho.
6. Khóa threshold; đánh giá test đúng một lần.
7. Lưu `fraud-test-metrics.json` gồm threshold, TP/FP/TN/FN, accuracy, precision, recall, F1, total_cost và `test_evaluation_count=1`.

Không cần huấn luyện model mới; nhiệm vụ đánh giá operational heuristic.

### 5. `fraud-slices.csv` — 2 điểm

Trên test tại threshold đã khóa, báo `n`, prevalence, precision, recall, predicted-positive rate và cost cho:

- mỗi giá trị `country_risk`;
- mỗi giá trị `card_present`.

Nếu denominator bằng 0, metric tương ứng là 0 và phải ghi chú limitation.

### 6. `fraud-decision.md` — 2 điểm

Viết tối đa 250 từ:

- heuristic có thắng majority baseline theo metric phù hợp không?
- threshold/cost trade-off là gì?
- slice nào cần điều tra?
- ba điều cần xác nhận trước production.

## Quy tắc test discipline

Trong code, đặt biến đếm hoặc comment rõ nơi test được mở. Không chạy loop lựa chọn nào trên test. Nếu phát hiện lỗi sau khi mở test, ghi incident và không tuyên bố đó là unbiased final estimate.

## Lệnh gợi ý

```bash
python assessment/submission/fraud-evaluation.py
```

## Critical fail

- Dùng `chargeback_amount`, `investigation_code` hoặc `analyst_notes_length` làm feature/decision input.
- Random split hoặc chọn threshold trên test.
- Không có decision/action hoặc operational/dummy baseline.
- Thay đổi cost/threshold grid sau khi xem test mà không khai báo.

