# Đáp án Practical — Fraud Review

## Canvas tham chiếu

- **Non-ML goal:** giảm tổn thất fraud trong khi giữ manual-review load và customer friction ở mức chấp nhận được.
- **Decision/action:** tại authorization, Fraud Operations chọn gửi manual review hoặc cho giao dịch đi qua; timeout dùng heuristic/fail-safe policy đã phê duyệt.
- **Unit/time:** một transaction, sau khi thu request/lookup snapshot và trước authorization response.
- **Target/window:** `confirmed_fraud=1` nếu điều tra xác nhận fraud trong 720 giờ sau authorization; binary classification probability/risk score.
- **Primary metric:** `8×FP + 300×FN`; guardrail gồm review rate/capacity, recall và latency.
- **Baselines:** majority-class dummy và `heuristic_risk_score` hiện hành.
- **Constraints:** authorization latency; analyst capacity; privacy/legal; appeal; fallback.
- **Slices:** `country_risk`, `card_present`; thêm account age/amount bands ở review thật.
- **Leakage:** chargeback amount, investigation code, analyst notes; aggregate nhìn qua event time.
- **Non-goals:** tự động cáo buộc người dùng; suy luận causal; thay thế analyst; đánh giá đạo đức/fairness chỉ từ synthetic data.
- **Kết luận:** Conditional Go cho shadow evaluation; chưa production-ready vì cost/capacity là giả định và validation/test positive counts nhỏ.

## Prediction contract

```json
{
  "unit_of_prediction": "one transaction authorization request",
  "prediction_time": "after request and approved snapshot lookups, before authorization response",
  "target_name": "confirmed_fraud",
  "target_definition": "1 if investigation confirms fraud within 720 hours after authorization; otherwise 0 after label maturity",
  "target_window_start_hours": 0,
  "target_window_end_hours": 720,
  "output_type": "binary",
  "latency_sla_ms": 150,
  "owner": "Fraud Operations Manager"
}
```

## Feature audit

| Feature | Risk | Quyết định |
|---|---|---|
| `chargeback_amount` | post-outcome | Drop; chỉ có sau chargeback |
| `investigation_code` | label-generation/post-outcome | Drop; gần như tiết lộ label |
| `analyst_notes_length` | post-outcome | Drop; được tạo trong review |

`heuristic_risk_score` là baseline-only và có tại +0.02h nên hợp lệ để đánh giá operational policy, nhưng không dùng như feature của “model mới” trong practical.

## Kết quả và memo

Code tại `practical_test_solution.py` tái tạo các file. Threshold 0.05 được chọn **chỉ trên validation**. Test cost là 1,324 so với majority negative cost 3,000; vì vậy heuristic tạo giá trị theo cost giả định dù accuracy 0.585 thấp hơn dummy accuracy 0.926. Đây là ví dụ vì sao accuracy không phải primary metric khi class hiếm và FN đắt.

Policy dự đoán positive cho 60/135 giao dịch test, tức 44.4%; cần so với analyst capacity. Slice `country_risk=1` có review rate 83.3% nhưng chỉ 18 record; đây là cảnh báo cần thêm dữ liệu và governance, không phải kết luận fairness. Trước production phải xác nhận FP/FN cost với Finance/Fraud Ops, latency/capacity/fallback với Engineering/Ops, và legal/fairness/appeal với Risk/Legal; sau đó shadow run trên dữ liệu tương lai.

