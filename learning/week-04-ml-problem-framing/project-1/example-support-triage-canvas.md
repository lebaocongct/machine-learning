# ML Problem Canvas — Support ticket specialist routing

**Phiên bản:** 1.0  
**Trạng thái:** Ví dụ tham chiếu

## A. Mục tiêu và decision

- **Problem name:** Early support escalation triage.
- **Non-ML goal:** Giảm escalation không được chuẩn bị trước mà không vượt specialist capacity.
- **Decision:** Ngay sau khi parse tin nhắn đầu tiên, chọn specialist queue hay standard queue cho từng ticket.
- **Decision owner:** Support Operations Manager.
- **Action positive:** specialist queue và ưu tiên review.
- **Action negative:** standard queue; vẫn áp dụng SLA/fallback hiện tại.
- **Phương án non-ML:** rule theo outage, billing keyword và customer tier; đây cũng là operational baseline.

## B. Prediction contract

- **Prediction unit:** Một support ticket mới.
- **Prediction time:** Sau initial message parsing và trước queue assignment; cho phép feature hoàn tất trong 0.1 giờ.
- **Target:** `escalated_within_48h`.
- **Target definition:** 1 nếu có manager-confirmed escalation từ 0 đến 48 giờ sau khi ticket được tạo.
- **Target window:** [0, 48] giờ.
- **Task type:** classification.
- **Model output:** Xác suất ticket escalation trong 48 giờ, khoảng [0,1].
- **Deployment mode:** human_in_loop.
- **Latency/capacity:** 500 ms; specialist queue không quá 20% ticket/ngày.

## C. Đo lường

- **Primary metric:** Tổng chi phí `10×FP + 200×FN` tại threshold chọn trên validation.
- **Guardrail:** recall ≥ 0.80 và predicted-positive rate ≤ 0.20, cần xác nhận với Operations.
- **Baseline:** Majority-class dummy và `triage_risk_score` heuristic hiện tại.
- **Success criterion:** Trên chronological holdout, giảm tổng chi phí ít nhất 15% so với heuristic trong khi đáp ứng guardrail; xác nhận bằng shadow run 4 tuần.
- **Trade-off 1:** Tăng recall giảm escalation bỏ sót nhưng tiêu hao specialist capacity.
- **Trade-off 2:** Tăng precision giảm review không cần thiết nhưng có thể tăng FN với chi phí cao.

## D. Dữ liệu và rủi ro

- **Constraints:** 500 ms; 20% specialist capacity; không dùng text nhạy cảm chưa được phê duyệt; phải có fallback.
- **Evaluation slices:** channel; customer tier; outage-active; ticket mới so với customer có lịch sử.
- **Leakage risks:** `resolution_hours`, `final_satisfaction`, `manager_override`; aggregate vượt prediction time; label policy đổi.
- **Non-goals:** Tự động đóng ticket; đánh giá hiệu suất agent; suy luận quan hệ nhân quả; thay thế manager escalation policy.
- **Fallback:** Khi feature thiếu/timeout, dùng heuristic hiện tại và log reason.

## E. Protocol và quyết định

- Sort theo `created_at`; train/validation/test = 70/15/15.
- Chọn threshold bằng validation; test mở đúng một lần sau khi khóa policy.
- **Kết luận:** Conditional Go — cần xác nhận cost/capacity và chạy shadow test trước production.

### Bản đồ field cho validator

```python
canvas = {
    "problem_name": "Early support escalation triage",
    "non_ml_goal": "Reduce unprepared escalations without exceeding specialist capacity",
    "decision": "Route each new ticket to specialist or standard queue",
    "decision_owner": "Support Operations Manager",
    "prediction_unit": "one newly created support ticket",
    "prediction_time": "after initial parsing and before queue assignment",
    "target": "escalated_within_48h",
    "target_window": "0 to 48 hours after ticket creation",
    "task_type": "classification",
    "model_output": "probability of escalation within 48 hours",
    "primary_metric": "10*FP + 200*FN on a chronological holdout",
    "baseline": "majority class and current triage_risk_score heuristic",
    "success_criterion": "15% lower cost than heuristic, recall >= 0.80, and predicted-positive rate <= 0.20",
    "deployment_mode": "human_in_loop",
    "constraints": ["500 ms latency", "specialist queue <= 20% per day"],
    "evaluation_slices": ["channel", "customer_tier", "outage_active"],
    "leakage_risks": ["post-outcome fields", "future aggregates"],
    "non_goals": ["automatic ticket closure", "agent performance scoring"],
}
```
