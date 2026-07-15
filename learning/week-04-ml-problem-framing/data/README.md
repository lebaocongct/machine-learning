# Datasets Tuần 4

Tất cả dữ liệu là synthetic và tái tạo bằng seed cố định:

```bash
python -m data.generate_datasets
```

## `scenario_cards.csv`

Tám tình huống dùng để quyết định: ML hay non-ML, supervised/unsupervised/generative, regression/classification/clustering/ranking và batch/online.

Không đọc expected task types trong metadata trước khi tự frame các tình huống.

## `support_triage.csv`

Case chính của guided lab/challenge: tại lúc ticket được tạo, quyết định đưa ticket vào specialist queue hay standard queue. Target là `escalated_within_48h`.

- `triage_risk_score` là heuristic baseline sẵn có, không phải feature của model mới.
- `resolution_hours`, `final_satisfaction`, `manager_override` là post-outcome leakage nếu dùng tại triage.
- Split theo `created_at`, không random split.

## `support_feature_catalog.csv`

Mô tả nguồn, thời điểm availability và ý định dùng feature. Ba cột post-outcome được đánh dấu `intended_for_model=True` có chủ đích để challenge phải phát hiện.

## `fraud_review.csv`

Dataset practical: tại authorization time, quyết định gửi giao dịch sang manual review hay cho đi qua. Target `confirmed_fraud` chỉ có sau điều tra.

- `heuristic_risk_score` là baseline score.
- `chargeback_amount`, `investigation_code`, `analyst_notes_length` là forbidden post-outcome fields.
- Error cost trong practical: false positive `8`, false negative `300`.

## `fraud_feature_catalog.csv`

Feature availability audit độc lập cho practical.

## Giới hạn

- Risk scores là synthetic heuristic, không phải hệ thống fraud/support thật.
- Chi phí sai lầm là giả định đào tạo; quyết định thực tế cần finance/operations/legal validation.
- Slice metrics mô phỏng không đủ để kết luận fairness.
- Không dùng dataset để suy diễn causal.
