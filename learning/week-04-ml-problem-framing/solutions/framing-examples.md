# Đáp án gợi ý — Problem framing và bài tự luận

Không có một câu chữ duy nhất đúng; đáp án đạt khi decision, contract và metric nhất quán.

## Frame 1 — Support triage

- Goal: giảm escalation không chuẩn bị mà không quá tải specialist.
- Decision/action: lúc ticket mới đến, route specialist hoặc standard queue; owner là Support Operations.
- Unit/time: một ticket, sau parse tin đầu và trước queue assignment.
- Target/output: escalation được xác nhận trong 48h; binary probability.
- Metric/baseline: `10×FP + 200×FN`; majority + heuristic hiện tại.
- Constraint/risk: specialist ≤20%; leakage từ resolution/satisfaction/override.
- Kết luận: Conditional Go, cần xác nhận cost/capacity và shadow test.

## Frame 2 — Delivery ETA

- Goal: giảm promise miss và tăng minh bạch checkout.
- Decision/action: hiển thị ETA và chọn promise band.
- Unit/time: một order tại checkout confirmation.
- Target: số giờ đến delivery completed, window tối đa 14 ngày; regression.
- Metric/baseline: MAE + P90 absolute error; median theo route/day-of-week baseline.
- Constraint/risk: output không âm; future scan events là leakage; slice vùng/shipper.
- Kết luận: Go nếu label timestamp đáng tin và có fallback khi route mới.

## Frame 3 — Invoice tax

- Goal/decision: tính số thuế đúng tại invoice creation.
- Dữ liệu: jurisdiction, product class, ngày hiệu lực; luật đầy đủ/kiểm toán được.
- Kết luận: No-Go cho ML; dùng versioned rule engine + tests. ML chỉ có thể hỗ trợ phát hiện mapping dữ liệu bất thường, không thay luật.

## Frame 4 — Customer discovery

- Goal: tìm cohort cho qualitative research, không tự động nhắm giá/ưu đãi.
- Unit/time: customer-month, batch hàng tháng.
- Output: cluster assignment + profile; unsupervised.
- Evaluation: stability, silhouette chỉ là metric kỹ thuật; quyết định cuối dựa actionability của researcher.
- Risk: cluster dễ bị gán ý nghĩa quá mức; feature nhạy cảm và scale; drift.
- Kết luận: exploratory Go với human interpretation, không dùng cluster như ground-truth segment.

## Bài 3 — ML suitability

1. VAT: rule engine vì quy tắc xác định, cần audit/versioning.
2. Demand: regression/forecasting có thể phù hợp; so seasonal naive và kiểm tra data coverage.
3. Từ chối hồ sơ y tế: No-Go cho tự động hóa; trước hết thiết kế human review, appeal, safety/legal governance.
4. Dashboard không action: not ready; xác định decision owner và thí nghiệm sử dụng output trước.

## Bài 8 — Operating points

| Policy | Accuracy | Precision | Recall | F1 | Cost (FP=10, FN=200) |
|---|---:|---:|---:|---:|---:|
| A | 0.760 | 0.267 | 0.800 | 0.400 | 6,200 |
| B | 0.895 | 0.481 | 0.650 | 0.553 | 7,700 |

Theo cost đã cho, A tốt hơn dù accuracy/F1 thấp hơn. Hòa vốn khi `220C_FP + 20C_FN = 70C_FP + 35C_FN`, nên `C_FN/C_FP = 10`. Nếu capacity specialist là constraint cứng, policy A có thể không khả thi; cần tối ưu trong tập policy thỏa capacity, không bỏ constraint.

