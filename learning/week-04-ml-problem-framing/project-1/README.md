# Project 1 — ML Problem Canvas

## Mục tiêu

Chứng minh một use case ML đáng làm và có thể đánh giá **trước khi** xây model. Người học có thể chọn case cá nhân hoặc dùng case support triage.

## Bộ nộp bắt buộc

```text
project-1/submission/
├── problem-canvas.md
├── prediction-contract.json
├── feature-catalog.csv
├── metric-baseline-plan.md
└── defense-notes.md
```

Sao chép ba template trong thư mục này; không sửa file ví dụ.

## Quy trình 8 bước

1. Viết non-ML goal và KPI hiện tại.
2. Xác định decision, decision owner và action cho positive/negative output.
3. Khóa unit of prediction, prediction time và latency/capacity.
4. Định nghĩa target, label source và cửa sổ quan sát.
5. Chọn task/output; ghi phương án non-ML.
6. Chọn primary metric, guardrail, dummy và operational baseline.
7. Audit feature catalog tại prediction time; ghi slices/leakage risks.
8. Viết success criterion và kết luận Go / Conditional Go / No-Go.

## `metric-baseline-plan.md`

Phải trả lời:

- Metric nào được tối ưu, metric nào là constraint?
- FP và FN gây ra điều gì, chi phí/độ nghiêm trọng do ai xác nhận?
- Dummy baseline và operational baseline là gì?
- Split theo thời gian/entity thế nào?
- Threshold được chọn ở đâu? Test được mở bao nhiêu lần?
- Hai trade-off và hai evaluation slices ưu tiên.

## Defense 5 phút

- 45 giây: goal → decision → action.
- 60 giây: unit/time → target/window → output.
- 60 giây: metric/baseline/trade-off.
- 60 giây: constraint/slices/leakage.
- 45 giây: Go/No-Go và next experiment.
- 30 giây dự phòng.

## Acceptance gate

Project không đạt nếu thiếu decision/action, dùng feature sau prediction time, không có baseline, dùng test để chọn threshold hoặc chỉ báo accuracy cho class hiếm.

Xem rubric tại `../assessment/rubric.md` và checklist defense tại `../assessment/canvas-defense.md`.

