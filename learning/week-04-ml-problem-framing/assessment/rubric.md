# Rubric Tuần 4 — 100 điểm

## Cơ cấu

| Hạng mục | Điểm | Bằng chứng |
|---|---:|---|
| Guided labs | 20 | Hai notebook chạy, assertions và outputs |
| 10 bài tập | 20 | 2 điểm/bài theo README |
| Coding challenge | 30 | API, 12 tests, code quality, protocol |
| Practical test | 20 | Fraud artifacts theo đề |
| Quiz | 10 | Quy đổi 20 câu × 0.5 |
| **Tổng** | **100** | |

## 1. Guided labs — 20 điểm

| Tiêu chí | Điểm tối đa |
|---|---:|
| Notebook 01 phân loại scenario, 4 frames, contract và leakage audit | 8 |
| Notebook 02 split đúng, baselines, validation sweep, test một lần | 8 |
| Kết quả tái lập, output files và reflection/trade-off | 4 |

## 2. Bài tập — 20 điểm

Mỗi bài 0/1/2 theo `exercises/README.md`. Điểm 2 cần đúng bản chất, artifact kiểm chứng được và nêu giả định/rủi ro; chỉ điền cho có tối đa 1.

## 3. Coding challenge — 30 điểm

| Tiêu chí | Điểm tối đa |
|---|---:|
| Task classification + canvas/contract validation | 6 |
| Feature audit bắt đúng availability/leakage/duplicate | 6 |
| Baseline và binary metrics đúng edge cases | 5 |
| Chronological split, threshold sweep/tie-break | 7 |
| Holdout workflow, test-once discipline | 4 |
| Validation/code clarity; không import đáp án | 2 |

Chấm tự động gợi ý: 12 public tests × 2 điểm = 24; 6 điểm review code/protocol. Hidden/adversarial test có thể kiểm tra invalid input.

## 4. Practical — 20 điểm

| Tiêu chí | Điểm tối đa |
|---|---:|
| Fraud Canvas | 4 |
| Prediction contract | 2 |
| Feature audit | 3 |
| Split/baseline/sweep/final test | 7 |
| Slice report | 2 |
| Decision memo | 2 |

## 5. Quiz — 10 điểm

`quiz_score / 20 × 10`.

## Mức kết quả

- 90–100: Vượt yêu cầu; framing sẵn sàng qua design review.
- 80–89: Tốt; còn sửa nhỏ.
- 70–79: Đạt; phải đóng các gap được ghi rõ trước Tuần 5.
- 50–69: Chưa đạt; làm lại contract, leakage và metric protocol.
- <50: Làm lại Tuần 4 có hướng dẫn.

## Critical fail

Bất kể tổng điểm, trạng thái là **chưa đạt** nếu có một trong các lỗi:

1. Không có decision/action/owner nhưng vẫn tuyên bố production-ready.
2. Dùng feature post-outcome hoặc label-generation mà không phát hiện.
3. Chọn threshold/model bằng test hoặc trộn test vào train.
4. Không có dummy/operational baseline.
5. Với class hiếm, chỉ dùng accuracy để kết luận.
6. Bịa cost/constraint như fact đã được stakeholder phê duyệt.

## Rubric riêng cho ML Problem Canvas — 24 điểm

Mỗi tiêu chí 0–3:

| Tiêu chí | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Decision/action/owner | Thiếu | Mơ hồ | Khá rõ | Cụ thể, actionable |
| Unit/time | Thiếu | Mâu thuẫn | Xác định | Có edge cases/SLA |
| Target/window | Thiếu | Proxy mơ hồ | Định nghĩa rõ | Có source/maturity/limitations |
| Metric/trade-off | Thiếu | Metric chung | Gắn error type | Cost + guardrail + sensitivity |
| Baseline/success | Thiếu | Baseline yếu | Dummy + operational | Delta/split/time horizon rõ |
| Constraint/deployment | Thiếu | Liệt kê chung | Capacity/latency rõ | Fallback/ownership rõ |
| Slices/risks | Thiếu | Chung chung | ≥2 slice và leakage | Ưu tiên + mitigation |
| Go/No-Go evidence | Khẳng định | Dựa trực giác | Có bằng chứng | Giả định và next experiment rõ |

Canvas đạt độc lập khi ≥17/24 và không critical fail.

