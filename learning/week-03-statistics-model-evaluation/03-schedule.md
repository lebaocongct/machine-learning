# Lịch học Tuần 3 — 10 giờ

| Phiên | Thời lượng | Nội dung | Đầu ra |
|---|---:|---|---|
| 0 | 30 phút | Setup và diagnostic | Environment, dataset checks, baseline kiến thức |
| 1 | 75 phút | Mean, variance, distribution, conditional probability | Bài thống kê mô tả + event masks |
| 2 | 105 phút | Sampling, CLT, standard error, bootstrap | Notebook 1 + confidence intervals |
| 3 | 75 phút | Train/validation/test và leakage | Split invariants + leakage audit |
| 4 | 120 phút | Bias–variance, polynomial regression, validation curve | Notebook 2 phần model comparison |
| 5 | 75 phút | Learning curves và diagnosis | Hai learning curves + kết luận |
| 6 | 90 phút | Bài tập/challenge | Tối thiểu 8 bài; 12 tests pass |
| 7 | 30 phút | Quiz và retrospective | Quiz ≥16/20; report |
| **Tổng** | **600 phút** |  | **10 giờ** |

Practical test 75 phút làm riêng hoặc thay 60 phút bài tập nếu quỹ thời gian tuần bị giới hạn.

## Phiên 0 — Setup và diagnostic

1. Tạo/activate environment.
2. Chạy `python -m data.generate_datasets`.
3. Kiểm tra ba CSV không missing/duplicate row.
4. Không tra cứu, trả lời:
   - Mean và median phản ứng với outlier thế nào?
   - `P(A|B)` dùng mẫu số nào?
   - Validation set dùng làm gì?
   - Test set được mở bao nhiêu lần?
5. Commit: `chore: initialize week 3 workspace`.

## Phiên 1 — Descriptive statistics và probability

- 20 phút đọc phần 2–6.
- 20 phút tính statistic bằng NumPy và bằng tay trên array nhỏ.
- 20 phút event masks/contingency table.
- 10 phút phân biệt `P(A|B)`/`P(B|A)`.
- 5 phút exit ticket.

Stop condition: giải thích được `ddof`, denominator của conditional probability và không suy causal.

## Phiên 2 — Sampling, CLT và bootstrap

- 20 phút phân biệt data distribution/sampling distribution.
- 30 phút mô phỏng sample sizes 5, 20, 100.
- 15 phút xác minh `SE ∝ 1/sqrt(n)`.
- 25 phút cài bootstrap mean/median CI.
- 15 phút coverage/limitation analysis.

Stop condition: code dùng `replace=True` trong bootstrap và cùng seed tái tạo interval.

## Phiên 3 — Split và leakage

- 15 phút lập prediction-time feature table.
- 20 phút cài three-way split indices.
- 15 phút viết disjoint/exhaustive asserts.
- 15 phút phân loại 8 leakage scenarios.
- 10 phút viết protocol “test locked”.

Không sang model selection nếu test set chưa được tách và khóa.

## Phiên 4 — Bias–variance lab

- 20 phút fit degree 1, 5, 15.
- 25 phút vẽ polynomial fits.
- 30 phút chạy degree 1–15 trên cùng split.
- 20 phút lập validation curve.
- 15 phút chọn degree và refit train+validation.
- 10 phút final test một lần.

## Phiên 5 — Learning curves

- 20 phút tự cài nested-subset learning curve.
- 20 phút curve cho degree 1.
- 20 phút curve cho selected/high-degree model.
- 10 phút phân tích bias/variance/noise.
- 5 phút ghi experiment decision.

## Phiên 6 — Bài tập và challenge

- 25 phút bài 1–5.
- 25 phút bài 6–10.
- 35 phút challenge/test loop.
- 5 phút commit: `feat: complete leakage-safe model selection`.

## Phiên 7 — Đánh giá

- 20 phút quiz đóng tài liệu.
- 10 phút retrospective/rubric.

## Bonus — 2 đến 4 giờ

1. Thêm repeated learning curves và error bands.
2. So sánh một split với 5-fold CV trên development set.
3. Mô phỏng validation overfitting khi số candidate tăng.
4. Bootstrap difference in RMSE giữa hai model bằng paired resampling.
5. Tạo group leakage rồi sửa bằng `GroupShuffleSplit`.
6. Tạo chronological split và so với random split trên drift data.
