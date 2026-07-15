# Bài tập độc lập Tuần 3

## Quy tắc

- Làm tối thiểu 8/10 bài.
- Ghi seed, sample size và split protocol cho mọi experiment.
- Không xem test set trong bài 7–10 trước bước final evaluation.
- Không dùng `post_event_target_proxy` làm feature.
- Mọi kết luận phải trỏ tới statistic, curve hoặc test/assert cụ thể.

## Bài 1 — Mean, median, variance và outlier (8 điểm)

Cho:

```python
x = np.array([8., 9., 10., 11., 12.])
```

1. Tính mean, median, population/sample variance bằng tay và NumPy.
2. Thêm `100` rồi tính lại.
3. So sánh absolute/relative change của mean và median.
4. Viết `descriptive_summary` trong starter.
5. Giải thích statistic nào phù hợp cho “typical value”.

## Bài 2 — Distribution audit (8 điểm)

Dùng `skewed_population.csv`:

1. Tính min/q25/median/q75/q95/max của `monthly_spend`.
2. Vẽ histogram và ECDF.
3. Tính IQR outlier fences.
4. So sánh raw và `log1p` distribution.
5. Không kết luận Normal chỉ từ histogram.

## Bài 3 — Conditional probability (10 điểm)

1. Tạo 2x2 count table cho `premium` và `converted`.
2. Tính `P(converted)`, `P(premium)`, `P(converted|premium)`, `P(premium|converted)`.
3. Xác minh Bayes bằng các đại lượng tính được.
4. Viết `probability_table`.
5. Nêu một confounder khả dĩ; không kết luận causal.

## Bài 4 — Sampling distribution và CLT (10 điểm)

Từ `monthly_spend`, mô phỏng 2.000 sample means cho `n = 5, 20, 100`.

1. Cùng một local RNG, lấy sample có replacement.
2. Vẽ ba histograms.
3. Báo mean/std của sampling distribution.
4. So sánh empirical std với `population_std / sqrt(n)`.
5. Giải thích raw distribution vẫn lệch dù sample means gần Normal hơn.

## Bài 5 — Bootstrap confidence interval (10 điểm)

Lấy sample 120 rows với seed `314`:

1. Cài percentile bootstrap cho mean và median.
2. Dùng 4.000 resamples, seed `2718`.
3. So sánh interval width.
4. Đối chiếu với `scipy.stats.bootstrap` sau khi code tự viết chạy đúng.
5. Viết diễn giải frequentist chính xác.

## Bài 6 — Three-way split invariants (10 điểm)

Với `n=103`, tỷ lệ 60/20/20:

1. Tạo indices bằng một permutation.
2. Ghi rõ quy tắc rounding.
3. Assert disjoint, exhaustive, in-range và reproducible.
4. Xác minh X/y alignment qua một ID column.
5. Viết `assert_split_invariants`.

## Bài 7 — Leakage audit (10 điểm)

Phân loại từng tình huống thành target leakage, preprocessing leakage, duplicate/entity leakage, temporal leakage, test contamination hoặc hợp lệ:

1. Standardize toàn data rồi split.
2. Cùng customer ở train và test.
3. Dùng payment status sau ngày dự đoán.
4. Chọn degree có test RMSE thấp nhất.
5. Fit scaler trên train, transform validation/test.
6. Random split time series có drift.
7. Dùng validation để chọn threshold.
8. Chạy 50 seeds rồi báo seed có test tốt nhất.

Với mỗi case, viết cách sửa.

## Bài 8 — Validation curve và bias–variance (12 điểm)

Dùng `nonlinear_regression.csv`:

1. Chia 60/20/20 với seed 42.
2. Fit degree 1–15 bằng cùng pipeline.
3. Lưu train/validation RMSE.
4. Vẽ curve và chọn degree bằng validation.
5. Đánh dấu một underfit và một high-variance candidate.
6. Không tính test metric ở bước này.

## Bài 9 — Learning curves (12 điểm)

Cho degree 1, selected degree và degree 15:

1. Dùng nested training subsets.
2. Fit model mới ở mỗi size.
3. Vẽ train/validation RMSE.
4. Viết diagnosis theo curve, không theo nhãn degree.
5. Nêu liệu thêm data có khả năng giúp từng model không.

## Bài 10 — Test contamination simulation (10 điểm)

Mô phỏng quy trình sai:

1. Tạo 100 candidate models tương đương nhưng có random perturbations.
2. Chọn candidate bằng cùng test set.
3. Đánh giá candidate thắng trên holdout mới độc lập.
4. Lặp experiment nhiều lần.
5. So sánh “best reused-test metric” với fresh-holdout metric.
6. Giải thích winner's curse/model-selection bias.

## Tự chấm

| Mức | Điều kiện |
|---|---|
| Chưa đạt | Dưới 60 hoặc có leakage/test contamination |
| Đạt | 60–79, split invariants và bài 8 đúng |
| Tốt | 80–89, bootstrap/learning curve có reasoning rõ |
| Xuất sắc | 90–100, hoàn thành contamination simulation và limitations |
