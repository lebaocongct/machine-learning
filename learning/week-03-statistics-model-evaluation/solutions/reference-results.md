# Kết quả tham chiếu Tuần 3

Chỉ mở sau lần thử đầu tiên.

## 1. Skewed population

Trên toàn bộ 5.000 rows:

- Mean monthly spend: `40.69777946`.
- Median: `32.98715000`.
- Population variance: `806.49388889`.
- Population standard deviation: `28.39883605`.
- Q25/Q75: `[21.266950, 51.403350]`.
- Overall conversion rate: `0.232600`.
- Premium rate: `0.158600`.
- `P(converted | premium)`: `0.37704918`.
- `P(premium | converted)`: `0.25709372`.

Hai conditional probabilities khác nhau vì denominator khác nhau.

### Sampling-distribution experiment

2.000 trials với seed stream `20260301`:

| Sample size | Mean of sample means | Empirical SE | Theoretical SE |
|---:|---:|---:|---:|
| 5 | 40.466160 | 12.691014 | 12.700346 |
| 20 | 40.834749 | 6.346930 | 6.350173 |
| 100 | 40.671217 | 2.882916 | 2.839884 |

Empirical/theoretical SE ratios lần lượt khoảng `0.9993`, `0.9995`, `1.0152`.

## 2. Bootstrap sample

Sample 120 rows không replacement từ population bằng seed `314`; bootstrap 4.000 resamples, seed `2718`:

| Statistic | Sample estimate | 95% percentile CI |
|---|---:|---:|
| Mean spend | 41.472496 | [37.132843, 46.059028] |
| Median spend | 36.656050 | [31.212400, 41.502250] |

Các số là deterministic với dataset/algorithm/seed của gói. SciPy BCa có thể cho bounds khác vì method khác.

## 3. Nonlinear polynomial model selection

Protocol:

- Split 60/20/20, seed `42`.
- Train/validation/test: `180/60/60`.
- Compare degree `1..15` trên train/validation.
- Chọn bằng validation RMSE; refit train+validation; test một lần.

| Degree | Train RMSE | Validation RMSE |
|---:|---:|---:|
| 1 | 3.659932 | 3.552766 |
| 2 | 1.525173 | 1.445716 |
| 3 | 1.524678 | 1.451712 |
| 4 | 1.521396 | 1.464734 |
| 5 | 1.278792 | 1.125360 |
| 6 | 1.275287 | 1.131889 |
| 7 | 1.253067 | 1.101178 |
| 8 | 1.252954 | 1.099628 |
| 9 | 1.249824 | 1.099170 |
| **10** | **1.248495** | **1.084348** |
| 11 | 1.229490 | 1.098218 |
| 12 | 1.226479 | 1.126019 |
| 13 | 1.226441 | 1.125345 |
| 14 | 1.219529 | 1.130273 |
| 15 | 1.205060 | 1.133019 |

Selected degree: **10**. Final test RMSE sau refit: **1.224706**.

Degree 1 underfits. Train RMSE tiếp tục giảm sau degree 10 nhưng validation RMSE tăng lại; không chọn degree 15 theo train result.

## 4. Learning curve — selected degree 10

| Train size | Train RMSE | Validation RMSE |
|---:|---:|---:|
| 30 | 1.021936 | 1.581728 |
| 60 | 1.072797 | 1.190936 |
| 90 | 1.215082 | 1.130173 |
| 120 | 1.299775 | 1.115265 |
| 180 | 1.248495 | 1.084348 |

Validation error giảm rõ khi thêm data và gap thu hẹp. Train error tăng nhẹ là bình thường khi subset đại diện đa dạng hơn.

## 5. Expected failure cases

- Bootstrap `replace=False`: interval thường quá hẹp/sai mục tiêu.
- Fit PolynomialFeatures/scaler toàn data trước split: preprocessing leakage.
- Degree comparison chứa test RMSE: test contamination.
- Chọn degree 15 vì train RMSE thấp hơn: overfit selection criterion.
- Trộn `(n,)` và `(n,1)`: broadcasting metric bug.
- Tạo từng sample với cùng seed mới: lặp lại đúng một sample thay vì sampling distribution.
- Dùng feature post-outcome: final score đẹp nhưng deployment không hợp lệ.
