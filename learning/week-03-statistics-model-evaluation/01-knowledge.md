# Kiến thức Tuần 3 — Thống kê và Generalization

## 1. Bản đồ khái niệm

Tuần này nối hai thế giới:

```text
Population → sample → statistic/uncertainty
                         │
                         ▼
Raw dataset → train/validation/test → model selection → final estimate
```

Thống kê giúp hiểu độ biến thiên do lấy mẫu. Model evaluation giúp ước lượng khả năng hoạt động trên dữ liệu chưa thấy. Cả hai đều chống lại một lỗi trực giác: coi một con số quan sát được là sự thật cố định.

## 2. Population, sample, parameter và statistic

- **Population:** toàn bộ đối tượng/quá trình ta muốn suy luận.
- **Sample:** tập quan sát lấy từ population.
- **Parameter:** đại lượng cố định nhưng thường chưa biết của population, ví dụ `μ`.
- **Statistic:** đại lượng tính từ sample, ví dụ `x̄`.
- **Estimator:** quy tắc biến sample thành estimate.
- **Estimand:** đại lượng chính xác ta muốn ước lượng.

Ví dụ:

```python
population_mean = population["monthly_spend"].mean()  # biết được vì đây là mô phỏng
sample = population.sample(n=100, random_state=42)
sample_mean = sample["monthly_spend"].mean()
sampling_error = sample_mean - population_mean
```

Trong dữ liệu thật ta thường chỉ có sample; population mean không hiện diện để đối chiếu.

## 3. Mean, median và quantile

Với `n` giá trị:

$$
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i
$$

Mean dùng toàn bộ độ lớn và nhạy với outlier. Median là quantile 0.5 và bền vững hơn trước tail dài.

```python
values = np.array([10, 11, 12, 13, 200])
print(values.mean())       # 49.2
print(np.median(values))   # 12.0
```

Không có statistic “luôn tốt nhất”. Chọn estimand theo câu hỏi:

- Tổng ngân sách thường liên quan mean.
- “Khách hàng điển hình” có thể phù hợp median.
- Capacity planning có thể cần quantile 0.95/0.99.

## 4. Variance, standard deviation và `ddof`

Population variance:

$$
\sigma^2=\frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^2
$$

Sample variance không chệch thường dùng:

$$
s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2
$$

Trong NumPy:

```python
np.var(x, ddof=0)   # mô tả chính array này/population variance
np.var(x, ddof=1)   # sample variance estimator
```

Variance có đơn vị bình phương; standard deviation trở về đơn vị gốc.

## 5. Distribution không chỉ là histogram

Một distribution mô tả giá trị khả dĩ và xác suất/tần suất của chúng. Khi quan sát dữ liệu, kiểm tra:

- Center: mean/median/mode.
- Spread: variance/std/IQR/range.
- Shape: symmetric, skewed, multi-modal.
- Tail/outlier.
- Conditional distribution theo segment/time/group.

Normal distribution là một mô hình cụ thể, không phải mặc định cho mọi feature. `monthly_spend` thường lệch phải; biến đếm có thể gần Poisson; target nhị phân gần Bernoulli.

## 6. Joint và conditional probability

Với events `A`, `B` và `P(B)>0`:

$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)}
$$

Code bằng masks:

```python
premium = df["segment"].eq("premium")
converted = df["converted"].eq(1)
p_convert_given_premium = (converted & premium).sum() / premium.sum()
```

`P(converted | premium)` và `P(premium | converted)` có mẫu số khác nhau nên thường không bằng nhau.

Bayes:

$$
P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}
$$

Base rate rất quan trọng. Test có sensitivity cao vẫn có thể cho nhiều false positives khi event hiếm.

Conditional probability không chứng minh quan hệ nhân quả; confounders và selection bias có thể tạo association.

## 7. Random sampling và reproducibility

Dùng local random generator:

```python
rng = np.random.default_rng(42)
indices = rng.choice(len(population), size=100, replace=False)
sample = population.iloc[indices]
```

- Sampling population thường dùng `replace=False`.
- Bootstrap resampling dùng `replace=True`.
- Seed giúp tái tạo một experiment, không loại bỏ sampling uncertainty.
- Không gọi lại cùng seed ở mọi loop nếu mục tiêu là tạo các sample độc lập; tạo một generator rồi dùng tiếp.

## 8. Sampling distribution

Một statistic cũng là random variable vì nó thay đổi theo sample.

Mô phỏng sampling distribution của mean:

```python
means = []
for _ in range(2_000):
    sample = rng.choice(population_values, size=30, replace=True)
    means.append(sample.mean())
```

Đừng nhầm:

- **Data distribution:** distribution của từng observation.
- **Sampling distribution:** distribution của statistic qua nhiều sample.

## 9. Central Limit Theorem và Standard Error

Trong phiên bản cơ bản, nếu observations IID, variance hữu hạn và sample size đủ lớn, standardized sample mean tiến gần Normal:

$$
\frac{\bar{X}-\mu}{\sigma/\sqrt{n}}\Rightarrow \mathcal{N}(0,1)
$$

Standard error của mean:

$$
SE(\bar{X})=\frac{\sigma}{\sqrt{n}}\approx\frac{s}{\sqrt{n}}
$$

Hệ quả thực hành:

- Tăng `n` làm sampling distribution hẹp hơn.
- Muốn SE giảm một nửa thường cần sample size gấp bốn.
- CLT không nói raw observations trở thành Normal.
- “n ≥ 30” không phải luật phổ quát; skew/tail/dependence ảnh hưởng tốc độ hội tụ.
- Dữ liệu phụ thuộc theo thời gian/group vi phạm IID đơn giản.

## 10. Bootstrap confidence interval

Bootstrap xấp xỉ sampling distribution bằng cách resample từ empirical sample.

Percentile bootstrap:

1. Từ sample kích thước `n`, lấy `B` resamples kích thước `n` **có hoàn lại**.
2. Tính statistic cho từng resample.
3. Lấy quantiles `α/2` và `1-α/2`.

```python
rng = np.random.default_rng(42)
bootstrap_means = np.empty(5_000)
for b in range(5_000):
    resample = rng.choice(sample, size=len(sample), replace=True)
    bootstrap_means[b] = resample.mean()
lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
```

Diễn giải frequentist chính xác: nếu lặp lại sampling và quy trình CI nhiều lần, khoảng 95% các interval tạo ra sẽ chứa parameter thật dưới các giả định. Không nói “parameter có xác suất 95% nằm trong interval cố định này”.

Bootstrap có thể không tốt khi:

- Sample quá nhỏ hoặc không đại diện.
- Observations phụ thuộc nhưng resample từng row.
- Statistic nằm ở boundary, cực trị, không trơn.
- Tail hiếm không xuất hiện trong sample.
- Sampling design phức tạp bị bỏ qua.

## 11. Generalization và ba split

### Train

Fit model parameters và fit preprocessing statistics.

### Validation

Chọn feature, degree, regularization, threshold, hyperparameter; so sánh experiments.

### Test

Ước lượng cuối cùng sau khi toàn bộ lựa chọn đã đóng băng.

Nếu xem test result rồi đổi model, test đã trở thành validation. Cần một holdout mới để có final estimate ít thiên lệch hơn.

Một protocol đơn giản:

```python
train_idx, val_idx, test_idx = three_way_split_indices(...)
comparison = evaluate_degrees(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
best_degree = select_best_degree(comparison)
final_model.fit(X[np.r_[train_idx, val_idx]], y[np.r_[train_idx, val_idx]])
final_test_rmse = rmse(y[test_idx], final_model.predict(X[test_idx]))
```

## 12. Split invariants

Luôn kiểm tra:

```python
all_idx = np.concatenate([train_idx, val_idx, test_idx])
assert len(np.unique(all_idx)) == len(all_idx)
np.testing.assert_array_equal(np.sort(all_idx), np.arange(n_samples))
```

Ngoài ra:

- Split trước các transformation học từ dữ liệu.
- Cùng entity không được rơi vào nhiều split khi rows có group.
- Time-dependent problem cần chronological split.
- Classification hiếm có thể cần stratification.
- Random split chỉ đúng khi deployment distribution phù hợp giả định IID.

Tuần này dùng IID random split; group/time split sẽ được mở rộng sau.

## 13. Data leakage và test contamination

### Target leakage

Feature chứa trực tiếp/gián tiếp target hoặc thông tin chỉ có sau outcome.

Ví dụ: dự đoán khả năng vỡ nợ tại thời điểm phê duyệt nhưng dùng “số tiền thu hồi sau vỡ nợ”.

### Preprocessing leakage

Fit scaler/imputer/feature selection trên train+validation+test trước khi split.

### Duplicate/entity leakage

Cùng khách hàng/tài liệu/ảnh gần trùng xuất hiện ở cả train và test.

### Temporal leakage

Dùng tương lai để dự đoán quá khứ hoặc random split time series.

### Test contamination

Xem test nhiều lần để chọn model, feature hoặc seed. Không cần target nằm trong feature; chỉ riêng feedback loop với test metric đã làm estimate lạc quan.

Checklist prediction-time availability:

1. Feature được tạo lúc nào?
2. Tại prediction timestamp nó đã tồn tại chưa?
3. Nó được update sau outcome không?
4. Cách aggregate có lấn sang tương lai không?
5. Có fit statistic trên holdout không?

## 14. Bias–variance trực giác

Expected prediction error thường được phân rã trực giác thành:

$$
\text{Expected error}=\text{Bias}^2+\text{Variance}+\text{Irreducible noise}
$$

- **High bias:** model quá đơn giản, train và validation đều kém.
- **High variance:** train tốt nhưng validation kém; gap lớn.
- **Noise:** phần không thể loại bỏ chỉ bằng model phức tạp hơn.

Đây là phân tích kỳ vọng qua nhiều training samples. Một train/validation gap quan sát được là tín hiệu thực hành, không phải phép đo trực tiếp đầy đủ của bias/variance.

## 15. Polynomial Regression như complexity dial

PolynomialFeatures biến một feature:

$$
x\mapsto[x,x^2,\ldots,x^d]
$$

Sau đó LinearRegression vẫn tuyến tính theo coefficients nhưng nonlinear theo input.

- Degree 1: có thể underfit curve.
- Degree vừa đủ: nắm signal.
- Degree quá cao: có thể học noise, đặc biệt ở biên và sample nhỏ.

Đừng chọn degree theo train error; train error thường giảm khi complexity tăng.

## 16. Validation curve

Lập bảng:

| degree | train RMSE | validation RMSE |
|---:|---:|---:|

Pattern:

- Train/validation cùng cao → underfit/high bias.
- Train thấp, validation cao → high variance.
- Validation thấp nhất → candidate tốt theo protocol hiện tại.
- Sai khác rất nhỏ có thể do sampling noise; tie-break model đơn giản hơn là hợp lý.

## 17. Learning curve

Learning curve thay đổi training set size trong khi giữ model family/hyperparameter cố định.

Mỗi điểm phải:

1. Lấy subset chỉ từ train.
2. Fit pipeline mới.
3. Tính train score trên subset đó.
4. Tính validation score trên validation cố định hoặc CV folds đúng.

Chẩn đoán:

### High bias

Train và validation error hội tụ ở mức cao; thêm data ít giúp. Thử feature/model linh hoạt hơn hoặc giảm regularization.

### High variance

Train error thấp, validation error cao, gap thu hẹp khi thêm data. Thêm data, regularization hoặc giảm complexity có thể giúp.

### Data quality/noise

Cả hai plateau; cần xem label noise, feature availability, segmentation và metric.

Learning curve không tự chứng minh nguyên nhân; kết hợp với residual/error analysis.

## 18. Model selection bias và multiple comparisons

Khi thử nhiều model trên cùng validation set, ta có thể “overfit validation”. Model thắng có thể thắng một phần do noise của split.

Giảm rủi ro bằng:

- Giới hạn search space có lý do.
- Dùng cross-validation trên development data.
- Ghi lại toàn bộ trials.
- Giữ final test kín.
- Không chọn seed cho kết quả đẹp.

## 19. Error bars cho metric

Một test RMSE đơn lẻ không cho biết uncertainty. Có thể bootstrap **paired rows** `(y_true, y_pred)` hoặc bootstrap per-row loss để ước lượng interval.

Với MAE:

```python
absolute_errors = np.abs(y_pred - y_true)
estimate, low, high = bootstrap_ci(absolute_errors, np.mean)
```

Điều này phản ánh uncertainty do finite test sample dưới giả định resampling phù hợp; không bao phủ data drift hay uncertainty từ retraining trừ khi thiết kế bootstrap rộng hơn.

## 20. Quy trình experiment đúng

Trước khi chạy:

- Hypothesis.
- Split protocol và seed.
- Metric.
- Candidate degrees.
- Stop rule.
- Quy tắc tie-break.

Sau khi chạy:

- Báo tất cả candidate results.
- Chọn bằng validation.
- Refit train+validation.
- Final test một lần.
- Bootstrap CI/error slices.
- Ghi limitation và quyết định.

## 21. Failure modes và cách debug

| Triệu chứng | Kiểm tra đầu tiên |
|---|---|
| Validation tốt bất thường | Target/post-outcome leakage, duplicates |
| Test kém hơn validation mạnh | Validation overfit, shift, small holdout |
| Train error tăng theo degree | Pipeline/optimization/metric bug |
| High-degree prediction bùng nổ ở biên | Polynomial conditioning/extrapolation |
| Bootstrap interval quá hẹp | Resampling sai, dependence, duplicate seed logic |
| Split counts sai | Rounding và exhaustive invariant |
| Kết quả đổi mỗi lần | Random state/ordering/data mutation |
| Learning curve zigzag | Subset nhỏ, một permutation, variance cao; thêm repeats |

## 22. Formula sheet

```text
Mean:                    x̄ = sum(x_i) / n
Population variance:     sum((x_i - μ)^2) / N
Sample variance:         sum((x_i - x̄)^2) / (n - 1)
Conditional probability: P(A|B) = P(A∩B) / P(B)
SE of sample mean:       σ / sqrt(n)
RMSE:                    sqrt(mean((ŷ - y)^2))
Bootstrap percentile CI: quantile(T*, α/2), quantile(T*, 1-α/2)
Expected error intuition: bias² + variance + irreducible noise
```

## 23. Câu hỏi tự kiểm tra

1. CLT áp dụng cho raw observations hay sampling distribution của mean?
2. Vì sao bootstrap phải sample có replacement?
3. `ddof=0` và `ddof=1` trả lời hai mục đích nào?
4. `P(A|B)` khác `P(B|A)` ở đâu?
5. Vì sao split phải diễn ra trước preprocessing?
6. Validation set và test set khác vai trò thế nào?
7. Learning curve high bias trông ra sao?
8. Thêm data có luôn sửa high bias không?
9. Feature nào có thể đúng về mặt dữ liệu nhưng không hợp lệ tại prediction time?
10. Khi đã xem test và đổi model, cần làm gì để có final unbiased holdout hơn?
