# Challenge — Leakage-safe Polynomial Model Selection

## Bối cảnh

Bạn có một dataset nonlinear regression. Mục tiêu không chỉ là tìm model có RMSE thấp mà còn phải chứng minh quy trình đánh giá không làm rò rỉ thông tin từ test set.

Hoàn thành API trong `challenge/submission.py` mà không import code từ `solutions/`.

## API bắt buộc

```python
population_summary(values, ddof=0)
conditional_probability(event, given)
bootstrap_ci(values, statistic, n_resamples, confidence_level, random_state)
three_way_split_indices(n_samples, train_fraction, validation_fraction, test_fraction, random_state)
root_mean_squared_error(y_true, y_pred)
build_polynomial_model(degree)
evaluate_degrees(X_train, y_train, X_validation, y_validation, degrees)
select_best_degree(results)
manual_learning_curve(X_train, y_train, X_validation, y_validation, degree, train_sizes)
LeakageSafePolynomialWorkflow.fit(X, y)
LeakageSafePolynomialWorkflow.predict(X)
```

## Invariants quan trọng

### Split

- Train, validation và test index phải disjoint.
- Hợp của ba tập phải đúng `0..n_samples-1`.
- Cùng `random_state` phải cho cùng split.
- Test set không được truyền vào `evaluate_degrees`.

### Model selection

1. So sánh degree trên train/validation.
2. Chọn degree có validation RMSE thấp nhất; nếu hòa chọn degree nhỏ hơn.
3. Sau khi chốt degree, refit trên train + validation.
4. Đánh giá test đúng một lần.

### Bootstrap

- Resample **có hoàn lại** và giữ nguyên sample size.
- Dùng local `np.random.default_rng(random_state)`.
- Percentile interval hai phía.
- Trả `(estimate, lower, upper)` là Python floats.

### Polynomial model

Pipeline tham chiếu:

```text
PolynomialFeatures(include_bias=False)
→ StandardScaler
→ LinearRegression
```

`X` phải có shape `(n_samples, 1)`; `y` phải `(n_samples,)`.

## Thứ tự làm

```bash
python -m pytest tests/test_model_evaluation.py -q
```

1. Statistics và conditional probability.
2. Bootstrap CI.
3. RMSE và split indices.
4. Polynomial model/evaluation table.
5. Learning curve.
6. End-to-end workflow.

Sau khi public tests pass, chạy experiment với `data/nonlinear_regression.csv` cho degree `1..15` và lưu:

```text
degree, train_rmse, validation_rmse
```

## Critical fail

- Chọn degree bằng test RMSE.
- Fit preprocessing/polynomial transformation trên toàn bộ data trước split.
- Chạy test nhiều lần rồi báo kết quả tốt nhất.
- Dùng `post_event_target_proxy` trong practical.
- Bootstrap không có replacement.

## Đầu ra

- `challenge/submission.py` hoàn chỉnh.
- 12/12 public test groups pass.
- Degree comparison và learning curve CSV.
- Bias–variance report có evidence.
- Test RMSE cuối cùng được ghi đúng một lần.
