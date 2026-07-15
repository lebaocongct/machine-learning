# Challenge — Linear Regression bằng NumPy và Gradient Descent

## Mục tiêu

Cài đặt API trong `challenge/submission.py` mà không gọi implementation trong `solutions/`:

```python
add_bias_column(X)
predict_linear(X_bias, theta)
mean_squared_error(y_true, y_pred)
mse_gradient(X_bias, y, theta)
finite_difference_gradient(loss_fn, theta, epsilon)
fit_standardizer(X)
apply_standardizer(X, mean, scale)
gradient_descent(X_bias, y, initial_theta, learning_rate, n_steps)
LinearRegressionGD.fit(X, y)
LinearRegressionGD.predict(X)
```

## Shape contract

| Biến | Shape |
|---|---|
| `X` | `(n_samples, n_features)` |
| `X_bias` | `(n_samples, n_features + 1)` |
| `y` | `(n_samples,)` |
| `theta` | `(n_features + 1,)` |
| predictions | `(n_samples,)` |
| gradient | `(n_features + 1,)` |
| loss history | `(n_steps + 1,)` |

Không tự động `ravel()` input 2D sai contract; raise lỗi rõ để tránh che bug shape.

## Contract chi tiết

### Input validation

- Chuyển numeric input hợp lệ sang `float` array.
- Reject empty, non-finite hoặc sai số chiều.
- Không mutate input.
- Hyperparameter phải dương; `n_steps` là positive integer.

### Bias column

- Tạo array mới.
- Cột đầu toàn `1.0`.
- Giữ nguyên thứ tự feature.

### Prediction

```text
y_hat = X_bias @ theta
```

### MSE

```text
mean((y_pred - y_true) ** 2)
```

Return Python `float`.

### Analytical gradient

```text
2 / n * X_bias.T @ (X_bias @ theta - y)
```

Không gọi numerical gradient bên trong.

### Finite differences

- Central difference.
- Copy theta cho mỗi perturbation.
- `epsilon > 0`.
- Output cùng shape theta.

### Standardizer

- Population standard deviation, `ddof=0`.
- Constant feature có safe scale `1.0`.
- Fit trả `(mean, scale)`.
- Apply dùng đúng parameter đã fit.

### Gradient descent

- Full batch.
- History chứa initial loss và loss sau mỗi step.
- Copy `initial_theta`.
- Nếu loss thành non-finite, raise `FloatingPointError` với thông báo hữu ích.

### `LinearRegressionGD`

`fit` phải:

1. Validate X/y.
2. Fit standardizer.
3. Standardize X.
4. Thêm bias.
5. Khởi tạo theta zero.
6. Chạy gradient descent.
7. Lưu `theta_`, `loss_history_`, `mean_`, `scale_`.
8. Chuyển về raw `coef_` và `intercept_`.
9. Return `self`.

`predict` phải dùng raw coefficients hoặc đúng standardizer đã fit và reject gọi trước `fit`.

## Vòng lặp phát triển

```bash
python -m pytest tests/test_linear_regression.py -q
```

Thứ tự khuyến nghị:

1. Bias, prediction, MSE.
2. Analytical gradient.
3. Numerical gradient.
4. Gradient check `<1e-6`.
5. Standardizer.
6. Gradient descent trên line không noise.
7. Model class.
8. Housing experiment.

## Housing experiment bắt buộc

Features:

```python
[
    "area_sqm",
    "bedrooms",
    "age_years",
    "distance_km",
    "energy_score",
]
```

Target: `price_thousand`.

Chạy ít nhất bốn learning rate và lưu:

```text
learning_rate, initial_loss, final_loss,
steps_to_1pct_optimum, stable, coefficient_error
```

So sánh final parameters với:

```python
np.linalg.lstsq(add_bias_column(X), y, rcond=None)[0]
```

## Public tests

12 nhóm test kiểm tra:

- Bias/prediction/MSE.
- Analytical và numerical gradient.
- Standardization và constant feature.
- Loss convergence.
- Recovery of noiseless parameters.
- Không mutate input.
- Pre-fit và invalid shape/hyperparameters.

## Hints

### Hint 1

Nếu gradient check fail nhưng shape đúng, kiểm tra mean/sum và factor `2/n`.

### Hint 2

Raw coefficient sau training standardized không bằng `theta_[1:]`; phải chia cho scale và điều chỉnh intercept.

### Hint 3

Loss history có `n_steps + 1` phần tử vì chứa trạng thái trước update đầu tiên.

## Đầu ra

- `challenge/submission.py` hoàn chỉnh.
- 12/12 public tests pass.
- Gradient relative error `<1e-6`.
- Learning-rate experiment CSV và loss plot.
- Housing coefficients/metrics report.
- Commit: `feat: implement numpy linear regression with gradient descent`.

