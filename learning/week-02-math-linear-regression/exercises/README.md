# Bài tập độc lập Tuần 2

## Quy tắc

- Làm tối thiểu 8/10 bài.
- Trước mỗi phép nhân matrix, ghi shape trên giấy.
- Không gọi Scikit-Learn để thay implementation.
- `np.linalg.lstsq` chỉ dùng ở bước đối chiếu.
- Gradient analytical phải được viết độc lập với numerical gradient.

## Bài 1 — Vector, dot product và norm (8 điểm)

Cho:

```python
a = np.array([1.0, 2.0, -1.0])
b = np.array([3.0, 0.0, 4.0])
```

1. Tính `a + b`, `2a - b`, `a @ b` bằng tay rồi bằng NumPy.
2. Tính L1 và L2 norm của `a`.
3. Tính Euclidean distance giữa `a` và `b`.
4. Cài `cosine_similarity` trong `starter.py`.
5. Raise `ValueError` khi vector zero hoặc shape không khớp.

## Bài 2 — Shape audit (8 điểm)

Không chạy code trước khi điền output shape:

| Biểu thức | Shape A | Shape B | Output/Fail |
|---|---|---|---|
| `A @ b` | `(5, 3)` | `(3,)` |  |
| `A.T @ y` | `(3, 5)` | `(5,)` |  |
| `A @ B` | `(5, 3)` | `(3, 2)` |  |
| `A * b` | `(5, 3)` | `(3,)` |  |
| `A @ c` | `(5, 3)` | `(5,)` |  |

Sau đó tạo array ngẫu nhiên và xác minh.

## Bài 3 — Design matrix và bias (8 điểm)

Cho `X.shape == (4, 2)`:

1. Thêm cột `1` ở đầu mà không dùng loop.
2. Tạo theta cho intercept `5`, coefficients `[2, -3]`.
3. Tính bốn predictions bằng một phép `@`.
4. Chứng minh kết quả giống tính từng row bằng công thức scalar.

## Bài 4 — Least-squares line bằng tay và NumPy (10 điểm)

Dùng `single_feature_regression.csv`:

1. Vẽ scatter.
2. Tính slope/intercept bằng `np.linalg.lstsq`.
3. Vẽ fitted line.
4. Tính MSE, RMSE và R² bằng NumPy.
5. Không diễn giải R² là bằng chứng nhân quả.

## Bài 5 — Numerical derivative (10 điểm)

Cài `numerical_derivative` với central differences. Test trên:

$$
f(x)=x^2+3x-4,\quad f'(x)=2x+3
$$

Tại `x=2`, thử epsilon từ `1e-1` đến `1e-12`. Vẽ hoặc lập bảng absolute error. Giải thích vì sao error không giảm mãi khi epsilon nhỏ.

## Bài 6 — Chain rule scalar (10 điểm)

Với:

$$
\hat y = wx+b,\quad r=\hat y-y,\quad L=r^2
$$

Tại `x=3`, `y=10`, `w=2`, `b=1`:

1. Tính forward values.
2. Tính $\partial L/\partial w$ và $\partial L/\partial b$ bằng chain rule.
3. Xác minh bằng finite differences.
4. Thử một gradient step với learning rate `0.01`; loss phải giảm.

## Bài 7 — Gradient MSE multi-feature (12 điểm)

Cho một `X` nhỏ `(5, 2)`, thêm bias và theta ngẫu nhiên:

1. Viết MSE dạng vectorized.
2. Viết gradient `2/n * X.T @ residual`.
3. Ghi shape của từng biến.
4. So sánh với numerical gradient.
5. Cài `relative_gradient_error`.

Điều kiện: relative error `<1e-6`.

## Bài 8 — Gradient descent và learning rate (12 điểm)

Trên single-feature dataset, chạy ít nhất:

```text
0.001, 0.01, 0.05, 0.2, 1.0
```

Ghi:

- Initial/final loss.
- Loss monotonic hay không.
- Step đạt gần optimum.
- Parameter cuối.
- Stable/diverged.

Vẽ loss trên log scale. Không chọn learning rate chỉ vì nó lớn nhất trong nhóm stable.

## Bài 9 — Standardized vs unscaled (12 điểm)

Dùng housing dataset:

1. Chạy GD trên feature raw với một learning rate nhỏ.
2. Chạy GD trên standardized feature.
3. So sánh loss sau cùng, step-to-threshold và gradient norm.
4. Chuyển standardized coefficients về raw units.
5. So sánh với metadata true coefficients và `lstsq`.

## Bài 10 — Collinearity và numerical stability (10 điểm)

Tạo:

```python
x2 = 2 * x1 + small_noise
```

1. Fit least squares với `x1`, `x2`.
2. Đo condition number bằng `np.linalg.cond`.
3. Thay `small_noise` nhỏ dần và quan sát coefficients.
4. So sánh prediction stability với coefficient stability.
5. Giải thích vì sao không nên explicit inverse.

## Điểm tự đánh giá

| Mức | Điều kiện |
|---|---|
| Chưa đạt | Dưới 60 hoặc không hoàn thành bài 5–8 |
| Đạt | 60–79, gradient check pass |
| Tốt | 80–89, experiment/reasoning rõ |
| Xuất sắc | 90–100, có collinearity analysis và robust guards |

