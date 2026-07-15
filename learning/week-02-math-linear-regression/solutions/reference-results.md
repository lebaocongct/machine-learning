# Kết quả tham chiếu Tuần 2

Chỉ mở sau lần thử đầu tiên.

## 1. Single-feature regression

Reference least-squares/GD:

- Intercept: `4.11596242`
- Coefficient x: `3.20655686`
- MSE: `0.58128797`
- RMSE: `0.76242244`
- MAE: `0.62862784`
- R²: `0.99742204`

True generating parameters là intercept `4.0`, slope `3.2`; ước lượng khác nhẹ do noise.

## 2. Housing regression

Gradient descent dùng standardized X, learning rate `0.05`, 2,000 steps.

| Parameter | Estimated | True generating value |
|---|---:|---:|
| Intercept | 35.277100 | 35.0 |
| area_sqm | 2.368555 | 2.4 |
| bedrooms | 18.524732 | 18.0 |
| age_years | -0.843861 | -0.9 |
| distance_km | -2.436786 | -2.2 |
| energy_score | 0.843696 | 0.8 |

Metrics trên toàn synthetic dataset:

- MSE: `137.069193`
- RMSE: `11.707655`
- MAE: `9.674602`
- R²: `0.987428`
- Initial loss: `160,792.668170`
- Final loss: `137.069193`

GD và `np.linalg.lstsq` có maximum absolute parameter difference khoảng `1.5e-14` trong môi trường kiểm tra.

## 3. Gradient check

Với theta kiểm tra `[10, 1, -2, 0.5, 3, -1]` trên standardized housing design matrix:

- Relative error tham chiếu: khoảng `1.72e-08`.
- Gate: `<1e-6`.

Giá trị cụ thể có thể thay đổi nhẹ theo epsilon/nền tảng, nhưng phải vượt gate.

## 4. Learning-rate experiment — housing, 500 steps

Optimum MSE từ least squares: `137.069193`.

| LR | Final loss | Step đạt trong 1% optimum | Stable/monotonic |
|---:|---:|---:|---|
| 0.001 | 21,779.257484 | Chưa đạt | Có, quá chậm |
| 0.005 | 144.285496 | Chưa đạt 1% | Có |
| 0.01 | 137.069566 | 291 | Có |
| 0.05 | 137.069193 | 56 | Có |
| 0.10 | 137.069193 | 27 | Có |
| 0.20 | 137.069193 | 12 | Có |
| 0.50 | 137.069193 | 3 | Có trên dataset này |
| 1.00 | khoảng `1.26e128` | Không | Không, diverged |

Không kết luận LR `0.5` là lựa chọn phổ quát. Stability phụ thuộc Hessian, feature scale và dataset.

## 5. Unscaled housing example — 1,000 steps

| LR | Final loss/kết quả |
|---:|---:|
| 1e-7 | 1,816.061357 |
| 1e-6 | 1,672.397725 |
| 1e-5 | 1,328.950657 |
| 1e-4 | `FloatingPointError`/divergence |

Standardized model dùng LR `0.05` đạt optimum nhanh hơn rất nhiều. Đây là evidence cho optimization, không phải bằng chứng standardization luôn tăng generalization.

## 6. Failure cases bắt buộc hiểu

- Bỏ factor `2/n`: gradient direction có thể giống nhưng scale sai; learning-rate behavior thay đổi.
- Dùng `X @ residual`: shape sai; đúng là `X.T @ residual`.
- Parse y thành `(n,1)` trong khi prediction `(n,)`: broadcasting có thể tạo residual `(n,n)` mà không lỗi ngay.
- LR `1.0`: loss tăng mạnh và không hội tụ.
- Không scale housing feature: stable LR phải rất nhỏ.

