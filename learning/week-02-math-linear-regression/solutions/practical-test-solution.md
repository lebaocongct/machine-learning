# Kết quả tham chiếu — Practical test Tuần 2

Chỉ mở sau khi hết 75 phút.

## Dataset

- Rows: `90`
- Features: `4`
- X shape: `(90, 4)`
- X_bias shape: `(90, 5)`
- y shape: `(90,)`

## Gradient check

Với theta `[5, 1, -1, 0.5, 2]`, epsilon `1e-6`:

- Relative error: khoảng `5.22e-09`.

## Learning-rate experiment — 500 steps

| LR | Final loss | Step đạt trong 1% optimum | Nhận xét |
|---:|---:|---:|---|
| 0.001 | 741.150883 | Chưa đạt | Quá chậm |
| 0.01 | 5.677879 | 290 | Stable |
| 0.05 | 5.677840 | 56 | Stable |
| 0.20 | 5.677840 | 12 | Stable |
| 1.00 | khoảng `5.96e175` | Không | Diverged |

## Final raw parameters

| Parameter | Estimate |
|---|---:|
| Intercept | 23.639931 |
| study_hours | 3.398837 |
| practice_tests | 1.630337 |
| sleep_hours | 2.295108 |
| absences | -1.877717 |

## Metrics

- MSE: `5.677840`
- RMSE: `2.382822`
- MAE: `1.902680`
- R²: `0.984693`
- Initial loss: `5,517.281901`

Maximum parameter difference giữa GD 2,000 steps và `lstsq` khoảng `1.1e-14` trong môi trường kiểm tra.

## Kết luận mẫu

Standardized GD với LR `0.05` hội tụ ổn định và cho cùng nghiệm với least squares. LR `0.001` chưa đủ trong 500 steps, còn LR `1.0` vượt vùng stable. Coefficients gần hệ số sinh dữ liệu nhưng không trùng hoàn toàn do noise.

