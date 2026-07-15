# Practical Test Tuần 3 — 75 phút

## Quy tắc

- Dùng `data/practical_model_selection.csv`.
- Không đọc `dataset_metadata.json` hoặc `solutions/` trong thời gian làm bài.
- Tạo `outputs/practical_test_submission.py`.
- Chỉ dùng `x` làm model feature; `sensor_quality` chỉ dùng error analysis.
- **Cấm** dùng `post_event_target_proxy`.
- Test indices phải bị khóa cho đến khi model/degree/learning-curve decision hoàn tất.
- Dùng NumPy, Pandas, Matplotlib và Scikit-Learn.

## Phần 1 — Data contract và leakage audit (8 phút, 10 điểm)

1. Load CSV, in shape/dtype/missing/duplicate.
2. Viết bảng feature availability.
3. Nêu vì sao `post_event_target_proxy` là leakage.
4. Assert danh sách model features đúng `['x']`.

## Phần 2 — Three-way split (10 phút, 15 điểm)

Dùng seed `73`, tỷ lệ 60/20/20.

1. Tạo một permutation và ba index arrays.
2. Assert disjoint, exhaustive, in-range, reproducible.
3. In sizes và giữ nguyên test indices chưa truy cập target.

## Phần 3 — Validation-only model selection (20 phút, 25 điểm)

1. Tạo Pipeline `PolynomialFeatures → StandardScaler → LinearRegression`.
2. So sánh degree `1..12`.
3. Lưu train/validation RMSE vào `outputs/practical_degree_comparison.csv`.
4. Vẽ validation curve.
5. Chọn degree có validation RMSE thấp nhất; tie-break degree nhỏ hơn.

Không được có cột test metric trong comparison table.

## Phần 4 — Learning curve (12 phút, 15 điểm)

Với selected degree, dùng nested training sizes:

```text
30, 60, 90, 120, toàn bộ train
```

1. Fit pipeline mới ở mỗi size.
2. Tính train/validation RMSE.
3. Vẽ learning curve.
4. Viết diagnosis 2–4 câu trước khi mở test.

## Phần 5 — Final test một lần (12 phút, 20 điểm)

1. Đóng băng selected degree.
2. Refit pipeline trên train + validation.
3. Mở test và tính RMSE, MAE, R² đúng một lần.
4. Lưu predictions CSV và predicted-vs-actual plot.
5. Không thay model sau khi xem metric.

## Phần 6 — Bootstrap và error slice (8 phút, 10 điểm)

1. Bootstrap 95% percentile CI cho test MAE với 4.000 resamples, seed `99`.
2. So sánh RMSE cho `sensor_quality` dưới/trên median trong test.
3. Nêu limitation do test sample nhỏ.

## Phần 7 — Code quality (5 phút, 5 điểm)

- Có `main()` và ít nhất sáu hàm nhỏ.
- Local RNG, không global random state.
- Ba invariant asserts trở lên.
- Output paths được tạo tự động.

## Điều kiện đạt

- Tối thiểu 60/100.
- Không dùng forbidden feature.
- Selected degree chỉ dựa trên validation.
- Test mở sau khi learning curve/model decision hoàn tất.
- Split invariants pass.
- Final metrics hữu hạn và prediction count đúng test rows.
