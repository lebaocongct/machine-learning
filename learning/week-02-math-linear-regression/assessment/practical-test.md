# Practical test Tuần 2 — 75 phút

## Quy tắc

- Dùng `data/practical_regression.csv`.
- Được xem NumPy API chính thức, không xem `solutions/` hoặc copy challenge implementation.
- Tạo `outputs/practical_test_submission.py`.
- Chỉ dùng NumPy, Pandas và Matplotlib; không dùng Scikit-Learn để fit model.
- Không đọc `dataset_metadata.json` trong thời gian làm bài.

## Bài toán

Huấn luyện Linear Regression dự đoán `score` từ:

```text
study_hours, practice_tests, sleep_hours, absences
```

### Phần 1 — Shape và baseline toán học (10 phút, 15 điểm)

1. Load CSV, tách ID/feature/target.
2. In shape của X, y, X_bias và theta.
3. Tính constant baseline prediction bằng mean(y) và baseline MSE.

### Phần 2 — Gradient check (15 phút, 25 điểm)

1. Standardize X bằng mean/std tự viết.
2. Cài prediction, MSE và analytical gradient.
3. Cài central finite-difference gradient.
4. Chạy ở một theta không phải zero.
5. Báo relative error.

Điều kiện bắt buộc: relative error `<1e-6`.

### Phần 3 — Gradient descent (25 phút, 30 điểm)

Chạy 500 steps cho ít nhất:

```text
0.001, 0.01, 0.05, 0.2, 1.0
```

Với mỗi run, ghi initial/final loss, monotonic/stable và loss history. Detect non-finite loss.

Chọn một stable learning rate có lý do và fit final model.

### Phần 4 — Reference và metrics (15 phút, 20 điểm)

1. Chuyển coefficients về raw units.
2. So sánh với `np.linalg.lstsq`.
3. Báo MSE, RMSE, MAE và R².
4. Lưu predictions CSV.

### Phần 5 — Visualization và code quality (10 phút, 10 điểm)

1. Loss curves trên log scale.
2. Predicted vs actual scatter có đường `y=x`.
3. Có `main()`, ít nhất năm hàm nhỏ và ba invariant asserts.

## Điều kiện đạt

- Tối thiểu 60/100.
- Gradient relative error `<1e-6`.
- Model MSE nhỏ hơn constant baseline rõ rệt.
- GD parameters gần `lstsq` trong tolerance hợp lý.
- Không hard-code metrics/parameters.

