# Báo cáo nghiệm thu — Tuần 2

Ngày kiểm tra: **2026-07-15**

## 1. Môi trường kiểm tra

| Thành phần | Phiên bản |
|---|---:|
| Python | 3.12.13 |
| NumPy | 2.3.5 |
| Pandas | 2.2.3 |
| Matplotlib | 3.10.8 |
| Pytest | 9.1.1 |

Các phiên bản này nằm trong khoảng cho phép của `requirements.txt`. Notebook khai báo kernel Python 3 và không phụ thuộc state có sẵn.

## 2. Kiểm thử tự động

Lệnh:

```bash
WEEK2_IMPL=solutions.linear_regression_solution python -m pytest tests/test_linear_regression.py -q
```

Kết quả:

```text
............                                                             [100%]
12 passed in 0.11s
```

Bộ test bao phủ prediction/MSE, analytical và numerical gradient, standardization, constant feature, convergence, recovery of parameters, không mutate input, lỗi pre-fit, shape/hyperparameter sai và non-finite loss guard.

Lưu ý: chạy test mặc định với `challenge/submission.py` sẽ fail có chủ đích cho đến khi người học hoàn thành starter.

## 3. Notebook

| Notebook | Tổng cell | Code cell | Kết quả chạy tuần tự |
|---|---:|---:|---|
| `01-linear-algebra-for-ml.ipynb` | 18 | 10 | PASS |
| `02-gradient-descent-linear-regression.ipynb` | 21 | 12 | PASS |

- Cả hai tệp hợp lệ theo nbformat 4.5.
- Mọi cell có ID duy nhất.
- Chạy lại từ namespace sạch, backend Matplotlib không giao diện.
- Lab 2 qua các assert về gradient check, loss reduction, `lstsq` parity, metrics và output artifacts.

## 4. Kết quả số tham chiếu

### Housing regression

- Gradient relative error: `1.754e-08` trong notebook với epsilon `1e-6`.
- Initial MSE: `160792.668170`.
- Final MSE: `137.069193`.
- GD và `np.linalg.lstsq`: maximum parameter difference dưới `1.5e-14`.
- Learning rate `0.05`: hội tụ đến optimum ổn định.
- Learning rate `1.0`: loss tăng mạnh, không hội tụ.

### Practical regression

- Gradient relative error: `5.2175e-09`.
- MSE: `5.677840`.
- RMSE: `2.382822`.
- MAE: `1.902680`.
- R²: `0.984693`.
- Intercept: `23.639931`.
- Coefficients: `[3.398837, 1.630337, 2.295108, -1.877717]`.

## 5. Dữ liệu

| Dataset | Shape | Missing | Duplicate row | SHA-256 |
|---|---:|---:|---:|---|
| `single_feature_regression.csv` | `(80, 3)` | 0 | 0 | `caeadaa8…cc9bb7` |
| `housing_regression.csv` | `(160, 7)` | 0 | 0 | `d1e3e8e0…86be86` |
| `practical_regression.csv` | `(90, 6)` | 0 | 0 | `c2415e71…f9afd` |

Dữ liệu synthetic được sinh lại bằng seed cố định qua:

```bash
python -m data.generate_datasets
```

## 6. Quality gates cuối

- [x] 12/12 reference tests pass.
- [x] Hai notebook chạy từ đầu đến cuối.
- [x] Gradient relative error `<1e-6`.
- [x] GD đạt cùng nghiệm với stable least squares reference.
- [x] Có learning-rate và feature-scaling experiments.
- [x] Có quiz, practical test, rubric, đáp án và kết quả tham chiếu.
- [x] Python source compile thành công.
- [x] Không dùng Scikit-Learn để fit trong lab/challenge reference.

