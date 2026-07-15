# Báo cáo nghiệm thu — Tuần 3

Ngày kiểm tra: **2026-07-15**

## 1. Môi trường

| Thành phần | Phiên bản kiểm tra |
|---|---:|
| Python | 3.12.13 |
| NumPy | 2.3.5 |
| Pandas | 2.2.3 |
| Matplotlib | 3.10.8 |
| SciPy | 1.17.0 |
| Scikit-Learn | 1.8.0 |
| Pytest | 9.1.1 |

Các phiên bản nằm trong khoảng cho phép của `requirements.txt`.

## 2. Public tests trên reference implementation

```bash
WEEK3_IMPL=solutions.model_evaluation_solution python -m pytest -q
```

Kết quả:

```text
............                                                             [100%]
12 passed in 1.24s
```

Bộ test bao phủ descriptive statistics, conditional probability, deterministic bootstrap, split disjoint/exhaustive/reproducible, RMSE, polynomial pipeline, validation-only degree selection, learning curve và end-to-end holdout protocol.

Chạy mặc định với `challenge/submission.py` sẽ fail có chủ đích cho tới khi người học hoàn thành starter.

## 3. Notebook validation

| Notebook | Tổng cell | Code cell | Kết quả chạy namespace sạch |
|---|---:|---:|---|
| `01-sampling-clt-bootstrap.ipynb` | 20 | 13 | PASS |
| `02-bias-variance-model-selection.ipynb` | 23 | 12 | PASS |

- Hợp lệ theo nbformat 4.5.
- Cell IDs đầy đủ và duy nhất.
- Chạy bằng non-interactive Matplotlib backend.
- Không phụ thuộc hidden notebook state.
- Lab 2 kiểm tra test-access count bằng `1`.

## 4. Reference gates

### Sampling/bootstrap

- Population mean: `40.69777946`.
- `P(converted | premium)`: `0.37704918`.
- Empirical/theoretical standard-error ratios: `0.9993`, `0.9995`, `1.0152`.
- Bootstrap mean estimate: `41.47249583`.
- 95% percentile CI: `[37.13284254, 46.05902840]`.

### Nonlinear model selection

- Split sizes: `180/60/60`.
- Selected degree: `10`, dùng validation only.
- Validation RMSE: `1.084348`.
- Final test RMSE: `1.224706`.
- Test evaluation count: `1`.

### Practical

- Split sizes: `156/52/52`.
- Selected degree: `9`.
- Test RMSE: `1.479823`.
- Test MAE: `1.128643`.
- Test R²: `0.898043`.
- MAE 95% bootstrap CI: `[0.883170, 1.395091]`.
- Forbidden post-event feature bị loại khỏi model contract.

## 5. Dataset validation

| Dataset | Shape | Missing | Duplicate row | SHA-256 |
|---|---:|---:|---:|---|
| `skewed_population.csv` | `(5000, 5)` | 0 | 0 | `45d0f0fe…312399a` |
| `nonlinear_regression.csv` | `(300, 3)` | 0 | 0 | `00170b26…616263f7` |
| `practical_model_selection.csv` | `(260, 5)` | 0 | 0 | `236f6de2…b04748b4` |

Generator:

```bash
python -m data.generate_datasets
```

## 6. Quality gates cuối

- [x] 12/12 reference tests pass.
- [x] Hai notebooks chạy từ đầu đến cuối.
- [x] Bootstrap dùng replacement, local RNG và percentile bounds.
- [x] Split disjoint, exhaustive và reproducible.
- [x] Degree chỉ chọn bằng validation.
- [x] Pipeline fit transformations trên development data phù hợp.
- [x] Test chỉ mở sau khi model decision đóng băng.
- [x] Có learning curves, leakage audit và uncertainty interval.
- [x] Quiz, practical, rubric, đáp án và kết quả tham chiếu đầy đủ.
- [x] Tất cả Python source compile thành công.
