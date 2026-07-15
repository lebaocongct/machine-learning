# Tuần 3 — Thống kê, Bias–Variance và Đánh giá Mô hình

> Thời lượng chính: 10 giờ  
> Tiên quyết: hoàn thành Tuần 1–2 hoặc biết NumPy, Pandas và Linear Regression  
> Trọng tâm: sampling, bootstrap, split đúng, leakage, polynomial regression và learning curves

## 1. Kết quả cần đạt

Sau tuần này, bạn phải có thể:

1. Phân biệt population, sample, parameter, statistic và estimator.
2. Tính/giải thích mean, median, variance, standard deviation, quantile và ảnh hưởng của outlier.
3. Tính conditional probability từ event masks và không nhầm `P(A|B)` với `P(B|A)`.
4. Mô phỏng sampling distribution và giải thích đúng Central Limit Theorem.
5. Tự cài percentile bootstrap confidence interval có seed.
6. Chia train/validation/test thành ba tập disjoint, exhaustive và reproducible.
7. Nêu đúng vai trò của từng split và bảo vệ test set khỏi model selection.
8. Nhận diện target leakage, preprocessing leakage và test contamination.
9. Tạo underfit/fit/overfit bằng Polynomial Regression.
10. Dùng validation curve và learning curve để chẩn đoán bias/variance.
11. Chọn model bằng validation, refit train+validation và đánh giá test đúng một lần.

## 2. Definition of Done

- [ ] Hoàn thành hai notebook bằng **Restart Kernel → Run All**.
- [ ] Hoàn thành tối thiểu 8/10 bài tập độc lập.
- [ ] Cài `challenge/submission.py`; 12 public test groups pass.
- [ ] Bootstrap interval tái tạo được với seed và dùng replacement.
- [ ] Split index disjoint, exhaustive và đúng tỷ lệ.
- [ ] So sánh ít nhất 8 polynomial degrees bằng validation RMSE.
- [ ] Có learning curves cho một model underfit và một model variance cao.
- [ ] Test set không tham gia chọn degree/preprocessing.
- [ ] Quiz đạt ít nhất 16/20.
- [ ] Practical test đạt ít nhất 60/100.
- [ ] Báo cáo nêu rõ protocol, kết quả và ba nguy cơ leakage.

Critical fail:

- Chọn degree/hyperparameter bằng test set.
- Fit transformation trên toàn bộ data trước split.
- Dùng feature có sau prediction time.
- Chạy test nhiều lần rồi chỉ báo kết quả tốt nhất.
- Gọi confidence interval là xác suất 95% parameter nằm trong interval đã quan sát.
- Kết luận causal từ conditional probability/association.

## 3. Bắt đầu nhanh

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Linux/macOS/WSL2
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m data.generate_datasets
jupyter lab
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m data.generate_datasets
jupyter lab
```

## 4. Thứ tự học

1. Đọc `03-schedule.md` và làm diagnostic.
2. Đọc từng phần tương ứng trong `01-knowledge.md`.
3. Làm `notebooks/01-sampling-clt-bootstrap.ipynb`.
4. Làm `notebooks/02-bias-variance-model-selection.ipynb`.
5. Hoàn thành `exercises/README.md`.
6. Cài challenge theo contract trong `challenge/README.md`.
7. Làm quiz và practical trước khi mở `solutions/`.
8. Hoàn thành `report-template.md` và tự chấm rubric.

## 5. Protocol bắt buộc

```text
Raw data
   │
   ├── train (60%) ── fit parameters/models ─┐
   │                                         ├── chọn degree bằng validation
   ├── validation (20%) ─────────────────────┘
   │
   └── test (20%) ── khóa kín ── mở đúng một lần sau khi chốt model
```

Sau khi chọn degree: refit model mới trên `train + validation`, sau đó tính final test metric một lần. Không quay lại thay model dựa trên test result.

## 6. Lệnh chính

Chạy public tests trên bài làm:

```bash
python -m pytest tests/test_model_evaluation.py -q
```

Kiểm tra reference solution:

```bash
WEEK3_IMPL=solutions.model_evaluation_solution python -m pytest -q
```

Windows PowerShell:

```powershell
$env:WEEK3_IMPL="solutions.model_evaluation_solution"
python -m pytest -q
```

Chạy practical reference sau khi đã nộp bài:

```bash
python -m solutions.practical_test_solution
```

## 7. Datasets

| Tệp | Vai trò |
|---|---|
| `skewed_population.csv` | Sampling, CLT, conditional probability, bootstrap |
| `nonlinear_regression.csv` | Guided lab và challenge bias–variance |
| `practical_model_selection.csv` | Practical độc lập, chứa một feature leakage cố ý |
| `dataset_metadata.json` | Seed, contract và cơ chế sinh dữ liệu |

Tất cả dữ liệu là synthetic và tái tạo được bằng generator.

## 8. Quy tắc dùng lời giải

Chỉ mở `solutions/` sau khi:

- Tự viết bootstrap resampling.
- Có split invariants bằng assert/test.
- Tạo degree comparison không có test metric.
- Viết kết luận learning curve của mình.
- Nộp lần practical đầu tiên.

## 9. Output bàn giao

```text
outputs/
├── sampling_summary.csv
├── bootstrap_intervals.csv
├── degree_comparison.csv
├── learning_curve.csv
├── final_test_metrics.json
├── practical_predictions.csv
├── figures/
│   ├── sampling_distributions.png
│   ├── bootstrap_distribution.png
│   ├── degree_validation_curve.png
│   ├── polynomial_fits.png
│   ├── learning_curves.png
│   └── predicted_vs_actual.png
└── week03_report.md
```

## 10. Điểm qua tuần

- Tổng điểm: tối thiểu 75/100.
- Practical: tối thiểu 60/100.
- Quiz: tối thiểu 16/20.
- Public tests: 100% pass.
- Không có test contamination/leakage.
- Notebook chạy lại được từ trạng thái sạch.

## 11. Bản đồ tài liệu

```text
week-03-statistics-model-evaluation/
├── 01-knowledge.md
├── 02-reading-list.md
├── 03-schedule.md
├── notebooks/                    # 2 guided labs
├── exercises/                    # 10 bài độc lập + starter
├── challenge/                    # Leakage-safe workflow challenge
├── tests/                        # 12 public test groups
├── assessment/                   # Quiz, practical, rubric
├── solutions/                    # Đáp án + expected results
├── data/                         # 3 synthetic datasets + generator
├── report-template.md
└── VALIDATION.md
```
