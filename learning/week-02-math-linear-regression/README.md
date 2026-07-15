# Tuần 2 — Toán cho ML và Linear Regression bằng NumPy

> Thời lượng chính: 10 giờ  
> Đối tượng: lập trình viên đã hoàn thành Tuần 1 hoặc có nền tảng NumPy tương đương  
> Trọng tâm: vector, matrix, dot product, norm, derivative, gradient, chain rule và gradient descent

## 1. Kết quả cần đạt

Sau tuần này, bạn phải có thể:

1. Phân biệt scalar, vector, matrix và tensor bằng shape.
2. Tính và giải thích dot product, matrix multiplication, transpose và norm.
3. Biểu diễn linear model dưới dạng `y_hat = X @ theta`.
4. Giải thích derivative, partial derivative, gradient và chain rule.
5. Tự suy ra gradient của Mean Squared Error.
6. Cài analytical gradient và kiểm tra bằng central finite differences.
7. Huấn luyện Linear Regression bằng full-batch gradient descent với NumPy.
8. Giải thích ảnh hưởng của learning rate và feature scale qua loss curves.
9. So sánh nghiệm gradient descent với `np.linalg.lstsq`.
10. Debug các lỗi shape, gradient sai, loss tăng hoặc loss không hữu hạn.

## 2. Definition of Done

- [ ] Hoàn thành hai notebook trong `notebooks/` bằng Restart/Run All.
- [ ] Hoàn thành tối thiểu 8/10 bài tập độc lập.
- [ ] Cài đặt `challenge/submission.py` và làm 12 public tests pass.
- [ ] Analytical và numerical gradient có relative error `< 1e-6`.
- [ ] Chạy controlled experiment với tối thiểu bốn learning rate.
- [ ] Chứng minh standardization thay đổi tốc độ/độ ổn định hội tụ.
- [ ] Practical test đạt tối thiểu 60/100.
- [ ] Quiz đạt ít nhất 16/20.
- [ ] Hoàn thành experiment report và phân tích ba failure modes.
- [ ] Có ít nhất ba Git commit có ý nghĩa.

Critical fail:

- Gradient được hard-code hoặc sao chép từ numerical gradient.
- Loss giảm nhưng gradient analytical không qua gradient check.
- Chọn learning rate chỉ từ final metric mà không xem curve/hội tụ.
- Dùng `np.linalg.inv(X.T @ X)` làm cách mặc định mà không xét numerical stability.
- Code phụ thuộc notebook state và không chạy lại từ đầu.

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

Xác nhận:

```bash
python -c "import numpy, pandas, matplotlib; print('environment: OK')"
python -m pytest --version
```

## 4. Thứ tự học

1. Đọc `03-schedule.md` và làm diagnostic.
2. Đọc các phần tương ứng trong `01-knowledge.md`.
3. Làm `notebooks/01-linear-algebra-for-ml.ipynb`.
4. Làm `notebooks/02-gradient-descent-linear-regression.ipynb`.
5. Hoàn thành `exercises/README.md`.
6. Đọc contract trong `challenge/README.md` và cài starter.
7. Chạy gradient check trước khi chạy gradient descent dài.
8. Làm quiz và practical test trước khi mở `solutions/`.
9. Hoàn thành `report-template.md` và tự chấm rubric.

## 5. Các lệnh chính

Chạy public tests trên bài làm:

```bash
python -m pytest tests/test_linear_regression.py -q
```

Kiểm tra reference solution:

```bash
WEEK2_IMPL=solutions.linear_regression_solution python -m pytest -q
```

Windows PowerShell:

```powershell
$env:WEEK2_IMPL="solutions.linear_regression_solution"
python -m pytest -q
```

Chạy practical reference sau khi đã nộp bài:

```bash
python -m solutions.practical_test_solution
```

## 6. Datasets

| Tệp | Vai trò |
|---|---|
| `single_feature_regression.csv` | Trực quan line, slope, intercept và loss surface |
| `housing_regression.csv` | Challenge multi-feature với feature scale khác nhau |
| `practical_regression.csv` | Dataset độc lập cho practical test |
| `dataset_metadata.json` | Seed, hệ số sinh dữ liệu và đơn vị tham chiếu |

Tất cả dữ liệu là synthetic và có thể tái tạo bằng `python -m data.generate_datasets`.

## 7. Quy tắc dùng lời giải

Chỉ mở `solutions/` sau khi:

- Viết analytical gradient của mình.
- Chạy gradient check và ghi relative error.
- Có ít nhất một test fail đã được phân tích root cause.
- Thử sửa ít nhất một lần.

Reference solution không phải cách triển khai duy nhất. Bài làm đạt nếu thỏa contract và invariants.

## 8. Output bàn giao

```text
outputs/
├── learning_rate_experiment.csv
├── housing_predictions.csv
├── practical_predictions.csv
├── figures/
│   ├── loss_by_learning_rate.png
│   ├── fitted_line.png
│   └── predicted_vs_actual.png
└── week02_report.md
```

## 9. Điểm qua tuần

- Tổng điểm: tối thiểu 75/100.
- Practical test: tối thiểu 60/100.
- Quiz: tối thiểu 16/20.
- Public tests: 100% pass.
- Gradient relative error: `< 1e-6`.
- Không vi phạm critical fail.

## 10. Bản đồ tài liệu

```text
week-02-math-linear-regression/
├── 01-knowledge.md                 # Giáo trình toán + optimization
├── 02-reading-list.md              # Nguồn chính thức, thứ tự đọc
├── 03-schedule.md                  # Lịch 10 giờ theo từng phiên
├── notebooks/                      # 2 guided labs có self-check
├── exercises/                      # 10 bài độc lập + starter
├── challenge/                      # Challenge API và code khởi đầu
├── tests/                          # 12 public test groups
├── assessment/                     # Quiz, practical và rubric
├── solutions/                      # Đáp án, code và kết quả tham chiếu
├── data/                           # 3 dataset synthetic + generator
├── report-template.md              # Mẫu báo cáo thí nghiệm
└── VALIDATION.md                   # Bằng chứng nghiệm thu gói tài liệu
```

Đọc `VALIDATION.md` để xem phiên bản môi trường, test result, dataset checksum và các metrics đã đối chiếu.
