# Tuần 1 — Python cho dữ liệu, EDA và code có thể kiểm thử

> Thời lượng chính: 10 giờ  
> Đối tượng: lập trình viên đã biết Python  
> Trọng tâm: NumPy, Pandas, Matplotlib, môi trường tái tạo được, Git và pytest

## 1. Kết quả cần đạt

Sau tuần này, bạn phải có thể:

1. Đọc và giải thích `shape`, `ndim`, `dtype`, `axis` của một `ndarray`.
2. Dùng slicing, boolean mask, vectorization và broadcasting thay cho loop cơ bản.
3. Nạp, kiểm tra, lọc, tổng hợp và kết hợp bảng bằng Pandas.
4. Nhận diện missing value, duplicate, kiểu dữ liệu sai và giá trị vi phạm business rule.
5. Thực hiện EDA có câu hỏi, không chỉ gọi `describe()` rồi dừng lại.
6. Viết transformation không làm thay đổi dữ liệu đầu vào và kiểm tra bằng pytest.
7. Tạo biểu đồ có tiêu đề, nhãn, đơn vị và kết luận đi kèm.
8. Tạo môi trường ảo, khóa dependency và lưu tiến độ bằng Git.

## 2. Definition of Done

Tuần 1 chỉ hoàn thành khi:

- [ ] Hoàn thành hai notebook trong `notebooks/`.
- [ ] Hoàn thành tối thiểu 8/10 bài trong `exercises/README.md`.
- [ ] Tự cài đặt `challenge/submission.py` và làm toàn bộ public tests pass.
- [ ] Hoàn thành practical test trong 60 phút trước khi xem lời giải.
- [ ] Quiz đạt ít nhất 16/20 câu.
- [ ] Có EDA report trả lời đủ năm câu hỏi trong `report-template.md`.
- [ ] Giải thích được `shape`, `dtype`, broadcasting và vì sao test transformation.
- [ ] Có ít nhất ba commit Git có ý nghĩa.

Critical fail:

- Sửa dữ liệu gốc bằng tay để làm test pass.
- Dùng loop cho bài toán vectorization bắt buộc mà không giải thích.
- Xóa hàng lỗi nhưng không ghi rõ rule.
- Code chỉ chạy khi thực hiện notebook theo một thứ tự trạng thái bí mật.

## 3. Bắt đầu nhanh

### Linux, macOS hoặc WSL2

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

Xác nhận:

```bash
python -c "import numpy, pandas, matplotlib; print('environment: OK')"
python -m pytest --version
```

Sau khi môi trường ổn định:

```bash
python -m pip freeze > requirements.lock.txt
git init
git add .
git commit -m "chore: initialize week 1 workspace"
```

Không commit thư mục `.venv/`.

## 4. Thứ tự học đề xuất

1. Đọc `03-schedule.md` và làm diagnostic đầu tuần.
2. Đọc `01-knowledge.md`, chỉ chạy các ví dụ ngắn cần thiết.
3. Hoàn thành `notebooks/01-numpy-vectorization.ipynb`.
4. Hoàn thành `notebooks/02-guided-eda.ipynb`.
5. Làm `exercises/README.md` không xem đáp án.
6. Đọc `challenge/README.md`, cài đặt `challenge/submission.py` và chạy test.
7. Làm quiz rồi practical test.
8. Chỉ sau đó mới mở `solutions/` để đối chiếu.
9. Hoàn thành `report-template.md` và tự chấm theo rubric.

## 5. Lệnh quan trọng

Chạy public tests trên bài làm:

```bash
python -m pytest tests/test_data_cleaning.py -q
```

Kiểm tra reference solution:

```bash
WEEK1_IMPL=solutions.data_cleaning_solution python -m pytest -q
```

Windows PowerShell:

```powershell
$env:WEEK1_IMPL="solutions.data_cleaning_solution"
python -m pytest -q
```

## 6. Bộ dữ liệu

| Tệp                              | Mục đích                                                   |
| -------------------------------- | ---------------------------------------------------------- |
| `data/customer_orders_raw.csv`   | Lab EDA và challenge làm sạch dữ liệu                      |
| `data/customers.csv`             | Thực hành `merge` và kiểm tra unmatched key                |
| `data/practical_test_orders.csv` | Dữ liệu kiểm tra cuối tuần; không dùng khi luyện challenge |

Dữ liệu được tạo riêng cho khóa học, không chứa thông tin cá nhân thật.

## 7. Quy tắc sử dụng đáp án

Đáp án là công cụ phản hồi, không phải đường tắt. Trước khi mở `solutions/`:

- Ghi lại cách làm đầu tiên.
- Ghi lại lỗi hoặc test đang fail.
- Dự đoán nguyên nhân.
- Thử sửa ít nhất một lần.

Khi đối chiếu, không chép nguyên lời giải. Hãy viết lại implementation bằng cấu trúc của mình và giải thích vì sao nó đúng.

## 8. Đầu ra bàn giao

```text
outputs/
├── cleaned_orders.csv
├── data_quality_report.json
├── figures/
│   ├── revenue_by_category.png
│   └── orders_by_region.png
├── eda_report.md
└── practical_test_submission.py
```

## 9. Điểm qua tuần

- Tổng điểm: tối thiểu 75/100.
- Practical test: tối thiểu 60/100.
- Quiz: tối thiểu 16/20.
- Public tests: 100% pass.
- Không vi phạm critical fail.
