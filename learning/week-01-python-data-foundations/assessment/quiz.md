# Quiz Tuần 1 — 20 câu, 20 phút

Đóng notebook và tài liệu. Mỗi câu 1 điểm. Đạt từ 16/20.

## Phần A — Trắc nghiệm (Câu 1–12)

### Câu 1

```python
x = np.zeros((2, 3, 4))
```

`x.ndim` bằng:

A. 2  
B. 3  
C. 4  
D. 24

### Câu 2

Với `x.shape == (2, 3, 4)`, `x.mean(axis=0).shape` là:

A. `(2,)`  
B. `(3, 4)`  
C. `(2, 4)`  
D. `(2, 3)`

### Câu 3

Phép toán nào broadcast hợp lệ?

A. `(100, 5) + (100,)`  
B. `(100, 5) + (5,)`  
C. `(100, 5) + (4,)`  
D. `(100, 5) + (5, 100)`

### Câu 4

Muốn trừ mean của từng dòng khỏi matrix `(100, 5)`, biểu thức an toàn nhất là:

A. `x - x.mean(axis=1)`  
B. `x - x.mean()`  
C. `x - x.mean(axis=1, keepdims=True)`  
D. `x - x.mean(axis=0, keepdims=True)`

### Câu 5

Phát biểu đúng về basic slicing NumPy:

A. Luôn tạo copy.  
B. Thường tạo view và có thể ảnh hưởng mảng gốc.  
C. Luôn chuyển dtype thành float.  
D. Chỉ hoạt động với mảng 1D.

### Câu 6

Một cột lẽ ra là số nhưng Pandas đọc thành `object`/string. Nguyên nhân có khả năng nhất:

A. File có quá ít dòng.  
B. Cột chứa ít nhất một giá trị không phải số như `free`.  
C. Có giá trị âm.  
D. Tên cột có khoảng trắng.

### Câu 7

`pd.to_numeric(series, errors="coerce")` làm gì với giá trị không parse được?

A. Xóa toàn bộ dòng.  
B. Giữ nguyên chuỗi.  
C. Chuyển thành missing/NaN.  
D. Chuyển thành 0.

### Câu 8

Mục đích chính của `merge(validate="many_to_one")` là:

A. Tự xóa duplicate.  
B. Xác nhận cardinality của join.  
C. Điền missing values.  
D. Sắp xếp kết quả.

### Câu 9

Trong GroupBy, phép nào đếm cả dòng có giá trị missing ở cột đang xét?

A. `count()`  
B. `mean()`  
C. `size()`  
D. `nunique()` mặc định

### Câu 10

Vì sao không nên gọi `df.dropna()` trên toàn bảng mà không có policy?

A. Vì Pandas không hỗ trợ.  
B. Vì có thể xóa dòng chỉ do một field không quan trọng bị thiếu.  
C. Vì nó luôn thay đổi input tại chỗ.  
D. Vì nó chỉ hoạt động với số.

### Câu 11

Hàm transformation nào dễ kiểm thử nhất?

A. Hàm sửa biến global và không return.  
B. Hàm phụ thuộc cell notebook trước.  
C. Hàm nhận input, copy có chủ đích và return output mới.  
D. Hàm đọc/ghi file ở mọi bước.

### Câu 12

Test nào bảo vệ data contract tốt nhất?

A. `assert len(df) == 41` và không có test khác.  
B. `assert df.head().equals(expected.head())`.  
C. Kiểm tra key unique, required non-null và numeric domain.  
D. In `df.info()` và nhìn bằng mắt.

## Phần B — Trả lời ngắn (Câu 13–16)

### Câu 13

Phân biệt missing value và invalid value. Cho một ví dụ mỗi loại từ orders dataset.

### Câu 14

Vì sao parse ngày lẫn định dạng bằng một lệnh tự động có thể nguy hiểm?

### Câu 15

Vì sao một cleaning function không nên mutate DataFrame đầu vào?

### Câu 16

Nêu bốn thành phần tối thiểu để một biểu đồ EDA có thể được dùng làm bằng chứng.

## Phần C — Đọc code (Câu 17–20)

### Câu 17

Không chạy code, viết output:

```python
x = np.array([[1, 2], [3, 4], [5, 6]])
print(x - x.mean(axis=0))
```

### Câu 18

Đoạn code sau có rủi ro gì và sửa thế nào?

```python
high_value = df[df["unit_price"] > 50]
high_value["flag"] = True
```

### Câu 19

Vì sao số `orders` nên dùng `nunique(order_id)` thay vì chỉ `count(order_id)` trong một bảng chưa loại duplicate?

### Câu 20

Một left join làm số dòng tăng từ 100 lên 108. Nêu nguyên nhân có khả năng nhất và một cách biến giả định thành kiểm tra thực thi.
