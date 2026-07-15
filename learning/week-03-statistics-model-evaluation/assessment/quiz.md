# Quiz Tuần 3 — 20 câu, 25 phút

Đóng notebook và lời giải. Mỗi câu 1 điểm. Đạt từ 16/20.

## Phần A — Trắc nghiệm (Câu 1–12)

### Câu 1

Mean của `[1, 2, 3, 10]` là:

A. 2,5  
B. 3  
C. 4  
D. 4,5

### Câu 2

NumPy sample variance không chệch thường dùng:

A. `np.var(x, ddof=0)`  
B. `np.var(x, ddof=1)`  
C. `np.mean(x)`  
D. `np.std(x) ** 0.5`

### Câu 3

Event `B` xảy ra 40 lần; `A ∩ B` xảy ra 10 lần. `P(A|B)` là:

A. 0,10  
B. 0,20  
C. 0,25  
D. 0,40

### Câu 4

Phát biểu đúng về CLT:

A. Mọi raw feature trở thành Normal khi có nhiều rows.  
B. Sampling distribution của mean tiến gần Normal dưới các điều kiện thích hợp.  
C. Sample mean luôn bằng population mean.  
D. CLT loại bỏ outlier.

### Câu 5

Nếu sample size tăng từ `n` lên `4n`, standard error của mean xấp xỉ:

A. Tăng 4 lần  
B. Tăng 2 lần  
C. Giảm một nửa  
D. Không đổi

### Câu 6

Bootstrap resample chuẩn trong bài dùng:

A. Không replacement, size nhỏ hơn sample.  
B. Có replacement, cùng size sample gốc.  
C. Chỉ sample từ test set.  
D. Luôn giả định Normal.

### Câu 7

Tập nào dùng để chọn polynomial degree?

A. Train  
B. Validation  
C. Test  
D. Toàn bộ data

### Câu 8

Hành động nào hợp lệ?

A. Fit scaler trên toàn data rồi split.  
B. Chọn seed có test RMSE thấp nhất.  
C. Fit scaler trên train, transform validation/test.  
D. Thêm target vào feature để kiểm tra pipeline.

### Câu 9

Train RMSE rất thấp, validation RMSE cao và gap lớn thường gợi ý:

A. High bias  
B. High variance  
C. Không có noise  
D. Perfect generalization

### Câu 10

Train và validation error cùng cao, gần nhau và plateau thường gợi ý:

A. High bias  
B. High variance  
C. Target leakage  
D. Test contamination chắc chắn

### Câu 11

Validation curve thay đổi chủ yếu:

A. Training set size  
B. Hyperparameter/model complexity  
C. Test target  
D. Random label

### Câu 12

Learning curve thay đổi chủ yếu:

A. Training set size  
B. Target definition mỗi điểm  
C. Test set  
D. Metric mỗi điểm

## Phần B — Trả lời ngắn (Câu 13–16)

### Câu 13

Phân biệt population parameter và sample statistic bằng một ví dụ.

### Câu 14

Viết diễn giải đúng cho một frequentist 95% confidence interval.

### Câu 15

Vì sao xem test metric rồi đổi model làm test estimate bị lạc quan?

### Câu 16

Nêu bốn invariants cần kiểm tra cho train/validation/test indices.

## Phần C — Tình huống/code (Câu 17–20)

### Câu 17

Đoạn code sau sai gì?

```python
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
```

### Câu 18

Một feature `days_until_payment` chỉ biết 30 ngày sau quyết định tín dụng. Có được dùng để dự đoán default tại ngày quyết định không? Phân loại lỗi.

### Câu 19

Bạn thử degree 1–20. Degree 12 có train RMSE `0.4`, validation RMSE `1.8`; degree 5 có `1.0` và `1.1`. Chọn degree nào và vì sao?

### Câu 20

Sau khi chọn model bằng validation, viết thứ tự hai bước cuối cùng trước khi bàn giao final metric.
