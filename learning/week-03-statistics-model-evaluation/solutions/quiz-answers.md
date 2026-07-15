# Đáp án Quiz Tuần 3

## Phần A

1. **C** — `(1+2+3+10)/4 = 4`.
2. **B** — `ddof=1` dùng mẫu số `n-1`.
3. **C** — `10/40 = 0,25`.
4. **B** — CLT nói về sampling distribution của statistic dưới điều kiện thích hợp.
5. **C** — `SE ∝ 1/sqrt(n)`, tăng size 4 lần làm SE giảm 2 lần.
6. **B** — bootstrap có replacement và giữ size.
7. **B** — validation chọn hyperparameter.
8. **C** — fit train; chỉ transform holdouts.
9. **B** — train–validation gap lớn là dấu hiệu high variance.
10. **A** — cả hai error cao/gần nhau thường là high bias.
11. **B** — validation curve thay model complexity/hyperparameter.
12. **A** — learning curve thay training set size.

## Phần B

### Câu 13

Population mean `μ` của toàn bộ khách hàng là parameter. Mean `x̄` tính từ 100 khách hàng được lấy mẫu là statistic dùng để estimate `μ`.

### Câu 14

Nếu lặp lại sampling và cùng quy trình tạo interval rất nhiều lần dưới các giả định, khoảng 95% các interval sẽ chứa parameter thật. Không nói parameter ngẫu nhiên có 95% xác suất nằm trong interval đã quan sát.

### Câu 15

Khi test feedback ảnh hưởng lựa chọn, model được tối ưu cả theo noise riêng của test. Metric test không còn là đánh giá độc lập và thường lạc quan.

### Câu 16

Các indices phải: không overlap; hợp đầy đủ mọi row; đều trong range; cùng seed tái tạo; thêm alignment X/y/ID là kiểm tra tốt.

## Phần C

### Câu 17

Scaler học mean/std từ cả future test rows trước split: preprocessing leakage. Sửa bằng split trước, fit scaler trên train, transform holdouts; tốt hơn dùng Pipeline trong model selection.

### Câu 18

Không được dùng. Đây là target/temporal leakage vì feature chưa có tại prediction timestamp.

### Câu 19

Chọn degree 5 vì validation RMSE `1.1` thấp hơn `1.8`. Degree 12 fit train tốt nhưng variance gap lớn.

### Câu 20

Refit model đã đóng băng trên train+validation; sau đó mở test và tính final metric đúng một lần, không quay lại thay model.
