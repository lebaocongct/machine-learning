# Rubric Tuần 2

## 1. Điểm tổng tuần — 100 điểm

| Thành phần | Điểm | Full credit |
|---|---:|---|
| Hai guided labs | 20 | Run All; shape audit; gradient/loss plots và kết luận |
| Bài tập độc lập | 20 | ≥8/10; bài 5–9 đúng; self-check rõ |
| Challenge + experiments | 30 | 12 tests pass; gradient check; LR/scaling experiments; reference comparison |
| Practical test | 20 | Quy đổi theo practical score; tối thiểu 60/100 |
| Quiz và giải thích | 10 | ≥16/20; giải thích lại câu sai |

Điều kiện qua: tổng ≥75, practical ≥60, gradient check pass, toàn bộ public tests pass và không critical fail.

## 2. Critical fail

- Analytical gradient gọi numerical gradient hoặc hard-code.
- Relative gradient error không đạt `<1e-6` nhưng vẫn tiếp tục kết luận model đúng.
- Sửa dataset/expected results để test pass.
- Sai dấu update hoặc sai shape nhưng được che bằng `ravel()` tùy ý.
- Loss non-finite mà training tiếp tục im lặng.
- Không so sánh với least-squares reference.
- Notebook không chạy lại từ trạng thái sạch.

## 3. Rubric practical test — 100 điểm

| Hạng mục | Điểm | Full credit |
|---|---:|---|
| Shape + baseline | 15 | X/y/bias/theta đúng; constant baseline đúng |
| Gradient check | 25 | Analytical/numerical độc lập; central difference; relative error đạt |
| Gradient descent | 30 | History đúng; 5 LR; detect divergence; lựa chọn có lý do |
| Parameters + metrics | 20 | Raw coefficients; `lstsq` parity; MSE/RMSE/MAE/R² đúng |
| Visualization/code | 10 | Hai figure đúng; hàm nhỏ, main, asserts |

## 4. Mức chất lượng

| Mức | Biểu hiện |
|---|---|
| 0–49 | Không hoàn thiện gradient/model hoặc output sai cơ bản |
| 50–64 | Chạy được một phần nhưng thiếu gradient check/shape discipline |
| 65–74 | Hầu hết đúng nhưng chưa đủ quality gate |
| 75–84 | Đạt chuẩn: code/test/experiments tái tạo được |
| 85–94 | Tốt: reasoning rõ, robust guards, curves/metrics đúng |
| 95–100 | Xuất sắc: thêm convergence/condition analysis có giá trị |

## 5. Feedback template

```text
Điểm mạnh:
-

Lỗi quan trọng nhất:
-

Evidence:
- Test/shape/curve:

Học bù bắt buộc:
-

Điều kiện re-check:
-
```

