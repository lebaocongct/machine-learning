# Báo cáo Tuần 2

## 1. Thông tin

- Người học:
- Ngày bắt đầu/kết thúc:
- Thời gian thực tế:
- Python/NumPy version:
- Commit hoàn thành:

## 2. Shape contract

| Biến | Ý nghĩa | Shape thực tế | Shape kỳ vọng |
|---|---|---|---|
| X | Design matrix raw |  |  |
| X_scaled | Standardized features |  |  |
| X_bias | Design matrix có bias |  |  |
| y | Target |  |  |
| theta | Parameters |  |  |
| predictions | Model output |  |  |
| residual | Prediction error |  |  |
| gradient | MSE gradient |  |  |

## 3. Derivation

Viết lại bằng ngôn ngữ/ký hiệu của bạn:

$$
J(\theta)=\frac{1}{n}\|X\theta-y\|_2^2
$$

$$
\nabla_\theta J(\theta)=
$$

Giải thích factor `2/n`:

## 4. Gradient check

- Theta dùng để test:
- Epsilon:
- Analytical gradient:
- Numerical gradient:
- Relative error:
- Kết luận:

| Epsilon | Relative error | Nhận xét |
|---:|---:|---|
| 1e-4 |  |  |
| 1e-5 |  |  |
| 1e-6 |  |  |
| 1e-7 |  |  |

## 5. Learning-rate experiment

Hypothesis:

| Learning rate | Initial loss | Final loss | Step đạt 1% optimum | Stable? | Kết luận |
|---:|---:|---:|---:|---|---|
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |

## 6. Standardization experiment

- Unscaled learning rate:
- Standardized learning rate:
- Unscaled final loss/steps:
- Standardized final loss/steps:
- Vì sao khác nhau:

## 7. Reference comparison

| Hạng mục | Gradient descent | `np.linalg.lstsq` | Sai khác |
|---|---:|---:|---:|
| Intercept |  |  |  |
| area_sqm |  |  |  |
| bedrooms |  |  |  |
| age_years |  |  |  |
| distance_km |  |  |  |
| energy_score |  |  |  |
| MSE |  |  |  |

## 8. Failure analysis

| Failure | Triệu chứng | Root cause | Test/guard ngăn lặp lại |
|---|---|---|---|
| Gradient sai |  |  |  |
| Learning rate quá lớn |  |  |  |
| Feature không scale |  |  |  |

## 9. Kết quả đánh giá

- Public tests: ___/12 pass
- Gradient relative error:
- Quiz: ___/20
- Practical test: ___/100
- Tổng điểm rubric: ___/100
- Critical fail: Có/Không

## 10. Retrospective

- Điều tôi tự suy ra được:
- Điều tôi vẫn phải tra cứu:
- Lỗi shape khó nhất:
- Một insight về optimization:
- Kế hoạch học bù:

