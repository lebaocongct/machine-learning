# Đáp án Quiz Tuần 2

## Phần A

1. **B** — mỗi trong 80 rows dot với vector 5 phần tử.
2. **A** — inner dimensions `k` triệt tiêu, còn `(m, n)`.
3. **B** — hai vector 1D dùng `@` cho scalar dot product.
4. **C** — với vector thực, `x @ x = sum(x²)`.
5. **B** — derivative `2x`, tại 3 bằng 6.
6. **C** — một partial derivative cho mỗi parameter.
7. **A** — chain rule truyền derivative qua composition.
8. **B** — factor `2` từ square, chia `n` từ mean.
9. **B** — central difference dùng perturbation hai phía.
10. **B** — đi theo âm gradient để giảm loss cục bộ.
11. **B** — overshoot dẫn đến dao động/tăng/divergence.
12. **B** — scale gần nhau làm optimization geometry bớt kéo dài.

## Phần B

13. Residual có một giá trị cho mỗi sample. Nhân `X.T` lấy weighted sum residual theo từng feature, vì vậy output có một phần tử cho mỗi parameter.
14. Numerical gradient cần hai lần tính loss cho mỗi parameter, nên chi phí tăng tuyến tính theo số parameter và bị ảnh hưởng bởi epsilon/floating-point. Nó phù hợp làm oracle kiểm tra trên bài nhỏ.
15. `MSE = mean(error²)` có đơn vị target bình phương; `RMSE = sqrt(MSE)` có cùng đơn vị target.
16. Kiểm tra gradient bằng finite differences; kiểm tra dấu/factor/shape của update; kiểm tra scale và giảm learning rate. Ngoài ra kiểm tra input finite.

## Phần C

17.

```text
[ 6. 22.]
```

18. Prediction `7`, residual `-3`, loss `9`:

```text
dL/dw = 2 * residual * x = 2 * (-3) * 3 = -18
dL/db = 2 * residual = -6
```

19. Phải dùng `X.T @ residual`, không phải `X @ residual`. Shape đúng là `(d,n) @ (n,) -> (d,)`.
20.

```text
coef_raw[j] = theta_z[j+1] / scale[j]
intercept_raw = theta_z[0] - sum(coef_raw * mean)
```

