# Quiz Tuần 2 — 20 câu, 25 phút

Đóng notebook và lời giải. Mỗi câu 1 điểm. Đạt từ 16/20.

## Phần A — Trắc nghiệm (Câu 1–12)

### Câu 1

Shape của `(80, 5) @ (5,)` là:

A. `(5,)`  
B. `(80,)`  
C. `(80, 5)`  
D. Không hợp lệ

### Câu 2

Phép `(m, k) @ (k, n)` tạo output:

A. `(m, n)`  
B. `(k, k)`  
C. `(m, k)`  
D. `(n, m)`

### Câu 3

Với vector 1D `a` và `b` cùng shape, `a @ b` trả:

A. Vector element-wise  
B. Scalar dot product  
C. Outer product matrix  
D. Transpose

### Câu 4

Biểu thức nào bằng squared L2 norm của `x`?

A. `np.linalg.norm(x)`  
B. `np.sum(np.abs(x))`  
C. `x @ x`  
D. `x.mean()`

### Câu 5

Nếu `f(x)=x²`, derivative tại `x=3` là:

A. 3  
B. 6  
C. 9  
D. 12

### Câu 6

Gradient của loss theo theta có shape:

A. Shape của y  
B. Shape của X  
C. Shape của theta  
D. Luôn scalar

### Câu 7

Chain rule dùng khi:

A. Hàm là composition của nhiều phép biến đổi.  
B. Chỉ có một scalar constant.  
C. Cần sort dữ liệu.  
D. Cần matrix inverse.

### Câu 8

Gradient đúng của MSE `mean((X @ theta - y)**2)` là:

A. `X.T @ residual`  
B. `2/n * X.T @ residual`  
C. `2/n * X @ residual`  
D. `residual.T @ X @ theta`

### Câu 9

Central finite difference cho parameter `j` là:

A. `(L(theta+eps)-L(theta))/eps`  
B. `(L(theta+eps)-L(theta-eps))/(2eps)`  
C. `(L(theta+eps)+L(theta-eps))/(2eps)`  
D. `L(theta)/eps`

### Câu 10

Gradient descent muốn giảm loss nên update:

A. `theta += lr * gradient`  
B. `theta -= lr * gradient`  
C. `theta = gradient`  
D. `gradient -= theta`

### Câu 11

Learning rate quá lớn thường gây:

A. Loss giảm chậm nhưng monotonic.  
B. Loss dao động/tăng hoặc thành non-finite.  
C. Gradient luôn bằng zero.  
D. Feature tự standardize.

### Câu 12

Vì sao standardization thường giúp gradient descent?

A. Làm target không còn noise.  
B. Đưa feature về scale gần nhau, cải thiện hình học loss surface.  
C. Tăng số sample.  
D. Tự thêm regularization.

## Phần B — Trả lời ngắn (Câu 13–16)

### Câu 13

Giải thích bằng lời vì sao `X.T @ residual` chuyển một residual trên mỗi sample thành một gradient trên mỗi parameter.

### Câu 14

Vì sao numerical gradient phù hợp để test nhưng không phù hợp để train model nhiều parameter?

### Câu 15

Phân biệt MSE và RMSE về công thức và đơn vị.

### Câu 16

Nêu ba bước debug đầu tiên khi loss không giảm.

## Phần C — Đọc code/toán (Câu 17–20)

### Câu 17

Không chạy code, viết output:

```python
X = np.array([[1., 2.], [3., 4.]])
theta = np.array([10., -2.])
print(X @ theta)
```

### Câu 18

Với một sample `x=3`, `y=10`, `w=2`, `b=1`, loss là `(wx+b-y)^2`. Tính `dL/dw` và `dL/db`.

### Câu 19

Đoạn code sau sai ở đâu?

```python
residual = X @ theta - y
gradient = (2 / len(y)) * (X @ residual)
```

### Câu 20

Một model train trên standardized X lưu `theta_z`, `mean`, `scale`. Viết công thức chuyển coefficients và intercept về raw feature units.

