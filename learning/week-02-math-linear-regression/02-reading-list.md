# Reading list Tuần 2

## Cách đọc

Mục tiêu là hiểu đủ để code và debug, không đọc hết giáo trình toán. Với mỗi nguồn, tự viết một ví dụ NumPy và ghi shape của mọi biến.

## Bắt buộc — khoảng 90 phút

### Đại số tuyến tính — 35 phút

1. [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
   - Đọc: `@`, matrix/vector products, norms, least-squares.
   - Ghi lại shape rule của matrix multiplication.
2. [`numpy.dot`](https://numpy.org/doc/stable/reference/generated/numpy.dot.html)
   - Phân biệt dot của hai vector và matrix multiplication.
3. [`numpy.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)
   - Tập trung vào vector L2 norm.

### Toán cho ML — 40 phút

1. [Mathematics for Machine Learning](https://mml-book.github.io/)
   - Chương 2: Linear Algebra — đọc chọn lọc vector, matrix, inner product.
   - Chương 5: Vector Calculus — derivative, partial derivative, gradient, chain rule.
   - Chương 7: Continuous Optimization — gradient descent trực giác.
   - Chương 9: Linear Regression — liên hệ công thức với bài lab.

Không cần làm toàn bộ chứng minh trong tuần này.

### Gradient và least squares — 15 phút

1. [`numpy.linalg.lstsq`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
   - Dùng làm reference solution, không dùng thay challenge gradient descent.
2. [TensorFlow automatic differentiation](https://www.tensorflow.org/guide/autodiff)
   - Chỉ đọc phần mở đầu để thấy chain rule được tự động hóa thế nào; chưa cần code TensorFlow.

## Mở rộng — sau khi hoàn thành phần bắt buộc

- [MIT OpenCourseWare — Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
- [Scikit-Learn `LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [NumPy `matmul`](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
- [NumPy `linalg.solve`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html)
- [TensorFlow GradientTape basics](https://www.tensorflow.org/guide/autodiff)

## Câu hỏi định hướng

1. Inner dimension trong matrix multiplication có vai trò gì?
2. Dot product liên quan weighted sum như thế nào?
3. Gradient là scalar, vector hay matrix trong regression của bài?
4. Tại sao đi theo âm gradient?
5. Numerical gradient có nguồn sai số nào?
6. Tại sao scaling ảnh hưởng gradient descent nhưng không thay đổi bài toán tuyến tính cơ bản?
7. `lstsq` trả nghiệm gì khi hệ overdetermined?

## Nhật ký đọc

```text
Khái niệm:
Shape example:
Công thức tự viết:
Điểm chưa chắc:
Test hoặc experiment để xác minh:
```

