# Lịch học Tuần 2 — 10 giờ

| Phiên | Thời lượng | Nội dung | Đầu ra |
|---|---:|---|---|
| 0 | 30 phút | Setup, regenerate datasets, diagnostic | Môi trường và baseline tự đánh giá |
| 1 | 90 phút | Vector, matrix, dot product, norm, shape audit | Lab 1 phần đại số tuyến tính |
| 2 | 90 phút | Derivative, partial derivative, gradient, chain rule | Bài đạo hàm và computational graph |
| 3 | 75 phút | Linear model và MSE gradient | Derivation + gradient check đầu tiên |
| 4 | 120 phút | Gradient descent, learning rate, standardization | Lab 2 + loss curves |
| 5 | 60 phút | Bài tập độc lập | Tối thiểu 8/10 bài |
| 6 | 90 phút | Challenge và public tests | 12 tests pass; experiment table |
| 7 | 45 phút | Quiz, report, retrospective | Quiz ≥16/20; report hoàn chỉnh |
| **Tổng** | **600 phút** |  | **10 giờ** |

## Phiên 0 — Setup và diagnostic

1. Tạo môi trường hoặc dùng môi trường Tuần 1.
2. Chạy `python -m data.generate_datasets`.
3. Không tra cứu, trả lời:
   - Shape của `(100, 5) @ (5,)`?
   - Dot product và element-wise multiplication khác nhau thế nào?
   - Derivative và gradient khác nhau thế nào?
   - Gradient descent cập nhật parameter theo dấu nào?
4. Commit: `chore: initialize week 2 workspace`.

## Phiên 1 — Linear algebra

- 20 phút đọc kiến thức.
- 55 phút notebook 1.
- 10 phút làm lại shape audit không chạy code.
- 5 phút commit.

Stop condition: ghi đúng shape của mọi operand/output trong `X.T @ (X @ theta - y)`.

## Phiên 2 — Calculus trực quan

- 25 phút derivative và partial derivative.
- 25 phút finite difference bằng code.
- 25 phút chain rule trên computational graph.
- 15 phút tự suy ra derivative của ba hàm đơn giản.

## Phiên 3 — MSE gradient

- 20 phút viết derivation trên giấy.
- 20 phút cài analytical gradient.
- 25 phút cài central finite differences.
- 10 phút thử ba epsilon và ghi relative error.

Không sang gradient descent nếu gradient check chưa đạt.

## Phiên 4 — Optimization lab

- 25 phút gradient descent trên single feature.
- 30 phút learning-rate experiment.
- 30 phút standardized vs unscaled features.
- 20 phút so sánh với `lstsq`.
- 15 phút vẽ và kết luận từ loss curves.

## Phiên 5 — Bài tập độc lập

- Làm theo thứ tự 1–10.
- Nếu kẹt trên 10 phút, ghi hypothesis và chuyển bài.
- Bài 5–9 là critical core; không được bỏ toàn bộ.

## Phiên 6 — Challenge

- 20 phút đọc contract và shape table.
- 50 phút cài đặt/test theo vòng lặp đỏ → xanh.
- 20 phút chạy housing experiment và xuất coefficients/metrics.

## Phiên 7 — Đánh giá

- 20 phút quiz đóng tài liệu.
- 15 phút report và failure analysis.
- 10 phút tự chấm rubric/học bù.

Practical test 75 phút nên làm riêng. Nếu quỹ thời gian giới hạn đúng 10 giờ, thay 60 phút bài tập ở Phiên 5 bằng practical test và chuyển bài chưa làm sang bonus.

## Bonus — 2 đến 4 giờ

1. Vẽ contour loss theo intercept/slope và đường đi gradient descent.
2. Cài momentum và so sánh số step.
3. Thêm early stopping theo gradient norm.
4. Tạo near-collinear features và quan sát coefficients/condition number.
5. So sánh `lstsq`, pseudo-inverse và gradient descent.

