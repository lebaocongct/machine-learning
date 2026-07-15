# Bài tập độc lập Tuần 1

## Quy tắc

- Làm tối thiểu 8/10 bài.
- Bài 1–8 là bắt buộc để đạt mức chuẩn; bài 9–10 củng cố EDA và performance.
- Trước khi chạy code, ghi shape/dtype/output dự đoán.
- Không dùng `DataFrame.apply(axis=1)` cho bài có thể vectorize.
- Lưu code tái sử dụng vào `exercises/starter.py`.

## Bài 1 — Array anatomy (5 điểm)

```python
import numpy as np

x = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
```

Không chạy code trước khi trả lời:

1. `x.shape`, `x.ndim`, `x.size`, `x.dtype` là gì?
2. Shape của `x[0]`, `x[:, 1]`, `x[..., -1]` là gì?
3. Tính tổng theo trục cuối và xác nhận shape.
4. Đổi shape thành `(6, 4)` mà không làm thay đổi thứ tự dữ liệu.

Self-check:

```python
assert x.sum(axis=-1).shape == (2, 3)
assert x.reshape(6, 4)[-1, -1] == 23
```

## Bài 2 — Boolean mask và copy/view (5 điểm)

Cho:

```python
prices = np.array([12.5, 55.0, 8.0, 120.0, 33.5, 75.0])
```

1. Lấy các giá trị trong `[20, 80]` bằng boolean mask.
2. Tạo mảng mới giảm 10% cho các giá trị trên 50, không sửa `prices`.
3. Tạo một slice là view, sửa view và chứng minh mảng gốc thay đổi.
4. Sửa lại bằng `.copy()`.

## Bài 3 — Broadcasting và chuẩn hóa (10 điểm)

Cài đặt `column_zscore(matrix)` trong `starter.py`:

```python
z = (x - mean_by_column) / std_by_column
```

Yêu cầu:

- Không loop.
- Dùng population standard deviation (`ddof=0`).
- Raise `ValueError` nếu input không phải ma trận 2D.
- Không chia cho 0: cột constant phải trả toàn số `0` sau chuẩn hóa.

Self-check:

```python
x = np.array([[1., 10., 5.], [3., 20., 5.], [5., 30., 5.]])
z = column_zscore(x)
np.testing.assert_allclose(z[:, :2].mean(axis=0), 0, atol=1e-12)
np.testing.assert_allclose(z[:, :2].std(axis=0), 1, atol=1e-12)
np.testing.assert_allclose(z[:, 2], 0)
```

## Bài 4 — Revenue vectorization (10 điểm)

Cài đặt `compute_net_revenue`:

```text
gross = quantity × unit_price
net = gross × (1 − discount_pct / 100)
```

Yêu cầu:

- Chấp nhận ba array broadcast-compatible.
- Không loop/list comprehension.
- Raise `ValueError` nếu discount ngoài `[0, 100]`, quantity âm hoặc price âm.

Self-check:

```python
q = np.array([2, 1, 4])
p = np.array([10.0, 50.0, 2.5])
d = np.array([0.0, 10.0, 20.0])
np.testing.assert_allclose(compute_net_revenue(q, p, d), [20.0, 45.0, 8.0])
```

## Bài 5 — Profile một bảng (10 điểm)

Cài đặt `profile_table(df)` trả về DataFrame có index là tên cột và các cột:

- `dtype`
- `missing_count`
- `missing_pct`
- `unique_count`
- `sample_value`

Chạy trên `customer_orders_raw.csv`. Trả lời:

1. Cột số nào bị đọc sai dtype và vì sao?
2. Cột nào có missing?
3. `Order ID` có bao nhiêu duplicate key?

## Bài 6 — Chuẩn hóa category (10 điểm)

Không dùng solution cleaning function:

1. Chuẩn hóa tên cột thành snake_case.
2. Chuẩn hóa `category`, `region`, `payment_method` ở mức string cơ bản.
3. In số unique trước và sau.
4. Lập danh sách giá trị chưa map được thay vì tự đoán.

## Bài 7 — GroupBy đúng semantics (10 điểm)

Sau khi có cleaned data, cài đặt `summarize_by_category` trả:

- `category`
- `orders`: số `order_id` unique
- `units`: tổng quantity
- `gross_revenue`
- `net_revenue`
- `avg_order_value`: `net_revenue / orders`

Sắp giảm dần theo `net_revenue`.

Self-check:

```python
assert result["orders"].sum() == len(cleaned)
assert result.iloc[0]["category"] == "electronics"
```

## Bài 8 — Merge có kiểm soát (10 điểm)

Cài đặt `join_customers`:

- Left join cleaned orders với `customers.csv`.
- `validate="many_to_one"`.
- `indicator=True`.
- Không làm thay đổi số dòng orders.

Trả lời:

1. Có bao nhiêu order không match customer master?
2. Order ID và customer ID nào gây ra?
3. Nếu `customers.csv` có duplicate `customer_id`, điều gì xảy ra?

## Bài 9 — Controlled performance experiment (15 điểm)

Tạo ba array ngẫu nhiên với ít nhất 1.000.000 phần tử. So sánh:

- Python loop tính net revenue.
- NumPy vectorized expression.

Quy tắc:

- Warm-up cả hai cách.
- Chạy ít nhất 7 lần.
- Báo median thay vì chọn lần nhanh nhất.
- Xác nhận output tương đương bằng `np.testing.assert_allclose`.
- Không khẳng định một speed ratio là phổ quát; ghi môi trường benchmark.

## Bài 10 — EDA bằng hai biểu đồ (15 điểm)

Tạo và lưu:

1. Bar chart net revenue theo category, sắp giảm dần.
2. Bar chart số order theo region.

Mỗi biểu đồ phải có:

- Tiêu đề trả lời một câu hỏi.
- Trục và đơn vị.
- Figure size hợp lý.
- File PNG độ phân giải tối thiểu 120 dpi.
- Một câu kết luận và một limitation.

## Điểm tự đánh giá

| Mức | Điều kiện |
|---|---|
| Chưa đạt | Dưới 60 điểm hoặc chưa làm bài 3/4/5 |
| Đạt | 60–79 điểm và bài bắt buộc chạy đúng |
| Tốt | 80–89 điểm, có self-check và giải thích |
| Xuất sắc | 90–100 điểm, code rõ, benchmark đúng và insight có bằng chứng |

