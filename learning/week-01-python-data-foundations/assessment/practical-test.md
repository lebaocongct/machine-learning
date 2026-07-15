# Practical test Tuần 1 — 60 phút

## Quy tắc

- Dùng `data/practical_test_orders.csv`.
- Được xem tài liệu API chính thức, không xem `solutions/` hoặc copy challenge implementation.
- Tạo `outputs/practical_test_submission.py`.
- Không sửa CSV gốc.
- Không dùng `apply(axis=1)` cho revenue.

## Bài toán

Bạn nhận một batch đơn hàng mới. Hãy tạo một script có thể chạy từ project root:

```bash
python outputs/practical_test_submission.py
```

Script phải thực hiện các phần sau.

### Phần 1 — Profile dữ liệu (10 phút, 15 điểm)

In hoặc lưu:

- Shape.
- Dtype từng cột.
- Missing count.
- Duplicate `Order ID` count.
- Danh sách giá trị category/region/payment thô.

Viết ba vấn đề quality quan trọng nhất.

### Phần 2 — Làm sạch (25 phút, 35 điểm)

Áp dụng cùng contract tuần 1:

- Snake-case column names.
- Không mutate input.
- String strip/canonicalization.
- Parse date chặt cho year-first; day-first cho phần còn lại.
- Numeric parsing.
- Missing discount → 0.
- Giữ duplicate order đầu tiên.
- Loại required missing và row vi phạm date/numeric/canonical rule.
- Tạo `gross_revenue`, `net_revenue` bằng vectorization.
- Sort theo date và ID.

Lưu `outputs/practical_cleaned_orders.csv`.

### Phần 3 — Quality report (5 phút, 10 điểm)

Tạo dict/JSON gồm:

```text
raw_rows, clean_rows, removed_rows,
duplicate_order_ids, missing_cells_raw
```

### Phần 4 — Tổng hợp và biểu đồ (15 phút, 25 điểm)

1. Tổng orders và net revenue theo category.
2. Tổng orders và net revenue theo region.
3. Bar chart net revenue theo category.
4. Một câu insight và một limitation.

### Phần 5 — Chất lượng code (5 phút, 15 điểm)

- Có ít nhất ba hàm nhỏ.
- Có `main()` và guard `if __name__ == "__main__"`.
- Có ba assert cho invariant.
- Không hard-code kết quả.

## Điều kiện đạt

- Tối thiểu 60/100.
- Không vi phạm critical gate.
- Cleaned output phải có đúng 10 rows.
- Key unique; numeric domain hợp lệ; revenue đúng công thức.

Không mở `solutions/practical-test-solution.md` hoặc `.py` trước khi hết giờ.

