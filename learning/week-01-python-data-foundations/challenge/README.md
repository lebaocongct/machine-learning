# Challenge — Xây pipeline làm sạch đơn hàng có kiểm thử

## Mục tiêu

Cài đặt ba hàm trong `challenge/submission.py`:

```python
to_snake_case(name: str) -> str
clean_orders(raw: pd.DataFrame) -> pd.DataFrame
build_quality_report(raw: pd.DataFrame, cleaned: pd.DataFrame) -> dict[str, int]
```

Không gọi implementation trong `solutions/`.

## Input contract

Tệp chính: `data/customer_orders_raw.csv`.

Các cột bắt buộc sau khi đổi sang snake_case:

```text
order_id, order_date, customer_id, product, category,
quantity, unit_price, discount_pct, region, payment_method
```

## Cleaning rules

### 1. Không mutate input

`raw` phải giống hệt trước và sau khi gọi hàm.

### 2. Column names

- Strip khoảng trắng.
- Thay nhóm ký tự không phải chữ/số bằng `_`.
- Lowercase.
- Xóa `_` thừa ở đầu/cuối.

### 3. Text

- Strip khoảng trắng.
- Chuỗi rỗng thành missing.
- `order_id`, `customer_id`: uppercase.
- `category`: lowercase; chỉ chấp nhận `electronics`, `home`, `office`, `fashion`.

### 4. Region canonicalization

| Raw variants                                     | Canonical |
| ------------------------------------------------ | --------- |
| Hanoi, Ha Noi                                    | `north`   |
| Da Nang, Danang                                  | `central` |
| HCM, HCMC, TP.HCM, Ho Chi Minh, Ho Chi Minh City | `south`   |

Giá trị ngoài mapping là invalid.

### 5. Payment canonicalization

| Raw variants     | Canonical       |
| ---------------- | --------------- |
| Card/CARD        | `card`          |
| Cash             | `cash`          |
| E-wallet/ewallet | `e_wallet`      |
| Bank transfer    | `bank_transfer` |

### 6. Date

- ISO `YYYY-MM-DD` và `YYYY/MM/DD` phải parse chặt.
- Các format còn lại dùng day-first.
- `2026-13-01` và `bad-date` phải invalid, không được reinterpret thành ngày khác.

### 7. Numeric rules

- `quantity`: số nguyên dương.
- `unit_price`: số dương.
- `discount_pct`: trong `[0, 100]`.
- Discount thiếu được điền `0` theo business rule của bài.

### 8. Duplicate và required fields

- Giữ dòng đầu cho duplicate `order_id`.
- Loại dòng thiếu `order_id`, `order_date`, `customer_id`, `product` hoặc `category`.
- Loại dòng vi phạm date/numeric/canonical rules.

### 9. Derived features

```text
gross_revenue = quantity × unit_price
net_revenue = gross_revenue × (1 − discount_pct / 100)
```

### 10. Output

- `quantity`: integer dtype.
- Các cột numeric còn lại: float.
- Sắp theo `order_date`, rồi `order_id`.
- Reset index.
- Kết quả chính có 41 dòng.

## Quality report contract

Trả đúng năm key integer:

```python
{
    "raw_rows": ...,
    "clean_rows": ...,
    "removed_rows": ...,
    "duplicate_order_ids": ...,
    "missing_cells_raw": ...,
}
```

## Vòng lặp phát triển

```bash
python -m pytest tests/test_data_cleaning.py -q
```

Làm theo thứ tự:

1. `to_snake_case`.
2. Schema validation.
3. Copy và string normalization.
4. Date/numeric parsing.
5. Validity mask.
6. Derived features.
7. Stable sorting.
8. Quality report.

Mỗi lần chỉ sửa nguyên nhân của một nhóm test failure.

## Hints theo cấp độ

### Hint 1

`pd.to_numeric(..., errors="coerce")` giúp tách parse khỏi rule hợp lệ.

### Hint 2

Nếu parse tất cả ngày với `dayfirst=True`, một chuỗi year-first không hợp lệ có thể bị hiểu lại. Tách year-first bằng regex và parse format chặt trước.

### Hint 3

Dùng một boolean mask gồm các invariant, sau đó `.loc[valid].copy()`.

## Đầu ra

- `challenge/submission.py` hoàn chỉnh.
- 11 public tests pass.
- `outputs/cleaned_orders.csv`.
- `outputs/data_quality_report.json`.
- Commit: `feat: implement tested order cleaning pipeline`.
