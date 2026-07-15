# Kết quả tham chiếu Tuần 1

Chỉ mở sau khi đã hoàn thành lần thử đầu tiên.

## 1. Main cleaning challenge

```python
{
    "raw_rows": 52,
    "clean_rows": 41,
    "removed_rows": 11,
    "duplicate_order_ids": 1,
    "missing_cells_raw": 9,
}
```

Output contract:

- Shape: `(41, 12)`.
- Date range: `2026-06-01` đến `2026-06-20`.
- `order_id` unique.
- Quantity integer và dương.
- `unit_price` dương.
- Discount trong `[0, 100]`.
- Category: electronics, home, fashion, office.
- Region: north, central, south.
- Payment: card, cash, e_wallet, bank_transfer.

## 2. Revenue summary

- Gross revenue: `1,978.75`.
- Net revenue: `1,833.40`.
- Gross-to-net gap do discount: `145.35`, khoảng `7.35%` gross revenue.

### Theo category

| Category    | Orders | Units | Gross revenue | Net revenue | Avg order value |
| ----------- | -----: | ----: | ------------: | ----------: | --------------: |
| electronics |     12 |    23 |        782.00 |     723.750 |          60.312 |
| home        |     10 |    16 |        590.25 |     531.725 |          53.172 |
| fashion     |     10 |    15 |        388.50 |     366.300 |          36.630 |
| office      |      9 |    40 |        218.00 |     211.625 |          23.514 |

### Theo region

| Region  | Orders | Net revenue |
| ------- | -----: | ----------: |
| south   |     18 |     937.275 |
| north   |     14 |     466.350 |
| central |      9 |     429.775 |

## 3. Customer join

Left join với `customers.csv` giữ nguyên 41 rows.

Một key không match:

| Order ID | Customer ID | Merge state |
| -------- | ----------- | ----------- |
| O026     | C999        | left_only   |

Không xóa O026 chỉ dựa trên unmatched master key nếu contract chưa quy định như vậy; hãy flag để điều tra.

## 4. Vectorization experiment

Không có một speed ratio chuẩn vì phụ thuộc CPU, NumPy build, kích thước array và cách benchmark. Kết quả hợp lệ phải:

- Hai cách cho output tương đương trong tolerance.
- Có warm-up.
- Có ít nhất 7 lần đo.
- Báo median.
- Ghi rõ array size và môi trường.

Với array đủ lớn, vectorized NumPy thường phải nhanh hơn rõ rệt so với Python loop. Nếu không, kiểm tra lại xem benchmark có bao gồm khởi tạo dữ liệu hoặc array quá nhỏ hay không.

## 5. Insight mẫu

1. Electronics đóng góp net revenue lớn nhất (`723.75`) và có average order value cao nhất.
2. South chiếm hơn một nửa net revenue, nhưng sample nhỏ và cách tạo dữ liệu không đại diện population thật.
3. Office có nhiều units nhất (`40`) nhưng net revenue thấp nhất, cho thấy units không thay thế được revenue metric.
4. Có một customer key không match; đây là vấn đề master-data quality cần flag, không nên tự sửa.

## 6. Ba error cases nên phân tích

- `O020`: chuỗi `2026-13-01` không được reinterpret thành 13/01/2026.
- `O018`: quantity `two` phải bị coerce thành missing rồi reject theo numeric rule.
- `O021`: discount `120` parse được thành số nhưng vẫn invalid theo business domain.
