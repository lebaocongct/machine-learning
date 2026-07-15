# Kết quả tham chiếu — Practical test

Chỉ mở sau khi hết 60 phút.

## Quality report

```python
{
    "raw_rows": 15,
    "clean_rows": 10,
    "removed_rows": 5,
    "duplicate_order_ids": 1,
    "missing_cells_raw": 3,
}
```

## Tổng doanh thu

- Gross revenue: `383.50`
- Net revenue: `359.70`

## Theo category

| Category | Orders | Net revenue |
|---|---:|---:|
| home | 2 | 124.00 |
| fashion | 3 | 104.20 |
| electronics | 2 | 85.00 |
| office | 3 | 46.50 |

## Theo region

| Region | Orders | Net revenue |
|---|---:|---:|
| central | 2 | 124.00 |
| north | 5 | 120.70 |
| south | 3 | 115.00 |

## Rows bị loại

- Một duplicate `T005`.
- `T006`: thiếu customer ID.
- `T007`: quantity `zero`.
- `T008`: date không hợp lệ.
- `T011`: discount `110` ngoài miền.

Lưu ý: `T014/C999` vẫn là order hợp lệ theo order contract; nó chỉ không match customer master nếu thực hiện join. Không được xóa chỉ vì khóa chưa có trong bảng master khi business rule chưa yêu cầu.

## Insight mẫu

Home có net revenue cao nhất dù chỉ có hai đơn, do order value lớn. Dataset chỉ có 10 đơn hợp lệ nên không thể suy rộng rằng home luôn là category quan trọng nhất.

