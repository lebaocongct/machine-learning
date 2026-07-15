# Data dictionary Tuần 1

Các dataset được tạo riêng cho bài học. Không có dữ liệu cá nhân hoặc giao dịch thật.

## `customer_orders_raw.csv`

Unit mong đợi: một dòng là một order.

| Cột raw        | Kiểu đích       | Required | Rule                                   |
| -------------- | --------------- | -------: | -------------------------------------- |
| Order ID       | string          |       Có | Unique sau duplicate policy; uppercase |
| Order Date     | datetime        |       Có | Ngày hợp lệ; year-first parse chặt     |
| Customer ID    | string          |       Có | Không trống; uppercase                 |
| Product        | string          |       Có | Không trống; strip whitespace          |
| Category       | category/string |       Có | electronics, home, office, fashion     |
| Quantity       | int64           |       Có | Số nguyên dương                        |
| Unit Price     | float64         |       Có | Số dương                               |
| Discount Pct   | float64         |    Không | Thiếu → 0; miền [0, 100]               |
| Region         | category/string |       Có | north, central, south sau mapping      |
| Payment Method | category/string |       Có | card, cash, e_wallet, bank_transfer    |

Derived features:

```text
gross_revenue = quantity × unit_price
net_revenue = gross_revenue × (1 − discount_pct / 100)
```

## `customers.csv`

Customer master dùng cho bài merge.

| Cột         | Kiểu            | Rule                      |
| ----------- | --------------- | ------------------------- |
| customer_id | string          | Unique, khóa bảng phải    |
| segment     | category/string | consumer, smb, enterprise |
| signup_date | datetime        | Ngày hợp lệ               |

Join contract:

```text
orders many-to-one customers
```

Không match master key phải được flag bằng merge indicator; không tự động xóa order khi contract chưa yêu cầu.

## `practical_test_orders.csv`

Batch độc lập dùng trong practical test. Không dùng để phát triển hoặc tuning cleaning function trước giờ kiểm tra.

## Câu hỏi data governance ban đầu

1. Ai sở hữu definition của discount thiếu?
2. Duplicate order nên giữ bản đầu, bản cuối hay reconcile theo timestamp?
3. Unmatched customer key là late-arriving dimension hay lỗi input?
4. Tiền tệ và timezone đang ở đâu trong contract?
5. Khi reject row, raw record và reason code sẽ được lưu ở đâu?
