# Rubric Tuần 1

## 1. Điểm tổng tuần — 100 điểm

| Thành phần                      | Điểm tối đa | Tiêu chí đầy đủ                                                   |
| ------------------------------- | ----------: | ----------------------------------------------------------------- |
| Hai guided labs                 |          20 | Run All thành công; dự đoán shape trước khi chạy; có kết luận     |
| Bài tập độc lập                 |          20 | Hoàn thành ≥8 bài; bài 3–8 đúng; code và self-check rõ            |
| Cleaning challenge + experiment |          30 | 11 tests pass; không mutate; contract đúng; benchmark có protocol |
| Practical test                  |          20 | Điểm bài practical quy đổi theo tỷ lệ; tối thiểu 60/100           |
| Quiz và giải thích              |          10 | Quiz ≥16/20 và giải thích được câu sai                            |
| **Tổng**                        |     **100** |                                                                   |

Điều kiện qua: tổng ≥75, practical ≥60/100, public tests pass toàn bộ và không có critical fail.

## 2. Mô tả mức chất lượng

| Mức    | Biểu hiện                                                                   |
| ------ | --------------------------------------------------------------------------- |
| 0–49   | Code rời rạc, không tái tạo, contract không rõ, nhiều output sai            |
| 50–64  | Làm được thao tác cơ bản nhưng phụ thuộc hướng dẫn; error handling yếu      |
| 65–74  | Hầu hết bài đúng nhưng thiếu test/giải thích hoặc còn critical gap          |
| 75–84  | Đạt chuẩn: artifact chạy lại, test pass, giải thích được quyết định         |
| 85–94  | Tốt: code rõ, experiment công bằng, insight có evidence, error analysis tốt |
| 95–100 | Xuất sắc: thêm edge tests, CLI/automation hoặc quality checks có giá trị    |

## 3. Critical fail

Bài chưa đạt dù điểm số đủ nếu có một trong các lỗi:

- Thay đổi CSV gốc để bỏ lỗi.
- Hard-code output/row count thay vì thực hiện transformation.
- Key vẫn duplicate hoặc required field vẫn missing.
- Revenue dùng loop trong bài bắt buộc vectorize.
- Không thể chạy từ môi trường/kernel sạch.
- Không có test hoặc sửa public test để hợp implementation.
- Kết luận EDA trái với bảng/biểu đồ đã tạo.

## 4. Rubric practical test — 100 điểm

| Hạng mục           | Điểm | Full credit                                                       |
| ------------------ | ---: | ----------------------------------------------------------------- |
| Data profile       |   15 | Shape, dtype, missing, duplicate và categories đúng; nêu 3 vấn đề |
| Cleaning contract  |   35 | Toàn bộ normalization/date/numeric/duplicate/rule đúng; 10 rows   |
| Quality report     |   10 | Đủ key và số liệu đúng; không hard-code                           |
| Summary            |   15 | Category/region orders và net revenue đúng                        |
| Visualization      |   10 | Bar chart đúng dữ liệu, label/đơn vị/title và lưu file            |
| Code quality       |   10 | Hàm nhỏ, `main`, không mutate, vectorized, tên rõ                 |
| Insight/limitation |    5 | Kết luận có evidence; limitation thực tế                          |

## 5. Feedback template

```text
Điểm mạnh:
-

Lỗi quan trọng nhất:
-

Evidence:
- Test/line/output:

Học bù bắt buộc:
-

Điều kiện re-check:
-
```
