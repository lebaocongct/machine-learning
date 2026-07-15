# Đáp án Quiz Tuần 1

## Phần A

1. **B** — `ndim` là số trục, ở đây có ba trục.
2. **B** — giảm trục 0 của `(2, 3, 4)` còn `(3, 4)`.
3. **B** — `(5,)` khớp chiều cuối `5` của `(100, 5)`.
4. **C** — `keepdims=True` tạo `(100, 1)`, broadcast theo 5 cột.
5. **B** — basic slice thường là view.
6. **B** — một token như `free` ngăn cột được suy luận hoàn toàn là số.
7. **C** — giá trị không parse được thành missing/NaN.
8. **B** — kiểm tra cardinality many-to-one và fail nếu vi phạm.
9. **C** — `size()` đếm số dòng của group; `count()` bỏ missing ở cột.
10. **B** — có thể xóa dòng vì field không liên quan bị thiếu.
11. **C** — input/output rõ và ít hidden state.
12. **C** — invariant bảo vệ contract, không chỉ một sample output.

## Phần B

13. **Missing** là không có giá trị, ví dụ `customer_id` trống ở O015. **Invalid** là có giá trị nhưng vi phạm contract, ví dụ quantity `two`, price `free`, discount `120`.
14. Một parser có thể diễn giải lại ngày mơ hồ hoặc year-first không hợp lệ thành một ngày khác mà không báo lỗi. Cần parse format cấu trúc chặt trước và test các edge case.
15. Không mutate giúp hàm độc lập thứ tự gọi, dễ test, dễ chạy lại và tránh làm thay đổi dữ liệu mà caller vẫn đang dùng.
16. Tối thiểu: câu hỏi/tiêu đề, trục có nhãn, đơn vị/scale đúng và kết luận gắn với evidence. Có thể cộng thêm limitation/source.

## Phần C

17.

```text
[[-2. -2.]
 [ 0.  0.]
 [ 2.  2.]]
```

Mean theo cột là `[3, 4]` và broadcast qua ba dòng.

18. `high_value` có thể là view hoặc copy không rõ ràng và gây `SettingWithCopyWarning`/hành vi khó đoán. Dùng:

```python
high_value = df.loc[df["unit_price"] > 50].copy()
high_value["flag"] = True
```

19. Duplicate order làm `count` đếm một đơn nhiều lần. `nunique(order_id)` gần semantics “số đơn” hơn, dù vẫn cần xử lý duplicate theo contract.
20. Bảng bên phải có nhiều hơn một dòng cho một key nên một dòng trái match nhiều dòng. Dùng `validate="many_to_one"` và kiểm tra key bảng phải unique trước merge.

