# Lịch học Tuần 1 — 10 giờ

## Tổng quan

| Phiên    |   Thời lượng | Nội dung                                          | Đầu ra                                                        |
| -------- | -----------: | ------------------------------------------------- | ------------------------------------------------------------- |
| 0        |      30 phút | Setup, diagnostic, Git init                       | Môi trường chạy; commit đầu tiên                              |
| 1        |      90 phút | NumPy: shape, dtype, indexing, axis, broadcasting | Hoàn thành notebook 1                                         |
| 2        |      75 phút | Pandas: inspect, filter, groupby, merge           | Bài tập Pandas 1–4                                            |
| 3        |     105 phút | Guided EDA trên orders dataset                    | Hoàn thành notebook 2 và hai biểu đồ                          |
| 4        |     120 phút | Data contract, cleaning function, pytest          | Challenge implementation bắt đầu; test chuyển từ đỏ sang xanh |
| 5        |      60 phút | Bài tập độc lập                                   | Tối thiểu 8/10 bài                                            |
| 6        |      75 phút | Challenge và controlled experiment                | Public tests pass; benchmark vectorization                    |
| 7        |      45 phút | Quiz, report, retrospective                       | Quiz ≥16/20; weekly report                                    |
| **Tổng** | **600 phút** |                                                   | **10 giờ**                                                    |

## Phiên 0 — Setup và diagnostic (30 phút)

1. Tạo `.venv`, cài dependency, mở JupyterLab.
2. Chạy import smoke check.
3. Trả lời diagnostic không tra cứu:
   - Khác nhau giữa list và NumPy array?
   - `df.shape` trả gì?
   - `axis=0` nghĩa là gì?
   - Vì sao cần test một hàm làm sạch dữ liệu?
4. `git init` và commit trạng thái ban đầu.

Không tính điểm diagnostic; dùng để so sánh với cuối tuần.

## Phiên 1 — NumPy lab (90 phút)

- 15 phút: đọc phần NumPy trong `01-knowledge.md`.
- 55 phút: chạy và tự viết lại notebook 1.
- 15 phút: làm lại ba bài không nhìn notebook.
- 5 phút: commit.

Stop condition: giải thích được shape của mọi operand trước phép toán broadcasting.

## Phiên 2 — Pandas core (75 phút)

- 20 phút: đọc có mục tiêu trong reading list.
- 40 phút: bài tập select/filter/groupby/merge.
- 10 phút: kiểm tra unmatched keys khi merge.
- 5 phút: ghi ba lỗi dễ mắc.

## Phiên 3 — Guided EDA (105 phút)

- 15 phút: viết năm câu hỏi EDA trước khi code.
- 70 phút: notebook 2.
- 15 phút: viết ba insight và hai limitation.
- 5 phút: restart kernel → Run All.

## Phiên 4 — Cleaning và pytest (120 phút)

- 20 phút: đọc contract trong `challenge/README.md`.
- 25 phút: thiết kế transformation steps trên giấy.
- 55 phút: cài đặt `challenge/submission.py`.
- 15 phút: đọc test failure và sửa nguyên nhân.
- 5 phút: commit.

Không sửa test để hợp với code của mình.

## Phiên 5 — Bài tập độc lập (60 phút)

- Chọn bài theo thứ tự 1–10.
- Nếu kẹt quá 10 phút, ghi hypothesis rồi chuyển bài.
- Hoàn thành ít nhất 8 bài; bài 9–10 là stretch nếu nền tảng còn yếu.

## Phiên 6 — Challenge và experiment (75 phút)

- 30 phút: hoàn thiện toàn bộ public tests.
- 20 phút: benchmark loop và vectorization ít nhất 7 lần.
- 15 phút: kiểm tra output và data quality report.
- 10 phút: viết kết luận, không chỉ ghi số nhanh hơn.

## Phiên 7 — Đánh giá và retrospective (45 phút)

- 20 phút: quiz đóng tài liệu.
- 15 phút: hoàn thành report template.
- 10 phút: tự chấm rubric và lập danh sách học bù.

Practical test 60 phút nên làm ở một phiên riêng ngay sau Phiên 7 hoặc đầu ngày kế tiếp. Nếu quỹ thời gian chỉ đúng 10 giờ, thay 60 phút bài tập độc lập ở Phiên 5 bằng practical test và chuyển bài tập còn lại sang phần bonus.

## Thực hành bonus — 2 đến 4 giờ

1. Tạo thêm 20 dòng dữ liệu lỗi và chứng minh test bắt được.
2. So sánh memory của `object`, `string` và `category` dtype.
3. Viết property-style tests bằng 10 DataFrame nhỏ tự tạo.
4. Tạo CLI nhận đường dẫn input/output cho cleaning function.
5. Dùng `merge(validate=...)` để cố ý tạo và sửa lỗi cardinality.
