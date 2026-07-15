# Cách sử dụng lời giải

Chỉ mở thư mục này sau khi:

1. Hoàn thành lần thử đầu tiên.
2. Ghi lại test đang fail và root-cause hypothesis.
3. Thử sửa ít nhất một lần.

| Tệp                          | Nội dung                                   |
| ---------------------------- | ------------------------------------------ |
| `data_cleaning_solution.py`  | Reference implementation của challenge     |
| `exercise_solutions.py`      | Hàm tham chiếu cho bài tập độc lập         |
| `quiz-answers.md`            | Đáp án và giải thích quiz                  |
| `practical-test-solution.md` | Expected results của practical test        |
| `practical_test_solution.py` | Reference script của practical test        |
| `reference-results.md`       | Expected metrics, summaries và insight mẫu |

Reference solution không phải cách duy nhất. Bài làm khác cấu trúc vẫn đúng nếu thỏa contract, invariants và test.

Kiểm tra reference implementation:

```bash
WEEK1_IMPL=solutions.data_cleaning_solution python -m pytest -q
python -m solutions.practical_test_solution
```
