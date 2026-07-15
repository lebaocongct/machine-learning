# Cách dùng lời giải Tuần 2

Chỉ mở sau khi đã hoàn thành lần thử đầu tiên và ghi lại test/gradient failure.

| Tệp | Nội dung |
|---|---|
| `linear_regression_solution.py` | Reference API challenge |
| `exercise_solutions.py` | Hàm tham chiếu bài tập |
| `quiz-answers.md` | Đáp án quiz |
| `practical-test-solution.md` | Expected practical results |
| `practical_test_solution.py` | Practical reference script |
| `reference-results.md` | Housing/LR/scaling metrics tham chiếu |

Kiểm tra:

```bash
WEEK2_IMPL=solutions.linear_regression_solution python -m pytest -q
python -m solutions.practical_test_solution
```

Nếu bài làm có cấu trúc khác nhưng thỏa contract, gradient check và tests, nó vẫn đúng.

