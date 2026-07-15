# Lời giải Tuần 3

Chỉ mở thư mục này sau lần nộp đầu tiên.

| Tệp | Nội dung |
|---|---|
| `model_evaluation_solution.py` | Reference API cho challenge |
| `exercise_solutions.py` | Hàm compact và leakage-case answers |
| `quiz-answers.md` | Đáp án 20 câu |
| `reference-results.md` | Population/bootstrap/challenge expected values |
| `practical-test-solution.md` | Đáp án số practical |
| `practical_test_solution.py` | Script practical có output/figures |

Chạy reference tests:

```bash
WEEK3_IMPL=solutions.model_evaluation_solution python -m pytest -q
```

Chạy practical reference:

```bash
python -m solutions.practical_test_solution
```

Reference solution là một implementation hợp lệ, không phải cách duy nhất. Bài làm phải giữ protocol và invariants, không cần trùng cấu trúc từng dòng.
