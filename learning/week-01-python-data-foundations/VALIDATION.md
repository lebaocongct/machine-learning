# Validation report

> Ngày kiểm tra: 14/07/2026

## Môi trường kiểm tra

- Python 3.12.13
- NumPy 2.3.5
- Pandas 2.2.3
- Matplotlib 3.10.8
- Pytest 9.1.1

Teaching setup dùng Python 3.11 và dependency ranges trong `requirements.txt`; reference package cũng đã chạy thành công trên môi trường kiểm tra nêu trên.

## Kết quả

| Hạng mục                       | Kết quả                                                        |
| ------------------------------ | -------------------------------------------------------------- |
| JSON/nbformat của 2 notebook   | Pass                                                           |
| Notebook 1 — 9 code cells      | Pass                                                           |
| Notebook 2 — 13 code cells     | Pass                                                           |
| Reference cleaning test suite  | 11/11 pass                                                     |
| Exercise reference self-checks | Pass                                                           |
| Main cleaned shape             | `(41, 12)`                                                     |
| Main quality report            | 52 raw, 41 clean, 11 removed, 1 duplicate key, 9 missing cells |
| Practical reference script     | Pass                                                           |
| Practical cleaned rows         | 10                                                             |
| Practical net revenue          | 359.70                                                         |

## Invariants đã kiểm tra

- Không mutate raw DataFrame.
- Missing required column raise lỗi rõ.
- `order_id` unique.
- Required fields không missing.
- Quantity là integer dương.
- Price dương; discount trong `[0, 100]`.
- Category/region/payment nằm trong canonical domain.
- Unknown categorical values bị reject.
- Gross/net revenue đúng công thức trong tolerance.
- Output sorted ổn định.
- Quality report đúng expected values.

## Lệnh tái kiểm tra

```bash
WEEK1_IMPL=solutions.data_cleaning_solution python -m pytest -q
python -m solutions.practical_test_solution
```
