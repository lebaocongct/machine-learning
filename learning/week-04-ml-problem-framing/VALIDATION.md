# Validation report — Week 4 package

**Release check:** 2026-07-15  
**Environment:** Python 3.12.13; NumPy 2.3.5; pandas 2.2.3; Matplotlib 3.10.8; scikit-learn 1.8.0.

## Kết quả

| Check | Kết quả |
|---|---|
| Public challenge tests | **12/12 passed** |
| Notebook 01 | **9/9 code cells passed** |
| Notebook 02 | **11/11 code cells passed** |
| Notebook schema | **2/2 valid nbformat v4** |
| Practical reference | **passed**, threshold 0.05, test cost 1,324 |
| Exercise reference assertions | **passed** |
| Python compilation | **12/12 files compiled** |
| JSON/notebook parsing | **9/9 files parsed** |
| Data contracts | shapes, unique keys, binary targets, time order **passed** |
| Dataset regeneration | hashes before/after generator **identical** |

## Lệnh tái kiểm tra

```bash
PYTHONDONTWRITEBYTECODE=1 \
WEEK4_IMPL=solutions.problem_framing_solution \
python -m pytest tests/test_problem_framing.py -q -p no:cacheprovider

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/week4-mpl \
PYTHONDONTWRITEBYTECODE=1 python tests/run_notebooks.py

PYTHONDONTWRITEBYTECODE=1 python solutions/exercise_solutions.py
PYTHONDONTWRITEBYTECODE=1 python solutions/practical_test_solution.py
PYTHONDONTWRITEBYTECODE=1 python -m data.generate_datasets
```

`WEEK4_IMPL` chỉ dùng trong release check để chạy public tests trên đáp án. Người học bỏ biến này để test `challenge/submission.py`.

## Reference assertions đã xác nhận

### Support

- Split 503/108/109; selected threshold 0.05.
- Validation cost 1,120.
- Test TP/FP/TN/FN = 14/76/17/2; recall 0.875; cost 1,160.
- Majority accuracy 0.8532 nhưng recall 0 và cost 3,200.
- Feature audit phát hiện đúng 3 post-outcome fields.

### Fraud

- Split 630/135/135; selected threshold 0.05.
- Validation cost 1,016.
- Test TP/FP/TN/FN = 7/53/72/3; recall 0.70; cost 1,324.
- Majority accuracy 0.9259 nhưng recall 0 và cost 3,000.
- Feature audit phát hiện đúng 3 post-outcome fields.

## Dataset hashes

```text
74e52e9e3fca94ca414683ffcce311043e2a28a2cea2461ec56baff0b5d56d39  scenario_cards.csv
4df7e2e625bc394148a373640ded14a355fc0749b24ae843cb50bf0ef5dbccd5  support_triage.csv
1ba9a8a1d0919c2210bfa769e1c025420a5b0776086e4153ace35841567af22d  support_feature_catalog.csv
02ce898ac9eb277fc2e45cbec0c6704c9e97597b02637c9b697612d67a7e936c  fraud_review.csv
3c743a136e5fac2aa2111de05b6d834c103044ba0f3611331e611394e2de4e2e  fraud_feature_catalog.csv
6a2220cd4f08922dde0bf474c158d5cfecb10716eeb0fddcdf519f862df0edaa  dataset_metadata.json
```

## Giới hạn kiểm thử

- Không đánh giá production fairness, causal impact hoặc legal compliance từ synthetic data.
- FP/FN cost và capacity là giả định đào tạo; phải được stakeholder xác nhận.
- Package cố ý không phụ thuộc TensorFlow ở Tuần 4: đây là problem-framing gate trước modeling. TensorFlow/Keras được dùng khi bước sang xây model trong các tuần tiếp theo.

