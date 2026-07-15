# Challenge — Problem Framing và Cost-sensitive Operating Point

## Bối cảnh

Một support team muốn dùng heuristic risk score để đưa ticket có khả năng escalation vào specialist queue. Trước khi huấn luyện model mới, bạn phải chứng minh bài toán được frame đúng, feature có tại prediction time, baseline được đo và threshold chỉ chọn bằng validation.

Hoàn thành `challenge/submission.py` mà không import từ `solutions/`.

## API

```python
classify_task(output_kind, has_labels)
validate_problem_canvas(canvas)
validate_prediction_contract(contract)
audit_feature_catalog(catalog, max_allowed_offset_hours)
time_ordered_split_indices(timestamps, train_fraction, validation_fraction)
regression_baseline(y_train, y_evaluation, strategy)
binary_classification_metrics(y_true, y_pred)
majority_classification_baseline(y_train, y_evaluation)
threshold_sweep(y_true, scores, thresholds, false_positive_cost, false_negative_cost)
select_operating_threshold(results)
CostSensitiveHoldoutWorkflow.fit(scores, y, timestamps)
CostSensitiveHoldoutWorkflow.predict(scores)
```

## Canvas contract

Canvas hợp lệ phải có:

- Non-ML goal, decision và decision owner.
- Prediction unit/time.
- Target/window và task type.
- Model output, primary metric, baseline, success criterion.
- Deployment mode.
- Constraints, slices, leakage risks và non-goals.

Validator trả list lỗi ổn định; canvas hợp lệ trả `[]`.

## Feature audit

Input catalog:

```text
feature_name, source_stage, available_offset_hours, intended_for_model
```

Phải phát hiện ít nhất:

- Feature có sau allowed decision time.
- Feature từ `post_outcome` hoặc `label_generation` nhưng bị định dùng cho model.
- Duplicate feature definitions.

## Holdout protocol

1. Sort theo timestamp.
2. Chia 70/15/15.
3. Train dùng để hiểu prevalence/baseline; không chọn threshold bằng test.
4. Sweep thresholds trên validation với FP/FN cost.
5. Tie-break: minimum cost → higher recall → lower threshold.
6. Test đúng một lần.

## Lệnh

```bash
python -m pytest tests/test_problem_framing.py -q
```

## Critical fail

- Canvas không có decision/action.
- Dùng post-outcome features.
- Random split dữ liệu chronological case.
- Chọn threshold bằng test.
- Chỉ báo accuracy khi positive class hiếm.
- Không có heuristic/dummy baseline.

## Đầu ra

- `challenge/submission.py` hoàn chỉnh, 12 tests pass.
- Canvas và prediction contract hợp lệ.
- Feature-risk table.
- Validation threshold sweep.
- Final test metrics/cost đúng một lần.
- 5 phút defense giải thích metric, baseline, constraint và leakage.
