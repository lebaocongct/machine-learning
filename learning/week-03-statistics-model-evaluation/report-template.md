# Experiment Report — Tuần 3

## 1. Câu hỏi và estimand

- Câu hỏi:
- Population/deployment population:
- Sample quan sát:
- Statistic/metric:
- Điều không được suy diễn:

## 2. Data contract

| Cột | Vai trò | Có tại prediction time? | Dùng để fit? | Leakage risk |
|---|---|---:|---:|---|
|  |  |  |  |  |

## 3. Split protocol

- Unit of split:
- Random/group/time split:
- Train/validation/test ratio:
- Seed:
- Disjoint/exhaustive evidence:
- Khi nào test được mở:

## 4. Sampling và uncertainty

- Population/sample statistic:
- Sample size:
- Bootstrap statistic:
- Resamples/seed:
- Confidence interval:
- Diễn giải đúng:
- Limitation:

## 5. Hypothesis

```text
Nếu tăng polynomial degree từ ... đến ...,
thì train RMSE sẽ ... và validation RMSE sẽ ...,
vì ...
```

## 6. Controlled variables

- Cố định: split, seed, metric, pipeline, training data.
- Thay đổi duy nhất: polynomial degree.
- Candidate degrees:
- Tie-break rule:

## 7. Degree comparison

| Degree | Train RMSE | Validation RMSE | Gap | Nhận xét |
|---:|---:|---:|---:|---|
|  |  |  |  |  |

- Selected degree:
- Evidence:
- Vì sao không chọn bằng train score:

## 8. Learning curve diagnosis

- Train sizes:
- Train curve pattern:
- Validation curve pattern:
- Bias/variance/noise hypothesis:
- Thêm data có khả năng giúp không? Vì sao?

## 9. Final evaluation

- Model đã refit trên:
- Test được mở lần thứ:
- Test RMSE:
- MAE/CI nếu có:
- Validation–test difference:
- Có thay model sau khi xem test không? **Không**.

## 10. Error slices

| Slice | N | Metric | So với overall | Hypothesis |
|---|---:|---:|---:|---|
|  |  |  |  |  |

## 11. Leakage audit

- Target leakage:
- Preprocessing leakage:
- Duplicate/entity leakage:
- Temporal leakage:
- Test contamination:

## 12. Kết luận và quyết định

- Kết luận được evidence hỗ trợ:
- Điều chưa biết:
- Quyết định:
- Thí nghiệm tiếp theo chỉ dùng development data:

## 13. Retrospective

- Sai lầm quan trọng nhất:
- Test/assert đã bắt lỗi:
- Quy tắc sẽ tái sử dụng ở tuần sau:
