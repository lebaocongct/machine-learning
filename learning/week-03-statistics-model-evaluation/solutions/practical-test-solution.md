# Kết quả tham chiếu — Practical Test Tuần 3

Chỉ mở sau khi hết 75 phút.

## Contract và split

- Model feature: `x`.
- Analysis-only: `sensor_quality`.
- Forbidden: `post_event_target_proxy` — post-outcome target leakage.
- Rows: `260`.
- Seed: `73`.
- Train/validation/test sizes: `156 / 52 / 52`.

## Degree comparison

| Degree | Train RMSE | Validation RMSE |
|---:|---:|---:|
| 1 | 3.163198 | 3.320173 |
| 2 | 1.666984 | 2.143153 |
| 3 | 1.648371 | 2.117598 |
| 4 | 1.644525 | 2.121646 |
| 5 | 1.493637 | 1.768843 |
| 6 | 1.487031 | 1.755578 |
| 7 | 1.341031 | 1.522699 |
| 8 | 1.338806 | 1.493048 |
| **9** | **1.290497** | **1.438370** |
| 10 | 1.265777 | 1.484986 |
| 11 | 1.246315 | 1.539067 |
| 12 | 1.245807 | 1.534146 |

Selected degree: **9**, chỉ từ validation.

## Learning curve — degree 9

| Train size | Train RMSE | Validation RMSE |
|---:|---:|---:|
| 30 | 1.095412 | 4.969720 |
| 60 | 1.288791 | 1.818439 |
| 90 | 1.260462 | 1.606224 |
| 120 | 1.269759 | 1.559580 |
| 156 | 1.290497 | 1.438370 |

Ở sample nhỏ, high-degree model có variance lớn. Validation RMSE giảm rõ khi thêm data và gap thu hẹp; thêm data có khả năng còn giúp, nhưng curve gần cuối bắt đầu chậm lại.

## Final test

Sau khi refit degree 9 trên train+validation:

- Test RMSE: `1.479823`.
- Test MAE: `1.128643`.
- Test R²: `0.898043`.
- Bootstrap 95% percentile CI cho MAE: `[0.883170, 1.395091]`.

## Error slice

Median `sensor_quality` trên test: `0.7995895`.

| Slice | N | RMSE |
|---|---:|---:|
| Quality dưới median | 26 | 1.555909 |
| Quality từ median trở lên | 26 | 1.399607 |

Slice thấp có error cao hơn, phù hợp cơ chế noise mô phỏng. Mỗi slice chỉ có 26 rows nên không kết luận rộng mà không có interval/repeated data.

## Kết luận mẫu

Degree 9 thắng trên validation, không phải train. Learning curve cho thấy variance cao ở sample nhỏ và cải thiện khi thêm data. Final test gần validation nhưng là một estimate trên 52 rows; MAE interval và slice sizes cho thấy uncertainty còn đáng kể. Không thay model sau khi xem test.
