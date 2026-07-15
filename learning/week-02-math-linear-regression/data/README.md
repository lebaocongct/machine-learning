# Datasets Tuần 2

Tất cả dataset là synthetic, không chứa dữ liệu cá nhân thật và được tái tạo bằng seed cố định.

```bash
python -m data.generate_datasets
```

## `single_feature_regression.csv`

| Cột | Ý nghĩa |
|---|---|
| sample_id | ID kỹ thuật |
| x | Một feature liên tục |
| y | Target từ quan hệ tuyến tính cộng noise |

Mục đích: fitted line, slope/intercept, loss surface.

## `housing_regression.csv`

| Cột | Đơn vị/vai trò |
|---|---|
| property_id | ID kỹ thuật, không dùng làm feature |
| area_sqm | mét vuông |
| bedrooms | số phòng ngủ |
| age_years | tuổi căn nhà |
| distance_km | km đến trung tâm |
| energy_score | điểm hiệu quả năng lượng |
| price_thousand | target, nghìn đơn vị tiền tệ |

Mục đích: multi-feature regression và standardization experiment.

Các feature cố ý có scale rất khác nhau để quan sát ảnh hưởng lên gradient descent.

## `practical_regression.csv`

| Cột | Vai trò |
|---|---|
| student_id | ID kỹ thuật |
| study_hours | feature |
| practice_tests | feature |
| sleep_hours | feature |
| absences | feature |
| score | target |

Chỉ dùng trong practical test. Không dùng để điều chỉnh challenge implementation trước giờ kiểm tra.

## `dataset_metadata.json`

Chứa seed, hệ số sinh dữ liệu và noise standard deviation để đối chiếu sau khi hoàn thành bài. Không xem true coefficients trước practical test nếu muốn đánh giá trung thực.

## Giới hạn

- Quan hệ được tạo tuyến tính nên không đại diện độ phức tạp dữ liệu thực.
- Noise là Gaussian và không có missing/outlier hệ thống.
- Metrics trong tuần này chủ yếu kiểm tra optimization, không đánh giá generalization.
- Không diễn giải coefficients như causal effects.

