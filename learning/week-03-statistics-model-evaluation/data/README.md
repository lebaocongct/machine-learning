# Datasets Tuần 3

Tất cả dataset là synthetic, không chứa dữ liệu cá nhân thật và được tạo bằng seed cố định.

```bash
python -m data.generate_datasets
```

## `skewed_population.csv`

Finite population 5.000 khách hàng dùng cho sampling, CLT, conditional probability và bootstrap.

| Cột | Vai trò |
|---|---|
| `customer_id` | ID kỹ thuật |
| `segment` | `basic`, `plus`, `premium` |
| `monthly_spend` | Biến liên tục lệch phải |
| `support_calls` | Biến đếm |
| `converted` | Biến nhị phân 0/1 |

## `nonlinear_regression.csv`

Dataset chính cho lab/challenge bias–variance.

| Cột | Vai trò |
|---|---|
| `sample_id` | ID kỹ thuật |
| `x` | Feature tại prediction time |
| `y` | Target phi tuyến có noise |

Không đọc signal thật trong metadata trước khi hoàn thành model selection nếu muốn tự đánh giá trung thực.

## `practical_model_selection.csv`

Dataset độc lập cho practical test.

| Cột | Có được dùng để fit? | Lý do |
|---|---:|---|
| `observation_id` | Không | ID |
| `x` | Có | Có ở prediction time |
| `sensor_quality` | Chỉ error analysis | Không nằm trong prediction contract |
| `post_event_target_proxy` | **Cấm** | Được đo sau outcome; target leakage |
| `y` | Target | Chỉ dùng làm nhãn |

## Giới hạn

- Các mẫu được giả định IID; time/group split sẽ học ở tuần sau.
- Dữ liệu polynomial là bài mô phỏng, không đại diện mọi dạng phi tuyến.
- Bootstrap percentile trong tuần này phục vụ trực giác; không mặc định phù hợp mọi statistic/dataset.
- Không diễn giải association hoặc conditional probability thành causality.
