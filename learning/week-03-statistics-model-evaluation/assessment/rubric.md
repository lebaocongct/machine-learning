# Rubric Tuần 3

## 1. Điểm tổng tuần — 100 điểm

| Thành phần | Điểm | Full credit |
|---|---:|---|
| Hai guided labs | 20 | Run All; simulation/curves đúng; kết luận có evidence |
| Bài tập độc lập | 20 | ≥8/10; bài 4–9 đúng; leakage audit đầy đủ |
| Challenge + report | 30 | 12 tests pass; split/model selection/learning curve đúng protocol |
| Practical test | 20 | Quy đổi theo practical score; tối thiểu 60/100 |
| Quiz và giải thích | 10 | ≥16/20; sửa và giải thích câu sai |

Điều kiện qua: tổng ≥75, practical ≥60, public tests pass và không critical fail.

## 2. Critical fail

- Dùng test set để chọn degree, feature, seed hoặc preprocessing.
- Fit preprocessing trên cả holdout trước split.
- Dùng `post_event_target_proxy` hoặc feature sau prediction time.
- Split overlap/mất rows nhưng không phát hiện.
- Bootstrap không replacement hoặc giả interval được hard-code.
- Notebook không chạy lại từ đầu.
- Báo một metric đẹp và ẩn các candidate đã thử.

## 3. Rubric Practical — 100 điểm

| Hạng mục | Điểm | Full credit |
|---|---:|---|
| Data contract/leakage | 10 | Availability đúng; forbidden feature bị loại/assert |
| Three-way split | 15 | Seed/tỷ lệ đúng; disjoint/exhaustive/reproducible |
| Model selection | 25 | Degree 1–12; train/val table; selected by val only |
| Learning curve | 15 | Nested subsets; pipeline mới; diagnosis trước test |
| Final evaluation | 20 | Refit train+val; một test evaluation; RMSE/MAE/R²/output đúng |
| Bootstrap/error slice | 10 | Paired/per-row resampling đúng; CI và slice limitation |
| Code quality | 5 | Hàm nhỏ, main, asserts, paths/seed rõ |

## 4. Mức chất lượng

| Mức | Biểu hiện |
|---|---|
| 0–49 | Không hoàn thành split/model hoặc có leakage nghiêm trọng |
| 50–64 | Chạy được một phần nhưng protocol/uncertainty yếu |
| 65–74 | Hầu hết đúng nhưng thiếu quality gate |
| 75–84 | Đạt chuẩn: tái tạo được, test locked, reasoning đủ |
| 85–94 | Tốt: uncertainty/curves/error slices rõ |
| 95–100 | Xuất sắc: protocol chặt, limitations và follow-up có giá trị |

## 5. Feedback template

```text
Điểm mạnh:
-

Lỗi quan trọng nhất:
-

Evidence:
- Split/test/curve/CI:

Học bù bắt buộc:
-

Điều kiện re-check:
-
```
