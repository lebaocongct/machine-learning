# Kết quả tham chiếu Tuần 4

Các số dưới đây dùng dữ liệu synthetic đi kèm, grid threshold `0.05, 0.10, …, 0.95`, chronological split 70/15/15 và tie-break cost → recall → threshold. Sai khác ngoài sai số làm tròn cho thấy dữ liệu/protocol đã đổi.

## Scenario cards

| ID | Kết quả |
|---|---|
| S01 | classification |
| S02 | regression |
| S03 | clustering |
| S04 | regression |
| S05 | non-ML rule engine |
| S06 | generation |
| S07 | not ready — no action |
| S08 | ranking |

## Support triage — Notebook 02/challenge

- Split sizes: train 503, validation 108, test 109.
- Prevalence: train `0.111332`, validation `0.185185`, test `0.146789`.
- Feature audit: đúng 3 post-outcome risks — `resolution_hours`, `final_satisfaction`, `manager_override`.
- Validation chọn threshold: **0.05**, cost **1,120**, recall **0.90**, precision **0.20**.
- Final test: TP=14, FP=76, TN=17, FN=2.
- Test accuracy `0.284404`, precision `0.155556`, recall `0.875`, F1 `0.264151`, total cost **1,160**.
- Majority test baseline: accuracy `0.853211`, recall `0`, F1 `0`; error cost nếu tính cùng giả định = **3,200**.
- Test được mở đúng 1 lần.

Diễn giải: accuracy của heuristic tại operating point thấp hơn dummy, nhưng cost thấp hơn vì FN đắt gấp 20 FP. Predicted-positive rate `0.825688` vượt constraint ví dụ 20%; vì vậy kết luận hợp lý là **Conditional Go/No-Go ở capacity hiện tại**, không phải production-ready.

## Fraud review — Practical

- Split sizes: train 630, validation 135, test 135.
- Prevalence: train `0.034921`, validation `0.044444`, test `0.074074`.
- Feature audit: đúng 3 post-outcome risks — `chargeback_amount`, `investigation_code`, `analyst_notes_length`.
- Validation chọn threshold: **0.05**, cost **1,016**, recall `0.666667`, precision `0.071429`.
- Final test: TP=7, FP=53, TN=72, FN=3.
- Test accuracy `0.585185`, precision `0.116667`, recall `0.70`, F1 `0.20`, total cost **1,324**.
- Majority test baseline: accuracy `0.925926`, recall `0`; error cost **3,000**.
- Test được mở đúng 1 lần.

### Fraud slices tại threshold 0.05

| Slice | n | Prevalence | Precision | Recall | Predicted positive | Cost |
|---|---:|---:|---:|---:|---:|---:|
| country_risk=0 | 117 | 0.0684 | 0.1111 | 0.6250 | 0.3846 | 1,220 |
| country_risk=1 | 18 | 0.1111 | 0.1333 | 1.0000 | 0.8333 | 104 |
| card_present=0 | 98 | 0.0918 | 0.1200 | 0.6667 | 0.5102 | 1,252 |
| card_present=1 | 37 | 0.0270 | 0.1000 | 1.0000 | 0.2703 | 72 |

Không kết luận fairness từ các slice nhỏ này. `country_risk=1` chỉ có 18 record; cần uncertainty, thêm dữ liệu và review tính phù hợp/pháp lý của feature.

## Bài 8

- Policy A: accuracy 0.760, precision 0.2667, recall 0.800, F1 0.400, cost 6,200.
- Policy B: accuracy 0.895, precision 0.4815, recall 0.650, F1 0.5532, cost 7,700.
- Hòa vốn khi `FN_cost / FP_cost = 10`.

## Kiểm thử gói

Kết quả phát hành kỳ vọng: 12/12 public tests pass, hai notebooks chạy không lỗi, practical solution tái tạo đúng số liệu trên. Xem `VALIDATION.md` cho log thực tế của bản phát hành.

