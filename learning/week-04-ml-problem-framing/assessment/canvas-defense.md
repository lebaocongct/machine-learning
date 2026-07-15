# Canvas Defense — Checklist và câu hỏi phản biện

## Checklist trước khi nói

- [ ] Tôi có thể mô tả goal → decision → action trong 30 giây.
- [ ] Unit, prediction time và target window không mơ hồ.
- [ ] Mọi feature chủ chốt có bằng chứng availability.
- [ ] Primary metric gắn với cost/constraint.
- [ ] Có dummy và operational baseline.
- [ ] Threshold/hyperparameter không chọn bằng test.
- [ ] Có ít nhất hai slice và failure/fallback policy.
- [ ] Kết luận có thể là Conditional Go hoặc No-Go.

## 12 câu reviewer có thể hỏi

1. Nếu không có model, quy trình hiện tại làm gì và tốn bao nhiêu?
2. Ai nhận prediction và họ thay đổi decision cụ thể nào?
3. Positive label được tạo bởi outcome thật hay policy cũ?
4. Label hoàn tất lúc nào; các record chưa mature xử lý ra sao?
5. Feature nào có nguy cơ nhìn tương lai?
6. Vì sao metric này phản ánh lỗi quan trọng hơn accuracy?
7. FP/FN cost do ai xác nhận; độ nhạy kết quả với cost ra sao?
8. Model phải thắng baseline nào và bao nhiêu mới đáng triển khai?
9. Capacity/latency/privacy constraint có thể bác bỏ giải pháp không?
10. Average metric che nhóm/tình huống nào?
11. Khi timeout, drift hoặc data missing, fallback là gì?
12. Bằng chứng nào khiến bạn đổi từ Go thành No-Go?

## Cách chấm defense — 10 điểm

| Tiêu chí | 0 | 1 | 2 |
|---|---|---|---|
| Goal/decision/action | Mơ hồ | Thiếu một liên kết | Chuỗi rõ, có owner |
| Contract | Sai/mâu thuẫn | Còn edge case | Unit/time/target/window khóa rõ |
| Metric/baseline | Không có | Có nhưng chưa gắn cost | Metric, trade-off, baseline thuyết phục |
| Risk/constraint | Bỏ qua | Nêu chung | Có leakage, slice, capacity/fallback |
| Evidence/judgment | Khẳng định quá mức | Có giả định | Phân biệt bằng chứng, giả định, Go gate |

