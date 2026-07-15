# Kiến thức Tuần 4 — Từ nhu cầu đến bài toán ML

## 1. Bản đồ AI → ML → Deep Learning

**Artificial Intelligence (AI)** là phạm vi rộng của hệ thống thực hiện nhiệm vụ thường cần trí tuệ con người: lập kế hoạch, suy luận, nhận biết, tạo nội dung hoặc ra quyết định. **Machine Learning (ML)** là một nhánh của AI, học quy luật từ dữ liệu thay vì mã hóa mọi luật bằng tay. **Deep Learning (DL)** là một nhánh của ML dùng mạng nơ-ron nhiều tầng để học biểu diễn.

Quan hệ tập hợp:

```text
AI
└── Machine Learning
    └── Deep Learning
        └── TensorFlow/Keras là một bộ công cụ triển khai
```

TensorFlow không quyết định ta có bài toán đúng hay không. Một pipeline huấn luyện hoàn hảo vẫn thất bại nếu target không gắn với decision, feature chưa có khi dự đoán hoặc metric không phản ánh chi phí.

### Predictive và generative

- Predictive ML ước lượng nhãn, số, thứ hạng hoặc rủi ro từ quan sát hiện có.
- Generative ML tạo dữ liệu mới như văn bản, ảnh hoặc âm thanh có điều kiện theo prompt/context.
- Một sản phẩm có thể phối hợp cả hai: classifier định tuyến yêu cầu, LLM soạn nháp câu trả lời, con người duyệt.

## 2. Họ bài toán

### Supervised learning

Dữ liệu có cặp đầu vào–nhãn `(x, y)`.

| Kiểu output | Bài toán | Ví dụ | Metric thường gặp |
|---|---|---|---|
| Nhãn rời rạc | Classification | Ticket có escalation trong 48h? | Precision, recall, F1, PR-AUC |
| Số liên tục | Regression | ETA giao hàng bao nhiêu phút? | MAE, RMSE, quantile loss |
| Thứ tự | Ranking | Sắp kết quả tìm kiếm | NDCG, MRR, Recall@K |

### Unsupervised learning

Không có nhãn định trước. Clustering tìm nhóm; dimensionality reduction tạo biểu diễn thấp chiều; anomaly detection tìm quan sát lệch chuẩn. Cluster không tự mang ý nghĩa nghiệp vụ: cần chuyên gia đặt tên, kiểm tra độ ổn định và gắn với hành động.

### Generative learning

Output là nội dung mới. Việc đánh giá thường cần nhiều lớp: chất lượng nhiệm vụ, tính đúng/sát nguồn, an toàn, chi phí, latency và human evaluation. Không ép bài toán tạo sinh vào metric classification nếu output không phải nhãn rời rạc.

## 3. Khi nào không nên dùng ML

Ưu tiên giải pháp đơn giản hơn khi:

- Luật đầy đủ, ổn định và xác định, ví dụ tính thuế theo bảng quy tắc đã công bố.
- Không có decision/action sau dự đoán.
- Không thu được label hoặc label đến quá muộn để học/giám sát.
- Lỗi hiếm nhưng hậu quả không chấp nhận được và không có human safeguard.
- Volume thấp, quy trình thủ công rẻ hơn vòng đời dữ liệu–model–monitoring.
- Chính sách/pháp lý cấm hoặc yêu cầu giải thích mà thiết kế hiện tại không đáp ứng.
- Một heuristic đã đạt mục tiêu với chi phí thấp hơn.

Checklist **ML suitability**:

1. Có pattern có thể học từ dữ liệu không?
2. Có đủ ví dụ đại diện và label đáng tin không?
3. Input có sẵn tại prediction time không?
4. Output thay đổi một decision cụ thể không?
5. Giá trị cải thiện lớn hơn chi phí xây dựng/vận hành không?
6. Có thể đo chất lượng và phản hồi sau triển khai không?

Nếu câu 3 hoặc 4 là “không”, dừng và frame lại.

## 4. Decision-first problem framing

Đừng bắt đầu bằng “hãy dự đoán churn”. Bắt đầu bằng chuỗi:

```text
Mục tiêu phi-ML → decision → action → prediction → target → dữ liệu → metric
```

Ví dụ support:

- Mục tiêu phi-ML: giảm escalation không được chuẩn bị trước và giữ SLA.
- Decision: khi ticket mới đến, đưa vào specialist queue hay standard queue.
- Decision owner: Support Operations Manager.
- Action: nếu score vượt threshold, gửi specialist queue; nếu không, standard queue.
- Prediction: xác suất ticket sẽ escalation trong 48 giờ.
- Target: `escalated_within_48h ∈ {0,1}`.

Một prediction không có consumer/action chỉ là dashboard, chưa phải sản phẩm ML.

## 5. Prediction contract

Prediction contract khóa ngữ nghĩa trước khi code. Tối thiểu gồm:

- **Unit of prediction**: một ticket, một giao dịch, một user-day…
- **Prediction time**: thời điểm hệ thống phải tạo output.
- **Input availability**: trường nào đã tồn tại ở thời điểm đó.
- **Target definition**: sự kiện nào tạo positive label.
- **Target window**: quan sát target từ mốc nào đến mốc nào.
- **Output type**: binary probability, numeric estimate, category, ranking score.
- **Latency/SLA** và owner.

Ví dụ chính xác:

```json
{
  "unit_of_prediction": "one newly created support ticket",
  "prediction_time": "after initial message parsing and before queue assignment",
  "target_name": "escalated_within_48h",
  "target_definition": "1 if a manager-confirmed escalation occurs within 48 hours",
  "target_window_start_hours": 0,
  "target_window_end_hours": 48,
  "output_type": "binary",
  "latency_sla_ms": 500,
  "owner": "Support Operations Manager"
}
```

### Unit, time và horizon

Ba lỗi phổ biến:

1. Dòng dữ liệu là event nhưng decision áp dụng cho customer → unit mismatch.
2. Feature tổng hợp dùng dữ liệu sau thời điểm dự đoán → time leakage.
3. Target “eventually churned” không có cửa sổ → không biết khi nào label hoàn tất.

## 6. Target, label và proxy

**Target** là đại lượng bài toán muốn ước lượng. **Label** là giá trị target quan sát trong dataset. **Proxy label** thay thế target khó đo, ví dụ click thay cho mức hữu ích.

Proxy chỉ hợp lệ khi ghi rõ:

- Quan hệ kỳ vọng giữa proxy và outcome thật.
- Nhóm nào có thể bị đo sai.
- Thời gian trễ và missingness.
- Cách kiểm tra proxy drift.
- Non-goal: điều model không được hiểu là đang tối ưu.

Label quality phải được audit: nguồn tạo label, hướng dẫn annotator, disagreement, coverage, latency, policy changes và noise.

## 7. Metric: model, decision và business

Ba tầng không đồng nhất:

- **Model metric**: precision, recall, MAE… đánh giá dự đoán.
- **Decision metric**: tổng chi phí, review capacity, SLA after routing.
- **Business KPI**: churn, revenue, satisfaction, harm rate.

Model metric tốt không bảo đảm KPI tốt; cần giả thuyết liên kết và đo sau triển khai.

### Confusion matrix

Với positive class là “cần specialist”:

| | Thực tế positive | Thực tế negative |
|---|---:|---:|
| Dự đoán positive | TP | FP |
| Dự đoán negative | FN | TN |

- `precision = TP / (TP + FP)`: trong số case đã đưa specialist, bao nhiêu thực sự positive?
- `recall = TP / (TP + FN)`: trong số positive thật, bắt được bao nhiêu?
- `F1`: trung bình điều hòa precision và recall.
- `accuracy`: dễ gây hiểu sai khi positive hiếm.

### Cost-sensitive operating point

Nếu chi phí FP là `C_FP`, FN là `C_FN`:

```text
TotalCost(threshold) = FP(threshold) × C_FP + FN(threshold) × C_FN
```

Threshold là policy, không phải hằng số toán học mặc định 0.5. Chọn threshold trên validation theo metric/cost đã thỏa thuận; chỉ sau đó đánh giá test một lần.

### Hai trade-off phải viết ra

Ví dụ:

1. Tăng recall làm giảm escalation bỏ sót nhưng tăng specialist load và có thể giảm precision.
2. Tăng precision bảo vệ capacity nhưng có thể tăng FN, gây escalation không chuẩn bị.

Luôn kèm constraint: “recall tối thiểu 0.85, số ticket specialist không vượt 20%/ngày”, thay vì tối ưu một metric vô điều kiện.

## 8. Baseline

Baseline trả lời “model mới có tốt hơn lựa chọn rẻ nhất hợp lý không?”.

- Classification: majority class, stratified random, rule/heuristic hiện tại.
- Regression: mean/median constant, last value, seasonal naive.
- Ranking: popularity/recentness.
- Generative: template, retrieval-only, human workflow hiện tại.

Nên có ít nhất hai baseline:

1. **Dummy baseline** kiểm tra model có học signal cơ bản.
2. **Operational baseline** so với hệ thống đang dùng.

So sánh cùng split, target, metric và cost. Không quảng cáo accuracy cao hơn nếu recall positive vẫn bằng 0.

## 9. Batch, online và human-in-the-loop

### Batch

Chạy theo lịch trên nhiều record. Phù hợp khi latency phút/giờ/ngày chấp nhận được, dữ liệu cập nhật theo lô, chi phí throughput quan trọng hơn phản hồi tức thời.

### Online

Dự đoán cho từng request gần thời gian thực. Cần feature online nhất quán, latency budget, fallback, timeout, monitoring và versioning.

### Human-in-the-loop

Model ưu tiên/đề xuất, con người quyết định. Cần:

- Ngưỡng hoặc vùng bất định để chuyển review.
- Capacity constraint và queue discipline.
- UI hiển thị bằng chứng phù hợp, không tạo automation bias.
- Lưu decision của người dùng và outcome thật tách biệt.

Mode triển khai là constraint của framing, không phải chi tiết để cuối dự án mới chọn.

## 10. Data leakage và feature availability

**Leakage** xảy ra khi dữ liệu lúc huấn luyện chứa thông tin không hợp lệ ở thời điểm dự đoán hoặc trực tiếp/gián tiếp tiết lộ label.

Các loại chính:

- Post-outcome: `resolution_hours`, `chargeback_amount` chỉ có sau outcome.
- Label-generation: mã điều tra dùng để xác nhận fraud.
- Time-window: aggregate vô tình nhìn qua prediction time.
- Split leakage: cùng entity gần như trùng xuất hiện ở train và test.
- Preprocessing leakage: fit scaler/imputer trên toàn bộ dữ liệu.
- Policy leakage: feature ghi lại decision cũ, nhưng bị hiểu là customer behavior.

Audit feature catalog bằng bốn câu:

1. Feature được tạo ở stage nào?
2. Available offset so với prediction time là bao nhiêu?
3. Có dùng output/outcome tương lai không?
4. Online và offline có cùng định nghĩa không?

Độ tương quan cao bất thường hoặc kết quả gần hoàn hảo là tín hiệu điều tra, không phải bằng chứng đủ để kết luận leakage.

## 11. Split và test discipline

Với dữ liệu có thời gian:

```text
quá khứ ── train 70% ── validation 15% ── test 15% ── tương lai
```

- Train: học tham số/hiểu prevalence.
- Validation: chọn feature, model, hyperparameter, threshold.
- Test: ước lượng cuối cùng sau khi mọi lựa chọn đã khóa.

Nếu xem test rồi đổi threshold, test đã trở thành validation. Cần một test mới hoặc đánh dấu kết quả là exploratory.

## 12. Evaluation slices

Average metric có thể che lỗi ở nhóm quan trọng. Chọn slice từ rủi ro và vận hành, không chỉ từ cột dễ có:

- channel, device type, geography, customer tier;
- new vs existing users;
- volume peak vs normal;
- language hoặc accessibility group khi phù hợp và được phép.

Với mỗi slice báo `n`, prevalence, TP/FP/TN/FN, precision, recall và cost. Không so metric khi mẫu quá nhỏ mà không nêu uncertainty. Slice evaluation là bước phát hiện, không thay thế đánh giá fairness đầy đủ.

## 13. ML Problem Canvas

Một canvas đạt yêu cầu phải trả lời:

1. Problem name và non-ML goal.
2. Decision, action, decision owner.
3. Prediction unit/time.
4. Target, target window, task/output.
5. Primary metric và trade-off.
6. Dummy + operational baseline.
7. Success criterion có số và thời hạn.
8. Deployment mode và latency/capacity.
9. Constraints.
10. Evaluation slices.
11. Leakage risks và kiểm soát.
12. Non-goals.

Canvas là giả thuyết phiên bản hóa. Khi policy, data source hoặc target definition đổi, phải cập nhật canvas/contract trước khi chạy lại pipeline.

## 14. Go/No-Go gate trước model

Chỉ chuyển sang tuần xây model khi tất cả câu sau là “có”:

- Decision và owner đã được xác nhận?
- Output dẫn đến action rõ ràng?
- Target có định nghĩa và cửa sổ đóng?
- Input cần thiết có sẵn tại prediction time?
- Có baseline đo trên cùng protocol?
- Metric/cost/constraint được chấp thuận?
- Có split chống leakage và test discipline?
- Có slices và failure policy?
- Giá trị kỳ vọng đủ lớn so với chi phí vận hành?

Nếu chưa, kết quả đúng của tuần này có thể là **No-Go** kèm thí nghiệm/thu thập dữ liệu cần thiết. Đó là một kết quả kỹ thuật tốt, không phải thất bại.

## 15. Câu hỏi tự kiểm tra

1. Vì sao “dự đoán churn” chưa phải problem statement hoàn chỉnh?
2. Một classifier accuracy 96% có thể tệ hơn majority baseline như thế nào?
3. Tại sao threshold 0.5 không tự động tối ưu business cost?
4. Khác biệt giữa prediction time và target window là gì?
5. Cho ba ví dụ post-outcome leakage.
6. Khi nào batch tốt hơn online?
7. Proxy label có thể gây hại cho nhóm nào và vì sao?
8. Vì sao test chỉ được mở một lần?
9. Operational baseline khác dummy baseline thế nào?
10. Khi nào “không dùng ML” là quyết định tốt nhất?

