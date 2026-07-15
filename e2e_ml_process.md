# Quy trình End-to-End Machine Learning

> Tiêu chuẩn thực hành từ xác định bài toán đến vận hành, tái huấn luyện và ngừng hệ thống

| Thuộc tính    | Giá trị                                                                                                       |
| ------------- | ------------------------------------------------------------------------------------------------------------- |
| Phiên bản     | 1.0                                                                                                           |
| Ngày cập nhật | 2026-07-14                                                                                                    |
| Trạng thái    | Tài liệu chuẩn tham chiếu                                                                                     |
| Đối tượng     | Product Owner, Domain Expert, Data Engineer, Data Scientist, ML Engineer, Platform/SRE, Security/Risk         |
| Phạm vi       | ML dự đoán: classification, regression, ranking, recommendation, forecasting, clustering và anomaly detection |
| Ngoài phạm vi | GenAI/LLM cần bổ sung evaluation, safety, red teaming và content governance riêng                             |

---

## 1. Mục đích

Quy trình End-to-End Machine Learning, viết tắt là **E2E ML**, bao phủ toàn bộ vòng đời của một hệ thống ML:

1. Xác định quyết định kinh doanh cần hỗ trợ.
2. Xây dựng dữ liệu và nhãn đúng theo thời điểm dự đoán.
3. Thiết kế phép đánh giá trước khi phát triển mô hình.
4. Huấn luyện, đánh giá và quản lý thí nghiệm có thể tái lập.
5. Đóng gói, kiểm thử và phát hành mô hình an toàn.
6. Giám sát dữ liệu, mô hình, dịch vụ và kết quả kinh doanh.
7. Tái huấn luyện, xử lý sự cố hoặc ngừng hệ thống khi cần.

Một mô hình có metric tốt trong notebook **chưa phải** là một hệ thống ML hoàn chỉnh. Hệ thống chỉ được xem là E2E khi prediction tạo ra hành động có giá trị, có thể vận hành, quan sát, rollback, kiểm toán và bảo trì.

### 1.1. Mục tiêu của tài liệu

- Chuẩn hóa thuật ngữ và trách nhiệm giữa các vai trò.
- Ngăn data leakage, training–serving skew và đánh giá sai.
- Định nghĩa artifact, quality gate và Definition of Done.
- Làm khung tham chiếu cho cả dự án thử nghiệm lẫn production.
- Không phụ thuộc framework; có thể áp dụng với scikit-learn, TensorFlow, PyTorch hoặc nền tảng managed ML.

### 1.2. Nguyên tắc cốt lõi

1. **Tối ưu quyết định, không tối ưu metric một cách cô lập.**
2. **Bắt đầu bằng baseline và giải pháp không dùng ML.**
3. **Định nghĩa prediction time trước khi định nghĩa feature và label.**
4. **Thiết kế evaluation trước khi train model.**
5. **Fit mọi preprocessing chỉ trên training data.**
6. **Test set là tài sản được kiểm soát, không phải validation set thứ hai.**
7. **Mọi kết quả phải truy vết được tới code, config, data và environment.**
8. **Training và serving phải dùng cùng định nghĩa feature/preprocessing.**
9. **Không deploy nếu chưa có rollback, fallback và monitoring.**
10. **Risk, privacy, security và fairness là yêu cầu xuyên suốt vòng đời.**
11. **Tự động hóa một quy trình đã ổn định; không tự động hóa sự hỗn loạn.**
12. **Cho phép kết luận “không nên dùng ML” tại mọi giai đoạn.**

---

## 2. Thuật ngữ chuẩn

| Thuật ngữ             | Định nghĩa                                                                 |
| --------------------- | -------------------------------------------------------------------------- |
| Business objective    | Kết quả kinh doanh hoặc vận hành cần cải thiện                             |
| Decision/action       | Hành động được thực hiện dựa trên prediction                               |
| Prediction unit       | Đối tượng của một prediction, ví dụ user, giao dịch, thiết bị              |
| Prediction time       | Thời điểm hệ thống phải tạo prediction                                     |
| Feature               | Thông tin có sẵn tại hoặc trước prediction time                            |
| Label/target          | Kết quả thực tế cần dự đoán, được xác định theo label window               |
| Label window          | Khoảng thời gian quan sát để gán label sau prediction time                 |
| Prediction horizon    | Khoảng cách từ prediction time tới thời điểm/khoảng cần dự đoán            |
| Training example      | Một cặp feature–label gắn với entity và timestamp cụ thể                   |
| Baseline              | Mốc so sánh tối thiểu: rule, constant, heuristic hoặc model đơn giản       |
| Champion              | Model hiện đang được chấp nhận hoặc phục vụ production                     |
| Challenger            | Model ứng viên được so sánh với champion                                   |
| Data drift            | Phân phối input thay đổi theo thời gian                                    |
| Concept drift         | Quan hệ giữa input và target thay đổi                                      |
| Training–serving skew | Dữ liệu hoặc logic feature khác nhau giữa train và inference               |
| Calibration           | Mức độ probability dự đoán phản ánh xác suất thực tế                       |
| Model registry        | Nơi quản lý model artifact, version, trạng thái và metadata                |
| Lineage               | Quan hệ truy vết giữa source data, dataset, code, run, model và deployment |
| SLI/SLO               | Chỉ số và mục tiêu chất lượng dịch vụ                                      |
| Rollback              | Quay lại model hoặc phiên bản hệ thống ổn định trước đó                    |
| Retirement            | Ngừng model/hệ thống có kiểm soát                                          |

---

## 3. Tổng quan vòng đời

Quy trình không phải một chuỗi tuyến tính chạy đúng một lần. Validation có thể trả dự án về bước dữ liệu; monitoring có thể kích hoạt điều tra, retraining hoặc retirement.

~~~mermaid
flowchart TD
    A["0. Problem framing"] --> B["1–2. Data readiness"]
    B --> C["3. Evaluation design"]
    C --> D["4–5. Features and experiments"]
    D --> E["6. Model validation"]
    E -->|Pass| F["7–8. Package and release"]
    E -->|Fail| B
    F --> G["9. Monitor outcomes"]
    G -->|New data or drift| H["10. Retrain"]
    H --> C
    G -->|Incident| I["11. Respond and rollback"]
    G -->|No longer valuable| J["12. Retire"]
~~~

### 3.1. Phase control matrix

|                Phase | Input chính                       | Hoạt động chính                                                   | Output/artifact                                           | Owner chính         | Exit/gate            |
| -------------------: | --------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------- | ------------------- | -------------------- |
|         0. Discovery | Business context, current process | Frame decision, prediction, KPI, feasibility và risk              | Problem brief, prediction/KPI contract, risk tier         | Product + DS        | G0                   |
|     1. Data sourcing | Approved use case                 | Inventory source, define label, governance và point-in-time joins | Data inventory, label spec, data contract                 | Data + Domain + DS  | Sang Phase 2         |
|   2. Data validation | Raw snapshot + contracts          | EDA, schema/quality/representativeness validation                 | Quality report, versioned schema, reference distributions | Data + DS           | G1                   |
| 3. Evaluation design | Data-ready decision               | Define split, baseline, metrics, slices và test policy            | Evaluation plan, metric contract                          | DS + Domain/Product | G2                   |
|          4. Features | Evaluation plan + training split  | Build and test point-in-time correct transformations              | Versioned preprocessing/feature pipeline                  | DS + MLE            | Sang Phase 5         |
|        5. Experiment | Training pipeline + baselines     | Train, tune, track, reproduce và select candidates                | Run records, candidate models                             | DS + MLE            | G3                   |
|  6. Model validation | Frozen candidate                  | Test, error/slice/risk/robustness/performance review              | Evaluation report, model card, acceptance decision        | DS + Risk + Product | G4                   |
|           7. Package | Approved model                    | Export, register, contract/integration/load/security tests        | Registered release candidate, test evidence               | MLE + SRE           | G5                   |
|           8. Release | Production-ready candidate        | Shadow/canary/A-B, observe, promote hoặc rollback                 | Versioned deployment, release record                      | MLE + SRE + Product | G6                   |
|           9. Monitor | Live deployment + logs/labels     | Observe data/model/service/business/risk                          | Dashboards, alerts, health reports                        | SRE + MLE + Product | Liên tục             |
|          10. Retrain | Approved trigger + mature labels  | Re-run validated pipeline, champion–challenger                    | New approved version hoặc no-change decision              | MLE + DS            | Quay lại G1–G6       |
|         11. Incident | Alert/report                      | Triage, contain, rollback, correct và postmortem                  | Incident/RCA/CAPA records                                 | SRE + system owner  | Service recovered    |
|       12. Retirement | Retirement decision               | Migrate, disable, archive/delete và revoke                        | Decommission record                                       | Product + SRE       | Consumers signed off |

---

## 4. Vai trò và trách nhiệm

### 4.1. Vai trò

| Vai trò                | Trách nhiệm chính                                                                   |
| ---------------------- | ----------------------------------------------------------------------------------- |
| Business/Product Owner | Giá trị kinh doanh, KPI, phạm vi, ngân sách và quyết định go/no-go                  |
| Domain Expert          | Định nghĩa nghiệp vụ, label, lỗi chấp nhận được và giải thích kết quả               |
| Data Owner/Steward     | Quyền sử dụng dữ liệu, chất lượng, retention và phê duyệt truy cập                  |
| Data Engineer          | Ingestion, data pipeline, schema, lineage, backfill và SLA dữ liệu                  |
| Data Scientist         | EDA, evaluation design, feature/model experiment và error analysis                  |
| ML Engineer            | Training pipeline, packaging, model registry, deployment và training–serving parity |
| Platform/SRE           | Hạ tầng, CI/CD, serving reliability, observability, capacity và incident response   |
| Security/Privacy/Risk  | Threat, privacy, compliance, fairness và risk acceptance                            |
| QA/Reviewer            | Independent review, test evidence và release verification                           |

### 4.2. Ma trận trách nhiệm tối thiểu

Ký hiệu: **A** = Accountable, **R** = Responsible, **C** = Consulted.

| Hoạt động              | Product | Domain | Data |   DS |  MLE |  SRE | Risk |
| ---------------------- | ------: | -----: | ---: | ---: | ---: | ---: | ---: |
| Problem framing và KPI |     A/R |      C |    C |    R |    C |      |    C |
| Label và data contract |       C |    A/C |    R |    R |    C |      |    C |
| Evaluation và model    |       C |      C |    C |  A/R |    R |      |    C |
| Production validation  |       C |        |    C |    R |  A/R |    R |    C |
| Release và rollback    |       A |      C |    C |    C |    R |    R |    C |
| Monitoring và incident |     A/C |      C |    R |    R |    R |  A/R |    C |
| Risk acceptance        |       A |      C |    C |    C |    C |    C |  A/R |
| Retirement             |     A/R |      C |    R |    C |    R |    R |    C |

Một người có thể giữ nhiều vai trò trong nhóm nhỏ, nhưng trách nhiệm và bằng chứng phê duyệt vẫn phải rõ.

---

## 5. Phase 0 — Discovery và problem framing

### 5.1. Mục tiêu

Chứng minh rằng bài toán cần giải quyết có giá trị, có hành động sau prediction, có dữ liệu khả thi và ML là lựa chọn hợp lý.

### 5.2. Câu hỏi bắt buộc

#### Business và decision

- Vấn đề hiện tại là gì? Ai chịu tác động?
- Quyết định/hành động nào sẽ dùng prediction?
- Nếu không có model, quy trình hiện tại hoạt động thế nào?
- Chi phí false positive và false negative là gì?
- Ai có quyền override model?
- Thành công được đo bằng business KPI nào?

#### Prediction contract

- Prediction unit là gì?
- Prediction được tạo tại thời điểm nào?
- Hệ thống được dùng feature nào tại thời điểm đó?
- Target được định nghĩa chính xác ra sao?
- Label window và prediction horizon là bao lâu?
- Prediction chạy theo batch, request hay event?
- Latency, throughput và freshness yêu cầu là gì?

#### Feasibility

- Có đủ historical data và label không?
- Label có đáng tin cậy hay chỉ là proxy?
- Hành động do model đề xuất có thể thay đổi outcome không?
- Model có tốt hơn rule hoặc heuristic đơn giản không?
- Lợi ích kỳ vọng có lớn hơn chi phí xây dựng và vận hành không?

#### Risk

- Use case có ảnh hưởng tới sức khỏe, tài chính, tuyển dụng, quyền lợi hoặc an toàn không?
- Có dữ liệu cá nhân, nhạy cảm hoặc thuộc phạm vi kiểm soát đặc biệt không?
- Cần human review hoặc khả năng giải thích ở mức nào?
- Điều gì xảy ra nếu model không trả kết quả hoặc trả kết quả sai hàng loạt?

### 5.3. Cây metric

Không dùng một model metric để đại diện cho toàn bộ thành công:

| Tầng             | Ví dụ                                                      |
| ---------------- | ---------------------------------------------------------- |
| Business outcome | doanh thu tăng, churn giảm, thời gian xử lý giảm           |
| Decision/process | số case xử lý đúng, tỷ lệ automation, acceptance rate      |
| Model quality    | PR-AUC, recall, MAE, NDCG, calibration                     |
| Service quality  | availability, p95 latency, error rate, cost/prediction     |
| Guardrail/risk   | complaint rate, fairness gap, override rate, harmful error |

### 5.4. Artifact

- ML Problem Brief.
- Prediction Contract.
- KPI và Metric Tree.
- Non-ML Baseline.
- Risk Tier và danh sách reviewer bắt buộc.
- Cost–benefit estimate.
- Decision log.

### 5.5. Gate G0 — Opportunity

**Pass khi:**

- Có decision/action rõ ràng sau prediction.
- Có owner cho KPI và vận hành.
- Có baseline hiện tại và giá trị kỳ vọng đo được.
- Dữ liệu/label có khả năng thu thập hợp pháp và thực tế.
- Risk tier và yêu cầu phê duyệt đã xác định.

**Fail/stop khi:**

- Không có hành động sau prediction.
- Rule đơn giản đã đủ tốt và rẻ hơn đáng kể.
- Không thể có label đáng tin cậy.
- Lợi ích không bù chi phí/rủi ro.

---

## 6. Phase 1 — Data sourcing, labeling và governance

### 6.1. Mục tiêu

Xác định dữ liệu nào được phép dùng, dữ liệu đó được tạo như thế nào và làm sao tái dựng đúng training example tại prediction time.

### 6.2. Data inventory

Với mỗi nguồn dữ liệu, ghi lại:

- Owner và hệ thống nguồn.
- Schema, khóa, timestamp và timezone.
- Tần suất cập nhật và data freshness.
- Coverage, historical depth và backfill policy.
- Quyền truy cập, license, consent và purpose limitation.
- PII/sensitive fields, encryption và retention.
- SLA, known issues và downstream consumers.

### 6.3. Label specification

Label phải là một hợp đồng có version, không chỉ là tên cột.

| Thành phần        | Câu hỏi                                             |
| ----------------- | --------------------------------------------------- |
| Entity            | Label thuộc về user, order, device hay session nào? |
| Observation point | Chốt feature tại thời điểm nào?                     |
| Outcome           | Sự kiện nào tạo positive/giá trị target?            |
| Window            | Quan sát outcome trong bao lâu?                     |
| Exclusion         | Trường hợp nào không đủ điều kiện gán label?        |
| Delay             | Sau bao lâu label mới hoàn chỉnh?                   |
| Source            | Ground truth đến từ hệ thống nào?                   |
| Quality           | Cách audit noise, missing và disagreement?          |

### 6.4. Point-in-time correctness

Một feature chỉ hợp lệ nếu tồn tại trước hoặc tại prediction time. Cần:

- Join theo entity **và** event time.
- Không dùng bản ghi được sửa/cập nhật sau prediction time.
- Không dùng thống kê tính trên tương lai.
- Không để label window chồng vào feature window.
- Xử lý late-arriving events và backfill theo quy tắc cố định.
- Lưu snapshot/version để tái dựng dataset.

### 6.5. Data contract

Data contract tối thiểu bao gồm:

- Tên, type, nullability và semantic của field.
- Allowed range/category và đơn vị.
- Primary/entity key và event timestamp.
- Freshness, volume và uniqueness expectation.
- Quy tắc PII, retention và access.
- Owner, version và backward compatibility.
- Hành động khi vi phạm: warn, quarantine hay block pipeline.

### 6.6. Artifact

- Data Source Inventory.
- Data Access Approval.
- Label Specification.
- Data Contract/Schema.
- Dataset Snapshot và lineage ID.
- Data retention/deletion policy.

---

## 7. Phase 2 — EDA và data validation

### 7.1. Mục tiêu

Đánh giá dữ liệu có đủ chất lượng, đại diện và ổn định để phát triển model hay không.

### 7.2. Kiểm tra cấu trúc và chất lượng

- Row count, feature count, schema và type.
- Missing, invalid, duplicate và contradictory records.
- Cardinality, range, unit và impossible values.
- Outlier thật so với lỗi đo/nhập liệu.
- Timestamp coverage, timezone và data gaps.
- Label prevalence/distribution và label noise.
- Duplicate entity hoặc near-duplicate giữa các giai đoạn thời gian.
- Coverage theo source, region, device, cohort và subgroup quan trọng.

### 7.3. Phân tích thống kê

- Distribution của feature và target.
- Quan hệ feature–target nhưng không diễn giải correlation thành causality.
- Multicollinearity và redundancy.
- Class imbalance và rare categories.
- Seasonal/time trend và regime change.
- Sample bias, survivorship bias và selection bias.
- So sánh historical data với expected serving population.

### 7.4. Kiểm tra label

- Audit thủ công một sample stratified.
- Đo inter-annotator agreement nếu có human label.
- Tách “unknown/not yet observed” khỏi negative.
- Kiểm tra label delay và censoring.
- Xác định proxy label có thể tạo vòng lặp phản hồi hay không.

### 7.5. Data validation tự động

Nên kiểm tra theo schema và thống kê:

- Missing/invalid rate.
- Range và categorical domain.
- Volume và freshness.
- New/dropped feature.
- Distribution shift.
- Training–serving skew.
- Label prevalence bất thường.

### 7.6. Artifact

- EDA Report.
- Data Quality Report.
- Versioned Schema.
- Data Issue Backlog có owner/severity.
- Reference distributions cho monitoring.
- Data Readiness Decision.

### 7.7. Gate G1 — Data readiness

**Pass khi:**

- Dataset có lineage và có thể tái tạo.
- Label definition được Domain Owner chấp nhận.
- Critical data quality issues đã sửa hoặc có mitigation.
- Population đủ đại diện cho intended use.
- Quyền sử dụng, privacy và retention đã được phê duyệt.

---

## 8. Phase 3 — Evaluation design

### 8.1. Mục tiêu

Khóa cách chia dữ liệu, baseline, metric, threshold và acceptance criteria **trước** khi tối ưu model.

### 8.2. Chọn chiến lược split

| Cấu trúc dữ liệu            | Split phù hợp              | Rủi ro cần tránh                  |
| --------------------------- | -------------------------- | --------------------------------- |
| IID, entity độc lập         | Random split               | duplicate/near-duplicate          |
| Classification mất cân bằng | Stratified split           | prevalence khác giữa các split    |
| Một entity có nhiều record  | Group split theo entity    | cùng user/device ở train và test  |
| Dữ liệu theo thời gian      | Temporal split/backtest    | dùng tương lai dự đoán quá khứ    |
| Dữ liệu không gian          | Spatial/group split        | vùng lân cận rò rỉ giữa split     |
| Recommendation              | User/item/time-aware split | đánh giá sai cold-start           |
| Forecasting                 | Rolling/expanding window   | random split phá thứ tự thời gian |

Tỷ lệ 70/15/15 hoặc 80/10/10 chỉ là điểm khởi đầu. Chiến lược split phải phản ánh cách model gặp dữ liệu mới trong production.

### 8.3. Vai trò của từng tập

- **Training:** fit model và tất cả preprocessing có learnable state.
- **Validation/Cross-validation:** chọn feature, model, hyperparameter và threshold.
- **Test/Holdout:** ước lượng cuối cùng sau khi mọi quyết định đã khóa.
- **Shadow/Online experiment:** xác nhận behavior và business impact trong môi trường thực.

Không được fit scaler, imputer, feature selection, target encoder hoặc oversampling trên toàn dataset trước khi split.

Nếu test set bị xem nhiều lần qua nhiều vòng release, tổ chức phải coi nó đã bị “tiêu thụ” và tạo holdout mới hoặc dùng sealed rolling windows.

### 8.4. Metric contract

Metric contract cần ghi:

- Tên metric chính và guardrail metrics.
- Công thức, averaging và sample weighting.
- Population/slice áp dụng.
- Evaluation window.
- Confidence interval hoặc độ biến thiên qua fold/time window.
- Baseline/champion cần vượt qua.
- Minimum acceptable value và non-inferiority margin.
- Threshold selection policy.
- Cách xử lý missing/delayed labels.

### 8.5. Chọn metric theo bài toán

| Bài toán               | Metric khởi đầu                            | Lưu ý                                                        |
| ---------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| Regression             | MAE, RMSE, quantile loss                   | Chọn theo chi phí lỗi; MAPE không phù hợp khi target gần 0   |
| Binary cân bằng        | F1, ROC-AUC, log loss                      | Đánh giá calibration nếu dùng probability                    |
| Binary mất cân bằng    | PR-AUC, recall tại precision tối thiểu     | Chọn threshold theo cost/capacity                            |
| Multiclass             | Macro-F1, per-class recall, log loss       | Không chỉ dùng accuracy                                      |
| Ranking/recommendation | NDCG@K, Recall@K, MAP@K                    | Bổ sung coverage, diversity và online KPI                    |
| Forecasting            | MAE/RMSE/WAPE/MASE theo horizon            | Dùng time backtest và naive baseline                         |
| Anomaly detection      | Precision@budget, recall khi có label      | Báo cáo label delay và review capacity                       |
| Clustering             | Stability, silhouette và domain validation | Không có metric nội tại nào tự chứng minh giá trị kinh doanh |

### 8.6. Baseline

Thứ tự baseline nên là:

1. Quy trình hiện tại.
2. Constant/majority/mean/last-value.
3. Rule hoặc heuristic.
4. Model đơn giản, dễ giải thích.
5. Champion production nếu đã có.

### 8.7. Gate G2 — Evaluation plan

**Pass khi:**

- Split phản ánh production và ngăn entity/time leakage.
- Baseline, metric, slice và acceptance threshold đã khóa.
- Test access policy rõ ràng.
- Có kế hoạch đánh giá uncertainty, calibration, fairness và business outcome khi phù hợp.

---

## 9. Phase 4 — Preprocessing và feature engineering

### 9.1. Mục tiêu

Tạo transformation pipeline có version, có thể tái sử dụng nhất quán ở training và serving.

### 9.2. Hoạt động

- Impute missing values.
- Encode categorical/text/image inputs.
- Scale/normalize khi thuật toán yêu cầu.
- Xử lý outlier theo quy tắc đã kiểm chứng.
- Tạo temporal, aggregate, interaction hoặc domain features.
- Feature selection.
- Class weight, sampling hoặc augmentation.
- Chuẩn hóa shape, dtype và range.

### 9.3. Quy tắc chống leakage

- Split trước, fit preprocessing sau.
- Mọi statistic phải học từ training fold tương ứng.
- Target encoding phải dùng cross-fitting.
- Oversampling chỉ áp dụng bên trong training fold.
- Feature selection và PCA nằm trong cross-validation pipeline.
- Aggregate feature phải point-in-time correct.
- Augmentation chỉ áp dụng cho training.
- Không dùng ID proxy hoặc post-outcome fields.

### 9.4. Training–serving parity

- Tái sử dụng cùng transformation code hoặc exported preprocessing graph.
- Version feature definition và lookup data.
- Kiểm thử cùng một input raw tạo feature giống nhau ở offline và online.
- Định nghĩa default/fallback cho missing hoặc unseen category.
- Log feature version kèm prediction.

### 9.5. Artifact

- Feature Specification.
- Preprocessing Pipeline.
- Feature/Data Transformation Tests.
- Feature version và lineage.
- Training-ready dataset manifest.

---

## 10. Phase 5 — Baseline, training và experiment

### 10.1. Mục tiêu

Phát triển model ứng viên qua các thí nghiệm có giả thuyết, tái lập được và so sánh công bằng.

### 10.2. Trình tự

1. Chạy baseline.
2. Chọn vài họ model phù hợp với dữ liệu và constraint.
3. Xác nhận model có thể overfit một sample nhỏ để kiểm tra pipeline.
4. Train với cùng split và metric contract.
5. Phân tích learning curve và train–validation gap.
6. Tuning có ngân sách và search space rõ ràng.
7. Chạy ablation để biết yếu tố nào tạo cải thiện.
8. Chọn candidate dựa trên quality, latency, size, cost và maintainability.

### 10.3. Metadata bắt buộc của một run

- Run ID và timestamp.
- Code commit.
- Dataset và schema version.
- Feature/preprocessing version.
- Config và hyperparameters.
- Random seed.
- Framework/environment/container version.
- Hardware và compute budget.
- Train/validation metrics và slice metrics.
- Checkpoint/model artifact.
- Parent run hoặc hypothesis.
- Owner và conclusion.

### 10.4. Hyperparameter tuning

- Không tune trên test set.
- Đặt budget theo số trial, thời gian hoặc compute cost.
- Dùng cross-validation đúng loại dữ liệu.
- Dùng early stopping/pruning khi hợp lý.
- Không thay nhiều yếu tố cùng lúc nếu cần suy luận nguyên nhân.
- So sánh với baseline/champion bằng cùng protocol.

### 10.5. Artifact

- Training Pipeline.
- Versioned Config.
- Experiment Tracker/Run Table.
- Baseline Report.
- Candidate Models.
- Reproducibility Manifest.

### 10.6. Gate G3 — Candidate readiness

**Pass khi:**

- Model vượt baseline trên validation theo metric contract.
- Kết quả tái lập trong tolerance.
- Không có dấu hiệu leakage.
- Có candidate đáp ứng sơ bộ latency, size và cost.
- Thí nghiệm có conclusion, không chỉ có bảng metric.

---

## 11. Phase 6 — Model evaluation, error analysis và risk validation

### 11.1. Mục tiêu

Xác nhận model đủ tốt, an toàn và phù hợp để trở thành release candidate.

### 11.2. Evaluation bắt buộc

#### Model quality

- Metric chính và confidence interval/variance.
- So sánh với baseline/champion.
- Per-class, per-horizon hoặc per-query metric.
- Confusion matrix và threshold curve khi phù hợp.
- Calibration và reliability curve nếu output là probability.

#### Error analysis

- False positive/false negative mẫu.
- Slice theo thời gian, region, source, device và subgroup liên quan.
- Head so với long-tail.
- New/rare category và missing-feature cases.
- Error severity thay vì chỉ error count.
- Root cause: data, label, feature, model, threshold hay workflow.

#### Robustness

- Missing/invalid/extreme input.
- Out-of-distribution và seasonal change.
- Noise, corrupted data hoặc adversarial behavior nếu use case yêu cầu.
- Sensitivity với seed và hyperparameter.
- Backtest qua nhiều time window.

#### Operational fitness

- Model size, startup/warm-up time.
- p50/p95/p99 latency.
- Throughput, memory, CPU/GPU và cost/prediction.
- Batch scalability.

#### Responsible ML

- Fairness/slice gaps dựa trên intended use và risk tier.
- Explainability phù hợp với người dùng/decision.
- Privacy leakage và membership/model inversion risk khi liên quan.
- Security/threat review.
- Human review, appeal và override path.
- Known limitation và out-of-scope use.

### 11.3. Test set

Chỉ chạy test sau khi:

- Feature, model và hyperparameter đã khóa.
- Threshold đã chọn trên validation.
- Acceptance criteria đã định nghĩa.
- Reviewer xác nhận không có test-driven tuning.

Nếu test fail, không sửa model rồi báo lại cùng test như một đánh giá độc lập. Cần ghi nhận test đã được dùng và quyết định protocol mới.

### 11.4. Artifact

- Final Evaluation Report.
- Error Analysis Report.
- Slice/Fairness Report khi áp dụng.
- Performance Benchmark.
- Model Card.
- Risk Assessment.
- Acceptance/Rejection Decision.

### 11.5. Gate G4 — Model acceptance

**Pass khi:**

- Đạt primary metric và tất cả guardrail thresholds.
- Vượt hoặc non-inferior với champion theo policy.
- Không có critical slice/risk failure.
- Đạt latency, resource và cost budget.
- Model card và limitation đầy đủ.
- Reviewer bắt buộc đã phê duyệt.

---

## 12. Phase 7 — Packaging và production validation

### 12.1. Mục tiêu

Biến model đã được chấp nhận thành release candidate bất biến, có contract và chạy được trong hạ tầng thật.

### 12.2. Package

- Serialize/export model theo format phù hợp.
- Đóng gói preprocessing hoặc ràng buộc version.
- Định nghĩa input/output signature.
- Pin dependency và runtime.
- Tạo model version duy nhất.
- Lưu artifact, metadata và approval vào model registry.
- Kiểm tra checksum/integrity.

### 12.3. Test pyramid cho ML

| Nhóm test   | Ví dụ                                                        |
| ----------- | ------------------------------------------------------------ |
| Unit        | schema parser, feature transform, metric, postprocessing     |
| Data test   | type, range, freshness, null, category, label prevalence     |
| Model smoke | load model, one-batch inference, no NaN/Inf                  |
| Parity      | prediction trước/sau export; offline/online feature parity   |
| Integration | raw input → feature → model → response                       |
| Contract    | request/response schema và backward compatibility            |
| Performance | latency, throughput, memory, startup và load                 |
| Resilience  | dependency timeout, missing feature, model unavailable       |
| Security    | malformed input, auth, secret scan, dependency vulnerability |

### 12.4. Production-like validation

- Chạy model trong container/runtime giống production.
- Dùng representative payload.
- Kiểm tra resource limit và concurrency.
- Kiểm tra logging, tracing và metrics.
- Kiểm tra model/feature version được ghi đúng.
- Kiểm tra fallback và rollback.

### 12.5. Artifact

- Immutable Model Artifact.
- Runtime/Container Image.
- API/Input–Output Contract.
- Test Evidence.
- Registry Entry.
- Release Notes.
- Rollback Procedure.

### 12.6. Gate G5 — Production readiness

**Pass khi:**

- Tất cả critical tests pass.
- Model chạy được trên production-like infrastructure.
- Training–serving parity trong tolerance.
- Security/privacy requirements đạt.
- Monitoring, fallback, rollback và on-call owner đã sẵn sàng.

---

## 13. Phase 8 — Deployment và rollout

### 13.1. Chọn kiểu inference

| Kiểu              | Dùng khi                                | Quan tâm chính                             |
| ----------------- | --------------------------------------- | ------------------------------------------ |
| Batch             | prediction theo lịch, latency không gắt | freshness, completion SLA, backfill        |
| Online API        | cần phản hồi theo request               | p95/p99 latency, availability, concurrency |
| Streaming         | prediction theo event liên tục          | ordering, exactly/at-least-once, state     |
| Edge/on-device    | privacy, offline hoặc latency rất thấp  | size, memory, quantization, update         |
| Human-in-the-loop | rủi ro cao hoặc confidence thấp         | queue SLA, override, feedback quality      |

### 13.2. Release strategy

- **Shadow:** model nhận traffic thật nhưng output không ảnh hưởng quyết định.
- **Canary:** phục vụ tỷ lệ nhỏ traffic và tăng dần khi guardrail đạt.
- **Blue–green:** hai environment hoàn chỉnh, chuyển traffic có kiểm soát.
- **A/B test:** đo causal business impact khi an toàn và đủ sample.
- **Champion–challenger:** so sánh model mới với model hiện tại.

Không dùng A/B test để thay thế model validation; hai lớp đánh giá giải quyết hai câu hỏi khác nhau.

### 13.3. Yêu cầu triển khai

- Version model, feature và config.
- Traffic allocation và eligibility rule.
- Idempotency/correlation ID.
- Input validation và auth.
- Timeout, retry và circuit breaker phù hợp.
- Fallback: rule, cached output, champion hoặc human queue.
- Prediction logging có sampling/retention phù hợp.
- Ground-truth join key.
- Rollback trigger và người có quyền rollback.

### 13.4. Artifact

- Deployment Manifest.
- Rollout Plan.
- Experiment/Traffic Allocation Plan.
- Smoke Test Result.
- Rollback/Fallback Runbook.
- Release Approval.

### 13.5. Gate G6 — Rollout

**Pass khi:**

- Shadow/canary không vi phạm service và risk guardrail.
- Prediction distribution hợp lý.
- Logging và dashboard hoạt động.
- Ground-truth feedback path đã xác nhận.
- Owner phê duyệt mở rộng traffic.

---

## 14. Phase 9 — Monitoring và feedback loop

### 14.1. Mục tiêu

Phát hiện sớm lỗi dữ liệu, giảm chất lượng, sự cố dịch vụ và tác động kinh doanh ngoài dự kiến.

### 14.2. Năm lớp monitoring

| Lớp             | Tín hiệu                                                         | Ví dụ cảnh báo                               |
| --------------- | ---------------------------------------------------------------- | -------------------------------------------- |
| Data            | schema, volume, freshness, missing, range, category, drift       | feature null tăng, source chậm, category mới |
| Model           | prediction distribution, confidence, quality, calibration, slice | recall giảm, score collapse, fairness gap    |
| Service         | availability, latency, throughput, errors, resource, cost        | p99 vượt SLO, OOM, timeout                   |
| Business        | conversion, churn, review capacity, revenue/cost                 | KPI không cải thiện hoặc guardrail xấu       |
| Risk/operations | override, complaint, harmful error, fallback, access             | complaint spike, manual override tăng        |

### 14.3. Data/model monitoring

- Schema violations.
- Training–serving skew.
- Feature/prediction distribution drift.
- Missing và unseen categories.
- Model quality khi label đến.
- Metric theo cohort/time/slice.
- Calibration và threshold stability.
- Label prevalence và label delay.

Drift là tín hiệu điều tra, không tự động chứng minh concept drift hoặc bắt buộc retrain.

### 14.4. Khi ground truth đến trễ

Theo dõi đồng thời:

- Proxy metrics có quan hệ đã kiểm chứng.
- Prediction distribution.
- Data quality và drift.
- Human review sample.
- Leading business indicators.
- Quality metric đầy đủ sau khi label mature.

Dashboard phải phân biệt rõ metric proxy với metric ground-truth.

### 14.5. Alert design

Mỗi alert cần:

- SLI và threshold.
- Evaluation window.
- Severity.
- Owner/on-call.
- Runbook link.
- Suppression/deduplication policy.
- Hành động: observe, investigate, stop rollout, fallback hoặc rollback.

Không cảnh báo mọi biến động nhỏ; alert phải gắn với hành động.

### 14.6. Prediction log tối thiểu

- Prediction/correlation ID.
- Timestamp.
- Model, feature và config version.
- Input fingerprint hoặc fields được phép log.
- Prediction, confidence và threshold/action.
- Latency, status và fallback.
- Experiment group.
- Ground-truth join key.

Không log dữ liệu nhạy cảm nếu không cần; áp dụng masking, access control và retention.

### 14.7. Artifact

- Monitoring Specification.
- Dashboards và Alerts.
- SLI/SLO.
- Prediction/Feedback Schema.
- On-call Runbook.
- Periodic Model Health Report.

---

## 15. Phase 10 — Retraining và continuous training

### 15.1. Trigger

| Trigger         | Ví dụ                                | Lưu ý                                       |
| --------------- | ------------------------------------ | ------------------------------------------- |
| Theo lịch       | hàng tuần/tháng/quý                  | đơn giản nhưng có thể train không cần thiết |
| Đủ dữ liệu mới  | thêm N label hoàn chỉnh              | kiểm soát label maturity                    |
| Performance     | metric thấp hơn SLO                  | cần ground truth đáng tin cậy               |
| Drift           | feature/prediction shift vượt ngưỡng | phải điều tra nguyên nhân trước             |
| Business        | policy, sản phẩm hoặc population đổi | có thể cần problem reframing                |
| Manual/incident | bug fix, label correction            | ghi decision và backfill policy             |

### 15.2. Quy tắc

- Retraining không được bỏ qua data, evaluation và model gates.
- Dùng version mới cho data, label, code, config và model.
- So sánh challenger với champion trên cùng evaluation protocol.
- Kiểm tra slice, robustness, latency và cost.
- Không tự động promote chỉ vì metric trung bình cao hơn.
- Có rollback tới champion trước đó.
- Nếu label definition thay đổi, coi là một bài toán/version mới.
- Nếu production policy thay đổi, đánh giá lại threshold và business KPI.

### 15.3. Continuous training pipeline

1. Ingest snapshot.
2. Validate data/schema.
3. Transform/version features.
4. Train/tune.
5. Evaluate và so sánh champion.
6. Validate infrastructure.
7. Register candidate.
8. Approval theo risk tier.
9. Shadow/canary.
10. Promote hoặc rollback.

### 15.4. Artifact

- Retraining Trigger Record.
- New Dataset/Label Version.
- Challenger Evaluation.
- Promotion Decision.
- Updated Model Card/Release Notes.
- Lineage từ production model về toàn bộ inputs.

---

## 16. Phase 11 — Incident response và maintenance

### 16.1. Các loại sự cố

- Data pipeline/schema failure.
- Silent feature corruption.
- Prediction quality collapse.
- Serving outage hoặc latency spike.
- Model/feature version mismatch.
- Security/privacy exposure.
- Harmful or unfair outcomes.
- Feedback loop hoặc automated action runaway.

### 16.2. Quy trình xử lý

1. **Detect:** alert, user report hoặc audit.
2. **Triage:** xác định severity, blast radius và model/version liên quan.
3. **Contain:** dừng rollout, disable action, fallback hoặc rollback.
4. **Preserve evidence:** log, dataset snapshot, config và deployment state.
5. **Diagnose:** data, feature, model, service hay process.
6. **Correct:** fix, backfill, retrain hoặc policy change.
7. **Validate:** chạy lại các gate liên quan.
8. **Recover:** rollout có kiểm soát.
9. **Postmortem:** nguyên nhân gốc, impact, action items và owner.

### 16.3. Runbook phải trả lời

- Ai là incident commander?
- Khi nào rollback tự động/thủ công?
- Fallback nào an toàn?
- Cách xác định affected predictions/users?
- Cần thông báo ai?
- Cách sửa hoặc thu hồi quyết định downstream?
- Bằng chứng nào phải lưu?

### 16.4. Artifact

- Incident Record.
- Timeline và Impact Assessment.
- Root Cause Analysis.
- Corrective/Preventive Actions.
- Updated Tests, Monitors và Runbook.

---

## 17. Phase 12 — Retirement và decommission

### 17.1. Trigger

- Use case không còn giá trị.
- Model bị thay thế hoàn toàn.
- Dữ liệu/feature không còn hợp lệ.
- Chi phí/rủi ro vượt lợi ích.
- Policy, product hoặc regulation thay đổi.
- Không thể duy trì chất lượng/SLO.

### 17.2. Hoạt động

- Xác định dependency và downstream consumer.
- Chuyển traffic về replacement/fallback.
- Dừng training, scheduling và serving.
- Thu hồi credential, endpoint và access.
- Archive model card, lineage, approvals và incident history.
- Xóa/retain data, log và artifact theo policy.
- Cập nhật registry status thành retired.
- Xác nhận không còn silent consumer.
- Ghi lessons learned.

### 17.3. Artifact

- Retirement Decision.
- Dependency/Consumer Sign-off.
- Archive/Deletion Record.
- Final Health/Impact Report.

---

## 18. Quality gates tổng hợp

| Gate                    | Thời điểm            | Câu hỏi quyết định                                 | Bằng chứng tối thiểu                    |
| ----------------------- | -------------------- | -------------------------------------------------- | --------------------------------------- |
| G0 Opportunity          | Sau problem framing  | Có nên dùng ML không?                              | Problem brief, KPI, risk tier, baseline |
| G1 Data readiness       | Sau EDA              | Dữ liệu/label có đủ và hợp lệ không?               | Contract, lineage, quality report       |
| G2 Evaluation plan      | Trước modeling       | Cách đánh giá có đúng và được khóa chưa?           | Split, metric, test policy              |
| G3 Candidate readiness  | Sau experiment       | Có ứng viên tái lập và hơn baseline không?         | Runs, config, validation report         |
| G4 Model acceptance     | Sau final evaluation | Model đủ tốt và an toàn không?                     | Test report, model card, risk review    |
| G5 Production readiness | Sau packaging        | Artifact có chạy an toàn trong hạ tầng thật không? | Tests, benchmark, registry, rollback    |
| G6 Rollout              | Trong deployment     | Có thể tăng traffic không?                         | Shadow/canary metrics, approval         |

Gate phải có:

- Owner phê duyệt.
- Pass/fail criteria định lượng.
- Link tới artifact.
- Timestamp và version.
- Exception/waiver có lý do, thời hạn và risk owner.

---

## 19. Artifact checklist

| Artifact            | Nội dung bắt buộc                                              | Owner chính   |
| ------------------- | -------------------------------------------------------------- | ------------- |
| ML Problem Brief    | decision, unit/time, target, value, constraint, non-goals      | Product + DS  |
| Prediction Contract | entity, prediction time, horizon, feature availability, action | DS + Domain   |
| Data Contract       | schema, semantic, quality, freshness, privacy, owner           | Data          |
| Label Specification | outcome, window, delay, exclusions, audit                      | Domain + DS   |
| Evaluation Plan     | split, baseline, metrics, slices, thresholds, test policy      | DS            |
| Experiment Record   | code/data/config/environment/metrics/artifact                  | DS/MLE        |
| Model Card          | intended use, data, evaluation, limitation, risk               | DS + Risk     |
| Registered Model    | immutable artifact, version, lineage, status                   | MLE           |
| Release Plan        | deployment mode, rollout, rollback, fallback                   | MLE + SRE     |
| Monitoring Spec     | SLIs, thresholds, owners, labels, runbooks                     | MLE + SRE     |
| Incident Runbook    | triage, containment, rollback, communication                   | SRE           |
| Retirement Record   | dependency, archive, deletion và sign-off                      | Product + SRE |

---

## 20. Reproducibility và lineage

Mỗi production prediction phải truy ngược được ít nhất theo chuỗi:

~~~text
Prediction
→ deployment version
→ model registry version
→ model artifact + preprocessing
→ training run
→ code commit + config + environment
→ dataset/label/feature versions
→ source data lineage
~~~

### 20.1. Yêu cầu tối thiểu

- Version control cho code, config, schema và infrastructure.
- Immutable dataset snapshot hoặc query + source snapshot có thể tái tạo.
- Dependency lock/container digest.
- Seed và hardware/runtime metadata.
- Central experiment tracking.
- Model registry có stage: candidate, approved, production, archived, retired.
- Deployment manifest ghi model/feature/config version.
- Prediction log ghi model version.

Reproducible không luôn có nghĩa bit-for-bit giống hệt trên mọi hardware; tolerance phải được định nghĩa.

---

## 21. CI, CD và CT cho ML

| Pipeline          | Trigger                           | Nội dung                                              |
| ----------------- | --------------------------------- | ----------------------------------------------------- |
| CI                | code/config change                | lint, unit, data/contract, model smoke, security scan |
| Training pipeline | data/code/config change           | validate, transform, train, evaluate, register        |
| CD                | approved model + release config   | deploy shadow/canary, smoke, observe, promote         |
| CT                | schedule/data/performance trigger | retrain toàn bộ và đi lại các quality gate            |

### 21.1. Phân tách trách nhiệm

- Code pass CI không đồng nghĩa model được chấp nhận.
- Model pass offline evaluation không đồng nghĩa service sẵn sàng.
- Service ổn định không đồng nghĩa business impact tích cực.
- Retraining thành công không đồng nghĩa được tự động deploy.

---

## 22. Kiến trúc logic tham chiếu

~~~mermaid
flowchart TD
    A["Data sources"] --> B["Ingest and validate"]
    B --> C["Versioned datasets and features"]
    C --> D["Train, evaluate and register"]
    D --> E["Batch, online or edge serving"]
    E --> F["Prediction and service logs"]
    F --> G["Monitoring and alerts"]
    H["Delayed labels and feedback"] --> G
    H --> C
    G -->|Approved trigger| D
~~~

### 22.1. Thành phần

- Source systems và event/log producers.
- Ingestion/orchestration.
- Data/schema validation.
- Versioned offline datasets và optional feature store.
- Training/tuning/evaluation pipeline.
- Metadata/experiment tracking.
- Model registry.
- Serving/batch inference.
- Prediction/ground-truth logging.
- Monitoring/alerting.
- CI/CD/CT và approval workflow.

Không bắt buộc dùng mọi nền tảng ngay từ MVP; bắt buộc giữ các contract, gate và lineage cốt lõi.

---

## 23. Cấu trúc repository gợi ý

~~~text
ml-system/
├── README.md
├── pyproject.toml
├── configs/
├── contracts/
│   ├── data/
│   ├── labels/
│   └── api/
├── docs/
│   ├── problem-brief.md
│   ├── evaluation-plan.md
│   ├── model-card.md
│   └── architecture.md
├── notebooks/
│   └── exploration/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── serving/
├── pipelines/
├── tests/
│   ├── unit/
│   ├── data/
│   ├── integration/
│   └── performance/
├── infra/
├── monitoring/
├── runbooks/
└── reports/
~~~

Notebook dùng cho khám phá; logic cần tái lập phải chuyển sang module và pipeline có test.

---

## 24. Ví dụ rút gọn — Dự đoán churn

### 24.1. Problem contract

| Thành phần         | Định nghĩa                                           |
| ------------------ | ---------------------------------------------------- |
| Business objective | Giảm churn của khách hàng trả phí                    |
| Prediction unit    | Một tài khoản đang active                            |
| Prediction time    | 23:59 Chủ nhật hằng tuần                             |
| Target             | Không còn trả phí trong 30 ngày tiếp theo            |
| Feature cutoff     | Không muộn hơn prediction time                       |
| Action             | Gửi case top-risk cho nhóm retention                 |
| Capacity           | Tối đa 5% active accounts mỗi tuần                   |
| Inference          | Weekly batch                                         |
| Model metric       | Recall@top-5%, PR-AUC, calibration                   |
| Business metric    | Incremental retained users qua controlled experiment |
| Guardrail          | Complaint/unsubscribe rate, fairness gap             |

### 24.2. Data và split

- Feature window: 90 ngày trước prediction time.
- Label window: 30 ngày sau prediction time.
- Temporal split; validation và test nằm sau training period.
- Không để feature window nhìn vào label window.
- Một account ở nhiều snapshot được nhóm/kiểm soát đúng theo protocol.

### 24.3. Baseline và model

- Baseline 1: inactive 14 ngày.
- Baseline 2: logistic regression.
- Challenger: gradient boosting hoặc neural network nếu dữ liệu/quy mô phù hợp.
- Chọn threshold/top-K theo capacity của retention team.

### 24.4. Deployment và monitoring

- Batch job ghi account, score, model version và reason codes.
- Shadow một chu kỳ trước khi gửi action.
- Monitor data freshness, score distribution, Recall@5% khi label mature.
- So sánh business lift bằng controlled experiment.
- Rollback về rule/previous champion khi data job hoặc guardrail fail.

Ví dụ này cho thấy model metric và business impact không giống nhau: prediction chính xác chưa chắc chương trình retention tạo incremental lift.

---

## 25. Anti-patterns cần tránh

1. Bắt đầu bằng chọn thuật toán trước khi định nghĩa decision.
2. Dùng accuracy duy nhất cho dữ liệu mất cân bằng.
3. Random split dữ liệu thời gian hoặc record lặp theo user.
4. Preprocess/feature selection trước khi split.
5. Dùng test set để chọn model hoặc threshold.
6. Không có baseline không-ML.
7. Chỉ lưu model file, không lưu data/code/config lineage.
8. Training và serving dùng hai logic feature khác nhau.
9. Deploy 100% traffic ngay lập tức.
10. Không log model version theo prediction.
11. Chỉ monitor latency, không monitor data/model/business.
12. Retrain tự động khi drift nhưng không biết nguyên nhân.
13. Tự động promote model chỉ vì metric trung bình cao hơn.
14. Không có fallback, rollback hoặc on-call owner.
15. Diễn giải correlation/feature importance thành causality.
16. Thu thập/log dữ liệu nhạy cảm “để sau này có thể cần”.
17. Giữ model sống mãi dù use case không còn giá trị.

---

## 26. Definition of Done

Một hệ thống ML chỉ được xem là E2E hoàn thành khi:

### Business và risk

- [ ] Decision, action, intended use và non-goals rõ ràng.
- [ ] Có KPI owner và business acceptance criteria.
- [ ] Risk tier, privacy, security và fairness review phù hợp.
- [ ] Có human override/appeal khi use case yêu cầu.

### Data

- [ ] Data và label contract có version/owner.
- [ ] Dataset có lineage và tái tạo được.
- [ ] Point-in-time correctness được kiểm tra.
- [ ] Split phản ánh production và không entity/time leakage.
- [ ] Data validation tự động chặn lỗi nghiêm trọng.

### Model

- [ ] Có non-ML và simple-model baseline.
- [ ] Evaluation plan khóa trước tuning.
- [ ] Run tái lập được từ code/data/config/environment.
- [ ] Error, slice, robustness và calibration analysis hoàn tất khi phù hợp.
- [ ] Test set chỉ dùng theo policy.
- [ ] Model card và limitations đầy đủ.

### Production

- [ ] Model/preprocessing artifact bất biến và nằm trong registry.
- [ ] Input/output contract và dependency được version.
- [ ] Unit, data, parity, integration, performance và resilience tests đạt.
- [ ] Shadow/canary/rollout plan đã thực hiện.
- [ ] Fallback và rollback đã được kiểm thử.

### Operations

- [ ] Monitoring bao phủ data, model, service, business và risk.
- [ ] Ground-truth/feedback loop hoạt động.
- [ ] Alert có owner, severity và runbook.
- [ ] Retraining trigger và promotion policy rõ ràng.
- [ ] Incident và retirement procedure tồn tại.
- [ ] Có thể truy prediction về model, run, code và data.

---

## 27. Tài liệu tham chiếu

- [Google — Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [scikit-learn — Common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn — Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html)
- [TensorFlow — TFX User Guide](https://www.tensorflow.org/tfx/guide)
- [TensorFlow — TensorFlow Data Validation](https://www.tensorflow.org/tfx/guide/tfdv)
- [TensorFlow — TFX Evaluator](https://www.tensorflow.org/tfx/guide/evaluator)
- [TensorFlow — TFX InfraValidator](https://www.tensorflow.org/tfx/guide/infra_validator)
- [TensorFlow — TFX Pusher](https://www.tensorflow.org/tfx/guide/pusher)
- [NIST — AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [NIST — AI RMF Core](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/)

---

## 28. Pipeline tóm tắt

~~~text
Business decision
→ Prediction contract
→ Governed data and labels
→ Point-in-time correct dataset
→ Locked evaluation plan
→ Baseline and reproducible experiments
→ Model, risk and infrastructure validation
→ Registry and controlled rollout
→ Data/model/service/business monitoring
→ Retrain, rollback or retire
~~~

Quy trình hoàn chỉnh là một vòng lặp có kiểm soát:

~~~text
Observe
→ Detect change or opportunity
→ Reframe/validate data
→ Train and evaluate
→ Approve and release
→ Measure real-world impact
→ Improve, rollback or retire
~~~
