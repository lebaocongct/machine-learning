# Nền tảng Trí tuệ nhân tạo, Machine Learning và Deep Learning — 2026

> Handbook khái niệm, phương pháp, đánh giá và vận hành hệ thống AI/ML hiện đại

| Thuộc tính         | Giá trị                                                                                       |
| ------------------ | --------------------------------------------------------------------------------------------- |
| Phiên bản tài liệu | 2.0                                                                                           |
| Cập nhật           | 2026-07-14                                                                                    |
| Trạng thái         | Tài liệu tham chiếu đã chuẩn hóa                                                              |
| Đối tượng          | Lập trình viên đã quen Python, người mới học ML/DL và kỹ sư cần hệ thống hóa kiến thức        |
| Phạm vi            | AI, ML cổ điển, Deep Learning, Generative AI, LLM, RAG, agent, evaluation và MLOps            |
| Mục tiêu           | Hiểu đúng khái niệm, chọn đúng phương pháp, đánh giá đúng và tránh các lỗi thực hành phổ biến |

## Lịch sử thay đổi

| Phiên bản | Ngày       | Thay đổi                                                                                                                                 |
| --------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 1.x       | 2026-06-29 | Tập hợp kiến thức nền tảng AI/ML/DL                                                                                                      |
| 2.0       | 2026-07-14 | Chuẩn hóa cấu trúc, thuật ngữ và kỹ thuật; bổ sung toán, uncertainty, calibration, foundation model, RAG, agent, MLOps và Responsible AI |

---

## 1. Cách sử dụng tài liệu

### 1.1. Kết quả học tập

Sau khi học tài liệu này, người đọc nên có thể:

1. Phân biệt AI, ML, DL, Generative AI, foundation model, LLM, RAG và agent.
2. Hiểu vai trò của đại số tuyến tính, xác suất, đạo hàm và tối ưu hóa trong ML.
3. Nhận diện đúng feature, label, prediction time, train/validation/test và data leakage.
4. Chọn learning paradigm, loại bài toán, thuật toán và metric phù hợp.
5. Giải thích cơ chế huấn luyện neural network và các kiến trúc DL chính.
6. Hiểu khi nào dùng prompting, RAG, fine-tuning hoặc agent.
7. Nhìn model như một thành phần của hệ thống production lớn hơn.
8. Nhận diện các rủi ro về dữ liệu, fairness, privacy, security và vận hành.

### 1.2. Phạm vi và mức độ

Tài liệu tập trung vào:

- Mental model chính xác.
- Trực giác toán học và kỹ thuật.
- Trade-off thực tế.
- Quy tắc tránh lỗi.
- Bản đồ công cụ và lộ trình học.

Tài liệu không thay thế:

- Giáo trình toán chuyên sâu.
- API reference của từng framework.
- Tài liệu pháp lý theo quốc gia/ngành.
- Quy trình validation chuyên biệt cho y tế, tài chính hoặc hệ thống an toàn cao.

### 1.3. Bản đồ nội dung

| Nhóm               | Chương | Câu hỏi chính                                                 |
| ------------------ | -----: | ------------------------------------------------------------- |
| Bức tranh tổng thể |      2 | AI, ML, DL, GenAI và LLM liên hệ thế nào?                     |
| Nền tảng           |    3–5 | Toán, dữ liệu và learning paradigm gồm những gì?              |
| Xây dựng ML        |    6–9 | Workflow, feature, thuật toán và training hoạt động ra sao?   |
| Đánh giá           |     10 | Metric nào đúng và kết quả có đáng tin không?                 |
| Deep Learning      |  11–12 | Neural network và các kiến trúc ứng dụng thế nào?             |
| GenAI hiện đại     |     13 | Foundation model, RAG, fine-tuning và agent khác nhau ra sao? |
| Production và risk |  14–15 | Deploy, monitor và quản trị AI như thế nào?                   |
| Thực hành          |  16–19 | Dùng công cụ gì, tránh lỗi nào và học tiếp ra sao?            |

---

## 2. Bức tranh tổng thể AI, ML và DL

### 2.1. Trí tuệ nhân tạo

**Trí tuệ nhân tạo (Artificial Intelligence — AI)** là lĩnh vực xây dựng hệ thống có thể thực hiện các nhiệm vụ thường cần đến nhận thức, suy luận, học, lập kế hoạch, ra quyết định, giao tiếp hoặc sáng tạo.

AI là phạm vi rộng, có thể sử dụng:

- Luật và hệ chuyên gia.
- Search, planning và constraint solving.
- Tối ưu hóa.
- Xác suất và thống kê.
- Machine Learning.
- Deep Learning.
- Kết hợp nhiều kỹ thuật trong một hệ thống.

Ví dụ:

- Hệ thống lập lịch dùng constraint solver.
- Công cụ phát hiện gian lận dùng ML.
- Trợ lý hội thoại dùng LLM, retrieval, tools và policy.
- Robot dùng perception, planning, control và learning.

### 2.2. Machine Learning

**Machine Learning (ML)** là nhánh của AI nghiên cứu các thuật toán học pattern hoặc hàm từ dữ liệu/kinh nghiệm để thực hiện dự đoán, mô tả cấu trúc hoặc chọn hành động.

Thay vì chỉ viết luật:

~~~text
Nếu email chứa A, B, C thì đánh dấu spam
~~~

Ta cung cấp ví dụ:

~~~text
Email + nhãn spam/không spam
→ thuật toán học model
→ model dự đoán email mới
~~~

ML không loại bỏ vai trò của con người. Con người vẫn định nghĩa:

- Bài toán và quyết định cần hỗ trợ.
- Dữ liệu, label và cách chia dữ liệu.
- Metric và chi phí sai lầm.
- Constraint, risk và acceptance criteria.
- Cách model được sử dụng trong hệ thống.

### 2.3. Deep Learning

**Deep Learning (DL)** là nhóm phương pháp ML dùng neural network nhiều tầng để học các biểu diễn phân cấp.

DL đặc biệt hiệu quả khi:

- Dữ liệu phi cấu trúc: ảnh, video, âm thanh, văn bản.
- Có nhiều dữ liệu hoặc pretraining.
- Quan hệ đầu vào–đầu ra phức tạp.
- Cần representation learning hoặc end-to-end learning.

DL không tự động tốt hơn ML cổ điển. Với tabular data nhỏ/vừa, dữ liệu ít hoặc yêu cầu giải thích mạnh, linear model hoặc tree ensemble có thể phù hợp hơn.

### 2.4. Generative AI

**Generative AI (GenAI)** là nhóm hệ thống tạo nội dung hoặc cấu trúc mới như văn bản, mã nguồn, ảnh, âm thanh, video hoặc dữ liệu tổng hợp.

Generative AI mô tả **khả năng/loại ứng dụng**, không phải một tầng hoàn toàn tách biệt khỏi ML/DL. Phần lớn GenAI hiện đại dùng Deep Learning, nhưng khái niệm mô hình sinh đã tồn tại lâu trong thống kê và ML.

### 2.5. Foundation model

**Foundation model** là model được pretrain trên dữ liệu rộng, có thể thích nghi cho nhiều downstream task qua prompting, retrieval, fine-tuning hoặc adapters.

Ví dụ nhóm foundation model:

- Large Language Model.
- Vision foundation model.
- Speech/audio foundation model.
- Multimodal model.

Không phải mọi model lớn đều là foundation model, và không phải mọi foundation model đều là LLM.

### 2.6. Large Language Model

**Large Language Model (LLM)** là language model quy mô lớn, thường dựa trên Transformer, được pretrain để mô hình hóa token sequence và sau đó có thể được instruction-tune/alignment cho tương tác.

LLM có thể:

- Sinh và biến đổi văn bản.
- Tóm tắt, phân loại, trích xuất.
- Hỗ trợ viết code.
- Gọi tool khi được tích hợp vào hệ thống.
- Xử lý nhiều modality nếu kiến trúc hỗ trợ.

LLM không phải database, search engine hoặc nguồn sự thật. Output là kết quả sinh có điều kiện theo input/context và có thể sai.

### 2.7. RAG và agent

**Retrieval-Augmented Generation (RAG)** là kiến trúc kết hợp retrieval với model sinh. RAG không phải một loại LLM.

**AI agent** là mẫu hệ thống trong đó model tham gia vòng lặp quan sát → lập kế hoạch/chọn hành động → gọi tool → nhận kết quả → tiếp tục hoặc dừng. Agent không đồng nghĩa với một model tự trị hoàn toàn.

### 2.8. Quan hệ giữa các khái niệm

~~~mermaid
flowchart TD
    A["Artificial Intelligence"] --> B["Rules, search and planning"]
    A --> C["Machine Learning"]
    C --> D["Classical ML"]
    C --> E["Deep Learning"]
    E --> F["Foundation models"]
    F --> G["LLM, vision, audio and multimodal"]
    G --> H["Systems: RAG, tools and agents"]
~~~

Generative AI chủ yếu nằm trong ML/DL hiện đại nhưng được xem là nhóm năng lực/ứng dụng. RAG và agent nằm ở cấp **hệ thống**, vì chúng kết hợp model với dữ liệu, tools, state, policy và orchestration.

### 2.9. Algorithm, model và system

| Khái niệm                | Ý nghĩa                             | Ví dụ                                                    |
| ------------------------ | ----------------------------------- | -------------------------------------------------------- |
| Algorithm                | Quy trình học hoặc suy luận         | gradient descent, Random Forest training                 |
| Model class/architecture | Họ hàm hoặc cấu trúc                | Logistic Regression, Transformer                         |
| Trained model            | Parameters đã học từ một run cụ thể | model churn version 12                                   |
| Pipeline                 | Chuỗi xử lý dữ liệu–feature–model   | imputer → encoder → classifier                           |
| AI/ML system             | Toàn bộ sản phẩm vận hành           | API, model, retrieval, logging, monitoring, human review |

### 2.10. Prediction không phải decision

Model có thể tạo:

~~~text
P(churn trong 30 ngày) = 0,82
~~~

Hệ thống vẫn phải quyết định:

- Threshold là bao nhiêu?
- Có đủ capacity xử lý không?
- Chi phí can thiệp là gì?
- Trường hợp nào cần human review?
- Khi model không khả dụng thì fallback thế nào?

Prediction là input cho decision; không phải toàn bộ decision.

---

## 3. Nền tảng toán học cho Machine Learning

Không cần hoàn thành toàn bộ toán cao cấp trước khi học ML, nhưng cần hiểu các khối sau.

### 3.1. Đại số tuyến tính

| Khái niệm             | Vai trò trong ML                            |
| --------------------- | ------------------------------------------- |
| Scalar                | Một giá trị, ví dụ loss hoặc learning rate  |
| Vector                | Một sample, embedding hoặc parameter vector |
| Matrix                | Batch dữ liệu, weights của layer            |
| Tensor                | Mảng nhiều chiều dùng trong DL              |
| Dot product           | Linear model, similarity và attention       |
| Matrix multiplication | Forward pass của neural network             |
| Norm L1/L2            | Khoảng cách và regularization               |
| Eigenvector/SVD       | PCA, latent factors và compression          |

Ví dụ linear model:

$$
\hat{y} = Xw + b
$$

Trong đó:

- $X$: ma trận feature.
- $w$: vector trọng số.
- $b$: intercept/bias term.
- $\hat{y}$: prediction.

### 3.2. Xác suất

Khái niệm cần nắm:

- Random variable.
- Probability distribution.
- Joint, marginal và conditional probability.
- Independence và conditional independence.
- Expectation và variance.
- Bayes theorem.
- Likelihood và posterior.

Bayes:

$$
P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}
$$

Xác suất model dự đoán không tự động là xác suất được calibrated. Cần kiểm tra calibration bằng dữ liệu giữ lại.

### 3.3. Thống kê

- Population và sample.
- Mean, median, variance, quantile.
- Covariance và correlation.
- Sampling distribution.
- Estimator, bias và variance.
- Confidence interval.
- Hypothesis test và multiple comparisons.
- Bootstrap.

**Correlation không chứng minh causation.** Feature importance hoặc hệ số model cũng không tự động chứng minh quan hệ nhân quả.

### 3.4. Giải tích và automatic differentiation

Khái niệm cốt lõi:

- Derivative và partial derivative.
- Gradient.
- Chain rule.
- Jacobian/Hessian ở mức khái niệm.

Training thường cần gradient của loss theo parameter:

$$
\nabla_{\theta}L(\theta)
$$

Backpropagation áp dụng chain rule hiệu quả trên computational graph. TensorFlow, PyTorch và JAX dùng automatic differentiation để tính gradient.

### 3.5. Tối ưu hóa

Mục tiêu phổ biến:

$$
\theta^* = \arg\min_{\theta}
\frac{1}{n}\sum_{i=1}^{n}L(f_{\theta}(x_i), y_i)
+ \lambda R(\theta)
$$

Trong đó:

- $L$: loss trên dữ liệu.
- $R$: regularization.
- $\lambda$: mức regularization.

Gradient descent:

$$
\theta_{t+1} = \theta_t - \eta \nabla_{\theta}L(\theta_t)
$$

$\eta$ là learning rate.

Tối ưu hóa là nền tảng để train model nhưng không phải lúc nào cũng là “một loại bài toán ML”. Route planning hoặc scheduling có thể là optimization thuần túy, không cần model học từ dữ liệu.

### 3.6. Lý thuyết thông tin

| Khái niệm          | Ứng dụng                                  |
| ------------------ | ----------------------------------------- |
| Entropy            | Độ bất định của phân phối                 |
| Cross-entropy      | Loss cho classification/language modeling |
| KL divergence      | Độ khác biệt giữa hai phân phối           |
| Mutual information | Phụ thuộc giữa feature và target          |

---

## 4. Dữ liệu trong Machine Learning

### 4.1. Dataset, sample, feature và target

| Khái niệm                  | Ý nghĩa                                                |
| -------------------------- | ------------------------------------------------------ |
| Dataset                    | Tập quan sát dùng cho phân tích, train hoặc evaluation |
| Sample/example/observation | Một đơn vị dữ liệu                                     |
| Feature                    | Thông tin đầu vào model được phép sử dụng              |
| Label/target               | Giá trị hoặc outcome cần học/dự đoán                   |
| Prediction time            | Thời điểm hệ thống phải tạo prediction                 |
| Feature window             | Khoảng quá khứ dùng để tạo feature                     |
| Label window               | Khoảng tương lai dùng để xác định label                |

Ví dụ churn:

| Thành phần | Định nghĩa                                   |
| ---------- | -------------------------------------------- |
| Sample     | Một khách hàng tại cuối mỗi tuần             |
| Features   | Hành vi trong 90 ngày trước prediction time  |
| Target     | Có rời dịch vụ trong 30 ngày tiếp theo không |
| Prediction | Xác suất churn                               |

Feature phải thực sự tồn tại tại prediction time. Đây là nguyên tắc **point-in-time correctness**.

### 4.2. Structured, semi-structured và unstructured data

| Loại            | Ví dụ                                    | Biểu diễn phổ biến                |
| --------------- | ---------------------------------------- | --------------------------------- |
| Structured      | bảng SQL, giao dịch, CRM                 | rows/columns                      |
| Semi-structured | JSON, XML, event log                     | key–value, nested records         |
| Unstructured    | text, ảnh, audio, video                  | token, pixel, waveform, embedding |
| Graph           | social network, phân tử, knowledge graph | node, edge, attributes            |
| Time series     | sensor, giá, nhu cầu                     | timestamp + value/features        |

### 4.3. Data schema và data contract

Data schema mô tả type/cấu trúc. Data contract rộng hơn, nên gồm:

- Tên và semantic của field.
- Type, unit, range, nullability.
- Entity key và event timestamp.
- Freshness, volume và uniqueness.
- Allowed categories.
- Owner và version.
- Privacy, access và retention.
- Hành động khi contract bị vi phạm.

### 4.4. Train, validation và test

| Tập              | Mục đích                                          |
| ---------------- | ------------------------------------------------- |
| Training set     | Fit model và mọi preprocessing có learnable state |
| Validation set   | Chọn feature, model, hyperparameter và threshold  |
| Test/holdout set | Ước lượng cuối sau khi quyết định đã khóa         |

Tỷ lệ 70/15/15 hay 80/10/10 chỉ là tham khảo. Cách split quan trọng hơn tỷ lệ.

### 4.5. Chọn split theo cấu trúc dữ liệu

| Dữ liệu                     | Split phù hợp              | Leakage cần tránh             |
| --------------------------- | -------------------------- | ----------------------------- |
| IID và entity độc lập       | Random split               | duplicate                     |
| Classification mất cân bằng | Stratified split           | prevalence lệch mạnh          |
| Nhiều record cùng entity    | Group split                | cùng user/device ở nhiều tập  |
| Theo thời gian              | Temporal split             | tương lai rò vào quá khứ      |
| Theo không gian             | Spatial/group split        | vùng gần nhau ở cả train/test |
| Forecasting                 | Rolling/expanding backtest | random shuffle                |
| Recommendation              | User/item/time-aware split | cold-start bị che giấu        |

### 4.6. Cross-validation

Cross-validation ước lượng generalization qua nhiều fold.

| Phương pháp             | Khi dùng                                                        |
| ----------------------- | --------------------------------------------------------------- |
| KFold                   | Regression/IID tương đối                                        |
| StratifiedKFold         | Classification cần giữ tỷ lệ lớp                                |
| GroupKFold              | Các record trong cùng group phải đi cùng nhau                   |
| TimeSeriesSplit/rolling | Dữ liệu có thứ tự thời gian                                     |
| Nested CV               | Tách model selection khỏi performance estimation khi dữ liệu ít |

Preprocessing và feature selection phải nằm **bên trong** từng training fold.

### 4.7. Sampling

- Random sampling.
- Stratified sampling.
- Bootstrap sampling.
- Reservoir sampling cho stream.
- Importance/weighted sampling.
- Mini-batch sampling trong DL.

Sample phải đại diện cho population mà model sẽ phục vụ. Dataset lớn nhưng lệch vẫn có thể tạo model kém.

### 4.8. Dữ liệu mất cân bằng

Class imbalance không chỉ là tỷ lệ lớp; còn liên quan chi phí lỗi và capacity.

Cách xử lý:

- Chọn metric phù hợp: PR-AUC, precision/recall tại threshold.
- Class/sample weights.
- Under/over-sampling chỉ trong training fold.
- Thu thập thêm positive examples.
- Hard-negative mining.
- Threshold theo cost/capacity.
- Calibration khi cần probability.

SMOTE và các kỹ thuật tổng hợp không phải mặc định; cần kiểm tra chúng có tạo sample hợp lệ trong domain hay không.

### 4.9. Missing values

#### Cơ chế thiếu

| Cơ chế | Ý nghĩa                                                                                |
| ------ | -------------------------------------------------------------------------------------- |
| MCAR   | Missing không phụ thuộc dữ liệu quan sát hay giá trị bị thiếu                          |
| MAR    | Sau khi điều kiện hóa trên biến quan sát, missing không còn phụ thuộc giá trị bị thiếu |
| MNAR   | Missing vẫn liên quan đến chính giá trị không quan sát hoặc cơ chế chưa đo             |

Tên “Missing At Random” dễ gây hiểu nhầm; MAR không có nghĩa dữ liệu thiếu hoàn toàn ngẫu nhiên.

#### Phương pháp xử lý

| Phương pháp              | Khi dùng                            | Rủi ro                                       |
| ------------------------ | ----------------------------------- | -------------------------------------------- |
| Xóa row/column           | Missing ít hoặc field không hữu ích | mất thông tin, selection bias                |
| Mean/median              | Baseline cho feature số             | giảm variance, bỏ quan hệ giữa biến          |
| Most frequent            | Categorical đơn giản                | phóng đại nhóm phổ biến                      |
| Constant/Unknown         | Missing có semantic riêng           | cần phân biệt với giá trị thật               |
| Missing indicator        | Missing mang tín hiệu               | có thể học artifact của pipeline             |
| KNN/iterative            | Quan hệ giữa feature có ích         | chậm, cần tránh leakage                      |
| MICE/multiple imputation | Phân tích cần uncertainty           | phức tạp hơn, cần tổng hợp kết quả đúng cách |
| Native missing handling  | Một số tree booster                 | vẫn cần hiểu và monitor missing              |

Quy tắc:

1. Split trước.
2. Fit imputer trên training data/fold.
3. Transform validation/test bằng imputer đã fit.
4. Monitor missing rate trong production.

### 4.10. Outlier

Outlier có thể là:

- Lỗi dữ liệu.
- Quan sát hiếm nhưng hợp lệ.
- Tín hiệu chính của bài toán anomaly/fraud.
- Contextual hoặc collective anomaly.

Phát hiện:

- Visualization.
- Z-score khi giả định gần phù hợp.
- IQR/MAD cho robust univariate checks.
- Isolation Forest.
- Local Outlier Factor.
- Domain rule và temporal pattern.

Xử lý:

- Sửa từ source nếu là lỗi.
- Xóa khi chắc chắn invalid.
- Capping/Winsorization.
- Log/Box-Cox/Yeo-Johnson.
- Robust scaling/model/loss.
- Giữ nguyên hoặc tạo flag khi có ý nghĩa.

Không loại outlier chỉ vì một rule thống kê gắn cờ.

### 4.11. Dataset shift

| Loại                  | Thay đổi                              |
| --------------------- | ------------------------------------- |
| Covariate/data drift  | $P(X)$ thay đổi                       |
| Label shift           | $P(Y)$ thay đổi                       |
| Concept drift         | $P(Y \mid X)$ thay đổi                |
| Training–serving skew | Feature/data logic train khác serving |

Drift là tín hiệu điều tra; không tự động chứng minh model đã hỏng hoặc cần retrain.

---

## 5. Learning paradigms và loại bài toán

### 5.1. Supervised learning

Dữ liệu có input và label. Model học ánh xạ:

$$
f: X \rightarrow Y
$$

Ví dụ:

- Classification.
- Regression.
- Ranking có relevance labels.
- Forecasting có historical targets.

### 5.2. Unsupervised learning

Dữ liệu không có target trực tiếp; mục tiêu là khám phá cấu trúc hoặc biểu diễn.

- Clustering.
- Dimensionality reduction.
- Density estimation.
- Một số dạng anomaly detection.

Kết quả unsupervised cần domain validation; metric nội tại không tự chứng minh giá trị.

### 5.3. Semi-supervised learning

Kết hợp ít dữ liệu có nhãn với nhiều dữ liệu không nhãn.

Kỹ thuật:

- Pseudo-labeling.
- Consistency regularization.
- Graph-based label propagation.

### 5.4. Self-supervised learning

Tín hiệu học được tạo từ chính cấu trúc dữ liệu:

- Next-token prediction.
- Masked prediction.
- Contrastive learning.
- Reconstruction.

Self-supervised pretraining là nền tảng của nhiều foundation model.

### 5.5. Reinforcement learning

Agent tương tác với environment để tối đa hóa cumulative reward.

Các khái niệm:

- State/observation.
- Action.
- Reward.
- Policy.
- Value function.
- Exploration–exploitation.

RL phù hợp với decision tuần tự nhưng khó vì reward design, sample efficiency, safety và off-policy evaluation.

### 5.6. Các chiến lược học khác

| Chiến lược                  | Ý nghĩa                                                             |
| --------------------------- | ------------------------------------------------------------------- |
| Offline/batch learning      | Train trên snapshot rồi deploy                                      |
| Online/incremental learning | Cập nhật liên tục hoặc theo mini-batch mới                          |
| Transfer learning           | Tái sử dụng representation/weights từ task khác                     |
| Multi-task learning         | Học nhiều target/task chung representation                          |
| Active learning             | Chọn sample có giá trị để yêu cầu label                             |
| Federated learning          | Train qua nhiều thiết bị/silo mà không tập trung raw data hoàn toàn |
| Continual learning          | Học qua chuỗi task/data mà hạn chế catastrophic forgetting          |

### 5.7. Loại bài toán và metric khởi đầu

| Bài toán                 | Output                         | Baseline                 | Metric khởi đầu                              |
| ------------------------ | ------------------------------ | ------------------------ | -------------------------------------------- |
| Binary classification    | label/probability              | majority, rule, logistic | PR-AUC, ROC-AUC, F1, log loss                |
| Multiclass               | một trong nhiều class          | majority                 | macro-F1, log loss, per-class recall         |
| Multilabel               | nhiều label đồng thời          | frequency                | micro/macro-F1, mAP                          |
| Regression               | số liên tục                    | mean/median              | MAE, RMSE, R²                                |
| Quantile regression      | quantile                       | historical quantile      | pinball loss, coverage                       |
| Ranking/search           | danh sách có thứ tự            | popularity/BM25          | NDCG@K, MRR, Recall@K                        |
| Recommendation           | items                          | popularity               | Recall@K, NDCG@K, coverage                   |
| Forecasting              | giá trị tương lai              | last/seasonal naive      | MAE, RMSE, MASE/WAPE                         |
| Clustering               | cluster assignment             | domain segmentation      | stability, silhouette + domain utility       |
| Anomaly detection        | score/flag                     | rule                     | precision@budget, recall, time-to-detect     |
| Association-rule mining  | frequent itemsets/rules        | item frequency           | support, confidence, lift                    |
| Dimensionality reduction | low-dimensional representation | feature subset           | reconstruction/variance + downstream utility |
| Generative modeling      | content/sample                 | template/retrieval       | task-specific human/model eval + safety      |

---

## 6. Quy trình Machine Learning từ bài toán đến production

### 6.1. Luồng chuẩn

~~~mermaid
flowchart TD
    A["Frame decision and metric"] --> B["Collect and validate data"]
    B --> C["Split and build baseline"]
    C --> D["Train, tune and analyze errors"]
    D --> E["Validate and release"]
    E --> F["Monitor outcomes"]
    F -->|Drift, labels or new goal| B
    F -->|No longer useful| G["Retire"]
~~~

### 6.2. Problem framing

Trước khi chọn thuật toán, trả lời:

- Business objective là gì?
- Prediction hỗ trợ decision/action nào?
- Prediction unit và prediction time là gì?
- Target, feature window và label window là gì?
- False positive/negative gây chi phí gì?
- Baseline hiện tại là gì?
- Metric model, service và business là gì?
- Constraint latency, cost, privacy và fairness là gì?

Kết luận hợp lệ có thể là “không cần ML”.

### 6.3. Baseline first

Thứ tự baseline:

1. Quy trình hiện tại.
2. Constant/majority/mean/last value.
3. Rule/heuristic.
4. Model đơn giản.
5. Champion production nếu đã có.

Baseline giúp biết model phức tạp có tạo giá trị thật hay không.

### 6.4. Experiment loop

~~~text
Hypothesis
→ controlled change
→ train/evaluate
→ error analysis
→ conclusion
→ next hypothesis
~~~

Một experiment cần lưu:

- Code commit.
- Data/label version.
- Config/hyperparameters.
- Seed và environment.
- Metric và slice results.
- Model artifact.
- Kết luận.

### 6.5. Data leakage

Data leakage xảy ra khi model được dùng thông tin không có trong điều kiện dự đoán thật hoặc evaluation bị nhiễm thông tin từ tập giữ lại.

| Loại                      | Ví dụ                             |
| ------------------------- | --------------------------------- |
| Target leakage            | feature chỉ xuất hiện sau outcome |
| Temporal leakage          | dùng tương lai dự đoán quá khứ    |
| Train–test contamination  | test tham gia train/preprocess    |
| Preprocessing leakage     | fit scaler/imputer trên toàn data |
| Feature-selection leakage | chọn feature bằng test            |
| Group leakage             | cùng user/patient ở train và test |
| Duplicate leakage         | near-duplicate ở nhiều split      |
| Human/process leakage     | analyst vô thức tune theo test    |

Phòng tránh:

- Xác định prediction time.
- Split trước preprocessing.
- Dùng pipeline.
- Group/time-aware split.
- Khóa test policy.
- Audit feature lineage.

### 6.6. Deployment không phải bước cuối

Sau deploy cần:

- Input/output contract.
- Model/feature version.
- Shadow/canary hoặc controlled rollout.
- Logging và ground-truth join.
- Monitoring.
- Fallback/rollback.
- Retraining, incident response và retirement.

---

## 7. Preprocessing và feature engineering

### 7.1. Pipeline

Một pipeline điển hình:

~~~text
Raw input
→ schema validation
→ missing handling
→ encoding/scaling
→ feature construction
→ model
→ calibration/threshold
→ decision output
~~~

Pipeline giúp đảm bảo cùng preprocessing giữa train, validation, test và serving.

### 7.2. Scaling

| Kỹ thuật        | Công thức/ý tưởng    | Khi dùng                        |
| --------------- | -------------------- | ------------------------------- |
| Standardization | $(x-\mu)/\sigma$     | linear model, SVM, KNN, NN, PCA |
| Min–Max         | đưa về range cố định | input cần bounded range         |
| Robust scaling  | median và IQR        | có outlier                      |
| Unit norm       | vector có norm 1     | cosine similarity, text vectors |

Tree-based model thường ít nhạy với monotonic scaling hơn distance/gradient-based model.

### 7.3. Transformation

- Log/log1p cho dữ liệu lệch phải không âm.
- Box-Cox cho positive values.
- Yeo-Johnson cho dữ liệu có zero/negative.
- Binning khi cần quan hệ piecewise hoặc interpretability.
- Cyclical encoding cho hour/day/month khi phù hợp.

Transformation phải fit trên training data và được version.

### 7.4. Categorical encoding

| Kỹ thuật        | Khi dùng                        | Lưu ý                                |
| --------------- | ------------------------------- | ------------------------------------ |
| One-hot         | cardinality thấp                | dimension tăng                       |
| Ordinal         | category có thứ tự thật         | không áp đặt thứ tự giả              |
| Frequency/count | cardinality cao                 | có thể drift                         |
| Target encoding | cardinality cao                 | bắt buộc cross-fitting chống leakage |
| Hashing         | vocabulary lớn/streaming        | collision                            |
| Embedding       | DL/recommender/high-cardinality | cần đủ dữ liệu                       |

### 7.5. Feature creation

Ví dụ:

- Date → day_of_week, month, is_holiday.
- Debt/income → debt_to_income.
- Latitude/longitude → distance.
- Event log → count/recency/frequency.
- Text → TF-IDF/embedding.
- Image/audio → pretrained embedding.

Mọi aggregate feature phải point-in-time correct.

### 7.6. Feature interactions

Feature cross giúp model đơn giản học tương tác:

~~~text
area_per_room = area / number_of_rooms
city × product_category
~~~

Tạo quá nhiều interactions làm tăng dimension và overfitting.

### 7.7. Feature selection

| Nhóm     | Ví dụ                                    | Lưu ý                                         |
| -------- | ---------------------------------------- | --------------------------------------------- |
| Filter   | variance, mutual information, chi-square | độc lập với model hoặc dùng thống kê đơn giản |
| Wrapper  | forward/backward selection, RFE          | tốn compute                                   |
| Embedded | L1, tree-based selection                 | phụ thuộc model                               |

Feature importance không đồng nghĩa causal importance. Impurity-based tree importance có thể thiên lệch; permutation importance cũng cần cẩn thận khi feature tương quan.

### 7.8. Dimensionality reduction

| Phương pháp | Loại              | Dùng cho                                                   |
| ----------- | ----------------- | ---------------------------------------------------------- |
| PCA/SVD     | tuyến tính        | compression, decorrelation, latent factors                 |
| t-SNE       | phi tuyến         | visualization; không nên mặc định cho production transform |
| UMAP        | phi tuyến         | visualization/representation, cần validate                 |
| Autoencoder | learned nonlinear | compression/representation/anomaly                         |

PCA thường cần scaling nếu đơn vị feature khác nhau.

### 7.9. Class imbalance

- Không đánh giá chỉ bằng accuracy.
- Chọn threshold theo cost.
- Sampling nằm trong training fold.
- So sánh với class weights.
- Đánh giá per-class và calibration.
- Kiểm tra shift của prevalence.

---

## 8. Các thuật toán Machine Learning cổ điển

### 8.1. Linear Regression

$$
\hat{y} = w^Tx + b
$$

Điểm mạnh:

- Nhanh, baseline tốt.
- Hệ số có thể giải thích khi giả định và feature phù hợp.
- Có nghiệm/tối ưu ổn định.

Hạn chế:

- Quan hệ tuyến tính nếu không feature engineering.
- Nhạy outlier với squared loss.
- Hệ số khó diễn giải khi multicollinearity hoặc preprocessing phức tạp.

### 8.2. Ridge, Lasso và Elastic Net

| Model       | Regularization | Tác động                                   |
| ----------- | -------------- | ------------------------------------------ |
| Ridge       | L2             | shrink weights, xử lý collinearity tốt hơn |
| Lasso       | L1             | có thể đưa hệ số về 0                      |
| Elastic Net | L1 + L2        | kết hợp sparsity và stability              |

### 8.3. Logistic Regression

Logistic Regression là linear classifier theo log-odds, thường tạo probability qua sigmoid.

Điểm mạnh:

- Baseline classification mạnh.
- Nhanh, hỗ trợ regularization.
- Dễ phân tích coefficient và calibration hơn nhiều model phức tạp.

Tên có “Regression” nhưng mục đích chính là classification.

### 8.4. Decision Tree

Tree chia feature space bằng các rule để giảm impurity/loss.

Điểm mạnh:

- Phi tuyến và interactions.
- Ít cần scaling.
- Dễ visualize khi cây nhỏ.

Hạn chế:

- Variance cao.
- Dễ overfit.
- Split không ổn định khi dữ liệu thay đổi nhỏ.

### 8.5. Random Forest

Random Forest kết hợp bagging và random feature subsets.

Điểm mạnh:

- Giảm variance so với một tree.
- Baseline tốt cho tabular data.
- Xử lý phi tuyến/interactions.

Hạn chế:

- Model lớn, inference có thể tốn hơn.
- Probability có thể cần calibration.
- Feature importance cần diễn giải cẩn thận.

### 8.6. Gradient Boosting

Boosting xây learner tuần tự để sửa residual/error.

Các implementation phổ biến:

- GradientBoosting/HistGradientBoosting.
- XGBoost.
- LightGBM.
- CatBoost.

Chúng thường mạnh cho tabular data nhưng cần kiểm soát:

- Leakage.
- Depth/leaves/learning rate.
- Early stopping.
- Categorical handling.
- Calibration và latency.

### 8.7. Support Vector Machine

SVM tối đa hóa margin; kernel cho phép ranh giới phi tuyến.

Phù hợp:

- Dataset nhỏ/vừa.
- Feature nhiều chiều.
- Sparse text với linear SVM.

Hạn chế:

- Cần scaling.
- Kernel SVM khó scale với rất nhiều samples.
- Probability không có tự nhiên, thường cần calibration.

### 8.8. K-Nearest Neighbors

KNN dự đoán dựa trên neighbors gần nhất.

- Train nhẹ.
- Inference và memory tốn khi dataset lớn.
- Nhạy scaling, irrelevant features và curse of dimensionality.

### 8.9. Naive Bayes

Dựa trên Bayes và conditional independence assumption.

Phù hợp:

- Sparse text/count features.
- Baseline nhanh.
- Dữ liệu nhỏ.

Giả định “naive” thường không đúng hoàn toàn nhưng model vẫn có thể hữu ích.

### 8.10. Clustering

| Thuật toán   | Điểm mạnh                         | Hạn chế                                  |
| ------------ | --------------------------------- | ---------------------------------------- |
| K-Means      | đơn giản, nhanh                   | cần K, nhạy scale/outlier, cụm gần cầu   |
| DBSCAN       | tìm noise, cụm hình dạng phức tạp | khó khi density khác nhau/high dimension |
| Hierarchical | dendrogram, nhiều mức             | tốn tài nguyên                           |
| GMM          | soft assignment, probabilistic    | giả định mixture form, local optimum     |

Cluster phải được kiểm tra stability và domain usefulness.

### 8.11. Anomaly detection

- Rule/statistical threshold.
- Isolation Forest.
- Local Outlier Factor.
- One-Class SVM.
- Autoencoder.
- Supervised classifier nếu có label đủ tốt.

Threshold phải gắn với review capacity và chi phí lỗi.

### 8.12. Chọn model

| Điều kiện                      | Model nên thử sớm              |
| ------------------------------ | ------------------------------ |
| Cần baseline dễ hiểu           | linear/logistic                |
| Tabular nhỏ/vừa, phi tuyến     | tree ensemble/boosting         |
| Sparse text                    | linear model/Naive Bayes       |
| Ảnh, audio, text raw           | pretrained DL                  |
| Dữ liệu rất ít                 | simple model + domain features |
| Latency/edge gắt               | model nhỏ, quantized/distilled |
| Probability dùng ra quyết định | model + calibration check      |

Không chọn model chỉ vì mới hoặc có benchmark cao ở dataset khác.

---

## 9. Training, tối ưu và generalization

### 9.1. Parameter và hyperparameter

| Loại           | Được xác định bởi           | Ví dụ                            |
| -------------- | --------------------------- | -------------------------------- |
| Parameter      | học từ training data        | weights, bias, tree splits       |
| Hyperparameter | người/search algorithm chọn | learning rate, depth, batch size |

### 9.2. Loss và metric

**Loss** là objective dùng để train.  
**Metric** là cách đo chất lượng theo mục tiêu.

| Bài toán              | Loss thường dùng                 | Metric                        |
| --------------------- | -------------------------------- | ----------------------------- |
| Regression            | MSE, MAE, Huber, quantile        | MAE, RMSE, R²                 |
| Binary classification | binary cross-entropy, focal loss | PR-AUC, ROC-AUC, F1, log loss |
| Multiclass            | categorical cross-entropy        | macro-F1, log loss, accuracy  |
| Ranking               | pairwise/listwise loss           | NDCG, MRR, Recall@K           |
| Language model        | token cross-entropy              | perplexity + task/human eval  |

Loss và metric không cần giống nhau nhưng phải liên hệ hợp lý.

### 9.3. Epoch, batch và step

- **Epoch:** một lượt qua training dataset.
- **Batch:** nhóm sample dùng trong một forward/backward pass.
- **Step/iteration:** một lần update parameter.

Nếu dataset có $N$ mẫu và batch size $B$, số step mỗi epoch xấp xỉ:

$$
\lceil N/B \rceil
$$

### 9.4. Learning rate

Learning rate quá lớn:

- Loss dao động/diverge.
- Gradient update vượt vùng tốt.

Quá nhỏ:

- Train chậm.
- Có thể dừng trước khi đạt vùng tốt.

Kỹ thuật:

- Warmup.
- Step/exponential/cosine schedule.
- Reduce on plateau.
- Adaptive optimizers.

### 9.5. Optimizer

| Optimizer         | Đặc điểm                                             |
| ----------------- | ---------------------------------------------------- |
| SGD               | đơn giản, cần tuning learning rate                   |
| Momentum/Nesterov | tăng tốc theo hướng ổn định                          |
| RMSprop           | adaptive scale theo gradient history                 |
| Adam              | adaptive, baseline tốt cho nhiều DL task             |
| AdamW             | tách weight decay tốt hơn cách L2 cổ điển trong Adam |

Không có optimizer tốt nhất cho mọi bài toán.

### 9.6. Overfitting và underfitting

| Hiện tượng   | Train | Validation                 | Hướng xử lý                                               |
| ------------ | ----- | -------------------------- | --------------------------------------------------------- |
| Underfitting | kém   | kém                        | model/feature tốt hơn, train lâu hơn, giảm regularization |
| Overfitting  | tốt   | kém hơn rõ                 | thêm dữ liệu, regularization, augmentation, model nhỏ hơn |
| Good fit     | tốt   | gần train trong mức hợp lý | tiếp tục error/slice analysis                             |

### 9.7. Bias–variance

- **Bias cao:** giả định quá đơn giản, underfitting.
- **Variance cao:** nhạy với training sample, overfitting.

Bias trong bias–variance khác với bias term/intercept và khác social bias.

### 9.8. Regularization

- L1/L2/weight decay.
- Early stopping.
- Dropout.
- Data augmentation.
- Label smoothing khi phù hợp.
- Batch/layer normalization có thể có hiệu ứng regularization nhưng mục đích chính khác.
- Model capacity control.
- Ensembling.

### 9.9. Initialization, normalization và gradient

Trong DL cần theo dõi:

- Weight initialization.
- Activation saturation/dead units.
- Vanishing/exploding gradients.
- Gradient clipping.
- Batch Normalization/Layer Normalization.
- Mixed precision và numerical stability.

### 9.10. Threshold và calibration

Classifier thường tạo score/probability; decision cần threshold.

Threshold thay đổi:

- Precision.
- Recall.
- Workload/capacity.
- Business cost.

Calibration tốt nghĩa là trong các prediction khoảng 0,8, tỷ lệ positive thực tế xấp xỉ 80% trong population/evaluation phù hợp.

Công cụ:

- Reliability/calibration curve.
- Brier score.
- Log loss.
- Platt scaling.
- Isotonic regression.
- Temperature scaling.

Calibration phải fit trên dữ liệu tách khỏi training model và đánh giá trên dữ liệu chưa dùng.

### 9.11. Reproducibility

Một run cần gắn:

- Code commit.
- Data/label version.
- Config.
- Seed.
- Environment/container.
- Hardware.
- Metric/artifact.

Deterministic hoàn toàn không phải lúc nào khả thi trên accelerator; cần định nghĩa tolerance.

---

## 10. Đánh giá mô hình

### 10.1. Nguyên tắc

1. Metric phản ánh chi phí/giá trị của decision.
2. Luôn có baseline.
3. Evaluation data phải đại diện cho production.
4. Báo cáo uncertainty, không chỉ một con số.
5. Đánh giá theo slice, không chỉ trung bình.
6. Tách model selection khỏi final test.
7. Offline metric không tự chứng minh business impact.

### 10.2. Confusion matrix

|                 | Predicted positive | Predicted negative |
| --------------- | -----------------: | -----------------: |
| Actual positive |      True Positive |     False Negative |
| Actual negative |     False Positive |      True Negative |

$$
Precision = \frac{TP}{TP+FP}
$$

$$
Recall = \frac{TP}{TP+FN}
$$

$$
F1 = 2\frac{Precision \cdot Recall}{Precision+Recall}
$$

### 10.3. Classification metrics

| Metric            | Dùng khi                              | Hạn chế                               |
| ----------------- | ------------------------------------- | ------------------------------------- |
| Accuracy          | lớp tương đối cân bằng, cost tương tự | đánh lừa khi imbalance                |
| Balanced accuracy | cần cân bằng recall các lớp           | không phản ánh probability            |
| Precision         | false positive đắt                    | phụ thuộc threshold/prevalence        |
| Recall            | false negative đắt                    | phụ thuộc threshold                   |
| F1                | cần cân bằng precision/recall         | bỏ qua true negative/cost cụ thể      |
| ROC-AUC           | ranking toàn dải threshold            | có thể lạc quan khi positive rất hiếm |
| PR-AUC            | positive hiếm                         | baseline phụ thuộc prevalence         |
| Log loss          | chất lượng probabilistic prediction   | phạt mạnh prediction tự tin nhưng sai |
| Brier score       | xác suất và calibration               | cần diễn giải theo prevalence         |

Log loss không chỉ dành cho probability đã calibrated; nó là proper scoring rule đánh giá probabilistic prediction.

### 10.4. Multiclass và multilabel

- Micro average: gộp toàn bộ decisions.
- Macro average: trung bình đều theo class.
- Weighted average: theo support.
- Top-K accuracy: khi nhiều class gần nhau.
- Per-class recall/precision: bắt buộc khi class quan trọng khác nhau.

### 10.5. Regression metrics

| Metric                | Tính chất                                 |
| --------------------- | ----------------------------------------- |
| MAE                   | dễ hiểu, ít nhạy outlier hơn squared loss |
| MSE                   | phạt lỗi lớn mạnh                         |
| RMSE                  | cùng đơn vị target                        |
| R²                    | so với baseline mean; có thể âm           |
| MAPE                  | không phù hợp khi actual bằng/gần 0       |
| Quantile/pinball loss | dự đoán quantile và cost bất đối xứng     |

Phải xem residual theo target range, time và subgroup.

### 10.6. Ranking và recommendation

- Precision@K.
- Recall@K.
- MAP.
- MRR.
- NDCG@K.
- Hit Rate.
- Coverage.
- Diversity/novelty.

Offline evaluation có selection/exposure bias; business impact thường cần online experiment an toàn.

### 10.7. Forecasting

- Time-based backtesting.
- Multiple forecast origins.
- Metric theo horizon.
- Seasonal-naive baseline.
- MAE, RMSE, MASE, WAPE hoặc quantile loss.
- Prediction interval coverage.

Tránh random split.

### 10.8. Clustering

- Silhouette score.
- Davies–Bouldin.
- Adjusted Rand/NMI nếu có reference labels.
- Stability qua seed/sample/time.
- Domain coherence và actionability.

Không tối ưu silhouette rồi mặc định cluster có ý nghĩa nghiệp vụ.

### 10.9. Uncertainty và confidence interval

Cách ước lượng:

- Cross-validation variability.
- Bootstrap.
- Repeated runs/seeds.
- Prediction interval.
- Conformal prediction khi giả định phù hợp.
- Bayesian/posterior methods.

Không gọi softmax score là “độ tin cậy tuyệt đối” nếu chưa kiểm tra calibration và distribution shift.

### 10.10. Slice evaluation

Đánh giá theo:

- Time period.
- Region/source/device.
- New versus returning users.
- Head versus long-tail.
- Missing/unseen categories.
- Protected/relevant demographic groups khi hợp pháp và cần thiết.

Model trung bình tốt có thể thất bại nghiêm trọng ở một slice nhỏ.

### 10.11. Error analysis

Quy trình:

1. Lấy mẫu lỗi có hệ thống.
2. Nhóm lỗi theo pattern/root cause.
3. Phân biệt lỗi data, label, feature, model, threshold hoặc workflow.
4. Ước lượng impact và frequency.
5. Chọn can thiệp có lợi nhất.
6. Chạy experiment mới.

### 10.12. Business evaluation

Model metric đo prediction quality; business metric đo tác động của action.

Có thể cần:

- A/B test.
- Randomized controlled rollout.
- Quasi-experiment.
- Incrementality/uplift modeling.

Không diễn giải association thành causal lift.

---

## 11. Deep Learning nền tảng

### 11.1. Tensor

Tensor là mảng nhiều chiều.

| Dữ liệu           | Shape ví dụ                       |
| ----------------- | --------------------------------- |
| Tabular batch     | batch × features                  |
| Ảnh               | batch × height × width × channels |
| Text tokens       | batch × sequence length           |
| Audio spectrogram | batch × time × frequency          |

Shape, dtype và device là nguồn lỗi phổ biến.

### 11.2. Neuron và layer

Một neuron:

$$
z = w^Tx + b
$$

$$
a = \phi(z)
$$

$\phi$ là activation function.

Neural network xếp nhiều layer để học composition của các phép biến đổi.

### 11.3. Activation functions

| Activation      | Dùng phổ biến                        |
| --------------- | ------------------------------------ |
| ReLU/Leaky ReLU | hidden layer                         |
| GELU/SiLU       | Transformer và model hiện đại        |
| Sigmoid         | binary/multilabel output             |
| Softmax         | mutually exclusive multiclass output |
| Linear          | regression output                    |
| Tanh            | một số RNN/legacy architecture       |

Output activation phải khớp label representation và loss.

### 11.4. Forward pass, loss và backpropagation

~~~text
Input
→ forward pass
→ prediction
→ loss
→ automatic differentiation/backpropagation
→ optimizer update
~~~

Backpropagation là cách áp dụng chain rule hiệu quả, không phải optimizer.

### 11.5. Kiến trúc chính

| Kiến trúc    | Điểm mạnh                              | Ứng dụng                          |
| ------------ | -------------------------------------- | --------------------------------- |
| MLP          | general nonlinear function             | tabular/feature vectors, heads    |
| CNN          | local spatial patterns, weight sharing | image, video, signal              |
| RNN/LSTM/GRU | sequential state                       | sequence/time series nhỏ/vừa      |
| Transformer  | attention, parallel sequence modeling  | text, vision, audio, multimodal   |
| GNN          | message passing trên graph             | molecule, network, recommendation |
| Autoencoder  | learned compression/reconstruction     | representation, anomaly           |
| GAN          | adversarial generation                 | image/data generation             |
| Diffusion    | iterative denoising generation         | image, video, audio               |

### 11.6. Embedding

Embedding ánh xạ đối tượng rời rạc/sparse vào vector dense.

Ứng dụng:

- Token/word.
- User/item.
- Image/audio/text representation.
- Semantic search.
- Retrieval và recommendation.

Khoảng cách trong embedding space chỉ có ý nghĩa theo cách model được train và metric được dùng.

### 11.7. Attention và Transformer

Scaled dot-product attention:

$$
Attention(Q,K,V) =
softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Các thành phần:

- Query, key, value.
- Multi-head attention.
- Positional information.
- Residual connection.
- Layer normalization.
- Feed-forward block.
- Padding/causal mask.

Self-attention chuẩn có chi phí bậc hai theo sequence length về attention matrix, dù implementation và biến thể có thể tối ưu.

### 11.8. CNN

Khái niệm:

- Filter/kernel.
- Stride.
- Padding.
- Feature map.
- Pooling.
- Receptive field.

Transfer learning từ pretrained backbone thường hiệu quả hơn train từ đầu khi dữ liệu ảnh hạn chế.

### 11.9. Sequence models

RNN/LSTM/GRU:

- Có state theo thời gian.
- Hữu ích với sequence/time series.
- Khó parallel hóa hơn Transformer.
- Có thể gặp vanishing/exploding gradients.

Không phải mọi time-series task đều cần deep learning; naive, statistical hoặc boosting baseline thường rất quan trọng.

### 11.10. Transfer learning và fine-tuning

Quy trình:

1. Chọn pretrained model phù hợp domain/license.
2. Thay task head.
3. Freeze backbone để train head.
4. Unfreeze một phần/toàn bộ.
5. Fine-tune với learning rate nhỏ.
6. Đánh giá catastrophic forgetting, domain shift và latency.

### 11.11. Data augmentation

- Image: crop, flip, color, mixup/cutmix khi phù hợp.
- Audio: noise, time/frequency masking.
- Text: cần cẩn thận để không đổi label/semantic.

Augmentation chỉ áp dụng training, không áp dụng validation/test theo cách làm thay đổi ground truth.

### 11.12. Hiệu năng huấn luyện

- GPU/TPU/accelerator.
- Vectorization và efficient input pipeline.
- Mixed precision.
- Gradient accumulation.
- Data/model parallelism.
- Checkpointing.
- Profiling.

Scale compute chỉ hữu ích khi pipeline, objective và data quality đúng.

---

## 12. Các miền ứng dụng Deep Learning

### 12.1. Computer Vision

| Task                  | Output                 | Metric                    |
| --------------------- | ---------------------- | ------------------------- |
| Image classification  | class/probability      | macro-F1, top-K, log loss |
| Object detection      | class + bounding boxes | mAP, precision/recall     |
| Semantic segmentation | class mỗi pixel        | IoU, Dice                 |
| Instance segmentation | mask mỗi object        | mask mAP                  |
| Image retrieval       | ranked images          | Recall@K, mAP             |
| OCR                   | text sequence          | character/word error rate |

Face recognition và biometric systems cần privacy, fairness, security và legal review nghiêm ngặt.

### 12.2. Natural Language Processing

- Text classification.
- Named Entity Recognition.
- Information extraction.
- Machine translation.
- Summarization.
- Question answering.
- Semantic search.
- Text generation.

Evaluation NLP cần chú ý ngôn ngữ, dialect, domain, long-tail và human judgment.

### 12.3. Speech và audio

| Task                 | Metric ví dụ                                       |
| -------------------- | -------------------------------------------------- |
| Speech-to-Text       | WER/CER                                            |
| Text-to-Speech       | human naturalness/intelligibility + signal metrics |
| Speaker verification | EER, false accept/reject                           |
| Audio classification | macro-F1, mAP                                      |
| Source separation    | SI-SDR và human evaluation                         |

### 12.4. Time series

DL options:

- RNN/LSTM/GRU.
- Temporal CNN.
- Transformer variants.
- Deep probabilistic forecasting.

Luôn so sánh với seasonal naive, exponential smoothing, linear/boosting baselines.

### 12.5. Recommendation

Các tầng:

1. Candidate generation/retrieval.
2. Ranking.
3. Re-ranking với constraint.
4. Policy/exploration.

Mô hình:

- Matrix factorization.
- Two-tower embeddings.
- Deep ranking model.
- Sequence recommender.

Cần monitor feedback loops, popularity bias, diversity và exploration.

### 12.6. Graph ML

Task:

- Node classification.
- Link prediction.
- Graph classification.
- Knowledge graph completion.

Rủi ro split: edge/node leakage và temporal leakage.

### 12.7. Multimodal

Model xử lý kết hợp text, image, audio, video hoặc sensor.

Thách thức:

- Alignment giữa modalities.
- Missing modality.
- Dataset licensing/consent.
- Evaluation tổ hợp.
- Chi phí inference.
- Cross-modal hallucination.

---

## 13. Foundation model, LLM, RAG và agent

### 13.1. Vòng đời LLM điển hình

~~~text
Raw corpora
→ tokenization
→ self-supervised pretraining
→ instruction tuning
→ preference/safety alignment
→ evaluation
→ prompting, RAG, tools or fine-tuning
→ deployment and monitoring
~~~

Không phải mọi model/product dùng đầy đủ các bước trên.

### 13.2. Token và context

- Text được chuyển thành token IDs.
- Context window giới hạn lượng input/history model xử lý trong một call.
- Tokenization khác nhau giữa model.
- Context dài hơn không bảo đảm model dùng đều mọi thông tin.
- Chi phí và latency thường tăng theo input/output tokens và kiến trúc serving.

### 13.3. Pretraining, instruction tuning và alignment

| Giai đoạn                         | Mục tiêu                                                     |
| --------------------------------- | ------------------------------------------------------------ |
| Pretraining                       | học pattern/representation rộng từ dữ liệu                   |
| Instruction tuning                | học tuân theo instruction/task format                        |
| Preference optimization/alignment | điều chỉnh hành vi theo preference/safety objectives         |
| Domain adaptation                 | thích nghi domain qua continued pretraining hoặc fine-tuning |

Alignment không loại bỏ hoàn toàn hallucination, bias hoặc misuse.

### 13.4. Inference controls

| Tham số/khái niệm        | Tác động                                   |
| ------------------------ | ------------------------------------------ |
| Temperature              | mức ngẫu nhiên/độ phẳng phân phối sampling |
| Top-p/top-k              | giới hạn candidate tokens                  |
| Max output tokens        | giới hạn độ dài                            |
| Stop sequences           | điều kiện dừng                             |
| Structured output/schema | ràng buộc hình thức output                 |

Deterministic settings không bảo đảm factual correctness.

### 13.5. Prompting, RAG và fine-tuning

| Phương pháp           | Thay đổi gì?                      | Phù hợp khi                               | Không giải quyết tốt                        |
| --------------------- | --------------------------------- | ----------------------------------------- | ------------------------------------------- |
| Prompting             | instruction/context trong request | task rõ, thay đổi nhanh                   | kiến thức lớn, hành vi cần ổn định sâu      |
| RAG                   | context từ nguồn ngoài            | dữ liệu cập nhật/nội bộ, cần citation     | lỗi retrieval, nguồn sai, reasoning/hành vi |
| Fine-tuning           | weights/adapters                  | format/style/behavior/domain task lặp lại | cập nhật knowledge động một cách kiểm chứng |
| Continued pretraining | representation/domain language    | domain corpus lớn                         | thay đổi nhanh, provenance theo từng câu    |
| Tool use              | hành động/tính toán/API           | cần dữ liệu thời gian thực hoặc thao tác  | policy/security/orchestration tự động       |

Kết hợp nhiều phương pháp thường cần thiết.

### 13.6. RAG pipeline

~~~mermaid
flowchart TD
    A["Documents and permissions"] --> B["Parse, chunk and index"]
    Q["User query"] --> C["Retrieve and rerank"]
    B --> C
    C --> D["Build grounded context"]
    D --> E["Generate answer"]
    E --> F["Citations, checks and feedback"]
~~~

Thành phần:

- Ingestion và document parsing.
- Chunking và metadata.
- Embedding/index.
- Query transformation.
- Retrieval.
- Reranking.
- Context construction.
- Generation.
- Citation/grounding check.
- Access control, logging và feedback.

RAG không bảo đảm factuality. Có thể retrieve sai, bỏ sót tài liệu, dùng nguồn stale hoặc model không tuân theo context.

### 13.7. Đánh giá RAG

Tách từng lớp:

| Lớp        | Metric/câu hỏi                                             |
| ---------- | ---------------------------------------------------------- |
| Corpus     | coverage, freshness, permissions                           |
| Retrieval  | Recall@K, MRR, NDCG, hit rate                              |
| Reranking  | relevant passage lên top không?                            |
| Context    | có đủ bằng chứng, có noise/contradiction không?            |
| Generation | correctness, completeness, groundedness, citation accuracy |
| System     | latency, cost, failure rate, safety                        |

### 13.8. Fine-tuning và PEFT

| Kỹ thuật                | Đặc điểm                                               |
| ----------------------- | ------------------------------------------------------ |
| Full fine-tuning        | cập nhật phần lớn/toàn bộ weights, tốn compute/storage |
| LoRA/adapters/PEFT      | cập nhật số parameter nhỏ hơn                          |
| Supervised fine-tuning  | học input–output examples                              |
| Preference optimization | học preference/ranking objectives                      |

Dataset quality và evaluation quan trọng hơn chỉ tăng số examples.

### 13.9. AI agent

Một agent system thường gồm:

- Model.
- Instructions/policy.
- Tools.
- State/memory.
- Planner/orchestrator.
- Environment.
- Termination/budget controls.
- Human approval.
- Logging/evaluation.

Vòng lặp:

~~~text
Observe
→ decide/plan
→ call tool
→ validate result
→ update state
→ continue, escalate or stop
~~~

Rủi ro:

- Prompt injection.
- Tool misuse.
- Excessive permissions.
- Infinite/expensive loops.
- State corruption.
- Irreversible actions.
- Data exfiltration.

Nguyên tắc:

- Least privilege.
- Allowlist tools/actions.
- Validate tool arguments/output.
- Budget và step limits.
- Human approval cho hành động rủi ro.
- Idempotency và rollback khi có thể.
- Full audit trail.

### 13.10. Đánh giá GenAI

Không dùng một metric duy nhất.

| Lớp          | Tiêu chí                                                 |
| ------------ | -------------------------------------------------------- |
| Task quality | correctness, completeness, relevance, format             |
| Grounding    | supported claims, citation accuracy                      |
| Safety       | harmful content, policy violations, jailbreak resilience |
| Fairness     | performance/behavior theo language và groups             |
| Robustness   | paraphrase, adversarial input, long context              |
| Tool use     | correct tool, arguments, result handling                 |
| Agent        | task success, steps, recovery, side effects              |
| Operations   | latency, tokens, cost, availability                      |

Evaluation set cần version, representative cases, hard cases và failure taxonomy. Model-as-judge hữu ích để scale nhưng phải được calibrated/validated với human judgments cho use case cụ thể.

### 13.11. Hạn chế của LLM/GenAI

- Hallucination/confabulation.
- Knowledge stale hoặc không có provenance.
- Sensitivity với prompt/context.
- Bias từ data và alignment.
- Prompt injection và data exfiltration.
- Non-determinism.
- Context window không tương đương memory đáng tin.
- Evaluation khó và dễ contamination.
- Chi phí, latency và vendor/model drift.

---

## 14. Production ML và MLOps

### 14.1. Model chỉ là một phần của hệ thống

Production ML còn có:

- Data ingestion.
- Validation.
- Feature/preprocessing.
- Training/tuning.
- Evaluation/approval.
- Registry.
- Serving.
- Logging.
- Monitoring.
- Feedback/labels.
- Retraining.
- Incident response.

### 14.2. Training mode

| Mode                    | Đặc điểm                                                                 |
| ----------------------- | ------------------------------------------------------------------------ |
| Offline/static training | retrain theo lịch/sự kiện, đơn giản hơn                                  |
| Online/dynamic training | cập nhật liên tục, thích nghi nhanh nhưng risk cao hơn                   |
| Continuous training     | pipeline tự động tạo candidate nhưng vẫn cần validation/promotion policy |

### 14.3. Inference mode

| Mode              | Khi dùng                                   |
| ----------------- | ------------------------------------------ |
| Batch/static      | có thể tính trước, freshness theo giờ/ngày |
| Online/dynamic    | prediction theo request, latency gắt       |
| Streaming         | prediction theo event liên tục             |
| Edge/on-device    | offline, privacy hoặc latency thấp         |
| Human-in-the-loop | risk cao hoặc confidence thấp              |

### 14.4. Artifacts cần quản lý

- Data/label contract.
- Dataset snapshot/version.
- Feature/preprocessing version.
- Training config.
- Experiment run.
- Model artifact.
- Evaluation report.
- Model card.
- Deployment manifest.
- Monitoring specification.
- Runbook và retirement record.

### 14.5. CI, CD và CT

| Pipeline | Mục đích                                           |
| -------- | -------------------------------------------------- |
| CI       | lint, unit/data/model smoke tests                  |
| Training | validate → transform → train → evaluate → register |
| CD       | deploy shadow/canary → observe → promote/rollback  |
| CT       | retrain theo trigger nhưng đi lại quality gates    |

Code pass CI không đồng nghĩa model đủ tốt. Model pass offline không đồng nghĩa system production-ready.

### 14.6. Deployment strategy

- Shadow.
- Canary.
- Blue–green.
- Champion–challenger.
- A/B test.
- Feature flag.

Luôn có:

- Fallback.
- Rollback.
- Version.
- Input validation.
- Health check.
- On-call owner.

### 14.7. Monitoring

| Lớp      | Tín hiệu                                            |
| -------- | --------------------------------------------------- |
| Data     | schema, volume, freshness, missing, drift           |
| Model    | score distribution, quality, calibration, slices    |
| Service  | availability, latency, throughput, errors, cost     |
| Business | KPI, conversion, capacity, complaints               |
| Risk     | override, harmful errors, fairness, security events |

Khi label đến trễ, phân biệt proxy metrics với ground-truth metrics.

### 14.8. Retraining

Trigger:

- Theo lịch.
- Đủ label mới.
- Performance giảm.
- Drift sau điều tra.
- Business/policy thay đổi.
- Bug/incident.

Retraining không tự động đồng nghĩa deploy. Candidate mới phải được so sánh với champion và qua lại data/model/infra/release gates.

### 14.9. Training–serving skew

Nguyên nhân:

- Hai code path preprocessing.
- Feature freshness khác.
- Join/time logic khác.
- Default/missing handling khác.
- Vocabulary/config version lệch.

Giảm skew bằng:

- Shared transformation artifact/code.
- Feature contracts.
- Offline–online parity tests.
- Version logging.
- Representative shadow traffic.

### 14.10. TFX và hệ sinh thái TensorFlow

TensorFlow Extended (TFX) cung cấp các component cho ingestion, statistics, schema, validation, transform, training, evaluation, infrastructure validation và deployment. Đây là một lựa chọn, không phải yêu cầu bắt buộc để làm MLOps đúng.

---

## 15. Responsible AI, governance và security

### 15.1. Responsible AI là yêu cầu vòng đời

Không đợi đến trước deployment mới kiểm tra risk. Risk cần được:

- Govern.
- Map theo context/use case.
- Measure.
- Manage.

### 15.2. Các thuộc tính cần xem xét

- Validity và reliability.
- Safety.
- Security và resilience.
- Accountability và transparency.
- Explainability/interpretability.
- Privacy.
- Fairness và harmful bias.
- Human oversight.

Không thể tối đa hóa mọi thuộc tính cùng lúc; cần trade-off theo use case.

### 15.3. Fairness

Quy trình:

1. Xác định affected groups và harm.
2. Kiểm tra representativeness/label bias.
3. Chọn fairness metric phù hợp context.
4. Đánh giá slice và intersection.
5. Mitigate ở data, model, threshold hoặc process.
6. Monitor sau deploy.

Không có một fairness metric đúng cho mọi bài toán; nhiều metric có thể mâu thuẫn.

### 15.4. Explainability

Phân biệt:

- Global model behavior.
- Local explanation.
- Feature importance.
- Counterfactual.
- Causal explanation.

SHAP/permutation importance/coefficient không tự động tạo causal explanation.

### 15.5. Privacy

- Data minimization.
- Purpose limitation.
- Consent/legal basis.
- Access control và encryption.
- Retention/deletion.
- De-identification với nhận thức re-identification risk.
- Differential privacy/federated approaches khi phù hợp.
- Không log secret/PII không cần thiết.

### 15.6. Security

Threats truyền thống:

- Insecure model deserialization.
- Dependency/supply-chain attack.
- Exposed endpoint/credential.
- Model extraction.
- Membership inference.
- Data/model poisoning.
- Adversarial examples.

GenAI/agent threats:

- Prompt injection.
- Indirect prompt injection từ tài liệu/web.
- Tool argument injection.
- Sensitive data disclosure.
- Excessive agency.
- Unsafe code/action execution.

### 15.7. Documentation

**Dataset/Data Card** nên mô tả:

- Motivation.
- Composition.
- Collection/annotation.
- Intended use.
- License/privacy.
- Known bias/limitations.
- Version/change history.

**Model Card** nên mô tả:

- Intended use và out-of-scope use.
- Data.
- Model/version.
- Evaluation protocol và slice metrics.
- Limitations.
- Ethical/risk considerations.
- Monitoring/owner.

### 15.8. Human oversight

Human-in-the-loop chỉ hữu ích khi:

- Người review có thông tin và quyền override.
- Workload/capacity thực tế.
- Escalation rõ.
- Feedback được audit.
- Automation bias được giảm.

Không dùng “có human review” như một tuyên bố an toàn nếu quy trình không thể hoạt động.

---

## 16. Công cụ và thư viện

Chọn công cụ theo nhu cầu, team và constraints; không chọn chỉ theo độ phổ biến.

### 16.1. Data và compute

| Công cụ         | Vai trò                                      |
| --------------- | -------------------------------------------- |
| NumPy           | array và numerical computing                 |
| pandas          | tabular analysis                             |
| Polars          | columnar dataframe hiệu năng cao             |
| PyArrow/Parquet | columnar format/interchange                  |
| DuckDB          | local analytical SQL                         |
| SciPy           | scientific computing/statistics/optimization |

### 16.2. Visualization

- Matplotlib.
- Seaborn.
- Plotly.
- Altair.

### 16.3. Classical ML

- scikit-learn.
- XGBoost.
- LightGBM.
- CatBoost.
- statsmodels.
- imbalanced-learn.
- OpenCV cho xử lý ảnh/video truyền thống và ứng dụng CV.

### 16.4. Deep Learning

| Framework  | Đặc điểm                                                                                                             |
| ---------- | -------------------------------------------------------------------------------------------------------------------- |
| TensorFlow | tensor/autodiff, Keras integration và production ecosystem                                                           |
| Keras 3    | high-level multi-backend API cho TensorFlow, JAX và PyTorch; một số inference backend khác được hỗ trợ tùy chức năng |
| PyTorch    | flexible tensor/autograd và production/research ecosystem                                                            |
| JAX        | composable transformations, autodiff, JIT và accelerators                                                            |

Trong dự án, chọn một stack chính và giữ import/runtime/version nhất quán.

### 16.5. Foundation model và GenAI

- Hugging Face Transformers/Datasets/Tokenizers.
- Sentence Transformers.
- Vector/search engines theo yêu cầu.
- Evaluation harness và tracing phù hợp hệ thống.
- Provider SDK nếu dùng hosted model.

Không để framework abstraction che mất model/version, prompt, retrieval, permissions và cost.

### 16.6. MLOps

| Nhu cầu                      | Công cụ ví dụ                                            |
| ---------------------------- | -------------------------------------------------------- |
| Experiment tracking/registry | MLflow hoặc managed platform                             |
| Data/model versioning        | DVC, lakehouse/catalog hoặc object versioning            |
| Orchestration                | Airflow, Prefect, Dagster, Kubeflow, TFX                 |
| Packaging                    | Docker                                                   |
| Serving                      | FastAPI, TensorFlow Serving, KServe hoặc managed serving |
| Model interoperability       | ONNX khi operator/model được hỗ trợ                      |
| Monitoring                   | metrics/log/tracing stack và ML-specific monitors        |
| CI/CD                        | GitHub Actions, GitLab CI hoặc platform tương đương      |

Kubernetes không phải mặc định cho mọi dự án; chỉ dùng khi scale và operational needs biện minh cho độ phức tạp.

---

## 17. Hiểu nhầm và anti-pattern phổ biến

1. **AI = ML = DL.**  
   Sai: ML là nhánh của AI; DL là nhóm phương pháp ML.

2. **GenAI là một tầng hoàn toàn tách khỏi DL.**  
   Phần lớn GenAI hiện đại dùng DL; GenAI mô tả năng lực sinh.

3. **RAG là một model.**  
   RAG là kiến trúc hệ thống retrieval + generation.

4. **Agent chỉ là LLM có prompt dài.**  
   Agent system còn cần tools, state, policy, budget, validation và controls.

5. **Dữ liệu càng nhiều càng tốt.**  
   Dữ liệu sai, lệch hoặc không đại diện vẫn gây hại.

6. **Accuracy cao nghĩa model tốt.**  
   Không đúng với imbalance, cost bất đối xứng hoặc probability decisions.

7. **Test set có thể dùng để thử nhiều lần.**  
   Dùng test để ra quyết định làm mất tính độc lập.

8. **Scaling/imputation trước split không sao.**  
   Có thể gây leakage.

9. **Feature importance chứng minh nguyên nhân.**  
   Importance mô tả model/data association, không tự chứng minh causality.

10. **DL luôn hơn tree model trên tabular data.**  
    Không; cần baseline và evaluation.

11. **Softmax score là confidence đáng tin.**  
    Không nếu chưa calibration và OOD evaluation.

12. **RAG loại bỏ hallucination.**  
    Không; retrieval và generation đều có failure modes.

13. **Fine-tuning là cách tốt nhất để cập nhật kiến thức thường xuyên.**  
    Knowledge động và cần provenance thường phù hợp hơn với retrieval/tool.

14. **Model-as-judge thay thế hoàn toàn human evaluation.**  
    Judge cần validation, bias checks và human anchors.

15. **Deploy xong là kết thúc.**  
    Production cần monitoring, feedback, incident, retraining và retirement.

16. **Drift tự động nghĩa phải retrain.**  
    Drift là tín hiệu điều tra.

17. **Human-in-the-loop luôn làm hệ thống an toàn.**  
    Chỉ đúng khi review process có năng lực, quyền và SLA thực.

18. **Kubernetes/MLOps platform càng phức tạp càng trưởng thành.**  
    Maturity là reproducibility, reliability và governance, không phải số lượng tool.

---

## 18. Bảng tóm tắt và glossary nhanh

### 18.1. AI landscape

| Thuật ngữ        | Một câu                                             |
| ---------------- | --------------------------------------------------- |
| AI               | lĩnh vực xây hệ thống thực hiện năng lực thông minh |
| ML               | thuật toán học từ dữ liệu/kinh nghiệm               |
| DL               | ML dùng neural network nhiều tầng                   |
| GenAI            | hệ thống tạo nội dung/dữ liệu mới                   |
| Foundation model | pretrained model thích nghi cho nhiều task          |
| LLM              | foundation/language model lớn cho token sequence    |
| RAG              | retrieval + generation system                       |
| Agent            | model-driven system có tools/state/action loop      |

### 18.2. Data

| Thuật ngữ    | Một câu                                        |
| ------------ | ---------------------------------------------- |
| Feature      | input có sẵn tại prediction time               |
| Target/label | outcome cần học/dự đoán                        |
| Train        | fit parameter/preprocessing                    |
| Validation   | chọn model/hyperparameter/threshold            |
| Test         | final independent estimate                     |
| Leakage      | thông tin không hợp lệ đi vào train/evaluation |
| Drift        | distribution/relationship thay đổi             |

### 18.3. Training

| Thuật ngữ      | Một câu                    |
| -------------- | -------------------------- |
| Parameter      | model học từ dữ liệu       |
| Hyperparameter | cấu hình được chọn/tune    |
| Loss           | objective train            |
| Metric         | chỉ số evaluation          |
| Epoch          | một lượt qua training set  |
| Batch          | nhóm sample mỗi step       |
| Learning rate  | kích thước update          |
| Optimizer      | quy tắc cập nhật parameter |

### 18.4. Generalization

| Thuật ngữ      | Một câu                                         |
| -------------- | ----------------------------------------------- |
| Underfitting   | model chưa biểu diễn/học đủ                     |
| Overfitting    | model khớp train nhưng generalize kém           |
| Regularization | hạn chế complexity/overfit                      |
| Calibration    | probability dự đoán phù hợp tần suất thật       |
| Robustness     | giữ chất lượng dưới biến đổi/failure conditions |

### 18.5. Production

| Thuật ngữ | Một câu                               |
| --------- | ------------------------------------- |
| Registry  | quản lý model version/status/metadata |
| Serving   | cung cấp inference                    |
| Shadow    | chạy model không ảnh hưởng decision   |
| Canary    | mở traffic nhỏ rồi tăng dần           |
| SLI/SLO   | chỉ số và mục tiêu dịch vụ            |
| Rollback  | quay về phiên bản ổn định             |
| CT        | continuous training có kiểm soát      |

---

## 19. Lộ trình học gợi ý

### Giai đoạn 1 — Nền tảng

1. NumPy, pandas và visualization.
2. Đại số tuyến tính, xác suất/thống kê và gradient.
3. Dataset, split, leakage và metrics.

### Giai đoạn 2 — Classical ML

1. Linear/Logistic Regression.
2. Tree, Random Forest và Gradient Boosting.
3. Pipelines, cross-validation, tuning và calibration.
4. Tabular project có baseline/error analysis.

### Giai đoạn 3 — Deep Learning

1. Tensor, autodiff và MLP.
2. CNN và transfer learning.
3. Sequence, attention và Transformer.
4. TensorFlow/Keras hoặc PyTorch project.

### Giai đoạn 4 — Chuyên môn

1. Chọn CV, NLP, time series, recommendation hoặc graph.
2. Pretrained model và fine-tuning.
3. GenAI, RAG và evaluation nếu phù hợp mục tiêu.

### Giai đoạn 5 — Production

1. Packaging, API/batch serving và Docker.
2. Experiment tracking, registry và tests.
3. Monitoring, drift, CI/CD/CT.
4. Responsible AI, security và model/data cards.

Nguyên tắc: mỗi giai đoạn phải có project, baseline, evaluation report và error analysis.

---

## 20. Tài liệu tham chiếu

### Khái niệm và học ML

1. [Google Machine Learning Glossary](https://developers.google.com/machine-learning/glossary)
2. [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
3. [Google Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)
4. [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

### Data, preprocessing và evaluation

1. [scikit-learn — Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
2. [scikit-learn — Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
3. [scikit-learn — Imputation](https://scikit-learn.org/stable/modules/impute.html)
4. [scikit-learn — Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
5. [scikit-learn — Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)
6. [scikit-learn — Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
7. [scikit-learn — Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
8. [scikit-learn — Model inspection](https://scikit-learn.org/stable/inspection.html)

### Deep Learning

1. [TensorFlow Guides](https://www.tensorflow.org/guide)
2. [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
3. [Keras 3](https://keras.io/keras_3/)
4. [Keras Developer Guides](https://keras.io/guides/)
5. [PyTorch Tutorials](https://docs.pytorch.org/tutorials/)
6. [JAX Documentation](https://docs.jax.dev/)

### Foundation model, RAG và production

1. [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
2. [Hugging Face — Retrieval-Augmented Generation](https://huggingface.co/docs/transformers/model_doc/rag)
3. [Google — Production ML Systems](https://developers.google.com/machine-learning/crash-course/production-ml-systems)
4. [TensorFlow Extended](https://www.tensorflow.org/tfx)

### Responsible AI và documentation

1. [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
2. [NIST Generative AI Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)
3. [Google ML Fairness](https://developers.google.com/machine-learning/crash-course/fairness)
4. [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
5. [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)

---

## 21. Checklist tự đánh giá

Người học đã nắm nền tảng khi có thể trả lời:

- [ ] Vì sao RAG và agent là system pattern, không phải một model class?
- [ ] Prediction time ảnh hưởng feature và leakage như thế nào?
- [ ] Khi nào dùng group split hoặc time split thay cho random split?
- [ ] Loss khác metric ra sao?
- [ ] Vì sao accuracy có thể đánh lừa?
- [ ] Threshold và calibration khác nhau thế nào?
- [ ] Bias term, statistical bias và social bias khác nhau ra sao?
- [ ] Vì sao DL không luôn tốt hơn tree boosting trên tabular data?
- [ ] Backpropagation khác optimizer như thế nào?
- [ ] Embedding là gì và khoảng cách embedding có giới hạn gì?
- [ ] RAG có thể thất bại ở những lớp nào?
- [ ] Fine-tuning khác prompting và retrieval ra sao?
- [ ] Model metric khác business impact như thế nào?
- [ ] Data drift khác concept drift và training–serving skew ra sao?
- [ ] Tại sao deploy không phải bước cuối?
- [ ] Model card và data card cần ghi gì?

Nếu chưa trả lời rõ bằng ví dụ, hãy quay lại chương tương ứng và xây một mini-project để kiểm chứng.
