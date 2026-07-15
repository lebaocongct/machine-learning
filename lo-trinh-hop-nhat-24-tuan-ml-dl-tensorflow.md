# Lộ trình hợp nhất 24 tuần — Machine Learning và Deep Learning với TensorFlow

> Phiên bản hợp nhất: 1.0  
> Cập nhật: 15/07/2026  
> Đối tượng: lập trình viên đã quen Python  
> Thời lượng: 24 tuần × 10–12 giờ/tuần, tương đương khoảng 240–288 giờ  
> Định hướng: ML Engineer thực chiến, ưu tiên thực hành, khả năng tái tạo và sản phẩm có thể bàn giao

## 1. Mục đích và quyết định hợp nhất

Tài liệu này là bản chuẩn duy nhất thay cho hai tài liệu:

- `ke-hoach-24-tuan-ml-dl-tensorflow(1).md`;
- `lo-trinh-ml-dl-tensorflow-24-tuan(2).md`.

Hai tài liệu cũ cùng mô tả một lộ trình ML/DL với TensorFlow trong 24 tuần nhưng khác cách phân bổ tuần. Bản hợp nhất giải quyết sự lệch lịch theo các nguyên tắc:

1. Giữ cấu trúc triển khai, hệ thống đánh giá và ba checkpoint ở tuần 10, 15 và 24.
2. Giữ phần giải thích kiến thức, môi trường, metric, checklist và capstone chi tiết.
3. Loại bỏ các đoạn lặp về nhịp học, baseline, leakage, error analysis, reproducibility và Definition of Done.
4. Dùng một tiến trình kiến thức duy nhất: nền tảng → ML cổ điển → TensorFlow/Keras → kiến trúc DL → production.
5. Project 2 bắt đầu ở tuần 20 và được productionize thành capstone ở tuần 21–24; không mở một dự án hoàn toàn mới ở tuần 22.

Sau 24 tuần, người học cần có khả năng:

- chuyển một vấn đề thực tế thành prediction contract, target, metric, baseline và tiêu chí thành công;
- xây pipeline ML cổ điển đúng quy trình, không rò rỉ dữ liệu;
- xây, train và chẩn đoán mô hình DL bằng TensorFlow/Keras;
- tổ chức thí nghiệm có giả thuyết, log, so sánh và error analysis;
- lưu, export, kiểm thử và phục vụ model;
- bàn giao một hệ thống có source code, test, container, benchmark, model card và kế hoạch monitoring.

### Ba mốc sản phẩm

| Mốc | Sản phẩm | Mục tiêu |
|---|---|---|
| Tuần 10 | Project 1 v1.0 | ML cổ điển end-to-end |
| Tuần 20 | Project 2 candidate v0.1 | Mô hình DL theo một nhánh chuyên sâu |
| Tuần 24 | Capstone v1.0 | Productionize Project 2 và bàn giao hoàn chỉnh |

## 2. Nguyên tắc học và thực hành

- Dành khoảng 70–75% thời gian cho code, lab, thí nghiệm, dự án và phân tích lỗi.
- Mọi bài toán phải có baseline phù hợp trước khi tăng độ phức tạp.
- Mỗi thí nghiệm chỉ thay đổi một yếu tố chính và phải ghi hypothesis, cấu hình, metric, kết quả và kết luận.
- Tách test set sớm nhưng chỉ dùng ở bước đánh giá cuối; không dùng test set để chọn model, threshold hoặc hyperparameter.
- Fit preprocessing trên train hoặc từng fold, không fit trên toàn bộ dữ liệu.
- Notebook dùng cho khám phá; logic lặp lại phải chuyển sang module/script và có test.
- Mô hình DL phải được so sánh với baseline đơn giản hoặc ML cổ điển phù hợp.
- Không kết luận bằng một metric tổng; phải xem learning curve, lỗi cụ thể và các data slice quan trọng.
- Một tuần chỉ hoàn thành khi có artifact chạy lại được, không chỉ khi đã đọc hoặc xem nội dung.
- Từ tuần 20, mọi quyết định model phải cân bằng chất lượng, latency, kích thước, độ phức tạp và rủi ro vận hành.

### Chu trình thí nghiệm chuẩn

1. Xác định câu hỏi, giả thuyết và metric.
2. Tạo baseline đơn giản nhất.
3. Chạy một thí nghiệm chỉ thay đổi một yếu tố chính.
4. Ghi lại data version, split, seed, config, metric và thời gian chạy.
5. Phân tích lỗi theo class, subgroup, thời gian hoặc horizon phù hợp.
6. Viết kết luận và đề xuất thí nghiệm tiếp theo.

## 3. Công cụ, môi trường và cấu trúc dự án

### 3.1. Phạm vi công cụ

- ML cổ điển: NumPy, pandas và scikit-learn.
- Deep Learning: TensorFlow và Keras với TensorFlow backend.
- Dữ liệu và quan sát: `tf.data`, TensorFlow Datasets, TensorBoard.
- Kỹ nghệ phần mềm: Git, pytest, Ruff, Docker, FastAPI hoặc TensorFlow Serving.
- Production nâng cao: data/model validation, monitoring, TFX hoặc công cụ orchestration tương đương.

TensorFlow không phải lựa chọn chính để học Random Forest, Gradient Boosting, SVM hoặc clustering. Các thuật toán ML cổ điển dùng scikit-learn; TensorFlow tập trung vào tensor, automatic differentiation và neural network.

### 3.2. Môi trường đề xuất

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install tensorflow keras tensorflow-datasets scikit-learn pandas numpy +  matplotlib seaborn jupyterlab tensorboard pytest ruff fastapi uvicorn
```

Trên Windows, dùng lệnh kích hoạt virtual environment tương ứng. Chọn phiên bản Python được bản TensorFlow stable đang sử dụng hỗ trợ, sau đó khóa dependency. Không dùng nightly build trong lộ trình chính.

Phong cách import cần nhất quán trong toàn bộ dự án:

```python
import tensorflow as tf
import keras
from keras import layers
```

### 3.3. Cấu trúc repository chuẩn

```text
project/
├── README.md
├── pyproject.toml
├── configs/
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_baseline.ipynb
├── src/
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── tests/
├── reports/
└── artifacts/
```

### 3.4. Cấu trúc một gói học tuần

```text
week-XX-topic/
├── README.md
├── 01-knowledge.md
├── 02-reading-list.md
├── notebooks/
│   ├── 01-from-scratch.ipynb
│   └── 02-guided-lab.ipynb
├── exercises/
├── challenge/
├── src/
├── tests/
├── assessment/
│   ├── quiz.md
│   ├── practical-test.md
│   └── rubric.md
├── solutions/
└── report-template.md
```

Thư mục `solutions/` chỉ nên được mở sau lần nộp đầu tiên.

## 4. Nhịp học chuẩn mỗi tuần

| Khối học | Thời lượng cốt lõi | Đầu ra |
|---|---:|---|
| Kiến thức có chọn lọc | 1,5 giờ | Ghi chú khái niệm, giả định và lỗi thường gặp |
| Code từ đầu/toán trực quan | 1 giờ | Cài đặt hoặc mô phỏng phần cốt lõi |
| Lab có hướng dẫn | 2 giờ | Hai lab ngắn hoặc một lab có checkpoint |
| Bài tập độc lập | 2 giờ | 6–10 bài và một coding challenge |
| Experiment/dự án | 2,5 giờ | Baseline, controlled experiment và error analysis |
| Kiểm tra/báo cáo | 1 giờ | Quiz, practical test, README và retrospective |
| Mở rộng tùy chọn | 0–2 giờ | Tài liệu nâng cao hoặc cải thiện artifact |

Trong tuần dự án, thời gian bài tập nhỏ chuyển sang hoàn thiện source code, test, tài liệu và demo.

### Chỉ tiêu thực hành toàn lộ trình

- Tối thiểu 160 bài tập ngắn.
- Tối thiểu 45 lab chạy lại được từ đầu.
- Tối thiểu 24 coding challenge hoặc thí nghiệm độc lập.
- 24 quiz và 24 practical test theo tuần.
- Ba checkpoint tổng hợp ở tuần 10, 15 và 24.
- Hai dự án portfolio; Project 2 được phát triển thành capstone.

## 5. Lộ trình chi tiết 24 tuần

## Giai đoạn A — Nền tảng dữ liệu, toán và tư duy ML (Tuần 1–4)

### Tuần 1 — Môi trường, NumPy, pandas và chất lượng dữ liệu

**Năng lực cần đạt**

- Tạo môi trường tái tạo được, tổ chức Git repository và chạy test.
- Làm việc chắc chắn với shape, dtype, indexing, broadcasting và vectorization.
- Nạp, kiểm tra, làm sạch, tổng hợp và trực quan hóa một tập dữ liệu bảng.

**Kiến thức**

- Scalar, vector, matrix, tensor; shape, axis và dtype.
- NumPy indexing, broadcasting, vectorized operations và memory cơ bản.
- pandas: load, missing value, duplicate, group, join và type conversion.
- Data quality: schema, missing rate, range, category và outlier.
- Git, virtual environment, dependency và pytest.

**Thực hành tối thiểu**

- 8–10 bài NumPy/pandas.
- Benchmark vòng lặp Python và vectorization.
- EDA một CSV có missing value và duplicate.
- Viết `clean_data()` cùng ít nhất năm unit test.

**Đánh giá và đầu ra**

- Quiz khái niệm và coding challenge 45 phút về làm sạch/tổng hợp dữ liệu.
- Repository chạy trong môi trường sạch; có EDA report, source và test.
- Không dùng vòng lặp cho phép toán mảng cơ bản khi đã có phép vector hóa phù hợp.

### Tuần 2 — Đại số tuyến tính, đạo hàm và gradient descent

**Năng lực cần đạt**

- Theo dõi shape trong phép toán ma trận.
- Liên hệ đạo hàm, gradient và loss với cập nhật tham số.
- Cài linear regression bằng NumPy và kiểm tra gradient.

**Kiến thức**

- Dot product, matrix multiplication, transpose, norm L1/L2 và cosine similarity.
- Inverse, pseudo-inverse, eigenvalue/eigenvector và SVD ở mức trực giác.
- Derivative, partial derivative, gradient và chain rule.
- Loss, learning rate, batch, epoch, gradient descent và mini-batch.

**Thực hành tối thiểu**

- 6–8 bài toán-code.
- Linear regression bằng normal equation và gradient descent.
- Gradient checking bằng finite difference.
- Thử ít nhất bốn learning rate và vẽ loss curve.

**Đánh giá và đầu ra**

- Practical test 60 phút: cài linear regression không dùng scikit-learn.
- Gradient tự tính và gradient số khớp trong tolerance đã nêu.
- Báo cáo giải thích learning rate quá nhỏ, phù hợp và quá lớn.

### Tuần 3 — Xác suất, thống kê, chiến lược split và bias–variance

**Năng lực cần đạt**

- Chọn đúng cách chia dữ liệu theo cấu trúc bài toán.
- Phân biệt sampling uncertainty, model bias và variance.
- Nhận diện leakage và test contamination.

**Kiến thức**

- Mean, variance, covariance, correlation và phân phối.
- Conditional probability, Bayes ở mức ứng dụng.
- Sampling, confidence interval, bootstrap và class imbalance.
- Random, stratified, group và time-based split.
- Underfitting, overfitting, bias–variance và learning curve.

**Thực hành tối thiểu**

- Mô phỏng sampling distribution/CLT.
- Bootstrap confidence interval.
- Chọn split cho dữ liệu người dùng lặp lại, chuỗi thời gian và classification mất cân bằng.
- Polynomial regression để tạo underfit/overfit và learning curves.

**Đánh giá và đầu ra**

- Bài tình huống phát hiện leakage/test contamination.
- Notebook mô phỏng và báo cáo bias–variance.
- Giải thích được vì sao test set không dùng để chọn model hoặc hyperparameter.

### Tuần 4 — Problem framing và ML Problem Canvas

**Năng lực cần đạt**

- Chuyển một yêu cầu thực tế thành bài toán dự đoán có thể đo lường.
- Định nghĩa prediction contract, metric, baseline, constraint và slice.
- Chuẩn bị Project 1 trước khi modeling.

**Kiến thức**

- Quan hệ AI, ML và DL; supervised/unsupervised; regression/classification.
- Batch, online và streaming ở mức định hướng.
- Prediction unit, prediction time, target, decision và owner.
- Offline metric, guardrail metric, business metric và chi phí sai lầm.
- Data contract, leakage risk, non-goal và tiêu chí go/no-go.

**Thực hành tối thiểu**

- Frame bốn bài toán thực tế.
- Viết hai tình huống metric trade-off.
- Lập ML Problem Canvas và data/prediction contract sơ bộ cho Project 1.

**Đánh giá và đầu ra**

- Bảo vệ canvas theo rubric và quiz tình huống 45 phút.
- Canvas phải có decision, target, prediction unit/time, metric, baseline, constraint, slices và rủi ro leakage.

**Gate giai đoạn A:** chỉ sang tuần 5 khi có thể tự giải thích dữ liệu vào, feature/target, split, baseline, metric và leakage cho một bài toán mới.

## Giai đoạn B — Machine Learning cổ điển end-to-end (Tuần 5–10)

### Tuần 5 — Regression, preprocessing và pipeline

**Kiến thức**

- EDA có mục tiêu và split trước preprocessing.
- Imputation, encoding, scaling, `ColumnTransformer` và `Pipeline`.
- Dummy, linear, ridge/lasso và tree-based baseline.
- MAE, MSE, RMSE, R², residual analysis và learning curve.
- Cross-validation cơ bản.

**Thực hành**

- Dựng pipeline regression trên dữ liệu bẩn.
- So sánh dummy baseline và ít nhất hai model.
- Chạy 5-fold CV phù hợp và phân tích năm lỗi lớn nhất.

**Đánh giá và đầu ra**

- Practical test 60 phút: từ dữ liệu thô tạo pipeline có thể predict dữ liệu mới.
- Preprocessing chỉ fit trên train/fold; báo cáo có MAE, RMSE, baseline và residual.

### Tuần 6 — Classification, imbalance, calibration và threshold

**Kiến thức**

- Logistic regression, k-NN và SVM ở mức cơ chế/giả định.
- Confusion matrix, precision, recall, specificity và F1.
- ROC-AUC, PR-AUC, log loss và calibration.
- Class imbalance, class weight, resampling và cost-sensitive learning.
- Decision threshold khác model parameter.

**Thực hành**

- Dùng một dataset mất cân bằng.
- So sánh dummy và ít nhất hai classifier.
- Vẽ ROC/PR curve, threshold sweep và cost table.
- Phân tích lỗi theo subgroup.

**Đánh giá và đầu ra**

- Chọn threshold cho một business scenario và bảo vệ quyết định.
- Không dùng accuracy làm metric duy nhất khi dữ liệu mất cân bằng.

### Tuần 7 — Linear model từ đầu, regularization và chẩn đoán learning curve

**Kiến thức**

- Linear, logistic và softmax regression.
- Sigmoid, softmax và cross-entropy.
- Batch/mini-batch gradient descent.
- L1/L2, early stopping, learning curve và validation curve.
- Parameter, hyperparameter, loss và metric.

**Thực hành**

- Cài logistic hoặc softmax regression từ đầu.
- So sánh các learning rate, L1/L2 và early stopping.
- Debug ba ca: learning rate quá cao, underfit và overfit.

**Đánh giá và đầu ra**

- Model từ đầu có forward pass, loss, gradient update và test.
- Báo cáo định lượng tác động của learning rate và regularization.

### Tuần 8 — Decision Tree, ensemble, cross-validation và tuning

**Kiến thức**

- Decision Tree: split, impurity, depth và pruning.
- Bagging, Random Forest và gradient boosting.
- K-fold, StratifiedKFold, GroupKFold và TimeSeriesSplit.
- Grid/randomized search và preprocessing bên trong CV.
- Feature importance và giới hạn của impurity-based importance.

**Thực hành**

- So sánh linear model, tree, Random Forest và boosting trên cùng protocol.
- Chạy search nhỏ, có hypothesis và ngân sách rõ.
- Phân tích permutation importance và error slice.

**Đánh giá và đầu ra**

- Bảng so sánh công bằng về chất lượng, tốc độ, khả năng giải thích và overfit.
- Không diễn giải feature importance như quan hệ nhân quả.

### Tuần 9 — Scaling, PCA, clustering và anomaly detection

**Kiến thức**

- Khi nào scaling cần thiết; trực giác SVM.
- PCA, explained variance và reconstruction error.
- K-Means, hierarchical clustering và DBSCAN.
- Silhouette, stability và giới hạn đánh giá không nhãn.
- Anomaly detection và threshold.

**Thực hành**

- So sánh có/không scaling.
- Đo explained variance và reconstruction error của PCA.
- Chạy K-Means/DBSCAN, đánh giá bằng metric nội tại và kiểm tra nghiệp vụ.
- Tạo anomaly ranking và kiểm tra thủ công các trường hợp đầu bảng.

**Đánh giá và đầu ra**

- Challenge không nhãn và ba bài tình huống chọn phương pháp.
- Không kết luận cluster chỉ bằng biểu đồ hoặc một silhouette score.

### Tuần 10 — Project 1: ML cổ điển end-to-end

**Phạm vi**

- Problem/data contract, EDA và split.
- Pipeline từ dữ liệu thô.
- Dummy baseline và ít nhất ba họ model.
- CV/tuning, calibration/threshold nếu phù hợp.
- Error analysis theo slice.
- Final test đúng một lần sau khi chốt lựa chọn.
- Source, test, README, experiment report và model card ngắn.

**Checkpoint 1 — ML Engineer Foundation**

- Practical test 90 phút, không xem notebook cũ.
- Project 1 phải tái tạo được từ môi trường sạch.
- Lỗi leakage, thiếu baseline hoặc không chạy lại được là lỗi chặn.

**Đầu ra:** Project 1 release v1.0.

## Giai đoạn C — TensorFlow và Keras cốt lõi (Tuần 11–15)

### Tuần 11 — Tensor, autodiff và MLP với Keras

**Kiến thức**

- `tf.Tensor`, `Variable`, shape, dtype, broadcasting và device.
- Eager execution, graph execution và `tf.function` ở mức thực dụng.
- `GradientTape` và gradient debugging.
- Neuron, MLP, logits, softmax và cross-entropy.
- `compile`, `fit`, `evaluate` và `predict`.

**Thực hành**

- Chuyển logistic regression tuần 7 sang TensorFlow.
- Kiểm tra gradient bằng bài toán có nghiệm biết trước.
- Xây MLP cho Fashion-MNIST hoặc dữ liệu tương đương.
- So với baseline tuyến tính/softmax; thử overfit một batch nhỏ.

**Đánh giá và đầu ra**

- Viết MLP, chú thích shape/loss và phân tích ít nhất mười prediction sai.
- Giải thích được logits, output head và loss tương ứng.

### Tuần 12 — Keras APIs, callback, checkpoint và TensorBoard

**Kiến thức**

- Sequential, Functional API và subclassing: khi nào dùng.
- Multi-input/multi-output model.
- EarlyStopping, ModelCheckpoint, ReduceLROnPlateau và custom callback.
- TensorBoard, run comparison và resume training.
- Save/load cơ bản.

**Thực hành**

- Xây cùng một model bằng Sequential và Functional API.
- Tạo một model đa input nhỏ.
- Viết custom callback, lưu checkpoint và resume.
- So sánh các run trên TensorBoard.

**Đánh giá và đầu ra**

- Chuyển kiến trúc từ Sequential sang Functional và sửa lỗi shape.
- Checkpoint phục hồi được; learning curves và config của run được lưu.

### Tuần 13 — Optimization và regularization

**Kiến thức**

- Initialization; ReLU, GELU, sigmoid và softmax.
- SGD + momentum, RMSprop, Adam và learning-rate schedule.
- Batch normalization và layer normalization.
- L1/L2/weight decay, dropout, early stopping và augmentation.
- Vanishing/exploding gradient và gradient clipping.

**Thực hành**

- Chạy ít nhất sáu ablation runs.
- So sánh SGD/Adam, hai activation, normalization, dropout và LR schedule.
- Mỗi run chỉ thay đổi một nhóm yếu tố và dùng cùng seed/split/protocol.

**Đánh giá và đầu ra**

- Experiment report có bảng ablation và learning curves.
- Phân biệt được lỗi optimization với lỗi generalization.

### Tuần 14 — Custom component và custom training loop

**Kiến thức**

- Custom loss, metric, layer và model.
- `GradientTape`, training step và full training loop.
- Metric state/reset, gradient clipping, NaN/Inf và `tf.function`.
- Khi nào dùng custom loop, khi nào ưu tiên `model.fit()`.

**Thực hành**

- Viết forward/loss/gradient update.
- Tạo một custom metric và một custom layer.
- Viết loop train/validation rồi so với `model.fit()`.
- Sửa một ca NaN gradient.

**Đánh giá và đầu ra**

- Hoàn thiện training loop bị khuyết.
- Metric reset đúng; kết quả có xu hướng nhất quán với built-in training.

### Tuần 15 — tf.data, dữ liệu lớn và TensorFlow Core checkpoint

**Kiến thức**

- Source, `map`, `shuffle`, `batch`, `cache` và `prefetch`.
- Thứ tự transformation và tính xác định.
- Keras preprocessing layers và train–serve consistency.
- TensorFlow Datasets, TFRecord và pipeline profiling cơ bản.

**Thực hành**

- Dựng ba pipeline dữ liệu và benchmark throughput.
- Parse record; kiểm tra shape, dtype, range và label của một batch.
- Mini-project tabular: Functional MLP so với model tốt nhất Project 1.
- Kết hợp callback, checkpoint, TensorBoard và custom component.

**Checkpoint 2 — TensorFlow Core**

- Tự debug shape, NaN loss, overfit và input pipeline.
- MLP không bắt buộc thắng tree ensemble, nhưng phải dùng cùng split/metric và phân tích nguyên nhân.

## Giai đoạn D — Kiến trúc Deep Learning và chuyên ngành (Tuần 16–20)

### Tuần 16 — Computer Vision và CNN

**Kiến thức**

- Convolution, kernel, stride, padding, pooling và receptive field.
- Feature extractor, classification head và output spatial shape.
- Image normalization, augmentation và class imbalance.
- Confusion matrix, per-class metric và error gallery.

**Thực hành**

- Dense baseline và CNN nhỏ trên CIFAR-10 hoặc dataset tương đương.
- Augmentation ablation.
- Trực quan hóa filter/feature map và phân tích lỗi theo ít nhất ba slice.

**Đánh giá và đầu ra**

- Tính shape qua các convolution block và sửa một CNN lỗi.
- CNN chạy end-to-end; augmentation chỉ áp dụng cho training.

### Tuần 17 — Transfer learning và fine-tuning

**Kiến thức**

- Pretrained backbone, frozen feature extractor và fine-tuning.
- Freeze/unfreeze theo giai đoạn và learning rate thấp.
- Batch-normalization caveat, preprocessing contract và domain shift.
- Top-k, per-class metric và calibration cơ bản.

**Thực hành**

- Train classification head trên frozen backbone.
- Unfreeze một phần và fine-tune.
- So sánh với scratch baseline trên cùng split/metric.
- Export model và tạo inference script cho một ảnh.

**Đánh giá và đầu ra**

- Sửa một pipeline fine-tune có lỗi cố ý mà không phá pretrained weights.
- Có biểu đồ hai giai đoạn và error gallery theo nhóm lỗi.

### Tuần 18 — Sequence và time series

**Kiến thức**

- Sequence, embedding, state, RNN, LSTM và GRU.
- Time split, windowing, horizon và walk-forward evaluation.
- Naive/seasonal baseline, MLP, 1D CNN và recurrent model.
- Leakage trong forecasting, residual theo horizon và drift.

**Thực hành**

- Tạo windows và chronological split.
- Xây hai baseline và ít nhất hai neural model.
- Backtest, leakage checks và residual analysis.

**Đánh giá và đầu ra**

- Phát hiện/sửa leakage trong time-series pipeline.
- Không random split chuỗi thời gian; so sánh công bằng với naive baseline.

### Tuần 19 — NLP, attention và Transformer nền tảng

**Kiến thức**

- Text vectorization, vocabulary, OOV, embedding, padding và masking.
- TF-IDF baseline và recurrent text classifier.
- Query, key, value; scaled dot-product và multi-head attention.
- Positional information, encoder block, residual và layer normalization.
- Padding mask, causal mask và chi phí attention theo sequence length.

**Thực hành**

- Majority/TF-IDF baseline.
- `TextVectorization` + embedding classifier.
- RNN hoặc BiLSTM/GRU.
- Small Transformer encoder cho text classification.
- Kiểm tra tensor shape và mask; phân tích lỗi ngôn ngữ.

**Đánh giá và đầu ra**

- Vẽ/giải thích shape qua pipeline NLP.
- So sánh neural model với TF-IDF baseline trên cùng dữ liệu.
- Phân biệt encoder-only và autoregressive/causal modeling ở mức kiến trúc.

### Tuần 20 — Chọn chuyên ngành và tạo Project 2 candidate

Chọn một nhánh, không học dàn trải:

| Nhánh | Nội dung gợi ý | Dự án gợi ý |
|---|---|---|
| Computer Vision | Detection/segmentation, augmentation nâng cao | Phân đoạn lỗi sản phẩm hoặc vật thể |
| NLP | Pretrained encoder, fine-tuning, text pairs | Phân loại ticket hoặc semantic similarity |
| Time series | Multi-horizon, backtesting | Dự báo nhu cầu |
| Recommendation | Embedding, retrieval/ranking | Gợi ý nội dung/sản phẩm |
| Anomaly detection | Autoencoder, threshold và drift | Phát hiện bất thường cảm biến |

**Yêu cầu Project 2 candidate**

- Problem/data/prediction contract.
- Baseline đơn giản hoặc không-DL.
- Data pipeline tái tạo được.
- Ít nhất ba controlled experiments có giả thuyết.
- Error analysis theo subgroup, class hoặc horizon.
- Model artifact có version và model card draft.
- Chọn candidate theo metric, error slice, latency sơ bộ, kích thước và rủi ro.

**Gate giai đoạn D:** Project 2 candidate v0.1 phải được review và bảo vệ 30 phút trước khi productionize.

## Giai đoạn E — Production ML và capstone (Tuần 21–24)

### Tuần 21 — Reproducibility, testing, packaging và CI

**Kiến thức**

- Tách data, feature, model, train, evaluate và inference thành module.
- Config-driven training, seed, environment và artifact metadata.
- Data contract và schema validation.
- Unit, integration và smoke test.
- Packaging, lint và CI.

**Thực hành**

- Chuyển logic Project 2 khỏi notebook.
- Tạo CLI dạng `python -m src.train --config ...`.
- Viết test cho shape, dtype, preprocessing và smoke training.
- Tạo CI workflow.

**Đánh giá và đầu ra**

- Từ môi trường sạch, train sample bằng một lệnh.
- Repository có `src/`, `tests/`, config và lock file; CI chạy test.

### Tuần 22 — Serialization, export và inference contract

**Kiến thức**

- Lưu model ở định dạng Keras phù hợp và export phục vụ inference.
- Batch/online inference và tiêu chí lựa chọn.
- Input/output schema, validation và version metadata.
- Train–serve parity và prediction tolerance.
- Invalid, missing và out-of-range request.

**Thực hành**

- Save, khởi động process mới, reload và so sánh prediction.
- Export model và viết parity test.
- Tạo batch predictor hoặc inference function.
- Kiểm thử input hợp lệ và không hợp lệ.

**Đánh giá và đầu ra**

- Sửa một pipeline có parity mismatch và schema lỗi.
- Artifact có version; input sai bị từ chối với thông báo rõ.

### Tuần 23 — Serving, container, hiệu năng và monitoring

**Kiến thức**

- FastAPI so với TensorFlow Serving; REST/gRPC ở mức lựa chọn.
- Batching, concurrency, warm-up, memory và cold start.
- Latency p50/p95/p99, throughput và benchmark protocol.
- Docker, health check, structured logging và model version.
- Data drift, concept drift, training-serving skew và label delay.
- Offline model metric, service metric và business metric.
- Champion–challenger/canary, retraining trigger và rollback ở mức thiết kế.

**Thực hành**

- Tạo API hoặc batch service và Docker image.
- Viết integration/smoke test cho endpoint.
- Benchmark latency/throughput và concurrent requests.
- Ghi log request metadata an toàn, prediction và latency.
- Tạo dữ liệu drift giả, report phát hiện thay đổi và runbook xử lý.

**Đánh giá và đầu ra**

- Container chạy local; API contract và benchmark environment rõ.
- Demo cold start, invalid request và concurrent request.
- Có monitoring plan kể cả khi ground truth đến trễ.

### Tuần 24 — Final evaluation, release và bàn giao capstone

**Kiến thức và hoạt động**

- Đóng băng data/config/model candidate.
- Chạy final test đúng một lần và ghi kết quả.
- Fresh-clone rehearsal.
- Model card, limitations, intended use và responsible ML.
- Monitoring, rollback, privacy/fairness và human fallback khi phù hợp.
- Release tag, demo, presentation, handover và retrospective.

**Checkpoint 3 — Production Readiness**

- Người khác làm theo README có thể train/evaluate/infer hoặc chạy demo.
- Save/load/export đạt prediction parity.
- Test, container, benchmark, model card và monitoring plan đầy đủ.
- Có backlog v1.1 dựa trên bằng chứng và limitation.

**Đầu ra:** Capstone v1.0.

**Gate hoàn thành:** “chạy được trên máy tác giả” chưa đủ; fresh-clone demo phải thành công.

## 6. Hệ thống kiểm tra và chấm điểm

### 6.1. Điểm tuần

| Thành phần | Trọng số |
|---|---:|
| Bài tập và lab | 40% |
| Experiment/challenge hoặc phần dự án | 30% |
| Practical test | 20% |
| Quiz và giải thích bằng lời của mình | 10% |

Điều kiện qua tuần:

- Tổng điểm tối thiểu 75/100.
- Practical test tối thiểu 60%.
- Hoàn thành toàn bộ artifact bắt buộc.
- Không vi phạm critical gate.

### 6.2. Critical gates

Các lỗi sau khiến tuần hoặc project chưa đạt dù tổng điểm đủ:

- Data leakage hoặc dùng test set để tuning/chọn model/threshold.
- Không có baseline phù hợp.
- Code không chạy lại được từ đầu hoặc phụ thuộc trạng thái notebook ẩn.
- Không giải thích được feature/tensor shape, loss hoặc metric chính.
- Không có error analysis tối thiểu ba trường hợp hoặc ba nhóm lỗi.
- Từ tuần 21: thiếu test cho preprocessing/input contract.
- Từ tuần 22: save/load/export không đạt prediction parity.
- Tuần 24: người khác không chạy được demo theo README.

### 6.3. Học bù

Nếu chưa đạt, không cần học lại toàn bộ tuần. Gói học bù gồm:

1. Sửa artifact lỗi.
2. Làm một challenge tương đương trên dữ liệu khác.
3. Viết root cause và biện pháp ngăn lặp lại.
4. Chạy lại practical test với biến thể mới.

### 6.4. Ba checkpoint

| Checkpoint | Thời điểm | Năng lực xác nhận | Điều kiện đạt |
|---|---|---|---|
| ML Engineer Foundation | Cuối tuần 10 | Split, pipeline, baseline, model comparison, metric/threshold và leakage analysis | Project 1 tái tạo được và đạt gate |
| TensorFlow Core | Cuối tuần 15 | Functional API, shape/loss, callback, checkpoint, TensorBoard, custom component và `tf.data` | Tự debug shape, NaN, overfit và input pipeline |
| Production Readiness | Cuối tuần 24 | One-command train, final evaluation, export, inference, test, container, benchmark và model card | Fresh-clone demo và capstone đạt toàn bộ gate |

## 7. Capstone end-to-end

### 7.1. Gợi ý đề tài

1. Phân loại ticket hỗ trợ: text classification, confidence threshold và human fallback.
2. Phát hiện lỗi sản phẩm: image transfer learning, class imbalance và API inference.
3. Dự báo nhu cầu: time split, backtesting và thảo luận uncertainty.
4. Phát hiện bất thường: supervised baseline hoặc autoencoder, threshold theo chi phí.
5. Churn prediction: tabular baseline so với MLP, calibration và decision threshold.

### 7.2. Deliverable bắt buộc

```text
capstone/
├── README.md
├── pyproject.toml
├── data/
│   └── README.md
├── configs/
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── tests/
├── reports/
│   ├── experiment_report.md
│   ├── error_analysis.md
│   ├── model_card.md
│   └── monitoring_plan.md
├── Dockerfile
└── artifacts/
```

### 7.3. Rubric 100 điểm

| Hạng mục | Điểm | Tiêu chí |
|---|---:|---|
| Problem framing và dữ liệu | 15 | Target, split và constraint rõ; không leakage |
| Baseline và đánh giá | 20 | Metric đúng, CV/backtest đúng, threshold hợp lý |
| Modeling và experiment | 20 | Giả thuyết rõ, tái tạo được và có ablation |
| Error analysis | 10 | Phân tích lỗi thực tế theo subgroup/class/horizon |
| Software quality | 15 | Module hóa, config, test, lint và README |
| Deployment và hiệu năng | 10 | API/batch/container chạy và có benchmark |
| Monitoring và responsible ML | 10 | Drift, rollback, limitation, fairness/privacy phù hợp |

Mức đạt là 75/100 và không có lỗi nghiêm trọng về leakage, test contamination hoặc reproducibility.

## 8. Bảng chọn metric nhanh

| Bài toán | Metric bắt đầu | Khi cần bổ sung |
|---|---|---|
| Regression thông thường | MAE + RMSE | MAPE khi target không gần 0; quantile loss khi chi phí bất đối xứng |
| Binary cân bằng | F1 hoặc ROC-AUC + log loss | Calibration khi probability được dùng để ra quyết định |
| Binary mất cân bằng | PR-AUC + precision/recall tại threshold | Cost-weighted metric hoặc recall tại precision tối thiểu |
| Multiclass | Macro-F1 + per-class recall | Top-k khi nhiều class tương tự |
| Time series | MAE/RMSE theo horizon + backtest | MASE để so với naive baseline |
| Ranking/recommendation | Recall@K, NDCG@K | Coverage, diversity và online business metric |

Metric phải gắn với chi phí sai lầm và quyết định thực tế, không chọn chỉ vì thư viện cung cấp sẵn.

## 9. Checklist và thứ tự chẩn đoán

### 9.1. Trước khi train

- [ ] Target có tồn tại tại thời điểm dự đoán không?
- [ ] Prediction unit và prediction time đã rõ chưa?
- [ ] Cùng user/entity có xuất hiện ở cả train và test không?
- [ ] Dữ liệu thời gian đã split theo thời gian chưa?
- [ ] Preprocessing chỉ fit trên train/fold chưa?
- [ ] Có dummy/rule-based/simple-model baseline chưa?
- [ ] Metric và threshold có phản ánh chi phí thực tế không?

### 9.2. Trong khi train DL

- [ ] Kiểm tra một batch: shape, dtype, min/max và label.
- [ ] Thử overfit một tập rất nhỏ.
- [ ] Theo dõi train/validation curves.
- [ ] Lưu best checkpoint theo validation metric phù hợp.
- [ ] Chỉ thay đổi một nhóm biến trong mỗi experiment.
- [ ] Kiểm tra NaN/Inf và gradient bất thường.

### 9.3. Trước khi triển khai

- [ ] Model load được trong process mới.
- [ ] Prediction trước/sau export tương đương trong tolerance.
- [ ] Input schema và lỗi validation rõ ràng.
- [ ] Có smoke test, latency benchmark và health check.
- [ ] Có model version, rollback và monitoring plan.
- [ ] Model card nêu intended use, limitation và rủi ro.

### 9.4. Thứ tự chẩn đoán khi model kém

1. Bài toán: target/metric có đúng điều cần tối ưu không?
2. Dữ liệu: label sai, missing, duplicate, leakage hoặc distribution shift?
3. Pipeline: preprocessing train/serve có nhất quán không?
4. Sanity check: model có overfit được một batch nhỏ không?
5. Baseline: model có thực sự hơn dummy/simple model không?
6. Optimization: loss, learning rate và gradient có ổn không?
7. Generalization: dữ liệu, capacity, regularization và augmentation.
8. Error slices: model thất bại ở nhóm nào và vì sao?
9. Tuning: chỉ tune sau khi các bước trước đã hợp lý.

Không bắt đầu bằng tăng số layer hoặc chạy search lớn.

## 10. Bản đồ đầu ra và điều chỉnh thời gian

### 10.1. Bản đồ đầu ra

| Mốc | Đầu ra |
|---|---|
| Tuần 1–4 | Bốn artifact nền tảng và ML Problem Canvas |
| Tuần 5–9 | Bộ lab ML cổ điển, pipeline template và experiment reports |
| Tuần 10 | Project 1 v1.0 |
| Tuần 11–15 | TensorFlow/Keras core labs và mini-project tabular comparison |
| Tuần 16–19 | CNN, transfer learning, time-series và NLP/Transformer artifacts |
| Tuần 20 | Project 2 candidate v0.1 |
| Tuần 21–23 | Production repository, export, service/container, benchmark và monitoring prototype |
| Tuần 24 | Capstone v1.0, model card, demo và retrospective |

### 10.2. Điều chỉnh theo quỹ thời gian

| Quỹ thời gian | Cách học | Tổng thời gian dự kiến |
|---|---|---:|
| 5–6 giờ/tuần | Mỗi tuần kế hoạch kéo dài khoảng 1,5 tuần; giữ đủ project | 34–36 tuần |
| 10–12 giờ/tuần | Theo đúng lộ trình | 24 tuần |
| 15–18 giờ/tuần | Gộp một số tuần nền tảng liền kề; không bỏ project/checkpoint | 16–18 tuần |

Nếu cần rút ngắn, giảm số thuật toán và số nhánh chuyên sâu; không bỏ baseline, evaluation, leakage checks, error analysis, testing và deployment.

## 11. Hướng chuyên môn sau 24 tuần

### Data Scientist

- Statistics, causal inference và A/B testing.
- Feature engineering, calibration, uncertainty và interpretability.
- Thêm dự án tabular/time-series có business framing mạnh.

### ML Engineer

- `tf.data` profiling, model optimization và distributed training.
- CI/CD/CT, data validation, registry và orchestration.
- TensorFlow Serving, TFX hoặc managed ML platform.

### Computer Vision Engineer

- Detection, segmentation, IoU/mAP và annotation quality.
- Compression, quantization và edge deployment.

### NLP Engineer

- Transformer, pretrained models, retrieval, evaluation và safety.
- Data curation, latency, context length và human evaluation.

## 12. Tài liệu chính thức theo thứ tự học

1. [NumPy documentation](https://numpy.org/doc/)
2. [pandas documentation](https://pandas.pydata.org/docs/)
3. [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html)
4. [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
5. [TensorFlow tutorials](https://www.tensorflow.org/tutorials)
6. [Keras developer guides](https://keras.io/guides/)
7. [Keras training and evaluation](https://keras.io/guides/training_with_built_in_methods/)
8. [Keras Functional API](https://keras.io/guides/functional_api/)
9. [TensorFlow tf.data guide](https://www.tensorflow.org/guide/data)
10. [TensorBoard getting started](https://www.tensorflow.org/tensorboard/get_started)
11. [Keras transfer learning and fine-tuning](https://keras.io/guides/transfer_learning/)
12. [Keras serialization and saving](https://keras.io/guides/serialization_and_saving/)
13. [TensorFlow TFX](https://www.tensorflow.org/tfx)

Tài liệu chi tiết từng tuần phải chỉ rõ phần bắt buộc đọc, phần mở rộng và mục tiêu sau khi đọc; không giao toàn bộ một tài liệu dài mà không có phạm vi.

## 13. Definition of Done của toàn lộ trình

Lộ trình hoàn thành khi người học có thể độc lập:

- [ ] Chuyển vấn đề thực tế thành prediction contract, metric, baseline và constraint.
- [ ] Chọn split phù hợp cho tabular, image, text hoặc time series.
- [ ] Xây scikit-learn pipeline không leakage và có cross-validation đúng.
- [ ] So sánh ML cổ điển bằng protocol công bằng và phân tích lỗi theo slice.
- [ ] Xây Keras model bằng Sequential/Functional API và giải thích shape/output/loss.
- [ ] Xây, train và debug MLP, CNN và một sequence model.
- [ ] Dùng `tf.data`, callback, checkpoint, TensorBoard và custom training loop.
- [ ] Dùng transfer learning và fine-tune an toàn.
- [ ] Tái tạo experiment từ data version, config, seed và environment.
- [ ] Lưu/export/load model và kiểm tra prediction parity.
- [ ] Cung cấp batch inference, API hoặc TensorFlow Serving container.
- [ ] Viết unit/integration/smoke test, model card, monitoring và rollback plan.
- [ ] Bàn giao Project 1 và Capstone đạt tối thiểu 75/100, không vi phạm critical gate.

## 14. Kế hoạch khởi động trong bảy ngày đầu

### Ngày 1

- Cài môi trường, tạo Git repository và chạy một TensorFlow quickstart.
- Ghi phiên bản Python, TensorFlow, Keras và dependency.

### Ngày 2

- Ôn scalar, vector, matrix, shape và broadcasting.
- Hoàn thành mười phép toán NumPy không dùng vòng lặp.

### Ngày 3

- Nạp và kiểm tra một CSV.
- Viết báo cáo ngắn về missing value, duplicate, dtype và range.

### Ngày 4

- Viết `clean_data()` và unit test.
- Benchmark loop so với vectorization.

### Ngày 5

- Thực hiện EDA có mục tiêu và tạo hai biểu đồ có kết luận.
- Ghi rủi ro data quality và leakage ban đầu.

### Ngày 6

- Chuyển logic làm sạch khỏi notebook sang `src/`.
- Chạy test từ command line trong process mới.

### Ngày 7

- Viết learning log một trang.
- Commit code và lập backlog cho tuần 2.

Mục tiêu tuần đầu là hình thành vòng lặp: hiểu dữ liệu → cài đặt → kiểm tra → giải thích → ghi chép.
