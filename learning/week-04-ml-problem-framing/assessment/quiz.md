# Quiz Tuần 4 — 20 câu / 25 phút

Không dùng tài liệu. Mỗi câu 1 điểm. Với câu trắc nghiệm, chọn **một** đáp án tốt nhất.

## Phần A — Nền tảng

**1.** Phát biểu nào đúng nhất?

A. Mọi hệ AI đều học từ dữ liệu.  
B. Deep Learning là tập con của ML, ML là tập con của AI.  
C. TensorFlow là một họ bài toán ML.  
D. Regression là unsupervised learning.

**2.** Dự đoán thời gian giao hàng theo phút từ dữ liệu có nhãn thuộc loại nào?

A. Classification  
B. Clustering  
C. Regression  
D. Generation

**3.** Chia khách hàng chưa có nhãn thành các nhóm hành vi ban đầu là:

A. Clustering  
B. Binary classification  
C. Regression  
D. Rule engine

**4.** Trường hợp nào phù hợp nhất với rule engine thay vì ML?

A. Dự báo demand có mùa vụ.  
B. Tính phí theo bảng luật đầy đủ và ổn định.  
C. Nhận diện ảnh lỗi sản phẩm.  
D. Xếp hạng tìm kiếm theo relevance.

**5.** Thiếu yếu tố nào khiến “dự đoán khách hàng không hài lòng” chưa sẵn sàng nhất?

A. GPU  
B. TensorFlow  
C. Decision/action và owner  
D. Mạng nơ-ron sâu

## Phần B — Contract và leakage

**6.** Prediction unit mô tả:

A. Số cột trong bảng.  
B. Thực thể/sự kiện nhận một prediction.  
C. Số epoch.  
D. Máy chủ inference.

**7.** Với ticket được triage lúc tạo, feature nào là leakage rõ nhất?

A. `channel`  
B. `message_length`  
C. `resolution_hours`  
D. `customer_tier`

**8.** Target window cần được khóa chủ yếu để:

A. tăng GPU utilization;  
B. xác định khi nào outcome được tính và label hoàn tất;  
C. chọn optimizer;  
D. giảm số feature.

**9.** Một proxy label nên được dùng thế nào?

A. Coi là outcome thật mà không cần ghi chú.  
B. Chỉ dùng nếu accuracy đạt 100%.  
C. Ghi quan hệ với outcome, bias/coverage và kế hoạch kiểm chứng.  
D. Không bao giờ dùng proxy.

**10.** Dữ liệu support có thứ tự thời gian. Split mặc định hợp lý nhất là:

A. random toàn bộ rồi chia;  
B. sort thời gian, train quá khứ và test tương lai;  
C. dùng test để train;  
D. chỉ giữ positive rows.

## Phần C — Metric, baseline và vận hành

**11.** Class positive chiếm 4%. Model luôn dự đoán negative có accuracy 96%. Kết luận đúng là:

A. model production-ready;  
B. cần xem recall/precision/cost và so baseline;  
C. không cần confusion matrix;  
D. threshold chắc chắn tối ưu.

**12.** Precision trả lời câu nào?

A. Trong positive thật, bắt được bao nhiêu?  
B. Trong predicted positive, đúng bao nhiêu?  
C. Tổng dự đoán đúng là bao nhiêu?  
D. Mỗi inference mất bao lâu?

**13.** False negative có chi phí cao hơn false positive rất nhiều. Nếu các yếu tố khác giữ nguyên, policy thường ưu tiên:

A. recall cao hơn;  
B. recall bằng 0;  
C. ít feature hơn bằng mọi giá;  
D. random split.

**14.** Threshold phải được chọn bằng:

A. test set sau khi xem mọi kết quả;  
B. validation set theo metric/cost đã khóa;  
C. training loss duy nhất;  
D. mặc định 0.5 trong mọi case.

**15.** Cặp baseline tốt nhất cho support triage là:

A. model sâu và model sâu hơn;  
B. majority dummy và heuristic hiện hành;  
C. test labels và validation labels;  
D. không cần baseline.

**16.** Trường hợp nào phù hợp với batch nhất?

A. Authorization phải trả trong 100 ms.  
B. Replenishment plan cập nhật mỗi đêm.  
C. Emergency alert tức thời.  
D. Autocomplete từng phím.

**17.** Human-in-the-loop không tự giải quyết rủi ro nào?

A. Human capacity và automation bias.  
B. Không có owner.  
C. Leakage trong training data.  
D. Tất cả các ý trên.

**18.** Evaluation slice nên được chọn dựa trên:

A. chỉ cột có nhiều category;  
B. rủi ro, nhóm người dùng và quy trình vận hành;  
C. cột có tên ngắn;  
D. metric toàn cục cao nhất.

## Phần D — Trả lời ngắn

**19.** Viết công thức total cost khi `FP_cost = 8`, `FN_cost = 300`; tính cost cho FP=12, FN=3.

**20.** Trong tối đa ba câu, giải thích vì sao “test chỉ mở một lần” và phải làm gì nếu đã dùng test để đổi threshold.

## Mức đạt

- 18–20: xuất sắc.
- 14–17: đạt.
- 10–13: cần ôn lại metric/contract/leakage.
- <10: làm lại notebook và bài 5–8 trước khi tiếp tục.

