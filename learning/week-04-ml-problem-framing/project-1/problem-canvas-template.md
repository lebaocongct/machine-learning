# ML Problem Canvas — [Tên bài toán]

**Phiên bản:** 0.1  
**Ngày:** YYYY-MM-DD  
**Owner:**  
**Trạng thái:** Draft / Reviewed / Approved

## A. Mục tiêu và decision

- **Problem name (`problem_name`):**
- **Non-ML goal (`non_ml_goal`):** outcome/KPI cần cải thiện, không nhắc model.
- **Current process:**
- **Decision (`decision`):** ai quyết định gì, tại thời điểm nào?
- **Decision owner (`decision_owner`):**
- **Action nếu output positive/high:**
- **Action nếu output negative/low:**
- **Phương án non-ML:**
- **Vì sao ML có thể tạo thêm giá trị:**

## B. Prediction contract

- **Prediction unit (`prediction_unit`):**
- **Prediction time (`prediction_time`):**
- **Target (`target`):**
- **Target definition / label source:**
- **Target window (`target_window`):**
- **Task type (`task_type`):** classification / regression / clustering / ranking / generation
- **Model output (`model_output`):**
- **Deployment mode (`deployment_mode`):** batch / online / human_in_loop / hybrid
- **Latency/capacity:**

## C. Đo lường

- **Primary metric (`primary_metric`):**
- **Guardrail metric/constraint:**
- **Dummy + operational baseline (`baseline`):**
- **Success criterion (`success_criterion`):** metric + delta + dataset/split + thời hạn.
- **FP consequence/cost:**
- **FN consequence/cost:**
- **Trade-off 1:**
- **Trade-off 2:**

## D. Dữ liệu và rủi ro

- **Constraints (`constraints`):**
  -
- **Evaluation slices (`evaluation_slices`):**
  -
- **Leakage risks (`leakage_risks`):**
  -
- **Non-goals (`non_goals`):**
  -
- **Label/proxy limitations:**
- **Expected drift:**
- **Fallback/failure policy:**

## E. Protocol đánh giá

- Split strategy:
- Validation selections:
- Test opening rule:
- Slice report minimum sample size:
- Monitoring after launch:

## F. Quyết định dự án

- **Go / Conditional Go / No-Go:**
- **Bằng chứng:**
- **Giả định chưa kiểm chứng:**
- **Thí nghiệm hoặc dữ liệu tiếp theo:**
- **Người phê duyệt và ngày:**

