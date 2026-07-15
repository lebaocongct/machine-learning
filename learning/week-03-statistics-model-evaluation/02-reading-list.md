# Reading list Tuần 3

## Cách đọc

Không đọc hết mọi trang. Với mỗi nguồn, ghi lại:

```text
Khái niệm:
Ví dụ hoặc shape:
Giả định:
Một lỗi có thể mắc:
Một experiment để kiểm tra:
```

## Bắt buộc — khoảng 100 phút

### Thống kê và xác suất — 40 phút

1. [OpenIntro Statistics](https://www.openintro.org/book/os/)
   - Chapter 2: mean, spread, distribution.
   - Chapter 3: probability và conditional probability.
   - Chapter 5: sampling variability, confidence intervals.
2. [NumPy Random Sampling](https://numpy.org/doc/stable/reference/random/index.html)
   - Dùng `Generator`/`default_rng` và seed có chủ đích.

### Bootstrap — 15 phút

1. [SciPy `stats.bootstrap`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
   - Đọc quy trình resample, `confidence_level`, `method`, `rng`.
   - Lab vẫn yêu cầu tự cài percentile bootstrap trước khi dùng SciPy để đối chiếu.

### Split, leakage và model selection — 45 phút

1. [Scikit-Learn `train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
   - `test_size`, `random_state`, `shuffle`, `stratify`.
2. [Scikit-Learn Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
   - Đọc inconsistent preprocessing và data leakage.
3. [Underfitting vs. Overfitting](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html)
   - Quan sát train/validation behavior theo polynomial degree.
4. [Scikit-Learn `learning_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html)
   - `train_sizes`, cross-validation và scoring.

## Mở rộng

- [Seeing Theory](https://seeing-theory.brown.edu/) — trực quan probability/inference.
- [Scikit-Learn Cross-validation Guide](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Scikit-Learn PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- [Scikit-Learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [SciPy probability distributions](https://docs.scipy.org/doc/scipy/reference/stats.html)

## Câu hỏi định hướng

1. Seed làm experiment reproducible nhưng không giải quyết loại uncertainty nào?
2. Khi nào median phù hợp hơn mean?
3. Confidence interval 95% nên diễn giải thế nào?
4. Tại sao test preprocessing phải dùng parameters fit từ train?
5. Degree comparison table có được chứa test RMSE không?
6. Learning curve khác validation curve ở trục x nào?
7. Thử nhiều candidate có thể overfit validation ra sao?
8. Vì sao test score là estimate chứ không phải “performance thật” cố định?
