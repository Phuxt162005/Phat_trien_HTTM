# Project01 – Hybrid Multi-Agent System for Diabetes Decision Support

## Trạng thái hiện tại

Đây là bản tái cấu trúc giai đoạn 1. Notebook gốc được giữ trong `legacy/` để đối chiếu và chuyển chức năng dần; không xóa kết quả thí nghiệm cũ.

### Ba dataset chính thức

| Dataset | Vai trò | Target | Mục đích |
|---|---|---|---|
| `diabetes01.csv` | Original supervised | `diabetes` | Phân loại nhị phân |
| `diabetes02.csv` | External unlabeled | Không có | EDA/khảo sát dữ liệu; không dùng làm nhãn huấn luyện phân loại |
| `diabetes04.csv` | Original supervised | `Diabetes_012` | Phân loại đa lớp |

**Lưu ý quan trọng:** `diabetes02.csv` không có cột nhãn bệnh tiểu đường. Không tự suy diễn hoặc tạo nhãn từ glucose, HbA1c hay các chỉ số khác để đưa vào bài toán supervised learning. Chỉ sử dụng dataset này cho các mục tiêu phù hợp với dữ liệu không nhãn cho đến khi có nhãn hợp lệ và nguồn xác thực.

Đặt ba file CSV tại `data/raw/` trên máy local. Các file dữ liệu thô được loại khỏi Git theo `.gitignore`; không sửa nội dung CSV gốc. Dataset cũ `diabetes03` không còn thuộc bộ ba dataset chính thức của dự án.

## Các thành phần đã có

- Cấu trúc project Python độc lập với notebook.
- Cấu hình riêng cho ba dataset chính thức.
- Dataset loader hỗ trợ cả dữ liệu có target và không có target.
- Hợp đồng `AgentMessage`, `AgentResult`, `BaseAgent`.
- `PatientDataAgent` kiểm tra missing values, duplicate rows, duplicate columns, constant columns, infinite values và target distribution khi có target.
- `RuleBasedAgent` phân loại phát hiện thành passed/warning/failed; không tự động xóa bản ghi trùng.
- `CoordinatorAgent` điều phối audit và ghi execution trace.
- Script `scripts/audit_datasets.py` audit ba dataset và xuất JSON.

## Pipeline ML/DL

Notebook gốc được giữ nguyên trong \`legacy/\`. Phần supervised ML đã được tách thành các module Python:

- \`src/project01/ml/preprocessing.py\`: chia train/validation/test theo stratify; imputer, scaler và one-hot encoder chỉ fit trên train.
- \`src/project01/ml/models.py\`: Logistic Regression, Random Forest và Soft Voting.
- \`src/project01/ml/evaluation.py\`: Accuracy, Macro Precision/Recall/F1, ROC-AUC, confusion matrix và classification report.
- \`src/project01/ml/pipeline.py\`: điều phối thí nghiệm và lưu model artifacts/metrics.
- \`src/project01/ml/deep_learning.py\`: PyTorch MLP tùy chọn, class weights, ReduceLROnPlateau, checkpoint theo validation và Early Stopping.
- \`src/project01/agents/ml_agents.py\`: ModelAAgent (Logistic Regression), ModelBAgent (Random Forest), ConsensusAgent (Soft Voting), DeepLearningAgent (PyTorch MLP tùy chọn), ExplanationAgent, EvaluationAgent, SafetyReviewAgent và MLCoordinatorAgent.
- \`scripts/train_ml.py\`: điểm chạy luồng Multi-Agent ML/DL; ghi agent log và message log vào báo cáo tổng hợp.

Chỉ \`diabetes01\` và \`diabetes04\` được dùng cho supervised classification. \`diabetes02\` bị từ chối ở bước preprocessing nếu không có target hợp lệ.

Chạy baseline ML:

\`\`\`bash
python scripts/train_ml.py
\`\`\`

Chạy cả PyTorch MLP (cài \`pip install -e ".[ml]"\` trước):

\`\`\`bash
python scripts/train_ml.py --include-deep-learning
\`\`\`

Chạy một dataset:

\`\`\`bash
python scripts/train_ml.py --datasets diabetes01
\`\`\`

Artifacts được lưu trong \`models/<dataset>/\`, gồm model \`.joblib\`, preprocessing pipeline và \`metrics.json\`. Khi bật Deep Learning, state dict và preprocessing riêng của MLP cũng được lưu.


### Xử lý mất cân bằng lớp trong ML

- Với Random Forest trên bài toán đa lớp, chỉ tập train và tập train+validation được áp dụng random oversampling một phần; tập validation và test giữ nguyên phân phối gốc.
- Ngưỡng mặc định đưa lớp thiểu số lên tối thiểu 20% số mẫu của lớp đa số, tránh nhân bản toàn bộ lớp thiểu số đến mức cân bằng tuyệt đối.
- Báo cáo metrics lưu phân bố lớp trước và sau oversampling. Cần so sánh Macro F1, Recall từng lớp và confusion matrix; không đánh giá mô hình chỉ bằng Accuracy.

### So sánh chiến lược xử lý mất cân bằng

Không chọn mức oversampling dựa trên kết quả tập test. Với `diabetes04`, chạy thí nghiệm so sánh các phương án Random Forest sau trên cùng một lần chia dữ liệu:

- Baseline không dùng class weighting/oversampling.
- `class_weight=balanced_subsample`.
- Random oversampling với ngưỡng lớp thiểu số lần lượt 20%, 50% và 100% lớp đa số.

Chạy:

```bash
python scripts/compare_imbalance.py --dataset diabetes04 --rf-estimators 120
```

Các phương án được lựa chọn bằng Macro F1 trên validation; Recall của lớp thiểu số chỉ dùng để phá hòa. Sau khi chọn xong, mô hình được huấn luyện lại trên train+validation và tập test chỉ được đánh giá một lần. Báo cáo được lưu tại `models/diabetes04/imbalance_comparison.json`, mô hình được chọn tại `models/diabetes04/random_forest_selected_imbalance.joblib`. Đây là thí nghiệm riêng, không tự động ghi đè kết quả pipeline Multi-Agent hiện tại.

### Điều chỉnh điểm số dự đoán cho lớp thiểu số (multiclass)

Thí nghiệm này điều chỉnh điểm xác suất của lớp thiểu số bằng một hệ số nhân, sau đó vẫn dùng `argmax` trên **toàn bộ các lớp**. Vì vậy, lớp 0, lớp 1 và lớp 2 tiếp tục cạnh tranh với nhau; không dùng quy tắc one-vs-rest gán mọi mẫu không thuộc lớp thiểu số thành lớp 0.

- Hệ số được chọn trên validation, không dùng test để chọn.
- Ưu tiên F1 lớp thiểu số trong nhóm ứng viên đáp ứng đồng thời hai điều kiện bảo toàn: Macro F1 không giảm quá 0.03 so với baseline và Recall của từng lớp còn lại không thấp hơn 75% baseline.
- Nếu không có ứng viên đáp ứng điều kiện, fallback chọn theo Macro F1 trước, sau đó mới xét F1 lớp thiểu số.
- Script hiện tại chỉ đánh giá trên validation và **không đọc tập test**. Tập test đã được xem trong các lần thử trước nên không còn được xem là holdout hoàn toàn độc lập; không tiếp tục dùng nó để lựa chọn hệ số.

Chạy:

```bash
python scripts/tune_minority_threshold.py --dataset diabetes04 --rf-estimators 120
```

Báo cáo validation lưu tại `models/diabetes04/minority_threshold_tuning.json`; mô hình Random Forest đã fit trên train+validation lưu tại `models/diabetes04/random_forest_minority_factor.joblib`. Báo cáo ghi rõ hệ số được chọn, metrics từng lớp, các ứng viên và trạng thái không đánh giá test. Khi triển khai dự đoán, cần áp dụng đúng hệ số đã chọn lên cột xác suất tương ứng với nhãn lớp thiểu số trước khi lấy `argmax`; không gọi `predict()` trực tiếp nếu muốn dùng quy tắc đã tuning. Hệ số không cải thiện calibration và không chứng minh giá trị lâm sàng.

### Phân tích đặc trưng và lỗi phân loại đa lớp

Trước khi tiếp tục điều chỉnh hệ số lớp thiểu số, cần kiểm tra đặc trưng nào thực sự đóng góp vào dự đoán và các lớp đang bị nhầm lẫn ở đâu. Chạy:

```bash
python scripts/diagnose_features.py --dataset diabetes04 --rf-estimators 120 --validation-sample-size 10000 --permutation-repeats 3
```

Báo cáo được lưu tại `models/diabetes04/feature_diagnostics.json`, gồm:

- Permutation importance theo Macro F1 trên một mẫu validation.
- Các cặp nhãn thực tế/dự đoán bị nhầm nhiều nhất trong mẫu validation.
- Thống kê numeric theo lớp và standardized mean difference giữa từng cặp lớp trên train.
- Phân bố các giá trị categorical phổ biến theo lớp trên train.

Quy trình chỉ fit mô hình trên train, tính permutation importance và confusion pairs trên validation; không truy cập test. Permutation importance phản ánh mức đóng góp dự đoán trên mẫu validation, không phải quan hệ nhân quả. Standardized mean difference chỉ mô tả mức tách biệt biên của đặc trưng numeric, không chứng minh ý nghĩa lâm sàng. Các kết quả này dùng để định hướng thí nghiệm tiếp theo, không tự động loại bỏ đặc trưng.

### Phân loại kiểu đặc trưng và benchmark mô hình đa lớp

Báo cáo feature diagnostics phân loại các cột đã mã hóa số theo ngữ nghĩa của diabetes04:
- Binary: biến nhị phân như `HighBP`, `HighChol`, `Smoker`, `Sex`.
- Ordinal: `Age`, `Education`, `Income`, `GenHlth`.
- Continuous/count: `BMI`, `MentHlth`, `PhysHlth`.
- Các cột dạng chuỗi được xem là categorical; với dataset khác, biến số có đúng hai giá trị 0/1 được nhận diện là binary, các biến số khác mặc định continuous.

Báo cáo bổ sung `feature_types` và `typed_feature_profiles_train`, giúp không diễn giải mọi cột số như biến liên tục. Quy tắc trên dựa trên định nghĩa trường dữ liệu diabetes04; cần kiểm tra metadata nếu thay dataset hoặc schema.

Chạy benchmark so sánh Random Forest, Extra Trees và HistGradientBoosting:

```bash
python scripts/benchmark_models.py --dataset diabetes04 --estimators 120 --n-jobs 1
```

Benchmark thử các cấu hình baseline, class weight (với mô hình hỗ trợ) và random oversampling 20%/50%. Có 11 ứng viên được so sánh. Tiêu chí chọn là Macro F1 trên validation, sau đó dùng F1 và Recall lớp thiểu số để phá hòa. Mô hình thắng cuộc được fit lại trên train+validation. **Không truy cập test và không tạo test metrics**, vì tập test đã được xem trong các thí nghiệm trước.

Kết quả:
- `models/diabetes04/model_benchmark.json`
- `models/diabetes04/selected_benchmark_model.joblib`

Đây là benchmark phát triển mô hình, không phải đánh giá lâm sàng. Không chọn mô hình chỉ theo Accuracy hoặc Minority Recall riêng lẻ; phải xem đầy đủ precision/recall/F1 của từng lớp và confusion matrix.

## Những phần chưa tích hợp hoàn chỉnh

Neo4j/Cypher, ingestion PDF/web, Hybrid RAG, chatbot và giao diện chưa được chuyển hoàn chỉnh. Chúng sẽ được triển khai theo từng module sau khi đối chiếu notebook, tài liệu và artifact gốc; không xem việc tạo thư mục là hoàn thành chức năng.

## 1. Chuẩn bị môi trường

Python 3.10 trở lên được khuyến nghị.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[test]"
```

## 2. Đặt dataset

Đặt đúng các file CSV vào:

```text
data/raw/diabetes01.csv
data/raw/diabetes02.csv
data/raw/diabetes04.csv
```

Không đổi tên cột, nhãn hoặc nội dung file. Nếu file đang có tên/đường dẫn khác, cập nhật cấu hình có kiểm soát.

## 3. Chạy audit dữ liệu

```bash
python scripts/audit_datasets.py
```

Báo cáo được ghi vào `data/manifests/original_dataset_audit.json`. Trạng thái warning do missing values hoặc duplicate rows không làm audit thất bại; các vấn đề cấu trúc nghiêm trọng như duplicate columns/infinite values sẽ được đánh dấu failed.

## 4. Chạy test

```bash
pytest
```

## 5. Cấu hình bí mật

Sao chép `.env.example` thành `.env` rồi điền thông tin Neo4j/LLM trên máy cá nhân khi cần. Không commit `.env`.

## Nguyên tắc dữ liệu và an toàn

- Ba dataset chính thức được giữ riêng biệt, không tự động gộp.
- Không tự động xóa duplicate rows; cần đánh giá trước khi xử lý.
- Dataset bên ngoài phải có nguồn, giấy phép, định nghĩa target và metadata.
- Không trộn dữ liệu cá nhân cấp bệnh nhân với số liệu dịch tễ tổng hợp trong cùng tập huấn luyện nếu chưa có thiết kế phương pháp phù hợp.
- Hệ thống là prototype nghiên cứu/giáo dục, không phải công cụ chẩn đoán hoặc kê đơn lâm sàng.

## Tài liệu tham chiếu

- `legacy/project01_diabetes_ver_02_original.ipynb`: notebook được giữ làm bản tham chiếu.
- `project_architecture_multiagent_diabet_v1.pdf`: đặc tả kiến trúc Multi-Agent được cung cấp riêng.
