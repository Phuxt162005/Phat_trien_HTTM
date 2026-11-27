# Project01 – Hybrid Multi-Agent System for Diabetes Decision Support

## Trạng thái hiện tại

Đây là bản tái cấu trúc giai đoạn 1. Notebook gốc được giữ nguyên trong `legacy/` để đối chiếu và chuyển chức năng dần, không xóa kết quả thí nghiệm cũ.

### Đã tạo trong bản khởi đầu
- Cấu trúc project Python độc lập với notebook.
- Cấu hình riêng cho ba dataset gốc: `diabetes01`, `diabetes03`, `diabetes04`.
- Dataset loader không ghi đè hoặc sửa file CSV gốc.
- Hợp đồng `AgentMessage`, `AgentResult`, `BaseAgent`.
- `PatientDataAgent` kiểm tra chất lượng dữ liệu dạng bảng.
- `RuleBasedAgent` thực hiện rà soát kỹ thuật dữ liệu.
- `CoordinatorAgent` chạy audit ban đầu và ghi execution trace.
- Script `scripts/audit_datasets.py` xuất báo cáo JSON.
- Bộ test cơ bản cho hợp đồng Agent và kiểm tra dữ liệu.

### Chưa tích hợp trong bản khởi đầu
Các phần ML/DL, mô hình đã huấn luyện, Neo4j, Cypher, ingestion PDF/web, Hybrid RAG, chatbot và giao diện chưa được chuyển hoàn chỉnh. Chúng sẽ được chuyển theo từng module sau khi đối chiếu logic và artifact gốc; không giả định rằng chỉ tạo thư mục là đã hoàn thành chức năng.

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

## 2. Đặt ba dataset gốc

Đặt đúng các file CSV hiện có vào `data/raw/`:

```text
data/raw/diabetes01.csv
data/raw/diabetes03.csv
data/raw/diabetes04.csv
```

Không đổi tên cột, nhãn hoặc nội dung file. Nếu file của bạn đang có tên/đường dẫn khác, hãy cung cấp để cập nhật cấu hình có kiểm soát.

## 3. Chạy audit dữ liệu

```bash
python scripts/audit_datasets.py
```

Báo cáo được ghi vào `data/manifests/original_dataset_audit.json`.

## 4. Chạy test

```bash
pytest
```

## 5. Cấu hình bí mật

Sao chép `.env.example` thành `.env` rồi điền thông tin Neo4j/LLM trên máy cá nhân khi cần. Không commit `.env`.

## Nguyên tắc dữ liệu
- Ba dataset gốc được giữ riêng biệt.
- Không tự động gộp `diabetes01` và `diabetes03`; cần audit trùng lặp trước khi diễn giải kết quả tổng hợp.
- Dataset bên ngoài phải có nguồn, giấy phép, định nghĩa target và metadata.
- Không trộn dữ liệu cá nhân cấp bệnh nhân với số liệu dịch tễ tổng hợp trong cùng một tập huấn luyện nếu chưa có thiết kế phương pháp phù hợp.
- Hệ thống là prototype nghiên cứu/giáo dục, không phải công cụ chẩn đoán hoặc kê đơn lâm sàng.

## Tài liệu tham chiếu
- `legacy/project01_diabetes_ver_02_original.ipynb`: notebook mới nhất được cung cấp trong cuộc trò chuyện, giữ nguyên làm bản tham chiếu.
- `project_architecture_multiagent_diabet_v1.pdf`: đặc tả kiến trúc Multi-Agent của dự án (được cung cấp riêng, không sao chép vào source tree).
