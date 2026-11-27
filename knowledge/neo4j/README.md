# Neo4j Knowledge Graph

## Mục tiêu

Module này cung cấp kết nối Neo4j, khởi tạo schema idempotent, truy vấn full-text và KnowledgeGraphAgent theo hợp đồng AgentResult.

**Không có dữ liệu y khoa mẫu được tự tạo.** schema.cypher chỉ tạo uniqueness constraints và full-text index; không xóa graph, không tạo node/fact, không thay thế graph cũ hoặc file DB_diabet.cypher nếu nhóm đang giữ ở nơi khác.

## Cấu hình

Sao chép `.env.example` thành `.env` ở thư mục gốc repository. `Neo4jSettings` tự nạp file này khi module được import. Điền `NEO4J_URI` bằng Bolt URI trong Connect panel của Aura, `NEO4J_USERNAME` (thường là `neo4j`), `NEO4J_PASSWORD` của instance và `NEO4J_DATABASE` (thường là `neo4j`). Instance ID không phải username hoặc database name. Không commit `.env`.

Cài dependency:

```bash
pip install -e ".[neo4j]"
```

## Khởi tạo schema

```bash
python scripts/init_neo4j.py
```

Lệnh kiểm tra kết nối rồi áp dụng schema.cypher. Các lệnh dùng IF NOT EXISTS, có thể chạy lại. Không chạy thao tác reset/xóa dữ liệu.

## Mô hình dữ liệu dự kiến

Node labels: Disease, Symptom, RiskFactor, DiagnosticTest, Complication, Treatment, Medication, Nutrition, MedicalDocument, DocumentChunk, KnowledgeSource.

Các node cần có id ổn định. Facts phải có liên kết provenance, ưu tiên:
- Entity/fact -[:SUPPORTED_BY]-> MedicalDocument
- MedicalDocument -[:FROM_SOURCE]-> KnowledgeSource
- DocumentChunk -[:PART_OF]-> MedicalDocument

Các quan hệ miền như HAS_SYMPTOM, HAS_RISK_FACTOR, HAS_COMPLICATION, DIAGNOSED_BY, HAS_TREATMENT, RELATED_TO cần được tạo từ tài liệu có nguồn, không suy đoán từ tên quan hệ.

## Truy vấn và Agent

- queries.cypher: truy vấn kiểm tra graph và provenance.
- Neo4jGraphRepository.health_check(): kiểm tra kết nối.
- Neo4jGraphRepository.search(query, limit=8): tìm node bằng full-text index, lấy node liên quan và nguồn hỗ trợ.
- KnowledgeGraphAgent.retrieve(query): trả về AgentResult, giữ nguyên nguồn và đánh dấu warning nếu không có nguồn hoặc không tìm thấy bằng chứng.

Đây là graph retrieval cơ bản, chưa phải Hybrid RAG: vector retrieval, chunk embedding, reranking và answer synthesis sẽ được tích hợp ở giai đoạn sau. Kết quả truy xuất là bằng chứng tham khảo, không phải chẩn đoán hay chỉ định điều trị.
