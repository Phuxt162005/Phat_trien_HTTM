# Document ingestion

## Phạm vi hiện có

Module `src/project01/knowledge/ingestion.py` hỗ trợ ingest file cục bộ:
- PDF: trích xuất text theo từng trang bằng pypdf, giữ số trang.
- TXT/Markdown: đọc UTF-8.
- HTML: loại bỏ script/style rồi trích xuất text bằng BeautifulSoup.
- Chuẩn hóa whitespace, chia chunk có overlap.
- Tạo ID ổn định theo nội dung, lưu SHA-256, tên nguồn, publisher, license, URL và đường dẫn file.
- Xuất JSONL để kiểm tra trước khi nạp vào graph.
- Tùy chọn upsert KnowledgeSource, MedicalDocument và DocumentChunk vào Neo4j bằng MERGE có tham số.

Ingestion **không tự tạo facts y khoa**, không sửa file nguồn và không tự khẳng định license. Người vận hành phải cung cấp metadata nguồn chính xác và xác nhận quyền sử dụng tài liệu.

## Cài đặt

```bash
pip install -e ".[rag,neo4j]"
```

## Chạy local trước (khuyến nghị)

```bash
python scripts/ingest_documents.py data/sources --source-name "WHO" --publisher "World Health Organization" --license "review-required" --output data/processed/who_chunks.jsonl
```

Thay đường dẫn và metadata bằng tài liệu/nguồn thực tế. Chỉ ghi license khi đã xác minh; nếu chưa, dùng nhãn nội bộ `review-required`.

## Upsert metadata và chunks vào Neo4j

Sau khi kiểm tra JSONL và cấu hình Neo4j trong `.env`:

```bash
python scripts/ingest_documents.py data/sources --source-name "WHO" --publisher "World Health Organization" --license "review-required" --neo4j
```

Các thao tác dùng MERGE với ID ổn định, có thể chạy lại để tránh tạo node trùng. Tên quan hệ hiện tại:
- `(Document)-[:FROM_SOURCE]->(KnowledgeSource)`
- `(Document)-[:HAS_CHUNK]->(DocumentChunk)`
- `(DocumentChunk)-[:PART_OF]->(Document)`

## Giới hạn hiện tại

- Chưa tải nội dung từ URL; URL chỉ là metadata nguồn do người vận hành khai báo.
- PDF scan cần OCR riêng; pypdf chỉ trích xuất text có sẵn.
- Chưa tạo embedding/vector index, chưa có vector retrieval, hybrid fusion hoặc reranking.
- Page range của chunk là phạm vi trang buffer; với tài liệu dài, chunk boundary được ưu tiên theo kích thước ký tự và overlap.
- Cần kiểm tra bản quyền, phiên bản, ngày xuất bản, ngôn ngữ và tính cập nhật trước khi dùng tài liệu làm bằng chứng.
