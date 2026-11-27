# Vinmec source collection

Mở [SOURCE_CATALOG.md](SOURCE_CATALOG.md) để xem 53 mã tài liệu Vinmec (có URL trùng được ghi chú) đã phân loại theo chủ đề.

## Trạng thái
- 53 mã tài liệu/liên kết tiếng Việt được lập danh mục; một số mã trùng URL, xem ghi chú khử trùng trong SOURCE_CATALOG.md.
- Chưa tải toàn bộ bài viết thành PDF; đa số là bài HTML trên website.
- Chưa ingest danh mục này vào Neo4j.
- Một số bài có thể trùng nội dung hoặc trích dẫn nguồn thứ cấp; cần kiểm tra trước khi ingest.

## Quy trình đưa bài vào hệ thống
1. Mở URL trong danh mục và kiểm tra trang còn hoạt động.
2. Ghi nhận title, author/reviewer, updated date, canonical URL.
3. Kiểm tra quyền tái sử dụng và quyền text/data mining. Không sao chép toàn văn bài viết có bản quyền vào Git.
4. Nếu được phép lưu nội dung, lưu bản văn bản/PDF được cấp phép vào thư mục nguồn phù hợp.
5. Chạy `scripts/ingest_documents.py` riêng cho từng tài liệu để gán đúng `--source-url`, `--source-name`, `--publisher`, `--license`.
6. Kiểm tra `chunk_count` và `neo4j_upserted_chunks`; sau đó mới chạy embedding.

Ví dụ với một tài liệu đã được phép lưu ở local:

```powershell
python scripts/ingest_documents.py data/sources/Vinmec/<file-da-duoc-phep>.pdf --source-name "Vinmec" --publisher "Vinmec Healthcare System" --license "Use subject to Vinmec copyright terms; permission verified" --source-url "https://www.vinmec.com/vie/bai-viet/<canonical-slug>" --output data/processed/vinmec_<topic>_chunks.jsonl --neo4j
```

Không đặt `--license` là CC-BY hoặc public domain nếu chưa có bằng chứng.


## Kiểm tra liên kết và metadata trước ingestion

Đã có script kiểm tra danh mục URL theo hướng metadata-only. Script:
- Khử trùng URL trong bảng Markdown.
- Chỉ truy cập HTTPS trên domain `vinmec.com`.
- Ghi HTTP status, URL sau redirect, canonical URL, title và metadata ngày đăng/cập nhật nếu trang công khai chúng.
- Không lưu nội dung bài viết, không tải PDF và không ghi dữ liệu vào Neo4j.
- Tôn trọng lỗi 403/429; không tự động vượt cơ chế chặn truy cập.

Chạy toàn bộ danh mục:

```powershell
python scripts/audit_vinmec_sources.py
```

Chạy thử 5 URL đầu tiên:

```powershell
python scripts/audit_vinmec_sources.py --limit 5 --delay 1
```

Báo cáo mặc định được tạo tại `data/processed/vinmec_source_audit.json`. Hãy kiểm tra `status_counts`, `canonical_url` và các trang bị redirect/blocked trước khi chuẩn bị ingestion.

**Quan trọng:** HTTP 200 không đồng nghĩa với quyền sao chép hoặc text mining. Chỉ ingest nội dung khi đã xác minh điều khoản/quyền sử dụng. Không dùng metadata audit thay cho kiểm tra chuyên môn của tài liệu.
