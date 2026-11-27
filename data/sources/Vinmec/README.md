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


## Trích xuất bài HTML Vinmec đã audit (local-only)

Pipeline mới đọc báo cáo audit, chỉ chọn dòng `status=ok` và `page_valid=true`, sau đó:
- Kiểm tra `robots.txt`; nếu bị chặn hoặc không đọc được robots.txt thì bỏ qua URL.
- Giới hạn domain HTTPS ở `vinmec.com`, có timeout, delay và giới hạn dung lượng phản hồi.
- Trích nội dung từ `article`, `main` hoặc vùng bài viết; loại bỏ các vùng điều hướng phổ biến.
- Ghi canonical URL, title, H1, ngày đăng/cập nhật, tác giả nếu trang công khai, SHA-256 và ghi chú quyền sử dụng.
- Khử trùng theo canonical URL và hash nội dung trong một lần chạy.
- Không ghi nội dung bài viết vào Git; `data/documents/` được thêm vào `.gitignore`.

Cài dependency HTML nếu chưa có:

```powershell
pip install -e ".[rag]"
```

**Chỉ chạy khi đã xác minh có quyền lưu trữ/xử lý nội dung theo điều khoản nguồn hoặc được cấp phép.** Cờ `--rights-note` bắt buộc để ghi lại căn cứ đã xác minh; không nhập nội dung xác nhận nếu chưa có căn cứ thực tế.

Chạy thử tối đa 2 URL sau khi xác minh quyền:

```powershell
python scripts/fetch_vinmec_articles.py --limit 2 --delay 2 --rights-note "Ghi rõ căn cứ cho phép lưu trữ và xử lý nội dung"
```

Chạy toàn bộ các ứng viên đủ điều kiện:

```powershell
python scripts/fetch_vinmec_articles.py --delay 2 --rights-note "Ghi rõ căn cứ cho phép lưu trữ và xử lý nội dung"
```

Kết quả nằm ở `data/documents/Vinmec/`; mỗi bài có `content.md` và `metadata.json`. Báo cáo nằm tại `data/documents/Vinmec/extraction_report.json`. Các file này ở local, không được commit lên Git.

Sau khi nội dung được phép lưu và đã kiểm tra, dùng pipeline ingestion hiện có:

```powershell
python scripts/ingest_documents.py data/documents/Vinmec --source-name "Vinmec" --publisher "Vinmec" --license "See per-document metadata; permission verified" --output data/processed/vinmec_chunks.jsonl
```

Lệnh trên chỉ tạo JSONL chunk, **không ghi Neo4j**. Chỉ thêm `--neo4j` sau khi đã kiểm tra nội dung và metadata. Với dữ liệu nhiều nguồn/quyền khác nhau, nên ingest theo từng tài liệu hoặc nhóm có cùng điều kiện quyền thay vì dùng chung một license.


## Ingestion Vinmec theo metadata riêng từng bài

Không dùng `scripts/ingest_documents.py` trên cả thư mục Vinmec nếu metadata từng bài khác nhau, vì lệnh tổng quát nhận một bộ metadata chung. Dùng script chuyên biệt:

```powershell
python scripts/ingest_vinmec_documents.py --input data/documents/Vinmec --output data/processed/vinmec_chunks.jsonl
```

Script này:
- Đọc riêng `metadata.json` và `content.md` trong từng thư mục bài.
- Giữ canonical URL, title, publisher và rights note tương ứng từng tài liệu.
- Bỏ qua bài thiếu metadata/quyền lưu trữ hoặc thiếu nội dung.
- Khử trùng theo SHA-256 nội dung.
- Mặc định chỉ tạo JSONL; không kết nối Neo4j.

Sau khi xem báo cáo và kiểm tra JSONL, mới ghi vào Neo4j:

```powershell
python scripts/ingest_vinmec_documents.py --input data/documents/Vinmec --output data/processed/vinmec_chunks.jsonl --neo4j
```

Embedding là bước riêng, chạy sau khi đã xác nhận số chunk được lưu trong Neo4j:

```powershell
python scripts/embed_neo4j_documents.py --create-index --limit 100000 --batch-size 32
```

Không chạy embedding khi chưa có chunk đang chờ embedding. Báo cáo ingestion phải được xem trước; số chunk không đại diện cho số facts lâm sàng. Script không tạo clinical facts.
