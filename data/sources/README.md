# Tài liệu nguồn y khoa cho Diabetes Decision Support

Thư mục chứa ghi chú tri thức tóm tắt có dẫn nguồn, không phải bản sao toàn văn website.

## Quy tắc
- Giữ URL nguồn, tổ chức, ngày truy cập và giới hạn bản quyền trong từng file.
- Không sao chép toàn văn Vinmec hoặc ADA; chỉ lưu ghi chú tóm tắt và liên kết.
- Không mặc định mọi nội dung WHO đều có giấy phép CC BY; kiểm tra chính sách áp dụng cho từng tài liệu.
- Không dùng ghi chú để chẩn đoán hoặc kê đơn cá nhân.
- Không trộn nhiều nguồn URL trong một lệnh ingestion vì `--source-url` áp dụng cho toàn bộ lần chạy.

## Chạy ingestion
```powershell
python scripts/ingest_documents.py data/sources/WHO --source-name "WHO" --publisher "World Health Organization" --license "See source-specific WHO copyright terms" --source-url "https://www.who.int/news-room/fact-sheets/detail/diabetes" --output data/processed/who_chunks.jsonl --neo4j
python scripts/ingest_documents.py data/sources/NIDDK --source-name "NIDDK" --publisher "National Institute of Diabetes and Digestive and Kidney Diseases" --license "US government information; verify page-specific notices" --source-url "https://www.niddk.nih.gov/health-information/diabetes" --output data/processed/niddk_chunks.jsonl --neo4j
python scripts/ingest_documents.py data/sources/CDC --source-name "CDC" --publisher "Centers for Disease Control and Prevention" --license "US government work; verify page-specific notices" --source-url "https://www.cdc.gov/diabetes/php/data-research/index.html" --output data/processed/cdc_chunks.jsonl --neo4j
python scripts/ingest_documents.py data/sources/Vinmec --source-name "Vinmec" --publisher "Vinmec Healthcare System" --license "Copyrighted; curated summary only" --source-url "https://www.vinmec.com/vie/bai-viet/xet-nghiem-chan-doan-dai-thao-duong-vi" --output data/processed/vinmec_chunks.jsonl --neo4j
```

ADA được lưu link tham khảo riêng do Standards of Care có bản quyền.
