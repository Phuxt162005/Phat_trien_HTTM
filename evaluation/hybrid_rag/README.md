# Đánh giá offline Hybrid RAG

## Mục tiêu

Bộ đánh giá đo chất lượng danh sách kết quả truy xuất, tách biệt với chất lượng câu trả lời của LLM/Agent. Không dùng kết quả do chính retriever tạo ra làm nhãn chuẩn. Qrels phải được người đánh giá gán dựa trên nội dung câu hỏi và nội dung chunk/tài liệu.

## Định dạng dữ liệu

Dùng UTF-8 JSONL, mỗi dòng là một JSON object.

### Qrels

Mỗi dòng có `query_id` và một trong hai trường:

- `relevant_ids`: mảng ID chunk/tài liệu có liên quan (binary relevance).
- `relevance`: object ánh xạ ID sang mức liên quan dương (graded relevance), ví dụ 1 = liên quan một phần, 2 = liên quan trực tiếp, 3 = nguồn hướng dẫn/chính thống trả lời trực tiếp.

Ví dụ minh họa, không phải ground truth của dự án:

```json
{"query_id":"q001","relevant_ids":["chunk-001","chunk-017"]}
{"query_id":"q002","relevance":{"chunk-020":3,"chunk-021":1}}
```

### Runs

Mỗi dòng có `query_id` và `results` là danh sách kết quả theo đúng thứ tự retriever trả về. Mỗi kết quả tối thiểu cần `id`; nên giữ thêm `title`, `url`, `source`, `page_start`, `page_end`, `retrieval_methods` để kiểm tra provenance.

Ví dụ minh họa:

```json
{"query_id":"q001","results":[{"id":"chunk-001","title":"Diabetes guideline","url":"https://example.org/guideline","source":{"name":"WHO"},"page_start":4,"page_end":5},{"id":"chunk-099","title":"Background","url":"https://example.org/background"}]}
```

Không đưa dữ liệu bệnh nhân có định danh vào qrels/runs.

## Chạy đánh giá

Sau khi chuẩn bị hai file:

```bash
python scripts/evaluate_hybrid_retrieval.py --qrels evaluation/hybrid_rag/qrels.jsonl --runs evaluation/hybrid_rag/runs.jsonl --k 5 10 --output evaluation/hybrid_rag/report.json
```

Bộ script không tự truy cập Neo4j và không tự tạo ground truth. Cần chạy HybridRetriever trên bộ câu hỏi, lưu thứ tự kết quả vào runs JSONL, rồi đánh giá với qrels đã được gán/duyệt độc lập.

## Chỉ số

- **Precision@K**: số kết quả liên quan trong top K chia cho K (kể cả khi retriever trả ít hơn K).
- **Recall@K**: số tài liệu liên quan tìm thấy trong top K chia tổng số tài liệu liên quan trong qrels.
- **MRR**: trung bình nghịch đảo thứ hạng của kết quả liên quan đầu tiên; báo cáo tên trường `reciprocal_rank` ở từng query và macro ở toàn bộ tập.
- **NDCG@K**: chất lượng xếp hạng, có xét mức độ liên quan nhiều cấp nếu qrels dùng object relevance.
- **Citation coverage@K**: tỷ lệ kết quả top K có URL HTTP(S) hợp lệ và có title hoặc source name.
- **Relevant citation accuracy@K**: tỷ lệ kết quả vừa liên quan vừa có citation hợp lệ trong nhóm kết quả liên quan top K.

Citation metrics hiện kiểm tra sự hiện diện/hình thức metadata, **không xác minh nội dung URL, độ tin cậy của nhà xuất bản, tính đúng của trang PDF hoặc việc trích dẫn có thực sự hỗ trợ phát biểu**. Những yếu tố đó cần audit riêng bằng người đánh giá.

## Thiết kế bộ câu hỏi khuyến nghị

Tạo câu hỏi theo các nhóm, có ID ổn định:

1. Định nghĩa và phân loại đái tháo đường.
2. Triệu chứng, yếu tố nguy cơ và biến chứng.
3. Xét nghiệm, ngưỡng chẩn đoán và diễn giải.
4. Điều trị, thuốc và theo dõi (đánh giá nguồn và ngày cập nhật đặc biệt chặt).
5. Dinh dưỡng, vận động và phòng ngừa.
6. Câu hỏi cần đối chiếu WHO/ADA/NIDDK/CDC/Vinmec.
7. Câu hỏi tiếng Việt không dấu, lỗi gõ và viết tắt.
8. Câu hỏi tiếng Anh và câu hỏi Việt-Anh tương đương.
9. Câu hỏi đa ý, thiếu thông tin hoặc không có bằng chứng trong kho.

Mỗi query cần có: câu hỏi nguyên bản, ngôn ngữ, nhóm chủ đề, ngày tạo, người gán nhãn, tiêu chí liên quan và danh sách ID liên quan. Nên có cả câu hỏi không có tài liệu phù hợp; với nhóm này, qrels dùng mảng rỗng và đánh giá thêm khả năng từ chối/abstain ở tầng chatbot, không diễn giải Recall bằng 0 như lỗi nếu kho thực sự không có bằng chứng.

## Quy trình gán nhãn

1. Đóng băng snapshot corpus và ghi mã phiên bản/hash.
2. Hai người đánh giá độc lập nội dung chunk, không chỉ nhìn tiêu đề hoặc điểm similarity.
3. Gán mức liên quan theo rubric đã thống nhất.
4. Giải quyết bất đồng và lưu người duyệt cuối.
5. Chia query thành development và held-out evaluation; không điều chỉnh tham số trên held-out.
6. Khi corpus thay đổi đáng kể, ghi phiên bản corpus mới và xem xét cập nhật qrels.

## Giới hạn và diễn giải

- Macro average là trung bình không trọng số trên query; cần báo cáo thêm số query theo từng nhóm để tránh nhóm đông lấn át nhóm nhỏ.
- Precision@K dùng mẫu số K cố định.
- Query không có tài liệu liên quan không phù hợp để tính Recall/NDCG thông thường; nên báo cáo riêng nhóm no-answer.
- Không đặt ngưỡng PASS tùy ý trước khi có baseline và đánh giá thủ công.
- Điểm retrieval tốt không chứng minh chatbot an toàn hoặc chính xác về mặt y khoa.
