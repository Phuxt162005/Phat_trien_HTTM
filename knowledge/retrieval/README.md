# Retrieval

## Hiện trạng

- Graph retrieval: Neo4j full-text search qua `Neo4jGraphRepository.search()`, trả về entity, quan hệ lân cận và provenance nếu graph đã có dữ liệu liên kết nguồn.
- Document ingestion: trích xuất PDF/text/HTML, chia chunk, lưu JSONL và tùy chọn upsert metadata/chunks vào Neo4j.

## Lộ trình

1. Tạo embedding cho DocumentChunk bằng multilingual sentence-transformer.
2. Tạo vector index trong Neo4j với chiều vector khớp model.
3. Vector retrieval và graph retrieval song song.
4. Hợp nhất kết quả (RRF), loại trùng theo chunk/document ID.
5. Reranking và đánh giá Precision@K, Recall@K, MRR, NDCG, citation accuracy.

Không coi việc có full-text index là đã hoàn thành vector retrieval hoặc Hybrid RAG.
