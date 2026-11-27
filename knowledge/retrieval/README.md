# Hybrid retrieval

## Thành phần đã triển khai

- Embedding: `intfloat/multilingual-e5-small`, vector 384 chiều, hỗ trợ tiếng Việt/Anh; query prefix `query:`, document prefix `passage:`.
- Embedding được tải lazy; cài `pip install -e ".[rag,neo4j]"`. Lần đầu chạy cần internet để tải model nếu cache chưa có.
- Neo4j vector index: `diabetes_document_chunk_vector`, cosine similarity, mặc định 384 chiều.
- Vector retrieval trả chunk, document title/URL, page range và source metadata.
- Graph retrieval dùng full-text index hiện có để truy xuất entity/chunk cùng các quan hệ và provenance.
- HybridRetriever chạy hai nhánh và hợp nhất bằng Reciprocal Rank Fusion (RRF), deduplicate theo chunk/entity ID.

## Tạo embedding

Sau khi ingestion và kiểm tra các chunk:

```bash
python scripts/embed_neo4j_documents.py --create-index --limit 100000 --batch-size 32
```

Lệnh chỉ tạo embedding cho chunk chưa có vector. Có thể chạy lại để tiếp tục; vector đã tồn tại không bị tính lại. Không dùng `--dimension` khác 384 với model mặc định.

## Dùng HybridRetriever

```python
from project01.knowledge import (
    Neo4jGraphRepository, SentenceTransformerEmbedder, HybridRetriever,
)

repository = Neo4jGraphRepository.connect()
try:
    retriever = HybridRetriever(repository, SentenceTransformerEmbedder())
    result = retriever.retrieve("Các yếu tố nguy cơ của bệnh tiểu đường", limit=8)
finally:
    repository.close()
```

## Giới hạn cần biết

- Đây là hybrid retrieval (vector + graph full-text) với RRF; chưa có cross-encoder reranking.
- Vector index phải tồn tại và có dimension trùng với embedding model.
- Graph full-text index phải được tạo bằng `knowledge/neo4j/schema.cypher`.
- Chưa có benchmark chất lượng trên bộ câu hỏi gán nhãn; RRF không tự chứng minh kết quả đúng y khoa.
- Chưa tự sinh câu trả lời hoặc trích dẫn khi thiếu evidence.
- Nếu đổi embedding model/dimension, cần tạo index tương ứng và tái embed toàn bộ chunk; không trộn vector của các model khác nhau trong cùng index.
