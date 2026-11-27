# Backend API

Backend FastAPI cung cấp API truy vấn thông tin đái tháo đường dựa trên Hybrid RAG.

## Cài đặt

Tại thư mục gốc dự án:

```powershell
pip install -e ".[web,rag,neo4j]"
```

Cấu hình Neo4j trong `.env` theo `.env.example`. Backend sử dụng embedding model `intfloat/multilingual-e5-small` (384 chiều), phải khớp với vector đã lưu trong Neo4j.

## Chạy

```powershell
uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

- Health: `GET http://127.0.0.1:8000/health`
- Chat: `POST http://127.0.0.1:8000/api/chat`

Ví dụ request:

```json
{"query":"Bệnh đái tháo đường type 2 là gì?"}
```

API hiện trả về trích đoạn bằng chứng và metadata nguồn từ Hybrid RAG. Đây là lớp backend nền tảng, chưa phải LLM sinh câu trả lời tự nhiên; không đưa ra chẩn đoán hoặc chỉ định điều trị. Khi Neo4j không sẵn sàng, endpoint chat trả HTTP 503 thay vì tạo câu trả lời không có căn cứ.
