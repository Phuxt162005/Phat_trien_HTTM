# Chatbot

## Trạng thái hiện tại

Đã có lớp dịch vụ chatbot nền tảng tại `src/project01/chatbot/service.py` và API tại `app/backend/main.py`.

Luồng hiện tại:

1. Nhận câu hỏi.
2. Xử lý lời chào, câu hỏi ngoài phạm vi và dấu hiệu khẩn cấp cơ bản.
3. Với câu hỏi về đái tháo đường, gọi `HybridRetriever` (Vector + Keyword + Graph + RRF).
4. Trả về trích đoạn tài liệu cùng URL, nguồn, trang và phương thức truy xuất.
5. Nếu không có bằng chứng hoặc Neo4j lỗi, chatbot từ chối kết luận thay vì tự tạo thông tin.

## Giới hạn chưa triển khai

- Chưa tích hợp LLM để tổng hợp câu trả lời tự nhiên.
- Chưa có bộ nhớ hội thoại nhiều lượt hoặc coreference resolution.
- Chưa có intent classifier chuyên biệt; routing hiện dùng quy tắc từ khóa.
- Chưa có đánh giá chất lượng câu trả lời sinh bởi LLM.

Vì vậy, không coi lớp này là chatbot y khoa hoàn chỉnh. Đây là backend evidence-first để kết nối và kiểm soát đầu vào/đầu ra trước khi tích hợp mô hình sinh.
