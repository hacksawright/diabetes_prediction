# Dự đoán tiểu đường đa agent + Vinmec KB + Neo4j

Demo học tập gồm 4 agent: `IntakeAgent` kiểm tra và chuẩn hóa đầu vào; `PredictionAgent` chạy mô hình logistic regression; `KnowledgeAgent` tìm đoạn bài Vinmec trong Neo4j; `ExplanationAgent` tạo lời giải thích và dẫn nguồn. `Coordinator` chạy các agent theo thứ tự. Không cần API key LLM.

Ứng dụng còn có `RAGAgent` cho hỏi đáp: câu hỏi → tìm đoạn liên quan trong Neo4j → gửi các đoạn được đánh số cho Gemini API → trả lời tiếng Việt kèm ký hiệu nguồn `[1]`, `[2]`. Phần RAG cần API key; phần dự đoán vẫn chạy độc lập.

**Giới hạn:** Mô hình học từ [OpenML 37](https://www.openml.org/search?type=data&sort=runs&id=37), bộ Pima 768 bản ghi chỉ gồm phụ nữ Pima từ 21 tuổi. Điểm mô hình **không phải** xác suất mắc bệnh của người dùng và không dùng để chẩn đoán. Chẩn đoán cần xét nghiệm và đánh giá của bác sĩ; xem [Vinmec: xét nghiệm chẩn đoán đái tháo đường](https://www.vinmec.com/vie/bai-viet/xet-nghiem-chan-doan-dai-thao-duong-vi).

## Yêu cầu

- Python 3.10+ và Docker Desktop đang chạy.
- Internet khi huấn luyện (tải OpenML lần đầu) và khi nạp KB (đọc Vinmec).

## Chạy trên Windows PowerShell

```powershell
Copy-Item .env.example .env
# Có thể sửa NEO4J_PASSWORD trong .env trước khi tạo database.
docker compose --env-file .env up -d
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m diabetes_agents.train
python -m diabetes_agents.ingest
uvicorn diabetes_agents.api:app --reload
```

Mở giao diện tại <http://127.0.0.1:8000>, tài liệu API tại <http://127.0.0.1:8000/docs>, trạng thái hệ thống tại <http://127.0.0.1:8000/health>. Neo4j Browser: <http://localhost:7474>, đăng nhập `neo4j` và mật khẩu trong `.env`. Nếu đổi mật khẩu sau lần tạo đầu, đổi cả trong Neo4j bằng Browser hoặc xóa dữ liệu cũ có chủ ý rồi tạo lại.

### Bật hỏi đáp RAG

Tạo API key trong [Google AI Studio](https://aistudio.google.com/app/apikey), điền vào `GEMINI_API_KEY=` trong `.env`, rồi khởi động lại `uvicorn`. Có thể chọn model qua `RAG_MODEL` (mặc định `gemini-3.8-flash`). Không đưa `.env` lên Git và không nhập thông tin định danh bệnh nhân vào câu hỏi; câu hỏi và các đoạn KB truy xuất được gửi tới Gemini API để tạo câu trả lời. API key có thể phát sinh chi phí sử dụng.

Ví dụ hỏi qua API:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/rag/ask -Method Post -ContentType 'application/json' -Body '{"question":"Những xét nghiệm nào dùng để chẩn đoán bệnh tiểu đường?"}'
```

Kết quả gồm `answer`, `sources` (URL, tiêu đề, đoạn trích, số thứ tự) và `model`. Khi KB không có đoạn phù hợp, ứng dụng báo thiếu dữ liệu thay vì tạo câu trả lời. Câu trả lời RAG chỉ phục vụ học tập, không thay thế tư vấn y tế.

Nếu Gemini trả `503 UNAVAILABLE` do quá tải, ứng dụng thử lại tối đa 3 lần với khoảng chờ tăng dần. Nếu vẫn thất bại, giao diện hiển thị các đoạn KB đã truy xuất để bạn đọc và thử hỏi lại sau. Lỗi này đến từ dịch vụ tạo câu trả lời, không có nghĩa KB trống.

Truy xuất KB hiện dùng **TF-IDF** trên các đoạn `Chunk` lấy từ Neo4j, sau đó xếp hạng bằng cosine similarity. Mỗi câu hỏi tạo ma trận TF-IDF từ KB hiện có; với KB nhỏ hiện tại cách này đơn giản và không cần embedding/API bổ sung. TF-IDF dùng các nhóm 3–5 ký tự để phù hợp hơn với tiếng Việt chưa tách từ. Kết quả nguồn trả về có thêm `score`. Nếu Gemini trả `504 DEADLINE_EXCEEDED`, ứng dụng cũng thử lại và vẫn hiển thị các đoạn KB khi việc tạo câu trả lời thất bại.

## KB lưu ở đâu?

Neo4j lưu bền vững trong thư mục **`neo4j/data/`** của dự án, gắn vào `/data` trong container. Database mặc định tên `neo4j`. Cấu trúc graph: `(Article {url,title,fetched_at})-[:HAS_CHUNK]->(Chunk {id,text,normalized,position})`. Mỗi đoạn giữ URL bài gốc để dẫn nguồn. Nạp lại bằng `python -m diabetes_agents.ingest`; chương trình thay các đoạn cũ của từng bài đã tải thành công.

### Tạo knowledge graph y khoa

Sau khi nạp KB, chạy `python -m diabetes_agents.kg`. Lệnh này đọc các `Chunk` Vinmec đang có và thêm nút `Concept` (bệnh, triệu chứng, xét nghiệm), cạnh `Chunk-[:MENTIONS]->Concept` và cạnh `Đái tháo đường-[:HAS_KNOWLEDGE {predicate, evidence, source_url, evidence_chunk_id}]->Concept`. Các cạnh chỉ được tạo khi đoạn nguồn khớp quy tắc trích xuất; chạy lại lệnh sẽ làm mới các cạnh chứng cứ, tránh giữ dữ liệu cũ sau khi nạp lại bài viết. Đây là graph tri thức nhỏ có truy vết nguồn, chưa phải bộ ontology y khoa đầy đủ.

Có thể chạy cùng quy tắc trực tiếp bằng file [cypher/knowledge_graph.cypher](cypher/knowledge_graph.cypher): mở trong Neo4j Browser và chạy từng câu lệnh. Cách đơn giản nhất trong dự án này vẫn là `python -m diabetes_agents.kg`.

```cypher
MATCH (d:Concept {key:'diabetes'})-[r:HAS_KNOWLEDGE]->(c:Concept)
RETURN d.name, r.predicate, c.name, r.evidence, r.source_url LIMIT 30;
```

Truy vấn trong Neo4j Browser:

```cypher
MATCH (a:Article)-[:HAS_CHUNK]->(c:Chunk)
RETURN a.title, a.url, c.position, c.text LIMIT 20;
```

Mô hình lưu riêng ở `artifacts/diabetes_model.joblib`; chỉ số đánh giá ở `artifacts/diabetes_model.json`. KB **không** nằm trong file mô hình. Cả `artifacts/` và `neo4j/data/` được bỏ qua bởi Git vì có dữ liệu tạo lúc chạy.

## Lưu ý kỹ thuật

Input `glucose` là glucose huyết tương **2 giờ sau nghiệm pháp uống đường**, không phải đường huyết lúc đói. `diabetes_pedigree` là đặc trưng định nghĩa riêng trong bộ Pima, không thể tự suy từ việc gia đình có người bệnh. Các trường huyết áp, nếp gấp da, insulin có thể để trống và sẽ được điền bằng trung vị tập huấn luyện. Giá trị 0 bất hợp lý trong dữ liệu huấn luyện được xử lý là thiếu. Chỉ nhập dữ liệu đo được đúng đơn vị; không điền số giả để nhận kết quả.

Ứng dụng không lưu thông tin người nhập. KB dùng trích đoạn bài công khai, có dẫn URL nguồn. Nếu Vinmec thay cấu trúc HTML/URL hoặc chặn truy cập, lệnh nạp sẽ báo lỗi từng bài. Các nguồn được khai báo trong `diabetes_agents/kb.py`.
