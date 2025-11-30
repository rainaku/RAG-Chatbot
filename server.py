# uvicorn server:app --reload --host 0.0.0.0 --port 8000

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from testRAG import hybrid_context, create_rag_chain_with_memory, add_to_chat_history
import uuid

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)



class Question(BaseModel):
    question: str
    session_id: str | None = None  # Optional session_id, nếu không có sẽ tạo mới

@app.get("/", response_class=HTMLResponse)
def index():
    
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/ask")
def rag_endpoint(q: Question):
    # Tạo hoặc sử dụng session_id
    session_id = q.session_id or str(uuid.uuid4())
    
    # Lấy context từ RAG
    ctx = hybrid_context(q.question)
    
    # Tạo chain với memory
    chain = create_rag_chain_with_memory(session_id)
    
    # Collect response để lưu vào history
    collected_tokens = []
    
    def token_gen():
        try:
            for token in chain.stream({"question": q.question, "context": ctx}):
                collected_tokens.append(token)
                yield token
            
            # Sau khi stream xong, lưu vào history
            if collected_tokens:
                answer = "".join(collected_tokens)
                print(f"[SERVER] Saving to history - Q: {q.question[:50]}... A: {answer[:50]}...")
                try:
                    add_to_chat_history(session_id, q.question, answer)
                except Exception as e:
                    print(f"[SERVER] Error saving history: {e}")
        except Exception as e:
            print(f"[SERVER] Error during streaming: {e}")
            import traceback
            traceback.print_exc()
            yield f"\n[ERROR] {str(e)}"

    return StreamingResponse(
        token_gen(), 
        media_type="text/plain; charset=utf-8",
        headers={"X-Session-Id": session_id}  # Trả về session_id để client lưu
    )
