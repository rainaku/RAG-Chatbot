# uvicorn server:app --reload --host 0.0.0.0 --port 8000
# py run_app.py


from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from testRAG import hybrid_context, create_rag_chain_with_memory, add_to_chat_history
import uuid
import threading

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

class Question(BaseModel):
    question: str
    session_id: str | None = None

@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

def save_history_background(session_id: str, question: str, answer: str):
    """Lưu history trong background thread để không block response"""
    try:
        add_to_chat_history(session_id, question, answer)
        print(f"[BG] Saved: Q: {question[:30]}...")
    except Exception as e:
        print(f"[BG] Error saving history: {e}")

@app.post("/ask")
def rag_endpoint(q: Question):
    session_id = q.session_id or str(uuid.uuid4())
    ctx = hybrid_context(q.question)
    chain = create_rag_chain_with_memory(session_id)
    
    collected_tokens = []
    
    def token_gen():
        for token in chain.stream({"question": q.question, "context": ctx}):
            collected_tokens.append(token)
            yield token
        
        # Lưu history trong background thread - KHÔNG block stream
        if collected_tokens:
            answer = "".join(collected_tokens)
            threading.Thread(
                target=save_history_background,
                args=(session_id, q.question, answer),
                daemon=True
            ).start()

    return StreamingResponse(
        token_gen(), 
        media_type="text/plain; charset=utf-8",
        headers={"X-Session-Id": session_id}
    )
