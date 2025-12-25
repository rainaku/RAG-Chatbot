#import các thư viện cần thiết
from supabase import create_client
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationSummaryBufferMemory
import os, re, unicodedata, difflib
from dotenv import load_dotenv

load_dotenv()
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
base_url = os.getenv("BASE_URL")


emb = OllamaEmbeddings(model="embeddinggemma:300m", num_gpu=1, base_url=base_url)
vs = SupabaseVectorStore(
    client=sb,
    table_name="documents",
    query_name="match_documents",
    embedding=emb,
)
#Lấy 2 file có chunk gần nhất
retriever = vs.as_retriever(search_kwargs={"k": 2})

llm = ChatOllama(model="qwen3:4b-instruct", temperature=0.3, num_gpu=1, base_url=base_url, reasoning=False, num_predict=1024) 
# LLM riêng để summarize
summarize_llm = ChatOllama(model="qwen3:4b-instruct", temperature=0.1, num_gpu=1,base_url=base_url,reasoning=False)
#Pre-prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn là trợ lý tra cứu thông tin chính thức của trường Đại học Công Thương TP.HCM (HUIT). 
TRẢ LỜI đúng trọng tâm - chỉ nội dung chính, không giải thích dài dòng, không lặp lại câu hỏi.

Nếu không tìm thấy thông tin trong context được cung cấp, hãy trả lời theo mẫu sau:
"Thông tin về [chủ đề câu hỏi] không có trong cơ sở dữ liệu chính thức của trường HUIT.

Nếu bạn cần thêm thông tin chi tiết hoặc có câu hỏi khác, vui lòng cho tôi biết!"
"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "Context:\n{context}\n\nHỏi: {question}")
])
parser = StrOutputParser()

# Prompt không có lịch sử (dùng cho trường hợp không có session)
prompt_no_history = ChatPromptTemplate.from_messages([
    ("system", """Bạn là trợ lý tra cứu thông tin chính thức của trường Đại học Công Thương TP.HCM (HUIT).
TRẢ LỜI đúng trọng tâm  -  chỉ nội dung chính, không giải thích dài dòng, không lặp lại câu hỏi.

Nếu không tìm thấy thông tin trong context được cung cấp, hãy trả lời theo mẫu sau:
"Thông tin về [chủ đề câu hỏi] không có trong cơ sở dữ liệu chính thức của trường HUIT.

Nếu bạn cần thêm thông tin chi tiết hoặc có câu hỏi khác, vui lòng cho tôi biết!"
"""),
    ("human", "Hỏi: {question}\n\nContext:\n{context}")
])

# ========= Helpers =========
def normalize(s: str) -> str:
    s = s.lower()
    s = s.replace("đ", "d")
    s = ''.join(ch for ch in unicodedata.normalize('NFD', s)
                if unicodedata.category(ch) != 'Mn')
    s = re.sub(r'[^a-z0-9]+', ' ', s).strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def load_subject_tokens_from_db(max_rows: int = 5000):
    try:
        res = (sb.table("documents").select("metadata").limit(max_rows).execute())
        rows = res.data or []
        tokens = set()
        for r in rows:
            md = r.get("metadata") or {}
            src = (md.get("source") or "").strip()
            if not src: continue
            base = os.path.splitext(os.path.basename(src))[0]
            nbase = normalize(base)
            if nbase.startswith("nganh "):
                tokens.add(nbase)
                tokens.add(nbase.replace("nganh ", ""))
            else:
                tokens.add(nbase)
        return sorted(tokens)
    except Exception as e:
        print(f"[WARNING] Không thể load SUBJECT_TOKENS từ DB: {e}")
        return []

# Lazy load cache
_subject_tokens_cache: list[str] | None = None

def get_subject_tokens() -> list[str]:
    """Lazy load SUBJECT_TOKENS từ DB"""
    global _subject_tokens_cache
    if _subject_tokens_cache is None:
        _subject_tokens_cache = load_subject_tokens_from_db()
    return _subject_tokens_cache

def detect_subject_token(question: str):
    SUBJECT_TOKENS = get_subject_tokens()  # Lazy load
    if not SUBJECT_TOKENS:
        return None
    nq = normalize(question)
    for t in SUBJECT_TOKENS:
        if t in nq:
            return t
    best = difflib.get_close_matches(nq, SUBJECT_TOKENS, n=1, cutoff=0.6)
    return best[0] if best else None

def keyword_search(question: str, subject_token: str | None, max_rows: int = 6):
    terms = re.findall(r"[a-zA-ZÀ-ỹ0-9]+", question.lower())
    terms = [t for t in terms if len(t) >= 2]
    if subject_token:
        terms.extend([subject_token, f"nganh {subject_token}"])
    if not terms:
        return []

    # Giữ nguyên thứ tự terms và loại bỏ trùng
    ordered_terms: list[str] = []
    seen_terms: set[str] = set()
    for t in terms:
        if t not in seen_terms:
            seen_terms.add(t)
            ordered_terms.append(t)

    # Các thuật ngữ đã normalize để tìm trong metadata (vốn không dấu)
    normalized_terms: list[str] = []
    seen_norm: set[str] = set()
    for t in ordered_terms:
        nt = normalize(t)
        if nt and nt not in seen_norm:
            seen_norm.add(nt)
            normalized_terms.append(nt)

    ors_parts = []
    ors_parts.extend(f"content.ilike.*{kw}*" for kw in ordered_terms)
    for kw in normalized_terms:
        variants = {kw}
        if " " in kw:
            variants.add(kw.replace(" ", "-"))
            variants.add(kw.replace(" ", ""))
        ors_parts.extend(f"metadata->>source.ilike.*{v}*" for v in variants)

    if not ors_parts:
        return []

    ors = ",".join(ors_parts)
    if os.getenv("DEBUG_KEYWORD"):
        print("[KW] or_ =", ors)
    res = (sb.table("documents").select("content,metadata").or_(ors).limit(max_rows).execute())
    rows = res.data or []
    return [Document(page_content=r.get("content",""), metadata=r.get("metadata",{})) for r in rows]

# ========= Metadata focus =========
def load_metadata_index(max_rows: int = 5000):
    """Tải danh sách source + tokens để tìm kiếm động theo metadata"""
    try:
        res = sb.table("documents").select("metadata").limit(max_rows).execute()
    except Exception as e:
        print(f"[WARNING] Không thể load metadata index: {e}")
        return []

    entries = []
    rows = res.data or []
    for r in rows:
        md = r.get("metadata") or {}
        src = (md.get("source") or "").strip()
        if not src:
            continue
        base = os.path.splitext(os.path.basename(src))[0]
        words = [tok for tok in normalize(base).split() if tok]
        if not words:
            continue
        tokens = set(words)
        abbr_all = "".join(word[0] for word in words if word)
        if len(abbr_all) >= 2:
            tokens.add(abbr_all)
        if len(words) > 1:
            abbr_skip_first = "".join(word[0] for word in words[1:] if word)
            if len(abbr_skip_first) >= 2:
                tokens.add(abbr_skip_first)
        if not tokens:
            continue
        entries.append({"source": src, "tokens": list(tokens)})
    return entries

_METADATA_INDEX = load_metadata_index()
_METADATA_DOC_CACHE: dict[str, list[Document]] = {}

def fetch_docs_by_source(source: str, max_rows: int = 2):
    if source in _METADATA_DOC_CACHE:
        return _METADATA_DOC_CACHE[source]
    res = (sb.table("documents")
           .select("content,metadata")
           .eq("metadata->>source", source)
           .limit(max_rows)
           .execute())
    rows = res.data or []
    docs = [Document(page_content=r.get("content",""), metadata=r.get("metadata",{})) for r in rows]
    _METADATA_DOC_CACHE[source] = docs
    return docs

def metadata_focus_docs(question: str, max_sources: int = 3):
    """Tự động tìm các tài liệu metadata phù hợp nhất với câu hỏi"""
    if not _METADATA_INDEX:
        return []

    q_tokens = [tok for tok in normalize(question).split() if tok]
    if not q_tokens:
        return []

    q_set = set(q_tokens)
    scored: list[tuple[float, dict]] = []
    for entry in _METADATA_INDEX:
        tokens = set(entry["tokens"])
        overlap = q_set.intersection(tokens)
        if not overlap:
            continue
        # điểm dựa trên số token trùng và tỉ lệ trùng
        score = len(overlap) + len(overlap) / (len(tokens) or 1)
        scored.append((score, entry))

    if not scored:
        return []

    top_entries = [entry for _, entry in sorted(scored, key=lambda x: x[0], reverse=True)[:max_sources]]
    docs: list[Document] = []
    for entry in top_entries:
        docs.extend(fetch_docs_by_source(entry["source"]))
    return docs

def subject_direct_search(subject_token: str, max_rows: int = 4):
    like = f"*{subject_token}*"
    res = (sb.table("documents")
           .select("content,metadata")
           .or_(f"metadata->>source.ilike.{like},content.ilike.{like}")
           .limit(max_rows)
           .execute())
    rows = res.data or []
    return [Document(page_content=r.get("content",""), metadata=r.get("metadata",{})) for r in rows]

# ========= HYBRID =========
def hybrid_context(question: str):
    subj = detect_subject_token(question)

    # A) Vector retrieval
    vec_docs = retriever.invoke(question) or []
    if subj:
        nsubj = subj
        vec_docs = [d for d in vec_docs
                    if nsubj in normalize(d.metadata.get("source",""))
                    or nsubj in normalize(d.page_content or "")]
        if not vec_docs:
            vec_docs = subject_direct_search(nsubj, max_rows=6)

    print(f"[DBG] subject={subj or '-'} | vector={len(vec_docs)}")
    for i, d in enumerate(vec_docs, 1):
        preview = (d.page_content or "").replace("\n", " ")[:120]
        print(f"  [V{i}] {d.metadata.get('source','?')} | {preview}...")

    # B) Keyword retrieval
    kw_docs = keyword_search(question, subj, max_rows=1) 
    meta_docs = metadata_focus_docs(question, max_sources=2)  
    if meta_docs:
        print(f"[DBG] metadata_focus={len(meta_docs)}")
    print(f"[DBG] keyword={len(kw_docs)}")
    for i, d in enumerate(kw_docs, 1):
        preview = (d.page_content or "").replace("\n", " ")[:120]
        print(f"  [K{i}] {d.metadata.get('source','?')} | {preview}...")

    # C) Merge + dedup
    all_docs = vec_docs + kw_docs + meta_docs
    seen, merged = set(), []
    for d in all_docs:
        key = (d.metadata.get("source","") + "|" + (d.page_content or "")[:80])
        if key not in seen and d.page_content:
            seen.add(key); merged.append(d)
    print(f"[DBG] merged_docs={len(merged)} | sources={[d.metadata.get('source','?') for d in merged]}")
    return "\n\n---\n\n".join(d.page_content for d in merged) if merged else "—"

# ========= MEMORY =========
# Dictionary để lưu ConversationSummaryBufferMemory theo session_id
_memory_store: dict[str, ConversationSummaryBufferMemory] = {}

def get_memory(session_id: str) -> ConversationSummaryBufferMemory:
    """Lấy hoặc tạo ConversationSummaryBufferMemory cho session_id"""
    if session_id not in _memory_store:
        _memory_store[session_id] = ConversationSummaryBufferMemory(
            llm=summarize_llm,
            max_token_limit=2000,  # Tự động summarize khi vượt quá 2000 tokens
            return_messages=True,  # Trả về list messages thay vì string
            memory_key="chat_history"
        )
    return _memory_store[session_id]

def add_to_chat_history(session_id: str, question: str, answer: str):
    """Thêm Q&A vào memory của session"""
    memory = get_memory(session_id)
    # save_context tự động thêm vào buffer và summarize nếu cần
    memory.save_context(
        {"input": question},
        {"output": answer}
    )

def create_rag_chain_with_memory(session_id: str):
    """Tạo chain với ConversationSummaryBufferMemory cho session"""
    memory = get_memory(session_id)
    
    # Tạo chain với history từ memory
    def create_chain_input(inputs: dict):
        # load_memory_variables() trả về dict với key "chat_history"
        # Vì return_messages=True, nó sẽ trả về list[BaseMessage]
        memory_vars = memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])
        
        # Debug: In ra lịch sử
        print(f"[MEMORY] Session: {session_id}, History length: {len(chat_history)}")
        if chat_history:
            print(f"[MEMORY] Last message: {chat_history[-1].content[:100]}...")
        
        return {
            "question": inputs["question"],
            "context": inputs["context"],
            "chat_history": chat_history if chat_history else []
        }
    
    chain = RunnableLambda(create_chain_input) | prompt | llm | parser
    return chain
