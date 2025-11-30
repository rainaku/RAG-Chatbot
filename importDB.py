import os, glob
from dotenv import load_dotenv
from supabase import create_client
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def read_txt_md(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def read_csv(path: str) -> str:
    # Thử lần lượt UTF-8 có BOM -> UTF-8 -> cp1258 -> latin-1
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            loader = CSVLoader(file_path=path, encoding=enc)
            docs = loader.load() 
            return "\n".join(d.page_content for d in docs)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Không đọc được CSV {path} với các encoding thường gặp") from last_err

def read_docx(path: str) -> str:
    loader = Docx2txtLoader(file_path=path)
    docs = loader.load() 
    return "\n".join(d.page_content for d in docs)

def load_corpus(data_dir="data"):
    docs = []
    for p in glob.glob(os.path.join(data_dir, "**/*"), recursive=True):
        if os.path.isdir(p): 
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext in [".txt", ".md"]:
            text = read_txt_md(p)
        elif ext in [".pdf"]:
            text = read_pdf(p)
        elif ext in [".docx"]:
            text = read_docx(p)
        elif ext in [".csv"]:
            text = read_csv(p)
        else:
            continue
        docs.append((p, text))
    return docs

def main():
    # 1) Supabase client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 2) Embedding 768
    embed = OllamaEmbeddings(model="embeddinggemma:300m")

    # 3) VectorStore 
    vs = SupabaseVectorStore(
        client=sb,
        table_name="documents",
        query_name="match_documents",
        embedding=embed,
    )

    # 4) Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    # 5) Load & add
    raw_docs = load_corpus("data")
    all_chunks = []
    for path, text in raw_docs:
        if not text.strip():
            continue
        for chunk in splitter.split_text(text):
            all_chunks.append(Document(page_content=chunk, metadata={"source": path}))

    if not all_chunks:
        print(" Không có tài liệu hợp lệ trong ./data")
        return

    vs.add_documents(all_chunks)
    print(f"Đã ingest {len(all_chunks)} chunks vào Supabase.")

if __name__ == "__main__":
    main()
