from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5:3b",
    base_url="http://127.0.0.1:11434",
    temperature=0.2,
    validate_model_on_init=True,
    verbose=True,
)

print(llm.invoke("Xin chào, test 1 câu ngắn giúp mình."))
#uvicorn server:app --reload --host 0.0.0.0 --port 8000
#streamlit run streamlit_app.py
#ollama serve
