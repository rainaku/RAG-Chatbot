from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3:4b-instruct",
    base_url="http://127.0.0.1:11434",
    temperature=0.2,
    validate_model_on_init=True,
    verbose=True,
)

print(llm.invoke("Xin chào, test 1 câu ngắn giúp mình."))
