import streamlit as st
import requests
import uuid

# Cấu hình trang
st.set_page_config(
    page_title="RAG Chatbot - HUIT",
    page_icon="🤖",
    layout="centered"
)

# Header
st.title("RAG Chatbot - HUIT")
st.caption("Trợ lý AI hỗ trợ tra cứu thông tin Khoa Công nghệ thông tin")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("Cài đặt")
    
    # Hiển thị session ID
    st.info(f"**Session ID:**\n\n`{st.session_state.session_id[:16]}...`")
    
    # Nút tạo chat mới
    if st.button("Chat Mới", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Thông tin
    st.subheader(" Hướng dẫn")
    st.write("- Đặt câu hỏi về học vụ, quy chế, thông tin trường")
    st.write("- Nhấn 'Chat Mới' để bắt đầu cuộc trò chuyện mới")
    
    st.divider()
    
    st.subheader("Công nghệ")
    st.write("**RAG**: Hybrid Search (Vector + Keyword)")
    st.write("**LLM**: Qwen3 4B (Ollama)")
    st.write("**Memory**: ConversationSummaryBufferMemory")
    st.write("**Vector DB**: Supabase")

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # Thêm câu hỏi của user vào messages
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Hiển thị câu hỏi của user
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Gọi API và streaming response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Gọi API backend
            response = requests.post(
                "http://localhost:8000/ask",
                json={
                    "question": prompt,
                    "session_id": st.session_state.session_id
                },
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                # Stream response
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
            else:
                full_response = f" Lỗi: {response.status_code}"
                message_placeholder.markdown(full_response)
        
        except requests.exceptions.ConnectionError:
            full_response = "Không thể kết nối đến server. Hãy đảm bảo server đang chạy tại `http://localhost:8000`"
            message_placeholder.markdown(full_response)
        except requests.exceptions.Timeout:
            full_response = "Timeout: Server mất quá nhiều thời gian để trả lời."
            message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"Lỗi: {str(e)}"
            message_placeholder.markdown(full_response)
    
    # Lưu response vào session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer
st.divider()
st.caption("RAG Chatbot © 2025 | Powered by TT AILORD LangChain + Ollama + Supabase")
