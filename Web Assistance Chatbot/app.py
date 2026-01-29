"""
Streamlit Application for Website Assitanace Chatbot
Main UI for indexing websites and asking questions
"""

import streamlit as st
import os
import time

from src.web_crawler import WebCrawler
from src.text_processor import TextProcessor
from src.vector_store import VectorStore
from src.chatbot import WebsiteChatbot


# Page configuration
st.set_page_config(
    page_title="Web Assistance Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextInput > label {
        font-size: 1.1rem;
        font-weight: bold;
    }
    .source-box {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #1f77b4;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "current_website" not in st.session_state:
    st.session_state.current_website = ""


def initialize_components():
    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

    crawler = WebCrawler()
    processor = TextProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vector_store = VectorStore(
        persist_directory=persist_dir,
        embedding_model=embedding_model
    )

    return crawler, processor, vector_store


def index_website(url: str):
    with st.spinner("🔍 Crawling website..."):
        crawler, processor, vector_store = initialize_components()

        result = crawler.crawl(url)
        if not result:
            st.error("❌ Failed to crawl website.")
            return False

        st.success(f"✅ Successfully crawled: {result['metadata']['title']}")
        st.info(f"📄 Extracted {len(result['content'])} characters")

    with st.spinner("✂️ Processing text into chunks..."):
        chunks = processor.process(result["content"], result["metadata"])
        st.success(f"✅ Created {len(chunks)} text chunks")

    with st.spinner("🧠 Storing embeddings..."):
        vector_store.clear_collection()
        if not vector_store.add_documents(chunks):
            st.error("❌ Failed to store embeddings.")
            return False
        st.success("✅ Successfully indexed website!")

    # ✅ Initialize chatbot (Ollama – FREE)
    chatbot = WebsiteChatbot(
        vector_store=vector_store,
        model_name="llama3",
        temperature=0.1
    )

    st.session_state.vector_store = vector_store
    st.session_state.chatbot = chatbot
    st.session_state.indexed = True
    st.session_state.current_website = url
    st.session_state.chat_history = []

    return True


def display_chat_interface():
    st.subheader("💬 Ask Questions")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask a question about the website...")

    if question:
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.chatbot.answer_question(question)
                st.write(response["answer"])

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response["answer"]}
        )

        st.rerun()


def main():
    st.markdown('<h1 class="main-header">🤖 Web Assistance Chatbot</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.success("✅ Using FREE Ollama (No API Key)")
        if st.session_state.indexed:
            st.info(f"Indexed: {st.session_state.current_website}")

    if not st.session_state.indexed:
        st.header("🌐 Index a Website")

        url = st.text_input("Enter Website URL", placeholder="https://example.com")
        if st.button("🚀 Index Website"):
            if url:
                if index_website(url):
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
            else:
                st.warning("Please enter a valid URL")
    else:
        display_chat_interface()


if __name__ == "__main__":
    main()
