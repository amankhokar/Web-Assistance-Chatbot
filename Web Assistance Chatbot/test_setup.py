"""
Test script for Website Chatbot 
Run this to verify all components work correctly
"""

import os
from dotenv import load_dotenv

# Load environment
load_dotenv()


def test_imports():
    print("=" * 60)
    print("Testing Package Imports...")
    print("=" * 60)

    required_packages = [
        ('streamlit', 'Streamlit'),
        ('langchain', 'LangChain'),
        ('chromadb', 'ChromaDB'),
        ('sentence_transformers', 'SentenceTransformers'),
        ('bs4', 'BeautifulSoup4'),
        ('requests', 'Requests'),
    ]

    failed = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name} imported successfully")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            failed.append(name)

    if failed:
        print(f"\n⚠️ Failed to import: {', '.join(failed)}")
        return False

    print("\n✅ All packages imported successfully!\n")
    return True


def test_environment():
    print("=" * 60)
    print("Testing Environment Configuration...")
    print("=" * 60)

    required_vars = {
        'CHUNK_SIZE': 'Chunk Size',
        'CHUNK_OVERLAP': 'Chunk Overlap',
        'EMBEDDING_MODEL': 'Embedding Model',
        'OLLAMA_MODEL': 'Ollama Model'
    }

    missing = []
    for var, name in required_vars.items():
        value = os.getenv(var)
        if value:
            print(f"✅ {name}: {value}")
        else:
            print(f"❌ {name}: Not set")
            missing.append(name)

    if missing:
        print(f"\n⚠️ Missing variables: {', '.join(missing)}")
        print("Check your .env file")
        return False

    print("\n✅ Environment configured correctly!\n")
    return True


def test_web_crawler():
    print("=" * 60)
    print("Testing Web Crawler...")
    print("=" * 60)

    from src.web_crawler import WebCrawler

    crawler = WebCrawler()
    result = crawler.crawl("https://example.com")

    if result and "content" in result:
        print("✅ Web crawler works")
        print(f"   Title: {result['metadata']['title']}")
        return True

    print("❌ Web crawler failed")
    return False


def test_text_processor():
    print("=" * 60)
    print("Testing Text Processor...")
    print("=" * 60)

    from src.text_processor import TextProcessor

    processor = TextProcessor(chunk_size=500, chunk_overlap=50)

    text = "Artificial Intelligence is changing the world. " * 40
    metadata = {
        "url": "https://test.com",
        "title": "Test Page",
        "domain": "test.com"
    }

    chunks = processor.process(text, metadata)

    if chunks and len(chunks) > 0:
        print(f"✅ Created {len(chunks)} chunks")
        return True

    print("❌ Text processor failed")
    return False


def test_vector_store():
    print("=" * 60)
    print("Testing Vector Store...")
    print("=" * 60)

    from src.vector_store import VectorStore

    vector_store = VectorStore(
        persist_directory="./test_chroma_db",
        embedding_model="all-MiniLM-L6-v2"
    )

    test_chunks = [
        {
            "text": "Machine learning is a part of artificial intelligence.",
            "metadata": {
                "source_url": "test.com",
                "page_title": "ML",
                "chunk_index": 0,
                "total_chunks": 1,
                "domain": "test.com"
            }
        }
    ]

    success = vector_store.add_documents(test_chunks)

    if not success:
        print("❌ Failed to add documents")
        return False

    results = vector_store.similarity_search("What is machine learning?", n_results=1)

    if results:
        print("✅ Vector store similarity search works")
        vector_store.clear_collection()
        return True

    print("❌ Vector store search failed")
    return False


def test_chatbot():
    print("=" * 60)
    print("Testing Chatbot with Ollama...")
    print("=" * 60)

    from src.vector_store import VectorStore
    from src.chatbot import WebsiteChatbot

    vector_store = VectorStore(
        persist_directory="./test_chroma_db",
        embedding_model="all-MiniLM-L6-v2"
    )

    vector_store.add_documents([
        {
            "text": "Python was created by Guido van Rossum in 1991.",
            "metadata": {
                "source_url": "test.com",
                "page_title": "Python",
                "chunk_index": 0,
                "total_chunks": 1,
                "domain": "test.com"
            }
        }
    ])

    chatbot = WebsiteChatbot(
        vector_store=vector_store,
        model_name=os.getenv("OLLAMA_MODEL", "llama3"),
        temperature=0.1
    )

    response = chatbot.answer_question("Who created Python?")

    if response and "answer" in response:
        print(f"✅ Ollama answered: {response['answer'][:80]}...")
        vector_store.clear_collection()
        return True

    print("❌ Ollama chatbot failed")
    return False


def main():
    print("\n" + "=" * 60)
    print(" WEBSITE CHATBOT - OLLAMA TESTING")
    print("=" * 60 + "\n")

    tests = [
        ("Imports", test_imports),
        ("Environment", test_environment),
        ("Web Crawler", test_web_crawler),
        ("Text Processor", test_text_processor),
        ("Vector Store", test_vector_store),
        ("Chatbot (Ollama)", test_chatbot),
    ]

    passed = 0
    for name, test in tests:
        if test():
            passed += 1
        print()

    print("=" * 60)
    print(f"RESULT: {passed}/{len(tests)} tests passed")
    print("=" * 60)

    if passed == len(tests):
        print("\n🎉 All tests passed! Your Ollama chatbot is ready.")
        print("Run: streamlit run app.py")
    else:
        print("\n⚠️ Some tests failed. Fix issues before running the app.")


if __name__ == "__main__":
    main()
