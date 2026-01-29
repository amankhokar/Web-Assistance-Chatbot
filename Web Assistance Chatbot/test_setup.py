"""
Test script for Website Chatbot
Run this to verify all components work correctly
"""

import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_imports():
    """Test if all required packages are installed"""
    print("=" * 60)
    print("Testing Package Imports...")
    print("=" * 60)
    
    required_packages = [
        ('streamlit', 'Streamlit'),
        ('langchain', 'LangChain'),
        ('openai', 'OpenAI'),
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
        print(f"\n⚠️  Failed to import: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All packages imported successfully!\n")
    return True


def test_environment():
    """Test environment configuration"""
    print("=" * 60)
    print("Testing Environment Configuration...")
    print("=" * 60)
    
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API Key',
        'CHUNK_SIZE': 'Chunk Size',
        'CHUNK_OVERLAP': 'Chunk Overlap',
        'EMBEDDING_MODEL': 'Embedding Model',
        'LLM_MODEL': 'LLM Model'
    }
    
    missing = []
    for var, name in required_vars.items():
        value = os.getenv(var)
        if value and var != 'OPENAI_API_KEY':
            print(f"✅ {name}: {value}")
        elif value and var == 'OPENAI_API_KEY':
            # Don't show full API key
            masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"✅ {name}: {masked}")
        else:
            print(f"❌ {name}: Not set")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing variables: {', '.join(missing)}")
        print("Create .env file from .env.example and configure values")
        return False
    
    print("\n✅ All environment variables configured!\n")
    return True


def test_web_crawler():
    """Test web crawler module"""
    print("=" * 60)
    print("Testing Web Crawler...")
    print("=" * 60)
    
    try:
        from src.web_crawler import WebCrawler
        
        crawler = WebCrawler()
        print("✅ WebCrawler initialized")
        
        # Test URL validation
        valid_url = "https://example.com"
        if crawler.validate_url(valid_url):
            print(f"✅ URL validation works: {valid_url}")
        else:
            print(f"❌ URL validation failed")
            return False
        
        # Test crawling
        print("\n🔍 Testing crawl on example.com...")
        result = crawler.crawl("https://example.com")
        
        if result and 'content' in result and 'metadata' in result:
            print(f"✅ Successfully crawled!")
            print(f"   Title: {result['metadata']['title']}")
            print(f"   Content length: {len(result['content'])} characters")
            print(f"   Preview: {result['content'][:100]}...")
        else:
            print("❌ Crawling failed")
            return False
        
        print("\n✅ Web Crawler working correctly!\n")
        return True
        
    except Exception as e:
        print(f"❌ Error testing web crawler: {e}")
        return False


def test_text_processor():
    """Test text processor module"""
    print("=" * 60)
    print("Testing Text Processor...")
    print("=" * 60)
    
    try:
        from src.text_processor import TextProcessor
        
        processor = TextProcessor(chunk_size=500, chunk_overlap=50)
        print("✅ TextProcessor initialized")
        
        # Test chunking
        test_text = "Artificial Intelligence is transforming the world. " * 50
        test_metadata = {
            'url': 'https://test.com',
            'title': 'Test Page',
            'domain': 'test.com'
        }
        
        chunks = processor.process(test_text, test_metadata)
        
        if chunks and len(chunks) > 0:
            print(f"✅ Created {len(chunks)} chunks")
            print(f"   First chunk length: {len(chunks[0]['text'])} characters")
            print(f"   Metadata preserved: {chunks[0]['metadata']['page_title']}")
        else:
            print("❌ Chunking failed")
            return False
        
        print("\n✅ Text Processor working correctly!\n")
        return True
        
    except Exception as e:
        print(f"❌ Error testing text processor: {e}")
        return False


def test_vector_store():
    """Test vector store module"""
    print("=" * 60)
    print("Testing Vector Store...")
    print("=" * 60)
    
    try:
        from src.vector_store import VectorStore
        
        # Create test vector store
        vector_store = VectorStore(
            persist_directory="./test_chroma_db",
            embedding_model="all-MiniLM-L6-v2"
        )
        print("✅ VectorStore initialized")
        
        # Test adding documents
        test_chunks = [
            {
                'text': 'Machine learning is a subset of artificial intelligence.',
                'metadata': {
                    'source_url': 'test.com',
                    'page_title': 'ML Test',
                    'chunk_index': 0,
                    'total_chunks': 2,
                    'domain': 'test.com'
                }
            },
            {
                'text': 'Deep learning uses neural networks with multiple layers.',
                'metadata': {
                    'source_url': 'test.com',
                    'page_title': 'ML Test',
                    'chunk_index': 1,
                    'total_chunks': 2,
                    'domain': 'test.com'
                }
            }
        ]
        
        success = vector_store.add_documents(test_chunks)
        if success:
            print(f"✅ Added {len(test_chunks)} documents to vector store")
        else:
            print("❌ Failed to add documents")
            return False
        
        # Test similarity search
        results = vector_store.similarity_search("What is machine learning?", n_results=2)
        if results and len(results) > 0:
            print(f"✅ Similarity search found {len(results)} results")
            print(f"   Top result: {results[0]['text'][:80]}...")
        else:
            print("❌ Similarity search failed")
            return False
        
        # Clean up
        vector_store.clear_collection()
        print("✅ Cleaned up test database")
        
        print("\n✅ Vector Store working correctly!\n")
        return True
        
    except Exception as e:
        print(f"❌ Error testing vector store: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chatbot():
    """Test chatbot module (requires OpenAI API key)"""
    print("=" * 60)
    print("Testing Chatbot (RAG)...")
    print("=" * 60)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your_openai_api_key_here':
        print("⚠️  Skipping chatbot test - OpenAI API key not configured")
        print("   Configure OPENAI_API_KEY in .env to test chatbot functionality")
        return True
    
    try:
        from src.vector_store import VectorStore
        from src.chatbot import WebsiteChatbot
        
        # Create vector store with test data
        vector_store = VectorStore(
            persist_directory="./test_chroma_db",
            embedding_model="all-MiniLM-L6-v2"
        )
        
        test_chunks = [
            {
                'text': 'Python is a high-level programming language created by Guido van Rossum in 1991.',
                'metadata': {
                    'source_url': 'test.com',
                    'page_title': 'Python',
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'domain': 'test.com'
                }
            }
        ]
        
        vector_store.add_documents(test_chunks)
        print("✅ Test data indexed")
        
        # Initialize chatbot
        chatbot = WebsiteChatbot(
            vector_store=vector_store,
            openai_api_key=api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.1
        )
        print("✅ Chatbot initialized")
        
        # Test question answering
        print("\n🤔 Testing question: 'Who created Python?'")
        response = chatbot.answer_question("Who created Python?")
        
        if response and 'answer' in response:
            print(f"✅ Answer generated: {response['answer'][:100]}...")
            print(f"   Confidence: {response['confidence']}")
            print(f"   Sources: {len(response['sources'])}")
        else:
            print("❌ Failed to generate answer")
            return False
        
        # Test "not available" response
        print("\n🤔 Testing unavailable question: 'What is quantum computing?'")
        response = chatbot.answer_question("What is quantum computing?")
        
        if "not available" in response['answer'].lower():
            print("✅ Correctly returned 'not available' response")
        else:
            print("⚠️  May have hallucinated an answer")
        
        # Clean up
        vector_store.clear_collection()
        print("✅ Cleaned up test database")
        
        print("\n✅ Chatbot working correctly!\n")
        return True
        
    except Exception as e:
        print(f"❌ Error testing chatbot: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print(" WEBSITE CHATBOT - COMPONENT TESTING")
    print("=" * 60 + "\n")
    
    tests = [
        ("Package Imports", test_imports),
        ("Environment Config", test_environment),
        ("Web Crawler", test_web_crawler),
        ("Text Processor", test_text_processor),
        ("Vector Store", test_vector_store),
        ("Chatbot (RAG)", test_chatbot),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results[name] = False
        
        print()  # Spacing
    
    # Summary
    print("=" * 60)
    print(" TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:.<40} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your chatbot is ready to use.")
        print("\nRun: streamlit run app.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
