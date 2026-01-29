# 🤖 Web Assistance Chatbot Using Embeddings

A sophisticated AI-powered chatbot that crawls websites, creates embeddings, and answers questions based strictly on the website content using Retrieval Augmented Generation (RAG).

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Design Decisions](#design-decisions)
- [Limitations & Future Improvements](#limitations--future-improvements)

---

## 🎯 Overview

This project implements an intelligent question-answering system that:
1. Accepts a website URL from users
2. Crawls and extracts meaningful content
3. Processes text into semantic chunks
4. Generates embeddings and stores them in a vector database
5. Answers user questions using RAG with short-term conversational memory

---

## 🏗️ Architecture

```
┌─────────────┐
│   User      │
│  (Streamlit)│
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────┐
│         Application Layer                │
│  ┌────────────────────────────────────┐  │
│  │  Web Crawler                       │  │
│  │  - URL validation                  │  │
│  │  - Content extraction              │  │
│  │  - HTML cleaning                   │  │
│  └────────────────────────────────────┘  │
│                    │                      │
│                    ▼                      │
│  ┌────────────────────────────────────┐  │
│  │  Text Processor                    │  │
│  │  - Semantic chunking               │  │
│  │  - Metadata preservation           │  │
│  │  - Duplicate removal               │  │
│  └────────────────────────────────────┘  │
│                    │                      │
│                    ▼                      │
│  ┌────────────────────────────────────┐  │
│  │  Vector Store                      │  │
│  │  - Embedding generation            │  │
│  │  - ChromaDB storage                │  │
│  │  - Similarity search               │  │
│  └────────────────────────────────────┘  │
│                    │                      │
│                    ▼                      │
│  ┌────────────────────────────────────┐  │
│  │  Chatbot (RAG)                     │  │
│  │  - Context retrieval               │  │
│  │  - LLM generation                  │  │
│  │  - Conversation memory             │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
```

### Data Flow:
1. Input: User provides website URL
2. Crawling: Extract clean content from HTML
3. Processing: Split into semantic chunks with metadata
4. Embedding: Generate vector representations
5. Storage: Persist in ChromaDB
6. Query: User asks questions
7. Retrieval: Find relevant chunks via similarity search
8. Generation: LLM generates grounded answers
9. Output: Display answer with sources

---

## ✨ Features

### Core Features
 **URL Validation & Error Handling**
- Validates URL format
- Handles unreachable websites gracefully
- Checks for empty or unsupported content

 **Intelligent Web Crawling**
- Removes headers, footers, navigation, ads
- Extracts main content using semantic HTML tags
- Avoids duplicate content
- Preserves page metadata (title, URL, domain)

 **Semantic Text Chunking**
- Configurable chunk size and overlap
- Recursive character splitting for natural boundaries
- Metadata preservation per chunk
- Duplicate chunk removal

 **Persistent Vector Storage**
- Uses ChromaDB for efficient storage
- Embeddings generated via SentenceTransformers
- Reusable embeddings (no regeneration per query)
- Similarity search with metadata filtering

 **Grounded Question Answering**
- RAG (Retrieval Augmented Generation)
- Answers strictly from website content
- Explicit "not available" response when information is missing
- Source citations for transparency

 **Short-Term Conversational Memory**
- Maintains context across queries
- Session-based memory (cleared on new website)
- Last 3 conversation exchanges retained

  **User-Friendly Interface**
- Clean Streamlit UI
- Real-time progress indicators
- Chat interface with source viewing
- Easy website switching

---

## 🛠️ Technology Stack

### AI/ML Frameworks
- LangChain (v0.1.9): AI orchestration framework
  - Prompt templates
  - Conversation memory management
  - Chain orchestration
  
- **OpenAI GPT-3.5-Turbo**: Large Language Model
  - Temperature: 0.1 (focused, deterministic)
  - Context window: 16k tokens
  - Cost-effective for production

### Embeddings
- **SentenceTransformers (all-MiniLM-L6-v2)**:
  - 384-dimensional embeddings
  - Fast inference (~0.5s for 100 sentences)
  - Excellent semantic understanding
  - Free and open-source
  - Model size: 80MB

### Vector Database
- **ChromaDB (v0.4.22)**:
  - Persistent storage
  - Efficient similarity search
  - Metadata filtering
  - Low memory footprint
  - No external server required

### Web Scraping
- **BeautifulSoup4**: HTML parsing
- **Requests**: HTTP client
- **html2text**: HTML to markdown conversion

### UI Framework
- **Streamlit (v1.31.0)**: 
  - Rapid prototyping
  - Built-in chat interface
  - Easy deployment

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Ollama (Local LLM) ([Download here](https://ollama.com))

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd website_chatbot
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# Use any text editor:
nano .env
# or
vim .env
```

**Required `.env` configuration:**
```env
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=gpt-3.5-turbo
TEMPERATURE=0.1
```

### Step 5: Run Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## 🚀 Usage

### 1. Index a Website
1. Enter a valid website URL (e.g., `https://en.wikipedia.org/wiki/Artificial_intelligence`)
2. Click **"Index Website"** button
3. Wait for crawling, processing, and embedding generation
4. Success message will appear when ready

### 2. Ask Questions
1. Type your question in the chat input
2. Press Enter or click Send
3. View the AI-generated answer
4. Expand "View Sources" to see relevant content chunks

### 3. View Sources
- Each answer includes source citations
- Click "View Sources" to see exact content used
- Verify answer accuracy against source material

### 4. Clear Memory
- Click "Clear Memory" in sidebar to reset conversation
- Useful when starting a new topic

### 5. Index New Website
- Click "Index New Website" to start over
- Previous website data will be cleared

---

## 📁 Project Structure

```
website_chatbot/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env.example               # Example environment variables
├── README.md                  # This file
│
├── src/                       # Source code modules
│   ├── __init__.py
│   ├── web_crawler.py         # Website crawling & content extraction
│   ├── text_processor.py      # Text chunking & preprocessing
│   ├── vector_store.py        # Embeddings & ChromaDB operations
│   └── chatbot.py             # RAG & question answering
│
├── chroma_db/                 # Persistent vector database (auto-created)
└── venv/                      # Virtual environment (auto-created)
```

### Module Descriptions

#### `web_crawler.py`
- **Purpose:** Crawl websites and extract clean content
- **Key Features:**
  - URL validation
  - Irrelevant content removal (headers, footers, ads)
  - Main content extraction
  - Metadata preservation
- **Main Class:** `WebCrawler`

#### `text_processor.py`
- **Purpose:** Process text into semantic chunks
- **Key Features:**
  - Recursive character splitting
  - Configurable chunk size/overlap
  - Metadata attachment
  - Duplicate removal
- **Main Class:** `TextProcessor`

#### `vector_store.py`
- **Purpose:** Manage embeddings and vector database
- **Key Features:**
  - Embedding generation (SentenceTransformers)
  - ChromaDB persistence
  - Similarity search
  - Collection management
- **Main Class:** `VectorStore`

#### `chatbot.py`
- **Purpose:** RAG-based question answering
- **Key Features:**
  - Context retrieval
  - LLM integration (Ollama)
  - Conversation memory
  - Grounded response generation
- **Main Class:** `WebsiteChatbot`

---

## 🎨 Design Decisions

### 1. **Why ChromaDB?**
**Chosen:** ChromaDB  
**Alternatives Considered:** Pinecone, Weaviate, FAISS

**Justification:**
-  **No external server required** - Embedded database
-  **Persistent storage** - Data survives app restarts
-  **Metadata filtering** - Rich query capabilities
-  **Easy setup** - No configuration complexity
-  **Cost-effective** - Completely free
-  **Production-ready** - Used by major companies

**Trade-offs:**
- ⚠️ Not suitable for massive scale (millions of vectors)
- ⚠️ No built-in clustering for distributed systems

**For this use case:** Perfect fit - single website indexing, local storage, easy deployment.

---

### 2. **Why SentenceTransformers (all-MiniLM-L6-v2)?**
**Chosen:** SentenceTransformers - all-MiniLM-L6-v2  
**Alternatives Considered:** OpenAI embeddings, HuggingFace larger models

**Justification:**
-  **Free & open-source** - No API costs
-  **Fast inference** - 0.5s for 100 sentences
-  **Excellent quality** - 384-dim semantic embeddings
-  **Small model size** - 80MB (easy deployment)
-  **No rate limits** - Unlimited usage
-  **Offline capable** - Works without internet



**For this use case:** Optimal balance of speed, cost, and quality for a fresher project.

---

**Justification:**
-  **Cost-effective** - ~$0.002/1K tokens (20x cheaper than GPT-4)
-  **Fast inference** - 1-2 seconds response time
-  **16K context window** - Handles multiple chunks
-  **Proven reliability** - Industry standard
-  **Easy setup** - Simple API integration


**For this use case:** Best option for a demo/assignment - affordable, fast, reliable.

**Temperature Setting:** 0.1 (low)
- Ensures focused, deterministic answers
- Reduces creativity/hallucination
- Prioritizes accuracy over fluency

---

### 4. **Why LangChain?**
**Chosen:** LangChain  
**Alternatives Considered:** Custom implementation, LlamaIndex

**Justification:**
-  **Rapid development** - Pre-built components
-  **Conversation memory** - Built-in session management
-  **Prompt templates** - Structured prompt engineering
-  **Industry standard** - Well-documented, community support
-  **Extensible** - Easy to add new features

**What we use:**
- `ChatOpenAI` - LLM wrapper
- `ConversationBufferMemory` - Short-term memory
- `ChatPromptTemplate` - Structured prompts

**For this use case:** Accelerates development while maintaining code quality.

---

### 5. **Text Chunking Strategy**
**Configuration:**
- **Chunk Size:** 1000 characters
- **Chunk Overlap:** 200 characters
- **Splitter:** RecursiveCharacterTextSplitter

**Justification:**
-  **Semantic boundaries** - Splits at paragraphs, sentences, then words
-  **Context preservation** - 200-char overlap maintains continuity
-  **Optimal size** - Fits in embedding model context (512 tokens)
-  **Balance** - Not too small (fragmented) or too large (diluted)

**Example:**
```
Original: "AI is transforming... [1500 chars] ...the future."

Chunk 1: "AI is transforming... [1000 chars] ...machine learning."
Chunk 2: "...machine learning... [1000 chars] ...the future."
         ↑ 200-char overlap ↑
```

---

## 🚧 Assumptions & Limitations

### Assumptions
1. **Website Accessibility:**
   - Websites are publicly accessible
   - No authentication required
   - No rate limiting or CAPTCHA

2. **Content Type:**
   - Focus on HTML text content
   - Images/videos not processed
   - PDFs not supported (can be added)

3. **Single URL:**
   - Index one URL per session
   - No multi-page crawling (can be extended)

4. **English Language:**
   - Optimized for English content
   - Multilingual support possible with model changes

5. **API Availability:**
   - OpenAI API is accessible
   - Sufficient API credits available

### Limitations

#### 1. **Crawling Limitations**
- ❌ **Single Page Only:** Only crawls the provided URL, not linked pages
- ❌ **JavaScript-Heavy Sites:** May miss dynamically loaded content
- ❌ **Authentication:** Cannot crawl password-protected pages
- ❌ **Rate Limiting:** No built-in handling for rate limits

**Mitigation:**
- Use Selenium/Playwright for JavaScript rendering
- Implement login flows for authenticated sites
- Add retry logic with exponential backoff

#### 2. **Content Extraction**
- ❌ **Image/Video Content:** Text-only processing
- ❌ **PDF Documents:** Not supported
- ❌ **Tables/Structured Data:** May lose formatting

**Mitigation:**
- Add OCR for images (Tesseract)
- Integrate PDF parsing (PyPDF2)
- Preserve table structure with specialized parsers

#### 3. **Embedding & Search**
- ❌ **Semantic Limitations:** May miss paraphrased queries
- ❌ **Cross-Lingual:** No multilingual support
- ❌ **Domain-Specific:** Generic embeddings may underperform on technical content

**Mitigation:**
- Fine-tune embeddings on domain data
- Use multilingual models (multilingual-MiniLM)
- Implement hybrid search (semantic + keyword)

#### 4. **LLM Limitations**
- ❌ **Context Window:** Limited to ~8-10 chunks per query
- ❌ **Hallucination Risk:** May generate plausible but incorrect answers
- ❌ **Cost:** API costs scale with usage

**Mitigation:**
- Implement strict prompt engineering
- Add confidence scoring
- Consider local LLMs for cost reduction

#### 5. **Memory Constraints**
- ❌ **Short-Term Only:** No long-term user profiles
- ❌ **Session-Based:** Memory lost on restart
- ❌ **Limited History:** Only last 3 exchanges

**Mitigation:**
- Add database for persistent memory
- Implement user accounts
- Expand memory window

#### 6. **Scalability**
- ❌ **Single User:** Not optimized for concurrent users
- ❌ **Local Storage:** ChromaDB embedded mode
- ❌ **No Caching:** Regenerates responses

**Mitigation:**
- Deploy ChromaDB in server mode
- Add Redis caching layer
- Implement user session management

---

## 🔮 Future Improvements

### Phase 1: Enhanced Crawling
1. **Multi-Page Crawling**
   - Follow internal links
   - Configurable depth limit
   - Breadth-first traversal

2. **Advanced Content Extraction**
   - PDF support (PyPDF2, pdfplumber)
   - Image text extraction (OCR with Tesseract)
   - Table preservation with pandas

3. **JavaScript Rendering**
   - Selenium/Playwright integration
   - Handle SPAs (Single Page Apps)
   - Wait for dynamic content

### Phase 2: Improved Retrieval
1. **Hybrid Search**
   - Combine semantic + keyword search
   - BM25 + vector similarity
   - Reranking with cross-encoders

2. **Advanced Chunking**
   - Semantic segmentation (sentence boundaries)
   - Paragraph-aware splitting
   - Markdown structure preservation

3. **Query Enhancement**
   - Query expansion with synonyms
   - Multi-query generation
   - Hypothetical document embeddings (HyDE)

### Phase 3: Better Generation
1. **Answer Quality**
   - Confidence scoring
   - Source credibility ranking
   - Fact verification

2. **LLM Optimization**
   - Fine-tune on Q&A datasets
   - Few-shot learning examples
   - Chain-of-thought prompting

3. **Cost Optimization**
   - Response caching
   - Batch processing
   - Local LLM option (LLaMA, Mistral)

### Phase 4: Production Features
1. **User Management**
   - Authentication (OAuth)
   - User-specific indexes
   - Usage quotas

2. **Monitoring & Analytics**
   - Query analytics
   - Performance metrics
   - Cost tracking

3. **Deployment**
   - Docker containerization
   - Cloud deployment (AWS/GCP/Azure)
   - Auto-scaling

4. **API Development**
   - RESTful API
   - Webhook integrations
   - Rate limiting

### Phase 5: Advanced Features
1. **Multi-Modal**
   - Image understanding (CLIP)
   - Video transcription
   - Audio processing

2. **Multilingual**
   - Auto language detection
   - Translation layer
   - Language-specific models

3. **Collaboration**
   - Shared workspaces
   - Team annotations
   - Export reports

---

### Example Test Cases

**Test Case 1: Valid Wikipedia Page**
```
URL: https://en.wikipedia.org/wiki/Machine_learning
Question: "What is supervised learning?"
Expected: Accurate answer from Wikipedia content
```

**Test Case 2: Question Not in Content**
```
URL: https://en.wikipedia.org/wiki/Machine_learning
Question: "What is quantum computing?"
Expected: "The answer is not available on the provided website."
```

**Test Case 3: Conversation Context**
```
URL: https://en.wikipedia.org/wiki/Python_(programming_language)
Q1: "What is Python?"
Q2: "When was it created?"  # Should understand "it" refers to Python
Expected: Contextual understanding maintained
```

---
