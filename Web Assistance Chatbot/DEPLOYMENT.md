# 🚀 Deployment Guide

Complete guide for deploying the Website-Based Chatbot application.

## 📋 Table of Contents
1. [Local Deployment](#local-deployment)
2. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Production Considerations](#production-considerations)

---

## 🏠 Local Deployment

### Prerequisites
- Python 3.8+
- pip package manager
- OpenAI API key

### Step-by-Step Guide

#### 1. Clone Repository
```bash
git clone <repository-url>
cd website_chatbot
```

#### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- Streamlit for UI
- LangChain for AI orchestration
- OpenAI for LLM
- ChromaDB for vector storage
- SentenceTransformers for embeddings
- Web scraping libraries

#### 4. Configure Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit .env file and add your OpenAI API key
# On Windows: notepad .env
# On macOS/Linux: nano .env
```

**Required configuration in `.env`:**
```env
OPENAI_API_KEY=sk-your-actual-api-key-here
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=gpt-3.5-turbo
TEMPERATURE=0.1
```

**Get OpenAI API Key:**
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new secret key
5. Copy and paste into `.env` file

#### 5. Test Setup
```bash
# Run test script to verify everything works
python test_setup.py
```

This will test:
- ✅ Package imports
- ✅ Environment configuration
- ✅ Web crawler functionality
- ✅ Text processing
- ✅ Vector store operations
- ✅ Chatbot (if API key configured)

#### 6. Run Application
```bash
streamlit run app.py
```

The app will open in your browser at: `http://localhost:8501`

**Troubleshooting:**
- If port 8501 is busy: `streamlit run app.py --server.port 8502`
- If browser doesn't open: Manually navigate to the URL shown in terminal

---

## ☁️ Streamlit Cloud Deployment

Deploy your app for free on Streamlit Cloud (recommended for demo/sharing).

### Prerequisites
- GitHub account
- Streamlit Cloud account (free)
- Code pushed to GitHub repository

### Step-by-Step Guide

#### 1. Prepare Repository

**Create GitHub repository:**
```bash
# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Website Chatbot"

# Create GitHub repo and push
git remote add origin https://github.com/your-username/website-chatbot.git
git branch -M main
git push -u origin main
```

**Ensure these files are in your repo:**
- ✅ `app.py` - Main Streamlit app
- ✅ `requirements.txt` - Dependencies
- ✅ `src/` directory - All modules
- ✅ `.gitignore` - Exclude sensitive files
- ❌ `.env` - DO NOT commit (use secrets instead)

#### 2. Deploy to Streamlit Cloud

**Step 2.1:** Go to [share.streamlit.io](https://share.streamlit.io)

**Step 2.2:** Sign in with GitHub

**Step 2.3:** Click "New app"

**Step 2.4:** Configure deployment:
- **Repository:** your-username/website-chatbot
- **Branch:** main
- **Main file path:** app.py

**Step 2.5:** Add secrets (OpenAI API key)
- Click "Advanced settings"
- Go to "Secrets" section
- Add your environment variables:

```toml
# .streamlit/secrets.toml format
OPENAI_API_KEY = "sk-your-actual-api-key-here"
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
CHUNK_SIZE = "1000"
CHUNK_OVERLAP = "200"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-3.5-turbo"
TEMPERATURE = "0.1"
```

**Step 2.6:** Click "Deploy"

**Wait for deployment** (usually 5-10 minutes for first deployment)

**Your app URL:** `https://your-app-name.streamlit.app`

### Update Deployed App

When you make changes:
```bash
git add .
git commit -m "Update feature X"
git push origin main
```

Streamlit Cloud will automatically redeploy (takes ~2-5 minutes).

### Streamlit Cloud Limitations

**Free Tier:**
- ✅ 1 GB RAM
- ✅ 1 CPU core
- ✅ Unlimited visitors (with rate limits)
- ⚠️ Apps go to sleep after inactivity
- ⚠️ Wake-up time: ~30 seconds

**Storage:**
- ChromaDB data persists during session
- Lost on app restart (design consideration)
- For persistent storage, consider managed ChromaDB or external DB

---

## 🐳 Docker Deployment

Containerized deployment for production environments.

### Dockerfile

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for ChromaDB
RUN mkdir -p /app/chroma_db

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build Docker image
docker build -t website-chatbot .

# Run container
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your-key-here \
  -e CHROMA_PERSIST_DIRECTORY=/app/chroma_db \
  -v $(pwd)/chroma_db:/app/chroma_db \
  website-chatbot
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  chatbot:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - CHROMA_PERSIST_DIRECTORY=/app/chroma_db
      - CHUNK_SIZE=1000
      - CHUNK_OVERLAP=200
      - EMBEDDING_MODEL=all-MiniLM-L6-v2
      - LLM_MODEL=gpt-3.5-turbo
      - TEMPERATURE=0.1
    volumes:
      - ./chroma_db:/app/chroma_db
    restart: unless-stopped
```

**Run with Docker Compose:**
```bash
# Create .env file with OPENAI_API_KEY
echo "OPENAI_API_KEY=sk-your-key" > .env

# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 🏭 Production Considerations

### 1. Environment Configuration

**Separate environments:**
```
.env.development
.env.staging
.env.production
```

**Load based on environment:**
```python
import os
env = os.getenv('APP_ENV', 'development')
load_dotenv(f'.env.{env}')
```

### 2. Secrets Management

**Options:**
- **AWS Secrets Manager** - For AWS deployments
- **Google Secret Manager** - For GCP
- **Azure Key Vault** - For Azure
- **HashiCorp Vault** - Platform-agnostic

**Example with AWS Secrets Manager:**
```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

secrets = get_secret('website-chatbot/prod')
openai_api_key = secrets['OPENAI_API_KEY']
```

### 3. Database Configuration

**For production, consider:**
- **ChromaDB Server Mode** - Run as separate service
- **Managed Vector DB** - Pinecone, Weaviate, Qdrant Cloud

**ChromaDB Server Mode:**
```python
# Server deployment
chroma run --path /chroma/data --port 8000

# Client connection
import chromadb
client = chromadb.HttpClient(host='chroma-server', port=8000)
```

### 4. Monitoring & Logging

**Structured logging:**
```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module
        }
        return json.dumps(log_data)

# Configure
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger()
logger.addHandler(handler)
```

**Monitoring tools:**
- **Application:** Prometheus + Grafana
- **Logs:** ELK Stack (Elasticsearch, Logstash, Kibana)
- **APM:** New Relic, Datadog, AppDynamics

### 5. Scaling Considerations

**Horizontal Scaling:**
```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: website-chatbot
spec:
  replicas: 3  # Multiple instances
  selector:
    matchLabels:
      app: chatbot
  template:
    metadata:
      labels:
        app: chatbot
    spec:
      containers:
      - name: chatbot
        image: website-chatbot:latest
        ports:
        - containerPort: 8501
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: chatbot-secrets
              key: openai-api-key
```

**Load Balancing:**
- Use NGINX or cloud load balancers
- Session affinity for conversation memory
- Health checks endpoint

### 6. Caching Strategy

**Response caching:**
```python
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_response(ttl=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(query):
            # Check cache
            cached = redis_client.get(query)
            if cached:
                return json.loads(cached)
            
            # Generate response
            response = func(query)
            
            # Cache result
            redis_client.setex(query, ttl, json.dumps(response))
            return response
        return wrapper
    return decorator

@cache_response(ttl=3600)
def answer_question(query):
    # ... chatbot logic
    pass
```

### 7. Cost Optimization

**OpenAI API:**
- Monitor token usage
- Set spending limits in OpenAI dashboard
- Implement rate limiting per user
- Cache frequent queries

**Example token tracking:**
```python
import tiktoken

def count_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Track before API call
prompt_tokens = count_tokens(prompt)
logger.info(f"Prompt tokens: {prompt_tokens}")
```

### 8. Security Best Practices

**Input validation:**
```python
import validators
from urllib.parse import urlparse

def validate_url(url):
    # Check format
    if not validators.url(url):
        raise ValueError("Invalid URL format")
    
    # Block internal IPs
    parsed = urlparse(url)
    if parsed.hostname in ['localhost', '127.0.0.1']:
        raise ValueError("Localhost URLs not allowed")
    
    # Check scheme
    if parsed.scheme not in ['http', 'https']:
        raise ValueError("Only HTTP/HTTPS allowed")
    
    return True
```

**Rate limiting:**
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@limiter.limit("10/minute")
def ask_question(request):
    # ... chatbot logic
    pass
```

### 9. Backup & Recovery

**Vector DB backups:**
```bash
# Backup ChromaDB
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz chroma_db/

# Automate with cron
0 2 * * * /path/to/backup_script.sh
```

**Disaster recovery plan:**
1. Regular database backups
2. Configuration versioning (Git)
3. Infrastructure as Code (Terraform)
4. Documented recovery procedures

---

## 📊 Performance Benchmarks

**Expected Performance:**
- **Crawling:** 2-5 seconds per page
- **Chunking:** <1 second for 10K chars
- **Embedding:** ~2 seconds per 100 chunks
- **Query response:** 2-4 seconds total

**Optimization tips:**
- Batch embedding generation
- Use faster embedding models for large scale
- Implement query caching
- Optimize chunk size based on use case

---

## 🔗 Additional Resources

### Cloud Platforms
- [AWS App Runner](https://aws.amazon.com/apprunner/) - Easy container deployment
- [Google Cloud Run](https://cloud.google.com/run) - Serverless containers
- [Azure Container Instances](https://azure.microsoft.com/en-us/services/container-instances/) - Quick container hosting
- [Heroku](https://www.heroku.com/) - Simple platform (paid)
- [Railway](https://railway.app/) - Modern deployment platform

### Managed Vector Databases
- [Pinecone](https://www.pinecone.io/) - Managed vector DB
- [Weaviate Cloud](https://weaviate.io/developers/weaviate/installation/weaviate-cloud-services) - Open-source vector DB
- [Qdrant Cloud](https://qdrant.tech/documentation/cloud/) - High-performance vector DB

---

## 📞 Support

For deployment issues:
1. Check application logs
2. Verify environment variables
3. Test locally first
4. Review cloud platform documentation

---

**Last Updated:** January 2024  
**Version:** 1.0.0
