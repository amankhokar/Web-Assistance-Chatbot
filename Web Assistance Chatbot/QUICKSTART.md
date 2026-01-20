# 🚀 Quick Start Guide

Get your Website-Based Chatbot running in 5 minutes!

## ⚡ Fast Setup (5 Steps)

### Step 1: Download & Install
```bash
# Clone or download the repository
cd website_chatbot

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**This installs:**
- ✅ Streamlit (UI)
- ✅ LangChain (AI framework)  
- ✅ OpenAI (LLM)
- ✅ ChromaDB (Vector database)
- ✅ SentenceTransformers (Embeddings)
- ✅ BeautifulSoup (Web scraping)

**Installation time:** ~2-3 minutes

### Step 3: Get OpenAI API Key

**Free Trial:** OpenAI provides $5 free credit for new accounts

1. Go to: https://platform.openai.com/signup
2. Create account (use email/Google/Microsoft)
3. Navigate to: https://platform.openai.com/api-keys
4. Click "Create new secret key"
5. Copy the key (starts with `sk-...`)

**Cost:** ~$0.002 per conversation (very cheap!)

### Step 4: Configure Environment
```bash
# Copy example file
cp .env.example .env

# Edit .env file (use any text editor)
# Windows: notepad .env
# macOS: open -e .env
# Linux: nano .env
```

**Paste your API key:**
```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Save and close** the file. Other settings are already optimized!

### Step 5: Run Application
```bash
streamlit run app.py
```

✅ **Browser opens automatically** at http://localhost:8501

---

## 🎯 First Use

### Index Your First Website

1. **Enter a URL** (try these examples):
   - https://en.wikipedia.org/wiki/Artificial_intelligence
   - https://en.wikipedia.org/wiki/Python_(programming_language)
   - https://www.python.org/about/

2. **Click "Index Website"**
   - Wait 10-30 seconds
   - Progress shown in real-time

3. **Ask Questions**:
   - "What is artificial intelligence?"
   - "What are the main applications?"
   - "Who invented Python?"

4. **View Sources**:
   - Click "View Sources" under answers
   - See exact content used

---

## 🧪 Test Your Setup

**Before running the app, verify everything works:**

```bash
python test_setup.py
```

This tests:
- ✅ All packages installed
- ✅ Environment configured
- ✅ Web crawler works
- ✅ Text processor works
- ✅ Vector store works
- ✅ Chatbot works (if API key set)

**Expected output:**
```
Testing Package Imports...
✅ Streamlit imported successfully
✅ LangChain imported successfully
...
Total: 6/6 tests passed

🎉 All tests passed! Your chatbot is ready to use.
```

---

## 💡 Usage Tips

### Best Practices

**Good URLs to try:**
- ✅ Wikipedia articles
- ✅ Documentation sites
- ✅ Blog posts
- ✅ News articles
- ✅ Company about pages

**URLs to avoid:**
- ❌ Login-required pages
- ❌ Heavy JavaScript sites (SPAs)
- ❌ PDF-only pages
- ❌ Download links

### Asking Questions

**Good questions:**
- ✅ "What is [topic] according to this website?"
- ✅ "Who created/invented [something]?"
- ✅ "What are the main benefits of [topic]?"
- ✅ "How does [process] work?"

**Questions that won't work:**
- ❌ Questions about content NOT on the website
- ❌ Current events (unless in content)
- ❌ Math calculations
- ❌ Personal opinions

### Features to Explore

1. **Conversation Memory**
   - Ask follow-up questions
   - Use "it", "this", "that" references
   - Context maintained across conversation

2. **Source Verification**
   - Always check sources
   - Verify accuracy
   - See exact text used

3. **Clear Memory**
   - Reset conversation
   - Start fresh topic
   - Click sidebar button

4. **Index New Website**
   - Switch to different site
   - Previous data cleared
   - Fresh start

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Error
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
```bash
# Ensure virtual environment is activated
pip install -r requirements.txt
```

#### 2. API Key Error
```
OpenAI API key not found
```

**Solution:**
- Check `.env` file exists
- Verify `OPENAI_API_KEY=sk-...` is correct
- No quotes needed around key
- Restart app after editing `.env`

#### 3. Crawling Fails
```
Failed to crawl website
```

**Solutions:**
- Check URL is valid and accessible
- Try different website
- Check internet connection
- Some sites block crawlers

#### 4. Slow Response
```
Taking too long to answer
```

**Causes:**
- OpenAI API latency (normal)
- Large website (many chunks)
- Slow internet

**Solutions:**
- Wait 5-10 seconds
- Try smaller websites first
- Check API status: https://status.openai.com/

#### 5. Port Already in Use
```
Port 8501 is already in use
```

**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

---

## 📱 Access from Other Devices

### Local Network Access

1. **Find your IP address**:
   ```bash
   # Windows
   ipconfig
   
   # macOS/Linux
   ifconfig
   ```

2. **Look for IPv4 address** (e.g., `192.168.1.100`)

3. **Run app with network access**:
   ```bash
   streamlit run app.py --server.address 0.0.0.0
   ```

4. **Access from other devices**:
   ```
   http://192.168.1.100:8501
   ```

---

## 🌐 Share Your App

### Option 1: Streamlit Cloud (Free)

**Easiest way to share publicly:**

1. Push code to GitHub
2. Go to: https://share.streamlit.io
3. Connect repository
4. Add API key in secrets
5. Deploy!

**Result:** Public URL like `https://yourapp.streamlit.app`

**See:** DEPLOYMENT.md for detailed guide

### Option 2: ngrok (Temporary)

**Quick public URL for testing:**

1. Download: https://ngrok.com/download
2. Run:
   ```bash
   # Terminal 1: Run app
   streamlit run app.py
   
   # Terminal 2: Create tunnel
   ngrok http 8501
   ```
3. Share the ngrok URL (e.g., `https://abc123.ngrok.io`)

**Note:** URL changes on restart (free tier)

---

## 📊 What's Happening Behind the Scenes?

### When You Index a Website:

1. **🔍 Crawling** (5-10s)
   - Fetches HTML from URL
   - Removes ads, menus, footers
   - Extracts main content

2. **✂️ Chunking** (1-2s)
   - Splits text into 1000-char chunks
   - Overlaps by 200 chars for context
   - Preserves metadata

3. **🧠 Embedding** (10-20s)
   - Converts text to numbers (vectors)
   - Uses SentenceTransformers model
   - Stores in ChromaDB

4. **💾 Storage** (<1s)
   - Saves to local database
   - Persists on disk
   - Ready for queries

### When You Ask a Question:

1. **🔎 Retrieval** (1s)
   - Converts question to vector
   - Finds 4 most similar chunks
   - Ranks by relevance

2. **🤖 Generation** (2-3s)
   - Sends chunks + question to OpenAI
   - GPT-3.5 generates answer
   - Ensures grounded in content

3. **📝 Response** (<1s)
   - Displays answer
   - Shows sources
   - Saves to memory

**Total time:** ~4-5 seconds per question

---

## 💰 Cost Estimation

### OpenAI API Costs

**GPT-3.5-Turbo Pricing:**
- Input: $0.0015 per 1K tokens
- Output: $0.002 per 1K tokens

**Typical Usage:**
- Question + Context: ~1000 tokens input
- Answer: ~200 tokens output
- **Cost per question: ~$0.002** (0.2 cents!)

**With $5 free credit:**
- ~2,500 questions free
- Perfect for testing/demo

**Monthly estimates:**
- Light use (50 Q/day): ~$3/month
- Medium (200 Q/day): ~$12/month
- Heavy (1000 Q/day): ~$60/month

**Tip:** Monitor usage at https://platform.openai.com/usage

---

## 🎓 Learn More

### Understanding the Tech

**RAG (Retrieval Augmented Generation):**
- Combines search + AI generation
- Prevents hallucination
- Grounds answers in real data

**Vector Embeddings:**
- Converts text to numbers
- Captures semantic meaning
- Enables similarity search

**LangChain:**
- Framework for AI apps
- Handles memory, prompts, chains
- Industry standard

### Resources

**Documentation:**
- Streamlit: https://docs.streamlit.io/
- LangChain: https://python.langchain.com/
- ChromaDB: https://docs.trychroma.com/
- OpenAI: https://platform.openai.com/docs/

**Tutorials:**
- RAG explained: https://www.pinecone.io/learn/retrieval-augmented-generation/
- Building chatbots: https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

---

## 🤝 Get Help

### Support Channels

1. **Check README.md** - Comprehensive documentation
2. **Run test_setup.py** - Diagnose issues
3. **Review logs** - Terminal output shows errors
4. **Check .env file** - Common configuration issues

### Error Reporting

If you find a bug:
1. Note the error message
2. Check what you were doing
3. Try to reproduce
4. Document steps

---

## ✅ Checklist

Before considering setup complete:

- [ ] Virtual environment created and activated
- [ ] All packages installed (`pip list` shows them)
- [ ] `.env` file created with API key
- [ ] `test_setup.py` passes all tests
- [ ] App runs (`streamlit run app.py`)
- [ ] Successfully indexed a website
- [ ] Asked questions and got answers
- [ ] Viewed sources
- [ ] Conversation memory works

**All checked?** 🎉 You're ready to use the chatbot!

---

## 🎯 Next Steps

1. **Try different websites** - See how it handles various content
2. **Explore edge cases** - Ask questions not in content
3. **Test conversation** - Multiple questions in sequence
4. **Review sources** - Verify answer accuracy
5. **Read README.md** - Understand architecture
6. **Deploy to cloud** - Share with others (see DEPLOYMENT.md)

---

## 🌟 Pro Tips

**Optimize for your use case:**
- Adjust chunk size in `.env` (smaller = more precise, larger = more context)
- Change embedding model (larger models = better quality, slower)
- Modify temperature (lower = focused, higher = creative)

**Save costs:**
- Cache common questions
- Use gpt-3.5-turbo (not gpt-4)
- Clear unused databases

**Improve accuracy:**
- Index high-quality sources
- Ask specific questions
- Verify sources always

---

**Happy Chatting! 🤖**

*Built for Humanli.ai Assignment | Data Science Fresher | January 2024*
