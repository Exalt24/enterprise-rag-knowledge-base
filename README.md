# Enterprise RAG Knowledge Base

Production-ready Retrieval-Augmented Generation system with advanced search capabilities.

## 🎯 Project Goals

Build an intelligent knowledge base that:
- Achieves 90%+ retrieval relevance across 1000+ documents
- Responds to queries in <2s P95 latency
- Supports multi-format documents (PDF, DOCX, TXT, Markdown, CSV)
- Implements advanced RAG techniques (HyDE, hybrid search, reranking)
- Runs on 100% free/open-source technologies

## 🏗️ Architecture

```
┌─────────────┐
│  Documents  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Document Parser │ (PDF, DOCX, TXT)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Chunking   │ (Semantic, Sliding Window)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Embeddings      │ (Sentence Transformers - Local)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vector DB       │ (Chroma - Local Storage)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Hybrid Search   │ (Vector + BM25)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Reranking       │ (Cross-Encoder)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM Generation  │ (Ollama - gpt-oss:20b)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Answer       │
└─────────────────┘
```

## 🛠️ Tech Stack (100% Free!)

**LLM & AI:**
- Ollama (Local LLMs) - gpt-oss:20b, llama3, mixtral
- Sentence Transformers - all-MiniLM-L6-v2 (local embeddings)
- LangChain - RAG orchestration
- LlamaIndex - Advanced retrieval patterns

**Vector Database:**
- Chroma - Open source, local storage
- pgvector - PostgreSQL extension (optional)

**Backend:**
- FastAPI - Python web framework
- Pydantic - Data validation

**Document Processing:**
- pypdf - PDF parsing
- python-docx - DOCX parsing

**Frontend:**
- Next.js (to be added in Week 3-4)

**Deployment:**
- Docker
- Render/Railway (free tier)

## 📁 Project Structure

```
enterprise-rag/
├── app/
│   ├── main.py              # FastAPI application
│   ├── api/
│   │   ├── routes/          # API endpoints
│   │   └── schemas/         # Pydantic models
│   ├── core/
│   │   ├── config.py        # Configuration
│   │   └── rag.py           # RAG pipeline
│   ├── services/
│   │   ├── ingestion.py     # Document ingestion
│   │   ├── embeddings.py    # Embedding generation
│   │   ├── retrieval.py     # Hybrid search
│   │   └── generation.py    # LLM generation
│   └── utils/
│       ├── chunking.py      # Text chunking strategies
│       └── parsers.py       # Document parsers
├── data/
│   ├── chroma/              # Vector database storage
│   └── documents/           # Uploaded documents
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_rag.py
├── .env                     # Environment variables
├── requirements.txt         # Python dependencies
└── test_setup.py           # Setup validation script
```

## 🚀 Quick Start

### 1. Install Ollama

Download from: https://ollama.ai/download

Pull the model:
```bash
ollama pull gpt-oss:20b
# Or: ollama pull llama3
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows Git Bash)
source venv/Scripts/activate

# Activate (Windows CMD)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 4. Test Setup

```bash
python test_setup.py
```

You should see:
```
🚀 Testing Enterprise RAG Setup...
1️⃣ Testing Ollama connection... ✅
2️⃣ Testing Sentence Transformers embeddings... ✅
3️⃣ Testing Chroma vector database... ✅
4️⃣ Testing complete RAG pipeline... ✅
🎉 ALL TESTS PASSED!
```

## 📚 Development Roadmap

### Week 1: Setup & Learning ✅
- [x] Environment setup
- [x] Dependencies installed
- [x] Ollama running
- [x] Basic RAG proof of concept
- [ ] LangChain tutorial
- [ ] RAG architecture design

### Week 2: Core Development
- [ ] Document ingestion pipeline
- [ ] Vector database integration
- [ ] Basic RAG query system
- [ ] Test with sample documents

### Week 3: Advanced Features
- [ ] Hybrid search (vector + BM25)
- [ ] Query optimization (rewriting, expansion)
- [ ] Conversation memory
- [ ] Admin dashboard

### Week 4: Production Polish
- [ ] Comprehensive testing
- [ ] Performance optimization
- [ ] Deployment (Docker)
- [ ] Documentation & demo video

## 🎯 Success Metrics

- ✅ 90%+ retrieval relevance
- ✅ <2s P95 query latency
- ✅ 1000+ documents supported
- ✅ 50+ concurrent users
- ✅ $0 cost per query (using free tools!)

## 📖 Learning Resources

**LangChain:**
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangChain Academy](https://academy.langchain.com/)

**RAG:**
- [Pinecone Learning Center](https://www.pinecone.io/learn/)
- [Advanced RAG Techniques](https://blog.langchain.dev/deconstructing-rag/)

**Ollama:**
- [Ollama Documentation](https://github.com/ollama/ollama)

## 💡 Zero-Cost Advantage

This project runs entirely on free/open-source tools:
- **LLM**: Ollama (local, unlimited usage)
- **Embeddings**: Sentence Transformers (local)
- **Vector DB**: Chroma (open source)
- **Hosting**: Render/Railway free tier

**Total monthly cost: $0** (vs. $140-320 for paid services)

This demonstrates:
- Cost optimization skills
- Resourcefulness
- Self-hosting knowledge
- Production-ready engineering without expensive APIs

## 📝 Notes

- Python 3.13 compatible
- All dependencies have precompiled wheels
- No C++ compiler needed
- Windows, macOS, Linux supported

## 🔗 Links

- Portfolio: [coming soon]
- Live Demo: [coming soon]
- Case Study: [coming soon]

---

Built as part of AI Automation Portfolio Transformation Plan
