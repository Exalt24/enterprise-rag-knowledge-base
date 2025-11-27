# Week 1 Learning Guide - Enterprise RAG

## Goal
Understand LangChain fundamentals and RAG architecture deeply enough to build production-ready systems.

---

## Day 1-2: LangChain Fundamentals (4-6 hours)

### Part 1: Core Concepts (2 hours)

**Read & Follow Along:**
1. **LangChain Introduction** (30 min)
   - https://python.langchain.com/docs/introduction/
   - Understand: What LangChain solves, why it exists

2. **LangChain Expression Language (LCEL)** (1 hour)
   - https://python.langchain.com/docs/concepts/lcel/
   - This is how you chain operations together
   - **Practice**: Open Python in your terminal and try examples

3. **Prompts & Prompt Templates** (30 min)
   - https://python.langchain.com/docs/concepts/prompt_templates/
   - Learn to structure prompts dynamically

**Hands-On Practice:**
Create `exercises/01_basics.py` and experiment with:
```python
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

llm = OllamaLLM(model="llama3")

# Exercise 1: Simple prompt
response = llm.invoke("Explain RAG in one sentence")
print(response)

# Exercise 2: Template with variables
template = ChatPromptTemplate.from_template(
    "You are a {role}. Explain {topic} in simple terms."
)
chain = template | llm
result = chain.invoke({"role": "teacher", "topic": "vector databases"})
print(result)
```

### Part 2: Chains & Output Parsers (2 hours)

**Read:**
1. **Chains** (45 min)
   - https://python.langchain.com/docs/concepts/chains/
   - Understand: Sequential processing, data flow

2. **Output Parsers** (45 min)
   - https://python.langchain.com/docs/concepts/output_parsers/
   - Learn to structure LLM responses

3. **Runnable Interface** (30 min)
   - https://python.langchain.com/docs/concepts/runnables/
   - The foundation of LCEL

**Hands-On Practice:**
Create `exercises/02_chains.py`:
```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3")

# Exercise: Multi-step chain
step1 = ChatPromptTemplate.from_template("Generate 3 topics about {subject}")
step2 = ChatPromptTemplate.from_template("Pick the most interesting from: {topics}")

chain = (
    {"topics": step1 | llm}
    | step2
    | llm
    | StrOutputParser()
)

result = chain.invoke({"subject": "AI automation"})
print(result)
```

### Part 3: Retrieval & Vector Stores (2 hours)

**Read:**
1. **Embeddings** (30 min)
   - https://python.langchain.com/docs/concepts/embeddings/
   - Understand: How text becomes vectors

2. **Vector Stores** (45 min)
   - https://python.langchain.com/docs/concepts/vectorstores/
   - Learn: Storage, retrieval, similarity search

3. **Retrievers** (45 min)
   - https://python.langchain.com/docs/concepts/retrievers/
   - Different retrieval strategies

**Hands-On Practice:**
Create `exercises/03_retrieval.py`:
```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Sample knowledge base
docs = [
    Document(page_content="LangChain is a framework for LLM apps", metadata={"source": "doc1"}),
    Document(page_content="RAG combines retrieval with generation", metadata={"source": "doc2"}),
    Document(page_content="Vector databases enable semantic search", metadata={"source": "doc3"}),
    Document(page_content="Embeddings convert text to numbers", metadata={"source": "doc4"}),
]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./data/learning_chroma"
)

# Try different searches
query = "What is RAG?"
results = vectorstore.similarity_search(query, k=2)

for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Source: {doc.metadata['source']}\n")
```

---

## Day 3: RAG Architecture (2-3 hours)

### Part 1: RAG Fundamentals (1.5 hours)

**Read These Articles:**
1. **What is RAG?** (30 min)
   - https://www.pinecone.io/learn/retrieval-augmented-generation/
   - Understand: Problem RAG solves, how it works

2. **RAG Architecture Patterns** (30 min)
   - https://blog.langchain.dev/deconstructing-rag/
   - Learn: Different RAG approaches

3. **Advanced RAG Techniques** (30 min)
   - https://www.pinecone.io/learn/advanced-rag-techniques/
   - Study: HyDE, Query rewriting, Reranking

**Key Concepts to Master:**
- Retrieval vs Generation
- Chunking strategies
- Embedding quality
- Context window management
- Source attribution

### Part 2: Build Complete RAG Chain (1.5 hours)

**Hands-On Practice:**
Create `exercises/04_rag_complete.py`:
```python
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# Initialize components
llm = OllamaLLM(model="llama3")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create sample knowledge base about your project
knowledge = [
    Document(page_content="This RAG system uses Llama 3 as the LLM, running locally via Ollama for unlimited free usage."),
    Document(page_content="Chroma is used as the vector database, providing semantic search capabilities."),
    Document(page_content="Sentence Transformers generate embeddings locally, avoiding API costs."),
    Document(page_content="The system achieves 90%+ retrieval relevance with sub-2s query latency."),
    Document(page_content="FastAPI serves the backend, providing REST endpoints for document upload and queries."),
]

vectorstore = Chroma.from_documents(
    documents=knowledge,
    embedding=embeddings,
    persist_directory="./data/rag_learning"
)

# Build RAG chain
template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer: Provide a clear, concise answer based on the context. If the answer isn't in the context, say "I don't have that information."
"""

prompt = ChatPromptTemplate.from_template(template)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Test queries
questions = [
    "What LLM does this system use?",
    "How does the system achieve cost efficiency?",
    "What is the query latency?",
    "Does it support video processing?",  # Not in context
]

for q in questions:
    print(f"\nQ: {q}")
    answer = rag_chain.invoke(q)
    print(f"A: {answer}")
    print("-" * 80)
```

**Run it and observe:**
- How retrieval works
- Context passed to LLM
- Responses when info is/isn't available

---

## Day 4: Design Your Architecture (2-3 hours)

### Task: Design Enterprise RAG System

**Create `ARCHITECTURE.md` with:**

1. **System Overview**
   - High-level diagram (draw.io or text-based)
   - Component responsibilities
   - Data flow

2. **Components:**
   - Document Ingestion Pipeline
   - Chunking Strategy
   - Embedding Generation
   - Vector Database
   - Retrieval Logic
   - LLM Integration
   - API Layer

3. **Technology Choices:**
   - Why Llama 3?
   - Why Chroma?
   - Why FastAPI?
   - Alternatives considered

4. **Advanced Features:**
   - Hybrid Search (Vector + BM25)
   - Query Optimization
   - HyDE technique
   - Reranking
   - Conversation Memory

5. **Success Metrics:**
   - Retrieval relevance: 90%+
   - Query latency: <2s P95
   - Documents supported: 1000+
   - Concurrent users: 50+

**Template:**
```markdown
# Enterprise RAG Architecture

## System Diagram
[Document Upload] → [Parser] → [Chunker] → [Embeddings] → [Chroma]
                                                               ↓
[User Query] → [Query Optimizer] → [Retriever] → [Reranker] → [LLM] → [Answer]

## Components Detail

### 1. Document Ingestion
- **Parsers**: pypdf (PDF), python-docx (DOCX)
- **Chunking**: Semantic chunking with 500-token chunks, 50-token overlap
- **Why**: Semantic chunking preserves context better than fixed-size

... (continue for each component)
```

---

## Checkpoint Exercises

Before moving to Week 2, you should be able to:

### ✅ Exercise 1: Explain These Concepts
- What is LCEL and why use it?
- Difference between embeddings and LLM?
- Why chunk documents before embedding?
- What makes a good RAG prompt?

### ✅ Exercise 2: Build from Scratch
Create `exercises/05_from_scratch.py`:
- Load 5 documents about any topic
- Create embeddings
- Store in Chroma
- Build RAG chain
- Answer 3 questions
- Add conversation memory (bonus)

### ✅ Exercise 3: Compare Approaches
Try different chunking sizes (100, 500, 1000 tokens):
- Which gives better retrieval?
- Which is faster?
- Document your findings

---

## Additional Resources

**Video Tutorials (Optional):**
- LangChain Crash Course: https://www.youtube.com/watch?v=LbT1yp6quS8
- RAG from Scratch: https://www.youtube.com/watch?v=sVcwVQRHIc8

**Documentation to Bookmark:**
- LangChain Docs: https://python.langchain.com/
- Chroma Docs: https://docs.trychroma.com/
- Sentence Transformers: https://www.sbert.net/

**Communities:**
- LangChain Discord: https://discord.gg/langchain
- Reddit r/LangChain: https://reddit.com/r/LangChain

---

## Time Tracking

- [ ] Day 1-2: LangChain fundamentals (4-6 hours)
- [ ] Day 3: RAG architecture (2-3 hours)
- [ ] Day 4: Design system (2-3 hours)

**Total: 8-12 hours over 4-5 days**

---

## Week 1 Completion Criteria

You're ready for Week 2 when you can:
- ✅ Build a RAG chain from scratch without tutorials
- ✅ Explain why each component exists
- ✅ Design your system architecture on paper
- ✅ Understand trade-offs (chunking size, retrieval strategies)

**After Week 1, you'll start building the production system!**
