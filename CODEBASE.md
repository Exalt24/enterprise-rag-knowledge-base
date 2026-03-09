ENTERPRISE RAG KNOWLEDGE BASE -- INTERVIEW CHEAT SHEET

ONE-LINER

"It's a RAG knowledge base with hybrid vector-plus-keyword search, cross-encoder reranking, and a 2-tier LLM fallback, deployed on Render's free tier with 512MB RAM constraints. Users upload documents, ask questions, and get cited answers."


QUICK FACTS

What it does: Upload documents, ask questions, get AI answers with source citations
Frontend: Next.js 16, React 19, Tailwind CSS 4
Backend: FastAPI (Python), LangChain for LLM abstraction
LLM (Production): Groq llama-3.3-70b-versatile (temperature 0.1)
LLM (Dev): Ollama llama3 (local, free)
Embedding model: all-MiniLM-L6-v2 (Sentence Transformers), 384 dimensions
Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (disabled on Render due to RAM)
Vector DB: Qdrant Cloud (migrated from Chroma)
Cache: Redis (Upstash), 1-hour TTL, with in-memory fallback
Key metric: 67.7% retrieval accuracy with hybrid search + reranking
Deployment: Backend on Render (Docker, 512MB RAM), Frontend on Vercel
Live Demo: https://enterprise-rag-knowledge-base.vercel.app
GitHub: https://github.com/Exalt24/EnterpriseRAGKnowledgeBase


ARCHITECTURE IN PLAIN ENGLISH

A user uploads a document (PDF, DOCX, TXT, or MD). The backend parses it, splits it into 500-character chunks with 50-character overlap using RecursiveCharacterTextSplitter, generates 384-dimensional embeddings with all-MiniLM-L6-v2, and stores the vectors plus metadata in Qdrant Cloud. When the user asks a question, the system picks a retrieval strategy (basic vector, BM25 hybrid, or reranking), finds the most relevant chunks, passes them as context to the LLM along with the question, and gets back a natural language answer with source citations. Redis caches query results so the same question returns instantly the second time.

Note on HyDE and Multi-Query: The codebase has implementations for HyDE (Hypothetical Document Embeddings) and Multi-Query retrieval in advanced_retrieval.py, but these are not exposed through the API. They were built during development and testing but never wired up to the query endpoint. The three strategies actually accessible to users are basic vector search, hybrid (BM25 + vector), and reranking.


EVERY POSSIBLE INTERVIEW QUESTION

What/How Questions

Q: What's your retrieval accuracy?

"67.7% tested accuracy with hybrid search plus cross-encoder reranking across evaluation queries. Basic vector search alone gets about 40%. Adding BM25 keyword matching bumps it to around 60%. Cross-encoder reranking on top gets the best results at 67.7%. Honestly, that number has room to grow. The biggest single improvement came from adding BM25 hybrid search, not from the reranker. If I were optimizing further, I'd look at chunking strategy and embedding model before anything else, since those are the foundation everything else sits on."

Q: What embedding model are you using and why?

"all-MiniLM-L6-v2 from Sentence Transformers. 384-dimensional vectors, about 22MB model size, runs on CPU at roughly 500 embeddings per second. I chose it because I'm on Render's 512MB RAM free tier, so the model needed to be tiny and run without a GPU.

The honest trade-off: this is a 2019-era model and it shows. On modern benchmarks like MTEB, it scores significantly lower than newer alternatives like nomic-embed-text-v1.5, BGE-base-en-v1.5, or even E5-small-v2, which offer much better retrieval accuracy in similar or slightly larger footprints. Its 512-token max sequence length is also limiting since newer models handle 2k-8k tokens. If I were rebuilding today, I'd probably go with nomic-embed-text or BGE-base since they're still small enough for CPU inference but meaningfully better at retrieval. The reason I stuck with MiniLM was that it was the most battle-tested option when I built this, and with 512MB RAM as my hard constraint, I was being conservative.

Normalization is enabled for cosine similarity, which is important because Qdrant's dot product similarity assumes normalized vectors."

Q: What retrieval strategies does the system support?

"Three strategies are accessible through the API. Basic vector search uses cosine similarity on the 384-dim embeddings, which is the simplest path. BM25 hybrid combines semantic vector search with keyword matching, weighted 70% vector and 30% BM25 by default. Reranking takes results from either of those and runs them through a cross-encoder that scores each query-document pair for relevance.

I should be upfront: the codebase also has implementations for HyDE (Hypothetical Document Embeddings) and Multi-Query retrieval in the advanced_retrieval service. HyDE generates a hypothetical answer with the LLM first, then searches for documents similar to that answer. Multi-Query generates 3 variations of the original question, searches with all of them, then deduplicates. But neither of these is wired up to the API endpoint. They were experiments during development that showed promise in testing but I never prioritized exposing them. So while the code exists, users can only access the three main strategies. If I had more time, HyDE in particular would be worth exposing since it showed real improvements for questions with vocabulary gaps."

Q: What's HyDE and why did you implement it? (if they dig into the code)

"Hypothetical Document Embeddings. Instead of searching with the user's question directly, I have the LLM generate a hypothetical answer first, then embed that answer and search for similar documents. It bridges the vocabulary gap between how people ask questions and how documents are written. A user might ask 'What's the company's revenue?' but the document says 'Annual gross income was $50M.' HyDE generates an answer that uses the document's vocabulary, so the embedding match is stronger.

The catch is it requires an LLM call before retrieval even starts, which adds latency and cost. That's part of why I didn't expose it in the API. For a free-tier deployment where I'm trying to keep Groq API calls minimal, adding a pre-retrieval LLM call for every query didn't feel justified. It's still in the codebase and works if you call the function directly, just not API-accessible."

Q: How does the 2-tier LLM fallback work?

"In development, it tries Ollama first since that's local and free. If Ollama isn't running or times out after 30 seconds, it falls back to Groq's API. In production on Render, it's Groq only since there's no local GPU. Both use temperature 0.1 for accuracy over creativity. The generation service catches exceptions on invoke() and tries the next provider automatically.

Why Groq specifically? At the time I built this, Groq was the clear winner for free-tier LLM inference. Their custom LPU hardware gives you sub-second latency on Llama 70B, which is dramatically faster than other hosted options. The trade-off versus OpenAI or Anthropic is quality: GPT-4o and Claude are better at nuanced reasoning and following complex instructions, but for RAG where you're basically asking 'summarize what's in these chunks and cite your sources,' Llama 70B is more than good enough, and the cost difference is massive. Groq's free tier gives you enough tokens for a demo project. OpenAI's equivalent would cost real money.

The limitation is that Groq's free tier has rate limits, so if multiple users hit the demo simultaneously, they'll see slower responses or errors. For a production system, I'd either pay for a Groq plan or switch to a self-hosted model."

Q: How does the cross-encoder reranker work?

"It uses cross-encoder/ms-marco-MiniLM-L-6-v2, a neural reranker trained on Microsoft's MARCO passage ranking dataset. Unlike the embedding model which encodes query and document separately (bi-encoder), the cross-encoder takes the query-document pair as a single input and outputs a relevance score. That joint encoding is more accurate because it can model the interaction between query terms and document terms directly, but it's much slower since you can't precompute document representations.

I normalize scores to 0-1 using min-max scaling, sort by score, and return the top results. It's lazy-loaded so it only initializes when someone actually requests reranking, which saves RAM in the common case.

Compared to alternatives: ColBERT uses 'late interaction' where you precompute token-level document embeddings and do lightweight scoring at query time, so it's much faster for large result sets. Cohere Rerank is a hosted API that's arguably better quality but costs money per call. For my use case with small result sets (reranking maybe 10-20 chunks), the cross-encoder latency is fine, and ms-marco-MiniLM is free and well-tested. If I were reranking hundreds of documents per query, I'd seriously look at ColBERT instead."

Q: Why is the reranker disabled on Render?

"512MB RAM limit. The embedding model takes about 300MB, the Python runtime and LangChain libraries take another 150MB, and there's barely any headroom. Loading the cross-encoder on top would push past the limit and crash the process. So on Render, I detect the environment and skip the reranker. Users still get hybrid search, just not the final reranking pass.

This is probably the biggest limitation of the production deployment. The reranker accounts for a meaningful chunk of the accuracy improvement, so production users get a worse experience than what the system is capable of. If I were spending money on this, moving to even a 1GB instance would let me run the reranker and get the full 67.7% accuracy in production."

Q: How does document processing work?

"Four steps. Parse: extract text from the uploaded file using pypdf for PDFs, python-docx for DOCX, or raw text for TXT/MD. PDFs get split by page by default. Chunk: RecursiveCharacterTextSplitter breaks text into 500-character chunks with 50-character overlap, using separators like double newlines, single newlines, periods, and spaces. Embed: all-MiniLM-L6-v2 processes chunks in batches of 32. Store: vectors plus metadata like filename, page number, word count, and chunk index go into Qdrant Cloud.

On the chunking specifically: 500 characters is quite small. That's roughly 100-125 tokens, which is well below the 256-512 token range that recent benchmarks suggest as optimal. I went conservative because smaller chunks mean more precise retrieval matches, and with only 384-dim embeddings, keeping chunks focused helps the embedding model represent them accurately. The downside is you sometimes lose paragraph-level context. With more time, I'd experiment with larger chunks (maybe 1000-1500 characters) combined with a better embedding model, or even try recursive/hierarchical chunking where you store both a large chunk and its sub-chunks and let the retriever decide which granularity is best.

Also worth noting: I'm using character count, not token count. RecursiveCharacterTextSplitter's chunk_size parameter is in characters, not tokens, despite 'token' being a common confusion. 500 characters is roughly 100-125 tokens depending on the text."

Q: How does caching work?

"Redis with a dual-backend strategy. Primary is Upstash Redis with connection pooling (max 10 connections, 5-second timeout). If Redis goes down, it falls back to an in-memory dict with TTL checking. Cache keys are MD5 hashes of the question plus search parameters (k, hybrid flag, reranking flag). TTL is 1 hour. The system tracks hit/miss rates automatically. Repeated queries are essentially free since they skip both retrieval and LLM generation.

This is exact-match caching, meaning the question has to be identical (after hashing) to get a cache hit. The more sophisticated approach would be semantic caching, where you embed the query and check if any cached query is semantically similar (cosine similarity above 0.9 or so). Redis actually supports this natively now with their vector search module. Semantic caching can cut LLM costs by up to 68% in production workloads because 'What is the revenue?' and 'How much revenue does the company have?' would hit the same cache entry. I didn't implement it because it adds complexity and the exact-match approach was sufficient for my demo use case, but it's the obvious next improvement.

One limitation: the in-memory fallback doesn't share state across Render's container restarts, so if Redis goes down and the container restarts, the fallback cache is empty. It's a stopgap, not a real solution."

Q: What's the database schema?

"Qdrant Cloud is the only database. Each document chunk is stored as a vector (384 floats) plus metadata: file_name, file_type, page number, total_pages, char_count, word_count, upload_date, file_size_kb, chunk_index, and total_chunks. Redis stores cache entries as JSON with answer, sources, model_used, and retrieval_scores. There's no relational database since the system is document-in, question-answer-out.

This simplicity is both a strength and a limitation. No relational DB means no user accounts, no document permissions, no audit trail. For an enterprise system, you'd absolutely need PostgreSQL (or similar) for user management and access control. But for a portfolio project demonstrating RAG architecture, keeping the data layer focused on vectors and cache makes the system easier to reason about."

Q: How do you handle security?

"Rate limiting at three tiers: 60 requests per minute for queries, 10 per minute for document ingestion, and 120 per minute for everything else. All implemented with Redis-backed sliding windows. File uploads are validated for size (10MB max) and extension (.pdf, .docx, .txt, .md only) with path traversal protection on filenames. Input validation via Pydantic on all API requests. Questions are limited to 500 characters.

What's missing: there's no authentication or authorization. Anyone with the URL can upload documents and query the system. For a real enterprise deployment, you'd need JWT-based auth, document-level access control (probably through Qdrant's payload filtering by user_id), and audit logging. The rate limiting helps prevent abuse on a public demo, but it's not a security boundary."

Q: How do you handle errors?

"Graceful degradation everywhere. Redis down? Falls back to in-memory cache. Ollama not running? Falls back to Groq. Hybrid search fails? Falls back to vector-only. Query optimization fails? Uses the original query as-is. Reranker unavailable? Returns results without reranking. The system always returns something useful even if individual components fail.

This was a deliberate design principle. In a RAG system, a slightly worse answer is almost always better than an error message. The user doesn't need to know that the reranker failed or that they're getting results from the in-memory cache instead of Redis. They just get their answer, maybe slightly lower quality."

Q: What API endpoints do you expose?

"Six main ones. POST /api/query for asking questions with options for hybrid search, query optimization, reranking, and conversation memory. POST /api/ingest for file uploads. GET /api/stats for collection info and cache metrics. GET /api/health for system status. GET /api/documents for listing uploaded files. DELETE /api/documents/{name} for removing documents."

Q: How does conversation memory work?

"LangGraph with MemorySaver, which is an in-memory checkpointer. Each conversation gets a conversation_id. The system maintains context across turns so you can ask follow-up questions like 'Tell me more about that' and it knows what 'that' refers to.

Important caveat: MemorySaver stores everything in-memory, which means conversation history doesn't survive server restarts. On Render's free tier, the container can spin down after inactivity and restart, at which point all conversation context is lost. For a production system, you'd swap MemorySaver for a persistent checkpointer backed by Redis or PostgreSQL. LangGraph supports this natively, I just didn't implement it because the in-memory version was simpler and sufficient for demos where conversations are short-lived anyway."


Why Questions (Architectural Decisions)

Q: Why Qdrant over Chroma?

"Started with Chroma but migrated for two reasons. Chroma stores everything locally on disk, which doesn't work on Render's ephemeral filesystem since files get wiped on every deploy. Qdrant Cloud is remote so no local storage needed. It's also about 2x faster for similarity search and has better metadata filtering.

The broader landscape here: Pinecone is the most popular managed vector DB but it's more expensive and opinionated. Weaviate has great built-in hybrid search support, which would have been nice, but Qdrant Cloud's free tier was generous enough for my use case. pgvector is what I'd probably use if I already had a PostgreSQL database in the stack, since it eliminates an entire service dependency. But since I have no relational DB in this project, standing up Postgres just for vector search would be overkill. Qdrant was the sweet spot: free managed tier, fast, good LangChain integration, and solid metadata filtering."

Q: Why all-MiniLM-L6-v2 instead of OpenAI's ada-002?

"Cost and deployment constraints. ada-002 (or its successor text-embedding-3-small) requires an API call per embedding, which costs money and adds latency. all-MiniLM-L6-v2 runs locally on CPU in about 300MB of RAM. For a system that needs to embed hundreds of document chunks during ingestion, local is way cheaper and faster. The trade-off is 384 dimensions vs ada-002's 1536, and honestly ada-002 produces better embeddings, especially for diverse content.

But here's the thing: for this project, the deployment constraint drove the decision more than quality. If I had a real budget, I'd strongly consider OpenAI's text-embedding-3-small (1536 dims, $0.02/1M tokens) or even a mid-tier open-source model like BGE-base-en-v1.5 or nomic-embed-text-v1.5 running on a GPU instance. The accuracy gains from a better embedding model would likely exceed what I got from adding the reranker, because embeddings are the foundation. Garbage embeddings in means garbage retrieval out, no matter how good your reranker is."

Q: Why 500-character chunks with 50-character overlap?

"I should clarify: the chunk_size in RecursiveCharacterTextSplitter is 500 characters, not tokens. That's roughly 100-125 tokens, which is actually on the small side compared to what current best practices recommend (256-512 tokens, or roughly 1000-2000 characters).

I went small because with a relatively weak embedding model (384 dimensions), keeping chunks focused means each embedding more accurately represents a specific piece of information. Larger chunks with MiniLM tend to produce 'diluted' embeddings where the vector tries to represent too many concepts at once.

The 50-character overlap (10%) ensures that sentences split across chunk boundaries are captured in at least one chunk. I use RecursiveCharacterTextSplitter because it respects natural boundaries like paragraphs and sentences rather than splitting mid-word, trying separators in order: double newlines, single newlines, periods, spaces.

What I'd do differently: if I upgraded the embedding model to something with 768+ dimensions and a longer context window, I'd increase chunk size to 1000-1500 characters. I'd also explore semantic chunking, which splits based on embedding similarity between adjacent sentences rather than fixed character count, though recent benchmarks from early 2026 show recursive splitting at 512 tokens actually outperforms semantic chunking in many cases (69% vs 54% accuracy in one study). So fixed-size recursive splitting isn't a bad choice, I'd just use larger chunks."

Q: Why hybrid search (70/30 vector/BM25) by default?

"Vector search is great for semantic meaning but misses exact keyword matches. If someone searches for 'PostgreSQL' and the document says 'PostgreSQL', BM25 catches that exact match even if the vector similarity isn't the highest. The 70/30 weighting keeps semantic understanding dominant while ensuring keyword precision.

This is actually the industry standard approach now. Pretty much every production RAG system in 2025-2026 uses hybrid retrieval. The debate is more about implementation details: Weaviate has this built into their database layer, which is cleaner than my approach of doing it in application code with rank-bm25. My BM25 index gets cached and only rebuilds when the document count changes, so it's fast on subsequent queries, but it lives in memory and rebuilds on every container restart.

The 70/30 ratio was tuned empirically. I tried 50/50 and 80/20. 70/30 gave the best accuracy on my eval set. The interesting thing is that BM25's contribution matters most for technical content with specific terminology. For general prose, pure vector search gets close to hybrid performance.

One thing I didn't implement that would help: Reciprocal Rank Fusion (RRF) as an alternative to weighted averaging. RRF is more robust because it combines rankings rather than raw scores, which avoids the calibration problem of combining scores from two very different systems."

Q: Why temperature 0.1 for the LLM?

"RAG answers need to be factually grounded in the retrieved documents. Higher temperature means more creative, more likely to hallucinate beyond what the context says. 0.1 keeps the LLM focused on what's actually in the documents while still allowing natural language generation. I could go to 0.0 for maximum determinism, but 0.1 gives slightly more natural-sounding responses without meaningfully increasing hallucination risk."


Walk Me Through Questions

Q: Walk me through what happens when a user asks a question.

"User types a question in the frontend and optionally selects search options. Frontend POSTs to /api/query with the question, k value (default 3), and flags for hybrid search, optimization, and reranking. Backend first checks Redis cache using an MD5 hash of the query plus parameters. If cache hit, returns immediately.

If miss, it runs the selected retrieval strategy. For hybrid, it does both vector similarity search in Qdrant and BM25 keyword search, combines results with 70/30 weighting. If reranking is on and we're not on Render, the cross-encoder rescores every result by encoding each query-document pair jointly. Top k chunks become the context.

The prompt template wraps the chunks with instructions to only answer from context and cite sources. LLM generates the answer. If using conversation memory, the conversation_id links this turn to previous context via LangGraph's MemorySaver. Response includes the answer, source documents with file names and page numbers, and model used. The result gets cached in Redis with 1-hour TTL."

Q: Walk me through the document ingestion pipeline.

"User uploads a file via the frontend. Backend validates: max 10MB, allowed extensions only, sanitized filename. Parser extracts text. For PDFs, it uses pypdf with split-by-page enabled so each page becomes a separate document. For DOCX, python-docx extracts paragraphs.

Chunker splits into 500-character pieces with 50-character overlap. Embeddings service processes chunks in batches of 32 through all-MiniLM-L6-v2. Each chunk gets metadata attached: filename, file type, page number, word count, upload date, chunk index. All vectors plus metadata go to Qdrant Cloud. Response tells the user how many chunks were stored.

One thing worth noting: this is a synchronous pipeline. The user waits for the entire ingestion to complete before getting a response. For small documents that's fine, but a large PDF could take 10+ seconds. In a production system, I'd use a job queue (like BullMQ or Celery) so the upload returns immediately and ingestion happens asynchronously with progress updates via WebSocket."


What Would You Change Questions

Q: What would you do differently?

"Honestly, quite a few things now that I've had time to reflect.

First, the embedding model. all-MiniLM-L6-v2 was the safe choice but it's showing its age. Models like nomic-embed-text-v1.5 or BGE-base-en-v1.5 offer meaningfully better retrieval quality in similar footprints. This would probably be the single highest-impact change.

Second, chunk size. 500 characters is too small. I'd bump to 1000-1500 characters with a better embedding model that can handle the longer context.

Third, semantic caching instead of exact-match caching. Redis supports vector search natively now, so you can cache based on query similarity rather than requiring exact string matches. This could cut LLM costs dramatically since most users ask variations of the same questions.

Fourth, expose HyDE and Multi-Query through the API. They're implemented but not accessible, which is wasted work.

Fifth, streaming responses. Right now the user waits for the full LLM generation before seeing anything. Streaming token-by-token would make the experience feel much faster.

Sixth, persistent conversation memory. MemorySaver loses everything on restart. Swapping to a Redis-backed or Postgres-backed checkpointer would take maybe an hour and make conversations actually persistent."

Q: What are the limitations?

"The 512MB RAM constraint on Render is the root cause of most limitations. It forces me to disable the cross-encoder reranker in production, use a small embedding model, and be careful about memory usage during ingestion. Production accuracy is lower than what the system achieves locally.

The embedding model is the weakest link quality-wise. 384 dimensions from a 2019-era model can't compete with modern alternatives, and that limitation cascades through everything downstream.

Conversation memory is in-memory only (MemorySaver), so it doesn't persist across container restarts. On Render's free tier, containers spin down after inactivity, meaning conversation context is regularly lost.

HyDE and Multi-Query retrieval exist in the codebase but aren't exposed through the API, so users can't access them.

No authentication or authorization. It's a public demo where anyone can upload and query.

The BM25 index lives in memory and rebuilds on every restart, which adds cold-start latency.

And the free tier on Qdrant Cloud has storage limits, so it's not suitable for enterprise-scale document collections."

Q: How would you scale this?

"The scaling path depends on what's breaking. For RAM constraints, move to a proper cloud instance (even 1GB would let the reranker run). For ingestion throughput, add an async job queue so large uploads don't block the API. For retrieval quality, upgrade the embedding model and increase chunk size.

For multi-tenancy, Qdrant supports multiple collections and payload-based filtering, so you could have per-user or per-tenant document isolation without separate databases.

For caching at scale, implement semantic caching with Redis vector search and consider tiered TTLs where popular queries cache longer.

For the LLM layer, implement a model router that picks the right model based on query complexity. Simple factual lookups don't need Llama 70B; a smaller 8B model would be 50x cheaper with the same quality for straightforward retrieval tasks.

Graph RAG is also interesting for the future. For documents with complex entity relationships (org charts, legal contracts, technical documentation with cross-references), a knowledge graph layer on top of vector search could significantly improve accuracy. NVIDIA reported 96% factual faithfulness using graph+vector on financial filings, compared to much lower numbers for vector-only. But that's a significant architecture change, not a simple scale-up."


MODELS AND LIBRARIES CHEAT SHEET

all-MiniLM-L6-v2 (Sentence Transformers): Embedding model, 384 dimensions, runs on CPU. Small enough for 512MB RAM, fast (~500 embeddings/sec), free. Outdated compared to newer alternatives but reliable.

cross-encoder/ms-marco-MiniLM-L-6-v2: Neural reranker for relevance scoring. Most accurate retrieval strategy. Disabled on Render due to RAM. Free and self-hosted alternative to Cohere Rerank.

llama-3.3-70b-versatile (Groq): Production LLM, temperature 0.1. Fastest open-source inference via Groq's custom LPU hardware. Free tier with rate limits.

llama3 (Ollama): Dev LLM, runs locally. Zero cost during development. Useful for iterating without burning API calls.

Qdrant Cloud: Vector database. Remote (no local disk needed), fast similarity search, good metadata filtering. Free tier sufficient for demo-scale.

rank-bm25: BM25 keyword search algorithm. Keyword matching component of hybrid search. In-memory, rebuilds on restart.

Redis (Upstash): Query result caching. 1-hour TTL, exact-match caching with MD5 keys. Connection pooling with in-memory fallback.

LangChain 1.1.0: LLM abstraction, prompt templates. Swap between Ollama/Groq without code changes.

langchain-groq: Groq provider. Connects to Groq API.

langchain-ollama: Ollama provider. Connects to local Ollama.

langchain-huggingface: HuggingFace integration. Loads Sentence Transformers for embeddings.

langchain-qdrant: Qdrant integration. Vector store operations through LangChain.

langchain-text-splitters: Text chunking. RecursiveCharacterTextSplitter with 500-character chunk size.

LangGraph 1.0.4: Conversation memory. MemorySaver (in-memory only, does not persist across restarts).

sentence-transformers 5.1.2: Embedding and reranker models. Loads both the embedding model and cross-encoder.

transformers 4.57.3: Hugging Face model library. Required by sentence-transformers.

torch 2.9.1: PyTorch (CPU-only on Render). Required by sentence-transformers, CPU-only to save RAM.

FastAPI 0.122.0: Backend framework. Async support, Pydantic validation, auto docs.

Next.js 16: Frontend framework. App Router, React 19.

pypdf: PDF parsing. Extract text from PDF documents.

python-docx: DOCX parsing. Extract text from Word documents.

Pydantic 2.12.5: Config and validation. Settings management, request/response schemas.


KEY FILES MAP

If they ask about retrieval strategies: backend/app/services/advanced_retrieval.py (note: HyDE and Multi-Query are here but not API-accessible)
Embedding model setup: backend/app/services/embeddings.py
LLM fallback and answer generation: backend/app/services/generation.py
RAG orchestrator (retrieval + generation): backend/app/services/rag.py
Model names and all settings: backend/app/core/config.py
Vector store (Qdrant operations): backend/app/services/vector_store.py
Document parsing (PDF/DOCX): backend/app/services/document_parser.py
Text chunking: backend/app/services/chunking.py
Ingestion pipeline orchestration: backend/app/services/ingestion.py
Redis/in-memory caching: backend/app/services/cache.py
Rate limiting: backend/app/core/rate_limiter.py
API endpoints: backend/app/api/routes.py
Request/response schemas: backend/app/api/schemas.py
Conversation memory: backend/app/services/conversation.py
Main app setup: backend/app/main.py
All Python dependencies: backend/requirements.txt
Render-specific dependencies (lighter): backend/requirements-render.txt
Docker deployment: backend/Dockerfile
RAG evaluation tests: backend/tests/test_rag_evaluation.py
