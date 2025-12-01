"""
Quick test to validate Ollama + LangChain + Chroma setup
Run this after installing dependencies to ensure everything works
"""

import os
import sys
from dotenv import load_dotenv

# Fix Windows console encoding for emojis
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Load environment variables
load_dotenv()

print("🚀 Testing Enterprise RAG Setup...\n")

# Test 1: Ollama Connection
print("1️⃣ Testing Ollama connection...")
try:
    from langchain_ollama import OllamaLLM

    llm = OllamaLLM(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "llama3")
    )

    response = llm.invoke("Say 'Ollama is working!' and nothing else.")
    print(f"   ✅ Ollama response: {response.strip()}\n")
except Exception as e:
    print(f"   ❌ Ollama error: {e}\n")
    exit(1)

# Test 2: Embeddings
print("2️⃣ Testing Sentence Transformers embeddings...")
try:
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Test embedding
    test_embedding = embeddings.embed_query("Hello world")
    print(f"   ✅ Embedding dimension: {len(test_embedding)}\n")
except Exception as e:
    print(f"   ❌ Embeddings error: {e}\n")
    exit(1)

# Test 3: Qdrant Vector Database
print("3️⃣ Testing Qdrant Cloud connection...")
try:
    from app.services.vector_store import vector_store

    # Test connection and get stats
    stats = vector_store.get_stats()
    print(f"   ✅ Connected to Qdrant Cloud")
    print(f"   ✅ Collection: {stats['collection_name']}")
    print(f"   ✅ Documents: {stats['total_documents']}\n")

except Exception as e:
    print(f"   ❌ Chroma error: {e}\n")
    exit(1)

# Test 4: Simple RAG Query
print("4️⃣ Testing RAG query...")
try:
    # Test semantic search
    test_query = "What skills does Daniel have?"
    results = vector_store.search(test_query, k=2)

    if results:
        print(f"   ✅ Search returned {len(results)} results")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    question = "What does RAG stand for?"
    answer = rag_chain.invoke(question)

    print(f"   Question: {question}")
    print(f"   ✅ Answer: {answer.strip()}\n")

except Exception as e:
    print(f"   ❌ RAG pipeline error: {e}\n")
    exit(1)

print("=" * 60)
print("🎉 ALL TESTS PASSED! Your RAG environment is ready!")
print("=" * 60)
print("\nNext steps:")
print("1. Start building the document ingestion pipeline")
print("2. Create the FastAPI backend")
print("3. Build the admin dashboard")
print("\nHappy building! 🚀")
