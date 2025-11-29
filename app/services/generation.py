"""
Generation Service

Handles LLM generation with retrieved context.

Uses Llama 3 via Ollama (local, unlimited usage).
Fallback options: Groq API, Gemini API (if needed later).

This is the "G" in RAG - generating answers from context!
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings


class GenerationResponse(BaseModel):
    """
    Structured generation response with Pydantic validation.

    Benefits:
    - Type-safe response handling
    - Easy to extend (add confidence, metadata, etc.)
    - Clear contract for API responses
    """
    answer: str = Field(..., description="Generated answer")
    query: str = Field(..., description="Original user query")
    model_used: str = Field(..., description="LLM model name")
    context_length: int = Field(..., description="Length of context provided to LLM")


class GenerationService:
    """
    Generates answers using LLM with retrieved context.

    Current: Llama 3 via Ollama
    Future: Add Groq/Gemini fallback for speed demos
    """

    def __init__(self):
        """Initialize Llama 3 LLM"""
        print(f"[i] Loading LLM: {settings.ollama_model}")

        self.llm = OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.1,  # Low temperature = more focused, less creative
        )

        print(f"[OK] LLM ready!")

        # Define RAG prompt template
        self.prompt_template = ChatPromptTemplate.from_template("""
You are a helpful AI assistant answering questions based on provided context.

IMPORTANT RULES:
1. Answer ONLY using information from the context below
2. If the answer is not in the context, say "I don't have that information in the provided documents."
3. Be concise and accurate
4. Cite sources when possible (e.g., "According to the document...")

Context:
{context}

Question: {question}

Answer:""")

        # Create generation chain
        self.chain = (
            self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def generate(
        self,
        query: str,
        context: str
    ) -> GenerationResponse:
        """
        Generate answer based on query and retrieved context.

        Args:
            query: User question
            context: Retrieved documents formatted as string

        Returns:
            GenerationResponse with answer and metadata
        """

        # Generate answer using LLM
        answer = self.chain.invoke({
            "context": context,
            "question": query
        })

        return GenerationResponse(
            answer=answer.strip(),
            query=query,
            model_used=settings.ollama_model,
            context_length=len(context)
        )


# Global instance
generation_service = GenerationService()


# =============================================================================
# Test Generation
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Generation Service Test")
    print("=" * 70)

    # Sample context (simulates retrieved documents)
    sample_context = """
[Source 1: rag_guide.txt]
RAG stands for Retrieval-Augmented Generation. It combines retrieval systems with large language models to provide accurate, context-aware responses.

[Source 2: architecture.txt]
The RAG system uses Llama 3 for generation, Sentence Transformers for embeddings, and Chroma for vector storage. It achieves sub-2 second query latency.

[Source 3: benefits.txt]
Benefits of RAG include: accurate answers grounded in documents, no hallucinations, up-to-date information, and cost-effectiveness compared to fine-tuning.
"""

    # Test queries
    test_queries = [
        "What does RAG stand for?",
        "What LLM is used in this system?",
        "What are the benefits of RAG?",
        "What is the capital of France?",  # NOT in context - should say "I don't have that information"
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: '{query}'")
        print("=" * 70)

        response = generation_service.generate(query, sample_context)

        print(f"Answer: {response.answer}")
        print(f"\nMetadata:")
        print(f"  Model: {response.model_used}")
        print(f"  Context length: {response.context_length} chars")

    print("\n" + "=" * 70)
    print("[OK] Generation working!")
    print("=" * 70)
    print("\nKey Learnings:")
    print("  - LLM generates answers from provided context")
    print("  - Temperature 0.1 = focused, factual answers")
    print("  - Prompt engineering guides LLM behavior")
    print("  - LLM should admit when answer not in context")
