"""
Generation Service with LLM Fallback

3-Tier LLM Strategy:
1. Ollama (local, unlimited) - Primary
2. Groq (350+ tokens/sec, free tier) - Fallback for speed
3. Gemini (generous free tier) - Final fallback

Benefits:
- Speed: Groq is 10-30x faster than local Ollama for demos
- Reliability: Fallback if Ollama is down
- Production pattern: Multi-provider resilience
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings


class GenerationResponse(BaseModel):
    """Structured generation response with Pydantic validation"""
    answer: str = Field(..., description="Generated answer")
    query: str = Field(..., description="Original user query")
    model_used: str = Field(..., description="LLM model name (ollama/groq/gemini)")
    context_length: int = Field(..., description="Length of context provided to LLM")


class GenerationService:
    """
    Generates answers using LLM with 3-tier fallback.

    Priority: Ollama (local) → Groq (fast) → Gemini (reliable)
    """

    def __init__(self):
        """Initialize all LLMs"""
        print(f"[i] Loading LLM providers...")

        # Primary: Ollama (local, unlimited)
        self.ollama = OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.1,
            timeout=30  # Timeout if Ollama is slow/down
        )

        # Fallback 1: Groq (350+ tokens/sec, free tier)
        self.groq = None
        if settings.groq_api_key:
            self.groq = ChatGroq(
                api_key=settings.groq_api_key,
                model="llama3-70b-8192",  # Fast, large context
                temperature=0.1
            )

        # Fallback 2: Gemini (generous free tier)
        self.gemini = None
        if settings.gemini_api_key:
            self.gemini = ChatGoogleGenerativeAI(
                google_api_key=settings.gemini_api_key,
                model="gemini-pro",
                temperature=0.1
            )

        print(f"[OK] LLMs ready!")
        print(f"  - Ollama: {settings.ollama_model} (primary)")
        print(f"  - Groq: {'Configured' if self.groq else 'Not available'}")
        print(f"  - Gemini: {'Configured' if self.gemini else 'Not available'}")

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

    def _generate_with_llm(self, llm, llm_name: str, query: str, context: str) -> Optional[GenerationResponse]:
        """
        Try to generate answer with specific LLM.

        Returns GenerationResponse or None if failed.
        """
        try:
            chain = self.prompt_template | llm | StrOutputParser()

            answer = chain.invoke({
                "context": context,
                "question": query
            })

            return GenerationResponse(
                answer=answer.strip(),
                query=query,
                model_used=llm_name,
                context_length=len(context)
            )

        except Exception as e:
            print(f"[!] {llm_name} failed: {e}")
            return None

    def generate(
        self,
        query: str,
        context: str
    ) -> GenerationResponse:
        """
        Generate answer with 3-tier LLM fallback.

        Tries: Ollama → Groq → Gemini

        Args:
            query: User question
            context: Retrieved documents formatted as string

        Returns:
            GenerationResponse with answer and metadata
        """

        # TIER 1: Try Ollama (local, unlimited)
        response = self._generate_with_llm(
            self.ollama,
            f"ollama/{settings.ollama_model}",
            query,
            context
        )

        if response:
            return response

        # TIER 2: Try Groq (fast, free tier)
        if self.groq:
            print("[i] Ollama unavailable, trying Groq...")
            response = self._generate_with_llm(
                self.groq,
                "groq/llama3-70b",
                query,
                context
            )

            if response:
                return response

        # TIER 3: Try Gemini (generous free tier)
        if self.gemini:
            print("[i] Groq unavailable, trying Gemini...")
            response = self._generate_with_llm(
                self.gemini,
                "gemini/gemini-pro",
                query,
                context
            )

            if response:
                return response

        # All failed - return error
        return GenerationResponse(
            answer="[X] All LLM providers unavailable. Please check your configuration.",
            query=query,
            model_used="none",
            context_length=len(context)
        )


# Global instance
generation_service = GenerationService()


# =============================================================================
# Test Generation with Fallback
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Generation Service Test (with Fallback)")
    print("=" * 70)

    sample_context = """
[Source 1: resume.pdf]
Daniel Alexis Cruz is a Full-Stack Developer specializing in AI, Blockchain & Cybersecurity. He has experience with React, Node.js, Python, and Solidity.

[Source 2: projects.txt]
Notable projects include AutoFlow Pro (browser automation with BullMQ and Redis) and an NFT Trading Platform (Solidity smart contracts).
"""

    test_queries = [
        "What technologies does Daniel work with?",
        "Tell me about AutoFlow Pro",
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
    print("[OK] Generation with fallback working!")
    print("=" * 70)
    print("\nFallback Strategy:")
    print("  1. Ollama (local, unlimited) - Primary")
    print("  2. Groq (350+ tokens/sec) - Speed fallback")
    print("  3. Gemini (generous free) - Final fallback")
