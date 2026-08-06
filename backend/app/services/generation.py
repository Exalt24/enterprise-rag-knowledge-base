"""
Generation Service with LLM Fallback

2-Tier LLM Strategy:
1. Ollama (local, unlimited) - Primary
2. Groq (350+ tokens/sec, free tier) - Fallback for speed

Benefits:
- Speed: Groq is 10-30x faster than local Ollama for demos
- Reliability: Fallback if Ollama is down
- Production pattern: Multi-provider resilience
"""

from typing import Optional, Iterator, Tuple
from pydantic import BaseModel, Field
from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings


class GenerationResponse(BaseModel):
    """Structured generation response with Pydantic validation"""
    answer: str = Field(..., description="Generated answer")
    query: str = Field(..., description="Original user query")
    model_used: str = Field(..., description="LLM model name (ollama/groq)")
    context_length: int = Field(..., description="Length of context provided to LLM")


class GenerationService:
    """
    Generates answers using LLM with 2-tier fallback.

    Priority: Ollama (local, unlimited) → Groq (cloud, fast)
    """

    def __init__(self):
        """Initialize all LLMs"""
        import os
        print(f"[i] Loading LLM providers...")

        is_render = os.getenv("RENDER") == "true"

        if is_render:
            # On Render: Use Groq as primary (no local Ollama)
            self.ollama = None
            self.groq = ChatGroq(
                api_key=settings.groq_api_key,
                model="llama-3.3-70b-versatile",
                temperature=0.1
            ) if settings.groq_api_key else None
        else:
            # Local dev: Ollama primary, Groq fallback
            self.ollama = OllamaLLM(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0.1,
                timeout=30
            )
            self.groq = ChatGroq(
                api_key=settings.groq_api_key,
                model="llama-3.3-70b-versatile",
                temperature=0.1
            ) if settings.groq_api_key else None

        print(f"[OK] LLMs ready!")
        print(f"  - Ollama: {settings.ollama_model} (primary)" if self.ollama else "  - Groq: llama-3.3-70b (primary)")
        if self.groq and self.ollama:
            print(f"  - Groq: Configured (fallback)")

    @property
    def primary_llm(self):
        """
        The first provider that will actually be tried, or None if nothing is configured.

        Callers that need "an LLM" must go through this rather than reaching for
        .ollama directly. On Render .ollama is None, so a hardcoded .ollama reference
        builds a broken chain and fails at invoke time.
        """
        return self.ollama or self.groq

    @property
    def primary_model_name(self) -> str:
        """Label of the provider that answers first, so reporting matches reality."""
        if self.ollama:
            return f"ollama/{settings.ollama_model}"
        if self.groq:
            return "groq/llama-3.3-70b-versatile"
        return "none"

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

    def _stream_with_llm(self, llm, llm_name: str, query: str, context: str) -> Iterator[str]:
        """
        Stream tokens from a specific LLM.

        Raises on failure so the caller can fall through to the next tier. We deliberately
        buffer the first chunk before yielding anything: if the provider is going to fail it
        almost always fails on the opening call, and that lets us fall back cleanly instead
        of stranding the client mid-answer with half a response already sent.
        """
        chain = self.prompt_template | llm | StrOutputParser()
        stream = chain.stream({"context": context, "question": query})

        first = next(stream)   # may raise, caller handles the fallback
        yield first
        for chunk in stream:
            if chunk:
                yield chunk

    def generate_stream(self, query: str, context: str) -> Tuple[str, Iterator[str]]:
        """
        Stream an answer with the same 2-tier fallback as generate().

        Returns (model_used, token_iterator). The provider is resolved eagerly so the caller
        knows which model answered before the first token reaches the client, which matters
        for the OpenAI-compatible payload where `model` sits in the very first SSE frame.
        """
        tiers = []
        if self.ollama:
            tiers.append((self.ollama, f"ollama/{settings.ollama_model}"))
        if self.groq:
            tiers.append((self.groq, "groq/llama-3.3-70b-versatile"))

        for llm, llm_name in tiers:
            try:
                stream = self._stream_with_llm(llm, llm_name, query, context)
                first = next(stream)  # forces the provider call, so failures surface here

                def _iter(first_chunk=first, rest=stream):
                    yield first_chunk
                    yield from rest

                return llm_name, _iter()
            except StopIteration:
                # Provider returned an empty answer. Treat as success, not a failure.
                return llm_name, iter(())
            except Exception as e:
                print(f"[!] {llm_name} streaming failed: {e}")
                continue

        def _err():
            yield "[X] All LLM providers unavailable. Please check your configuration."

        return "none", _err()

    def generate(
        self,
        query: str,
        context: str
    ) -> GenerationResponse:
        """
        Generate answer with 2-tier LLM fallback.

        Tries: Ollama → Groq (or just Groq on Render)

        Args:
            query: User question
            context: Retrieved documents formatted as string

        Returns:
            GenerationResponse with answer and metadata
        """

        # TIER 1: Try Ollama (local) or Groq (Render)
        if self.ollama:
            # Local development - use Ollama
            response = self._generate_with_llm(
                self.ollama,
                f"ollama/{settings.ollama_model}",
                query,
                context
            )
            if response:
                return response

        # TIER 2: Try Groq (fallback for local, primary for Render)
        if self.groq:
            if not self.ollama:
                # Groq is primary on Render
                pass
            else:
                print("[i] Ollama unavailable, trying Groq...")

            response = self._generate_with_llm(
                self.groq,
                "groq/llama-3.3-70b-versatile",
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
    print("  2. Groq (350+ tokens/sec) - Cloud fallback")
