"""
Text Chunking Service

Different strategies for splitting documents into chunks:
1. RecursiveCharacterTextSplitter - Smart splitting (respects paragraphs, sentences)
2. CharacterTextSplitter - Simple fixed-size splitting
3. SemanticChunker - Split by meaning (advanced)

Start with #1 (recommended for most cases).
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from app.core.config import settings


class TextChunker:
    """
    Chunk documents into smaller pieces for better retrieval.

    Why chunk?
    - LLMs have context window limits (e.g., 4096 tokens)
    - Smaller chunks = more precise retrieval
    - Better matching between queries and relevant content

    Chunking strategy matters!
    - Too small (50 tokens): Loses context, fragments meaning
    - Too large (2000 tokens): Less precise, retrieves irrelevant info
    - Sweet spot: 500 tokens with 50 overlap
    """

    @staticmethod
    def chunk_documents(
        documents: List[Document],
        chunk_size: int = None,
        chunk_overlap: int = None,
        strategy: str = "recursive"
    ) -> List[Document]:
        """
        Chunk documents using specified strategy.

        Args:
            documents: List of Document objects to chunk
            chunk_size: Size of each chunk in tokens (default from config)
            chunk_overlap: Overlap between chunks (default from config)
            strategy: "recursive" (smart) or "character" (simple)

        Returns:
            List of chunked Documents with preserved metadata
        """
        chunk_size = chunk_size or settings.chunk_size
        chunk_overlap = chunk_overlap or settings.chunk_overlap

        if strategy == "recursive":
            # Recommended: Respects paragraph/sentence boundaries
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]  # Try these in order
            )
        else:
            # Simple: Fixed-size splitting
            splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n"
            )

        # Chunk all documents
        chunked_docs = []

        for doc in documents:
            chunks = splitter.split_documents([doc])

            # Add chunk index to metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = len(chunks)
                chunked_docs.append(chunk)

        return chunked_docs

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List[str]:
        """
        Chunk raw text (without Document wrapper).

        Useful for quick chunking without metadata.
        """
        chunk_size = chunk_size or settings.chunk_size
        chunk_overlap = chunk_overlap or settings.chunk_overlap

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

        return splitter.split_text(text)


# =============================================================================
# Test Chunking
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Text Chunking Test")
    print("=" * 70)

    # Sample long text
    sample_text = """
    Retrieval-Augmented Generation (RAG) is a technique that enhances large language model responses by incorporating relevant information retrieved from external knowledge bases.

    The RAG process consists of several steps. First, documents are ingested and split into manageable chunks. These chunks are then converted into vector embeddings using models like Sentence Transformers.

    When a user asks a question, the query is also embedded and used to search the vector database for similar chunks. The most relevant chunks are retrieved and passed as context to the language model.

    Finally, the LLM generates a response based on both the query and the retrieved context, resulting in more accurate and grounded answers compared to pure generation.

    RAG is particularly useful for:
    - Question answering over proprietary documents
    - Customer support with knowledge base integration
    - Technical documentation search
    - Legal document analysis
    """

    # Test different chunk sizes
    for size in [100, 300, 500]:
        print(f"\nChunk size: {size} tokens, overlap: 50")
        print("-" * 70)

        chunks = TextChunker.chunk_text(sample_text, chunk_size=size, chunk_overlap=50)

        print(f"Total chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1}:")
            print(f"  Length: {len(chunk)} chars")
            print(f"  Preview: {chunk[:80]}...")

    # Test with Document objects
    print("\n" + "=" * 70)
    print("Testing with Document objects:")
    print("=" * 70)

    doc = Document(
        page_content=sample_text,
        metadata={"source": "test.txt", "file_type": "txt"}
    )

    chunked_docs = TextChunker.chunk_documents([doc], chunk_size=200)

    print(f"\nOriginal: 1 document")
    print(f"Chunked: {len(chunked_docs)} documents")

    for i, chunk in enumerate(chunked_docs[:2]):  # Show first 2
        print(f"\nChunk {i+1}:")
        print(f"  Content: {chunk.page_content[:100]}...")
        print(f"  Metadata: {chunk.metadata}")

    print("\n" + "=" * 70)
    print("[OK] Chunking working!")
    print("=" * 70)
    print("\nKey Learnings:")
    print("  - Chunk size affects retrieval precision")
    print("  - Overlap prevents information loss at boundaries")
    print("  - RecursiveCharacterTextSplitter respects structure")
    print("  - Metadata preserved through chunking")
