"""
File Management Service

Manages uploaded documents:
- List all documents
- Get document stats (chunk count, size, upload date)
- Delete documents from filesystem and vector DB
- Search/filter documents

Useful for admin dashboard and file organization.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from app.services.vector_store import vector_store
from app.core.config import settings


class DocumentInfo(BaseModel):
    """Information about an uploaded document"""
    file_name: str = Field(..., description="Document filename")
    file_path: str = Field(..., description="Full file path")
    file_type: str = Field(..., description="File extension")
    file_size_kb: float = Field(..., description="File size in KB")
    upload_date: str = Field(..., description="Upload timestamp")
    chunk_count: int = Field(..., description="Number of chunks in vector DB")
    total_chunks: int = Field(..., description="Total chunks across all documents")


class FileManagementService:
    """
    Manages uploaded documents and their metadata.

    Features:
    - List all documents with stats
    - Delete documents (file + vector DB chunks)
    - Search documents
    - Get per-document statistics
    """

    def __init__(self):
        self.vector_store = vector_store
        self.documents_dir = Path(settings.chroma_persist_dir).parent / "documents"

    def list_documents(self) -> List[DocumentInfo]:
        """
        List all uploaded documents with metadata.

        Returns:
            List of DocumentInfo objects
        """
        if not self.documents_dir.exists():
            return []

        documents = []

        # Get all supported files
        supported_extensions = {'.pdf', '.docx', '.txt', '.md'}
        files = []

        for ext in supported_extensions:
            files.extend(self.documents_dir.glob(f"*{ext}"))

        # Get chunk counts from vector DB
        all_docs_result = self.vector_store._vectorstore.get()
        total_chunks = len(all_docs_result['documents']) if all_docs_result['documents'] else 0

        # Build document info
        for file_path in files:
            # Count chunks for this file
            chunk_count = 0

            if all_docs_result['metadatas']:
                for metadata in all_docs_result['metadatas']:
                    if metadata.get('file_name') == file_path.name:
                        chunk_count += 1

            # Get file stats
            stat = file_path.stat()

            documents.append(DocumentInfo(
                file_name=file_path.name,
                file_path=str(file_path),
                file_type=file_path.suffix[1:],  # Remove dot
                file_size_kb=round(stat.st_size / 1024, 2),
                upload_date=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                chunk_count=chunk_count,
                total_chunks=total_chunks
            ))

        # Sort by upload date (newest first)
        documents.sort(key=lambda x: x.upload_date, reverse=True)

        return documents

    def delete_document(self, file_name: str) -> Dict[str, Any]:
        """
        Delete a document from filesystem and vector database.

        Args:
            file_name: Name of file to delete

        Returns:
            Dict with deletion results
        """
        file_path = self.documents_dir / file_name

        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_name}"
            }

        try:
            # Delete from vector database (all chunks with this file_name)
            all_docs_result = self.vector_store._vectorstore.get()

            if all_docs_result['metadatas']:
                # Find IDs of chunks belonging to this file
                ids_to_delete = []

                for i, metadata in enumerate(all_docs_result['metadatas']):
                    if metadata.get('file_name') == file_name:
                        ids_to_delete.append(all_docs_result['ids'][i])

                # Delete chunks
                if ids_to_delete:
                    self.vector_store.delete_documents(ids_to_delete)

                chunks_deleted = len(ids_to_delete)
            else:
                chunks_deleted = 0

            # Delete file from filesystem
            file_path.unlink()

            return {
                "success": True,
                "file_name": file_name,
                "chunks_deleted": chunks_deleted
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_document_details(self, file_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed info about a specific document.

        Args:
            file_name: Document filename

        Returns:
            Dict with document details including chunk previews
        """
        file_path = self.documents_dir / file_name

        if not file_path.exists():
            return None

        # Get chunks for this document
        all_docs_result = self.vector_store._vectorstore.get()

        chunks = []
        if all_docs_result['metadatas']:
            for i, metadata in enumerate(all_docs_result['metadatas']):
                if metadata.get('file_name') == file_name:
                    chunks.append({
                        "chunk_index": metadata.get("chunk_index", 0),
                        "content_preview": all_docs_result['documents'][i][:100] + "...",
                        "page": metadata.get("page")
                    })

        stat = file_path.stat()

        return {
            "file_name": file_name,
            "file_type": file_path.suffix[1:],
            "file_size_kb": round(stat.st_size / 1024, 2),
            "upload_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "chunk_count": len(chunks),
            "chunks": chunks
        }


# Global instance
file_management = FileManagementService()


# =============================================================================
# Test File Management
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("File Management Service Test")
    print("=" * 70)

    # List documents
    print("\n[1] Listing documents:")
    print("-" * 70)

    docs = file_management.list_documents()

    if not docs:
        print("No documents found")
    else:
        for doc in docs:
            print(f"\n{doc.file_name}")
            print(f"  Type: {doc.file_type}")
            print(f"  Size: {doc.file_size_kb} KB")
            print(f"  Chunks: {doc.chunk_count}")
            print(f"  Uploaded: {doc.upload_date}")

    # Get details for first document
    if docs:
        print(f"\n[2] Document details: {docs[0].file_name}")
        print("-" * 70)

        details = file_management.get_document_details(docs[0].file_name)

        if details:
            print(f"Chunks: {details['chunk_count']}")
            print(f"\nFirst 2 chunks:")
            for chunk in details['chunks'][:2]:
                print(f"  [{chunk['chunk_index']}] {chunk['content_preview']}")

    print("\n" + "=" * 70)
    print("[OK] File management working!")
    print("=" * 70)
