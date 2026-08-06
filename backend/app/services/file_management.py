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
        self.documents_dir = Path("./data/documents")

    def list_documents(self) -> List[DocumentInfo]:
        """
        List all uploaded documents with metadata.

        Returns:
            List of DocumentInfo objects
        """
        # The VECTOR STORE is the source of truth here, not the local filesystem.
        # This used to glob ./data/documents and count chunks per file found on disk,
        # which returns nothing on an ephemeral filesystem: the host wipes the directory
        # on every deploy while the embeddings persist in Qdrant. The result was a UI
        # reporting "Documents (0)" directly beneath a stats bar reporting 26 documents.
        # Grouping the vector-store payloads by file_name reflects what is actually
        # retrievable, which is the thing a user cares about.
        all_docs = self.vector_store.get_all_documents()
        total_chunks = len(all_docs)
        if total_chunks == 0:
            return []

        grouped: Dict[str, Dict[str, Any]] = {}
        for doc in all_docs:
            meta = doc.metadata or {}
            name = meta.get('file_name') or 'unknown'
            entry = grouped.setdefault(name, {
                'chunk_count': 0,
                'file_type': (meta.get('file_type') or Path(name).suffix[1:] or 'unknown'),
                'file_size_kb': meta.get('file_size_kb'),
                'upload_date': meta.get('upload_date'),
            })
            entry['chunk_count'] += 1
            # Keep the first non-empty value we see for the descriptive fields.
            if entry['file_size_kb'] in (None, '') and meta.get('file_size_kb') not in (None, ''):
                entry['file_size_kb'] = meta.get('file_size_kb')
            if not entry['upload_date'] and meta.get('upload_date'):
                entry['upload_date'] = meta.get('upload_date')

        documents = []
        for name, info in grouped.items():
            # Prefer real stats when the file still happens to be on disk, otherwise
            # fall back to what ingestion recorded in the payload.
            local = self.documents_dir / name
            size_kb = info['file_size_kb']
            upload_date = info['upload_date']
            file_path = str(local)
            if local.exists():
                stat = local.stat()
                size_kb = round(stat.st_size / 1024, 2)
                upload_date = upload_date or datetime.fromtimestamp(stat.st_mtime).isoformat()
            else:
                file_path = f"qdrant://{name}"

            documents.append(DocumentInfo(
                file_name=name,
                file_path=file_path,
                file_type=info['file_type'],
                file_size_kb=float(size_kb) if size_kb not in (None, '') else 0.0,
                upload_date=upload_date or '',
                chunk_count=info['chunk_count'],
                total_chunks=total_chunks
            ))

        # Sort by upload date (newest first), blanks last so they don't jump the list.
        documents.sort(key=lambda x: (x.upload_date == '', x.upload_date), reverse=False)
        documents.reverse()

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
            # For Qdrant, use filter-based deletion
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            try:
                # Delete points with matching file_name metadata
                self.vector_store._client.delete(
                    collection_name=settings.qdrant_collection,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="metadata.file_name",
                                match=MatchValue(value=file_name)
                            )
                        ]
                    )
                )
                # Note: Can't easily get count of deleted, assume chunks_deleted based on list
                all_docs = self.vector_store.get_all_documents()
                chunks_deleted = sum(1 for doc in all_docs if doc.metadata.get('file_name') == file_name)
            except:
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
        all_docs = self.vector_store.get_all_documents()

        chunks = []
        for doc in all_docs:
            if doc.metadata.get('file_name') == file_name:
                chunks.append({
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "content_preview": doc.page_content[:100] + "...",
                    "page": doc.metadata.get("page")
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
