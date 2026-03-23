# ABOUTME: ChromaDB interface for persistent vector storage and retrieval.
# ABOUTME: Manages document chunks, metadata, and similarity/keyword search operations.

"""ChromaDB interface for PDF document storage and retrieval."""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings


class PDFDatabase:
    """ChromaDB interface for PDF document storage and retrieval.

    Manages persistent storage of document chunks with embeddings and metadata.
    Provides similarity search and keyword search capabilities.

    Attributes:
        db_path: Path to ChromaDB persistence directory
        client: ChromaDB client instance
        collection: ChromaDB collection for PDF documents
    """

    def __init__(self, db_path: str):
        """Initialize database at specified path.

        Args:
            db_path: Path to ChromaDB database directory

        Example:
            >>> db = PDFDatabase("/path/to/chroma_db")
            >>> doc_count = len(db.list_documents())
        """
        self.db_path = db_path
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="pdf_documents",
            metadata={"hnsw:space": "cosine"}
        )

    def document_exists(self, document_id: str) -> bool:
        """Check if document exists in database.

        Args:
            document_id: Document ID to check

        Returns:
            True if document exists, False otherwise
        """
        results = self.collection.get(where={"document_id": document_id}, limit=1)
        return len(results['ids']) > 0

    def add_document(self, document_id: str, filename: str, chunks: List[Dict[str, Any]],
                    embeddings: List[List[float]]) -> int:
        """Add document chunks to database.

        Args:
            document_id: Unique document identifier (file hash)
            filename: Original filename
            chunks: List of chunk dicts with 'text', 'page', 'chunk_index'
            embeddings: List of embedding vectors for each chunk

        Returns:
            Number of chunks added

        Example:
            >>> chunks = [{"text": "...", "page": 1, "chunk_index": 0}]
            >>> embeddings = [[0.1, 0.2, ...]]
            >>> count = db.add_document("doc123", "file.pdf", chunks, embeddings)
        """
        chunk_texts = [chunk['text'] for chunk in chunks]
        ids = [f"{document_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "document_id": document_id,
                "filename": filename,
                "page": chunk['page'],
                "chunk_index": chunk['chunk_index']
            }
            for chunk in chunks
        ]

        # Add in batches
        batch_size = 5461  # ChromaDB's default batch size
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=chunk_texts[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )

        return len(chunks)

    def remove_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Remove all chunks for document.

        Args:
            document_id: Document ID to remove

        Returns:
            Dict with filename and chunk count if found, None if not found

        Example:
            >>> result = db.remove_document("doc123")
            >>> print(f"Removed {result['chunk_count']} chunks")
        """
        # Get chunks before deletion
        results = self.collection.get(where={"document_id": document_id})

        if not results['ids']:
            return None

        filename = results['metadatas'][0]['filename'] if results['metadatas'] else "Unknown"
        chunk_count = len(results['ids'])

        # Delete all chunks
        self.collection.delete(where={"document_id": document_id})

        return {
            "filename": filename,
            "chunk_count": chunk_count
        }

    def list_documents(self) -> List[Dict[str, str]]:
        """List all documents with metadata.

        Returns:
            List of document dicts with document_id, filename, chunk_count

        Example:
            >>> docs = db.list_documents()
            >>> for doc in docs:
            ...     print(f"{doc['filename']}: {doc['chunk_count']} chunks")
        """
        # Paginate to avoid SQLite's variable limit with large collections
        page_size = 5000
        offset = 0
        doc_map = {}
        while True:
            batch = self.collection.get(
                limit=page_size, offset=offset, include=['metadatas']
            )
            if not batch['ids']:
                break
            for metadata in batch['metadatas']:
                doc_id = metadata['document_id']
                if doc_id not in doc_map:
                    doc_map[doc_id] = {
                        'document_id': doc_id,
                        'filename': metadata['filename'],
                        'chunk_count': 0,
                        'added_date': 'N/A'
                    }
                doc_map[doc_id]['chunk_count'] += 1
            offset += page_size

        return list(doc_map.values())

    def search_similarity(self, query_embedding: List[float], top_k: int = 5,
                         document_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search by vector similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            document_filter: Optional document_id to filter results

        Returns:
            List of result dicts with text, metadata, and similarity scores

        Example:
            >>> embedding = generator.generate_single("machine learning")
            >>> results = db.search_similarity(embedding, top_k=5)
            >>> print(results[0]["text"])
        """
        where_filter = {"document_id": document_filter} if document_filter else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )

        if not results['ids'][0]:
            return []

        formatted_results = []
        for i, doc_id in enumerate(results['ids'][0]):
            formatted_results.append({
                'chunk_id': doc_id,
                'document': results['metadatas'][0][i]['filename'],
                'document_id': results['metadatas'][0][i]['document_id'],
                'page': results['metadatas'][0][i]['page'],
                'chunk_index': results['metadatas'][0][i]['chunk_index'],
                'text': results['documents'][0][i],
                'similarity': 1 - results['distances'][0][i]
            })

        return formatted_results

    def search_keywords(self, keywords: List[str], top_k: int = 5,
                       document_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search by keyword matching.

        Args:
            keywords: List of keywords to search for
            top_k: Number of results to return
            document_filter: Optional document_id to filter results

        Returns:
            List of result dicts with text, metadata, and keyword match scores

        Example:
            >>> results = db.search_keywords(["neural", "network"], top_k=5)
            >>> print(f"Found {len(results)} matches")
        """
        where_filter = {"document_id": document_filter} if document_filter else None
        lowered_keywords = [kw.lower() for kw in keywords]

        # Paginate to avoid SQLite's variable limit with large collections
        page_size = 5000
        offset = 0
        scored_results = []
        while True:
            batch = self.collection.get(
                where=where_filter,
                limit=page_size,
                offset=offset,
                include=['documents', 'metadatas']
            )
            if not batch['ids']:
                break

            # Score each chunk in this batch based on keyword matches
            for i, doc_id in enumerate(batch['ids']):
                text = batch['documents'][i].lower()
                score = sum(text.count(kw) for kw in lowered_keywords)
                if score > 0:
                    scored_results.append({
                        'chunk_id': doc_id,
                        'document': batch['metadatas'][i]['filename'],
                        'document_id': batch['metadatas'][i]['document_id'],
                        'page': batch['metadatas'][i]['page'],
                        'chunk_index': batch['metadatas'][i]['chunk_index'],
                        'text': batch['documents'][i],
                        'similarity': score / len(keywords),
                        'keyword_matches': score
                    })

            offset += page_size

        if not scored_results:
            return []

        # Sort by score descending
        scored_results.sort(key=lambda x: x['keyword_matches'], reverse=True)

        return scored_results[:top_k]
