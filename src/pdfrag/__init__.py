"""PDF RAG MCP Server - Semantic search and retrieval for PDF documents.

This package provides an MCP server that enables semantic search over PDF
documents using ChromaDB for vector storage and sentence-transformers for
embeddings.

Main components:
- server: FastMCP server with 5 tools for PDF management and search
- database: ChromaDB interface for persistent storage
- embeddings: Text embedding generation
- pdf: PDF text extraction with OCR support
- chunking: Semantic text chunking
- cli: Command-line interface for testing MCP servers

Usage:
    As MCP server:
        $ pdfrag --db-path /path/to/db

    As library:
        >>> from pdfrag import PDFDatabase, EmbeddingGenerator
        >>> db = PDFDatabase("/path/to/db")
        >>> generator = EmbeddingGenerator()
"""

__version__ = "1.0.0"

from .database import PDFDatabase
from .embeddings import EmbeddingGenerator

__all__ = [
    "PDFDatabase",
    "EmbeddingGenerator",
    "__version__",
]
