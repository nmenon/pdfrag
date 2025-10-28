#!/usr/bin/env python3
"""
PDF RAG MCP Server

An MCP server that provides RAG (Retrieval-Augmented Generation) capabilities for PDF documents.
Uses ChromaDB for vector storage, sentence-transformers for embeddings, and semantic chunking
for intelligent text segmentation.

Features:
- Add and remove PDFs from the database
- Semantic similarity search
- Keyword-based search
- Source document and page number tracking
- Semantic chunking for better context preservation
"""

import os
import sys
import json
import hashlib
import logging
import argparse
from typing import Optional, List, Dict, Any
from enum import Enum
from pathlib import Path
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field, field_validator, ConfigDict
import chromadb
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Constants
CHARACTER_LIMIT = 25000
DEFAULT_DB_PATH = "/Users/wgriffin/.dotfiles/files/mcps/pdfrag/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
DEFAULT_CHUNK_SIZE = 3  # Number of sentences per chunk
DEFAULT_OVERLAP = 1  # Sentence overlap between chunks
MIN_TEXT_THRESHOLD = 50  # Minimum characters to consider a page has text (not scanned)


# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


# Enums
class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    MARKDOWN = "markdown"
    JSON = "json"


class SearchType(str, Enum):
    """Type of search to perform."""
    SIMILARITY = "similarity"
    KEYWORD = "keyword"


# Global state for lifespan management
def create_lifespan(db_path: str):
    """Create a lifespan function with the specified database path.

    Args:
        db_path: Path to the ChromaDB database directory

    Returns:
        An async context manager for application lifespan
    """
    @asynccontextmanager
    async def app_lifespan(app):
        """Manage resources that live for the server's lifetime."""
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Initialize embedding model
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        # Get or create collection
        collection = chroma_client.get_or_create_collection(
            name="pdf_documents",
            metadata={"hnsw:space": "cosine"}
        )

        yield {
            "chroma_client": chroma_client,
            "embedding_model": embedding_model,
            "collection": collection
        }

        # Cleanup on shutdown (if needed)

    return app_lifespan


# Parse command line arguments to get database path
def get_db_path_from_args() -> str:
    """Parse command line arguments and return the database path.

    Returns:
        Database path from command line or DEFAULT_DB_PATH
    """
    # Only parse args if running as main script
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            description="PDF RAG MCP Server - Semantic search over PDF documents"
        )
        parser.add_argument(
            "--db-path",
            type=str,
            default=DEFAULT_DB_PATH,
            help=f"Path to ChromaDB database directory (default: {DEFAULT_DB_PATH})"
        )
        args = parser.parse_args()
        return args.db_path
    else:
        # When imported as a module, use default
        return DEFAULT_DB_PATH


# Initialize MCP server with database path
_db_path = get_db_path_from_args()
mcp = FastMCP("pdf_rag_mcp", lifespan=create_lifespan(_db_path))


# Helper Functions

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF with page numbers, using OCR for scanned pages.

    Tries standard text extraction first. If a page has minimal text (likely scanned),
    falls back to OCR using PyMuPDF's built-in Tesseract integration.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dicts with 'page', 'text', and 'ocr_used' keys
    """
    pages_text = []

    # Open PDF with PyMuPDF
    doc = fitz.open(pdf_path)

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            ocr_used = False

            # Try standard text extraction first
            text = page.get_text()

            # If minimal text found, likely a scanned page - use OCR
            if len(text.strip()) < MIN_TEXT_THRESHOLD:
                try:
                    # Use PyMuPDF's OCR capability (requires tesseract)
                    # get_textpage_ocr() returns a TextPage with OCR'd text
                    text_page = page.get_textpage_ocr()
                    text = page.get_text(textpage=text_page)
                    ocr_used = True
                except Exception as ocr_error:
                    # OCR failed (tesseract not installed or other error)
                    # Fall back to whatever text we got, even if minimal
                    pass

            # Only add pages with text content
            if text.strip():
                pages_text.append({
                    'page': page_num + 1,  # 1-indexed for user display
                    'text': text,
                    'ocr_used': ocr_used
                })

    finally:
        doc.close()

    return pages_text


def semantic_chunking(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, 
                     overlap: int = DEFAULT_OVERLAP) -> List[str]:
    """
    Perform semantic chunking by splitting text into sentences and grouping them.
    
    Args:
        text: Text to chunk
        chunk_size: Number of sentences per chunk
        overlap: Number of sentences to overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Split into sentences
    sentences = sent_tokenize(text)
    
    if not sentences:
        return []
    
    chunks = []
    i = 0
    
    while i < len(sentences):
        # Take chunk_size sentences
        chunk_sentences = sentences[i:i + chunk_size]
        chunk = ' '.join(chunk_sentences)
        chunks.append(chunk)
        
        # Move forward by (chunk_size - overlap) to create overlap
        i += max(1, chunk_size - overlap)
    
    return chunks


def create_chunks_from_pdf(pages_text: List[Dict[str, Any]], 
                           chunk_size: int = DEFAULT_CHUNK_SIZE,
                           overlap: int = DEFAULT_OVERLAP) -> List[Dict[str, Any]]:
    """
    Create semantic chunks from PDF pages with metadata.
    
    Args:
        pages_text: List of page dictionaries with 'page' and 'text'
        chunk_size: Number of sentences per chunk
        overlap: Sentence overlap between chunks
        
    Returns:
        List of chunk dictionaries with text, page number, and chunk index
    """
    all_chunks = []
    global_chunk_idx = 0
    
    for page_data in pages_text:
        page_num = page_data['page']
        page_text = page_data['text']
        
        # Create chunks for this page
        chunks = semantic_chunking(page_text, chunk_size, overlap)
        
        for chunk_text in chunks:
            all_chunks.append({
                'text': chunk_text,
                'page': page_num,
                'chunk_index': global_chunk_idx
            })
            global_chunk_idx += 1
    
    return all_chunks


def get_file_hash(filepath: str) -> str:
    """Generate SHA256 hash of file for unique identification."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def format_search_results_markdown(results: List[Dict[str, Any]], query: str) -> str:
    """Format search results as Markdown."""
    if not results:
        return f"No results found for query: '{query}'"
    
    output = [f"# Search Results for: '{query}'"]
    output.append(f"\nFound {len(results)} relevant chunks:\n")
    
    for i, result in enumerate(results, 1):
        output.append(f"## Result {i}")
        output.append(f"**Document:** {result['document']}")
        output.append(f"**Page:** {result['page']}")
        output.append(f"**Similarity Score:** {result['similarity']:.4f}")
        output.append(f"\n**Content:**")
        output.append(f"{result['text']}\n")
        output.append("---\n")
    
    return '\n'.join(output)


def format_search_results_json(results: List[Dict[str, Any]]) -> str:
    """Format search results as JSON."""
    return json.dumps({
        'count': len(results),
        'results': results
    }, indent=2)


def format_document_list_markdown(documents: List[Dict[str, Any]]) -> str:
    """Format document list as Markdown."""
    if not documents:
        return "No documents in the database."
    
    output = [f"# PDF Documents ({len(documents)} total)\n"]
    
    for doc in documents:
        output.append(f"## {doc['filename']}")
        output.append(f"**Document ID:** {doc['document_id']}")
        output.append(f"**Chunks:** {doc['chunk_count']}")
        output.append(f"**Added:** {doc['added_date']}")
        output.append("")
    
    return '\n'.join(output)


def format_document_list_json(documents: List[Dict[str, Any]]) -> str:
    """Format document list as JSON."""
    return json.dumps({
        'count': len(documents),
        'documents': documents
    }, indent=2)


# Pydantic Models

class PdfAddInput(BaseModel):
    """Input model for adding a PDF to the RAG database."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    pdf_path: str = Field(
        ...,
        description="Absolute path to the PDF file to add (e.g., '/home/user/documents/paper.pdf', '/tmp/report.pdf')",
        min_length=1
    )
    chunk_size: Optional[int] = Field(
        default=DEFAULT_CHUNK_SIZE,
        description="Number of sentences per chunk for semantic chunking (default: 3)",
        ge=1,
        le=20
    )
    overlap: Optional[int] = Field(
        default=DEFAULT_OVERLAP,
        description="Number of sentences to overlap between chunks (default: 1)",
        ge=0,
        le=10
    )
    
    @field_validator('pdf_path')
    @classmethod
    def validate_pdf_path(cls, v: str) -> str:
        """Validate that the PDF path exists and is a PDF file."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"File not found: {v}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {v}")
        if path.suffix.lower() != '.pdf':
            raise ValueError(f"File is not a PDF: {v}")
        return str(path.absolute())


class PdfRemoveInput(BaseModel):
    """Input model for removing a PDF from the RAG database."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    document_id: str = Field(
        ...,
        description="Document ID (file hash) of the PDF to remove. Get this from pdf_list tool.",
        min_length=1
    )


class PdfListInput(BaseModel):
    """Input model for listing PDFs in the database."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )


class PdfSearchInput(BaseModel):
    """Input model for searching PDFs."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    query: str = Field(
        ...,
        description="Search query text (e.g., 'machine learning algorithms', 'climate change impacts')",
        min_length=1,
        max_length=500
    )
    top_k: Optional[int] = Field(
        default=5,
        description="Number of top results to return (default: 5)",
        ge=1,
        le=50
    )
    document_filter: Optional[str] = Field(
        default=None,
        description="Optional document ID to search within a specific document only"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )


class PdfKeywordSearchInput(BaseModel):
    """Input model for keyword-based PDF search."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    keywords: str = Field(
        ...,
        description="Keywords to search for, space-separated (e.g., 'neural network training')",
        min_length=1,
        max_length=500
    )
    top_k: Optional[int] = Field(
        default=5,
        description="Number of top results to return (default: 5)",
        ge=1,
        le=50
    )
    document_filter: Optional[str] = Field(
        default=None,
        description="Optional document ID to search within a specific document only"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )


# MCP Tools

@mcp.tool(
    name="pdf_add",
    annotations={
        "title": "Add PDF to RAG Database",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def pdf_add(params: PdfAddInput, ctx: Context) -> str:
    """Add a PDF document to the RAG database with semantic chunking.
    
    This tool reads a PDF file, extracts text with page numbers, performs semantic
    chunking (grouping sentences intelligently), generates embeddings, and stores
    everything in ChromaDB for later retrieval.
    
    Args:
        params (PdfAddInput): Input parameters containing:
            - pdf_path (str): Absolute path to the PDF file
            - chunk_size (Optional[int]): Sentences per chunk (default: 3)
            - overlap (Optional[int]): Sentence overlap (default: 1)
    
    Returns:
        str: JSON response with document_id, filename, pages, chunks, and status
        
    Example:
        Input: {"pdf_path": "/home/user/research.pdf", "chunk_size": 3, "overlap": 1}
        Output: {"status": "success", "document_id": "abc123...", "chunks": 45}
    """
    try:
        # Get lifespan resources
        embedding_model = ctx.request_context.lifespan_context["embedding_model"]
        collection = ctx.request_context.lifespan_context["collection"]
        
        # Generate document ID from file hash
        document_id = get_file_hash(params.pdf_path)
        filename = Path(params.pdf_path).name
        
        # Check if document already exists
        existing = collection.get(where={"document_id": document_id})
        if existing['ids']:
            return json.dumps({
                "status": "already_exists",
                "message": f"Document '{filename}' is already in the database",
                "document_id": document_id,
                "existing_chunks": len(existing['ids'])
            }, indent=2)
        
        # Extract text from PDF
        ctx.report_progress(0.2, "Extracting text from PDF...")
        pages_text = extract_text_from_pdf(params.pdf_path)
        
        if not pages_text:
            return json.dumps({
                "status": "error",
                "message": "Could not extract any text from PDF. The file may be empty, corrupted, or OCR failed (ensure tesseract is installed for scanned PDFs)."
            }, indent=2)
        
        # Create semantic chunks
        ctx.report_progress(0.4, "Creating semantic chunks...")
        chunks = create_chunks_from_pdf(pages_text, params.chunk_size, params.overlap)
        
        if not chunks:
            return json.dumps({
                "status": "error",
                "message": "Could not create chunks from PDF text."
            }, indent=2)
        
        # Generate embeddings
        ctx.report_progress(0.6, f"Generating embeddings for {len(chunks)} chunks...")
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = embedding_model.encode(chunk_texts, show_progress_bar=False).tolist()
        
        # Prepare data for ChromaDB
        ctx.report_progress(0.8, "Storing in ChromaDB...")
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

        # Add to collection in batches (manually batch to avoid ChromaDB API changes)
        batch_size = 5461  # ChromaDB's default batch size
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=chunk_texts[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
        
        ctx.report_progress(1.0, "Complete!")
        
        return json.dumps({
            "status": "success",
            "message": f"Successfully added '{filename}' to the database",
            "document_id": document_id,
            "filename": filename,
            "pages": len(pages_text),
            "chunks": len(chunks),
            "chunk_size": params.chunk_size,
            "overlap": params.overlap
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error adding PDF: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": f"Failed to add PDF: {str(e)}"
        }, indent=2)


@mcp.tool(
    name="pdf_remove",
    annotations={
        "title": "Remove PDF from RAG Database",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def pdf_remove(params: PdfRemoveInput, ctx: Context) -> str:
    """Remove a PDF document and all its chunks from the RAG database.
    
    This tool deletes all chunks associated with a specific document ID. Use the
    pdf_list tool to find document IDs before removing.
    
    Args:
        params (PdfRemoveInput): Input parameters containing:
            - document_id (str): Document ID (file hash) to remove
    
    Returns:
        str: JSON response with deletion status and count of removed chunks
        
    Example:
        Input: {"document_id": "abc123..."}
        Output: {"status": "success", "removed_chunks": 45}
    """
    try:
        collection = ctx.request_context.lifespan_context["collection"]
        
        # Get all chunks for this document
        results = collection.get(where={"document_id": params.document_id})
        
        if not results['ids']:
            return json.dumps({
                "status": "not_found",
                "message": f"No document found with ID: {params.document_id}"
            }, indent=2)
        
        # Get filename before deletion
        filename = results['metadatas'][0]['filename'] if results['metadatas'] else "Unknown"
        chunk_count = len(results['ids'])
        
        # Delete all chunks
        collection.delete(where={"document_id": params.document_id})
        
        return json.dumps({
            "status": "success",
            "message": f"Successfully removed '{filename}' from the database",
            "document_id": params.document_id,
            "removed_chunks": chunk_count
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error removing PDF: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": f"Failed to remove PDF: {str(e)}"
        }, indent=2)


@mcp.tool(
    name="pdf_list",
    annotations={
        "title": "List All PDFs in Database",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def pdf_list(params: PdfListInput, ctx: Context) -> str:
    """List all PDF documents currently in the RAG database.
    
    This tool returns a list of all documents with their IDs, filenames, chunk counts,
    and metadata. Use this to discover document IDs for other operations.
    
    Args:
        params (PdfListInput): Input parameters containing:
            - response_format (ResponseFormat): Output format (markdown or json)
    
    Returns:
        str: Formatted list of documents (Markdown or JSON based on response_format)
        
    Example Output (JSON):
        {
          "count": 2,
          "documents": [
            {
              "document_id": "abc123...",
              "filename": "research.pdf",
              "chunk_count": 45,
              "added_date": "2024-01-15"
            }
          ]
        }
    """
    try:
        collection = ctx.request_context.lifespan_context["collection"]
        
        # Get all documents
        all_data = collection.get()
        
        if not all_data['ids']:
            if params.response_format == ResponseFormat.MARKDOWN:
                return "No documents in the database."
            else:
                return json.dumps({"count": 0, "documents": []}, indent=2)
        
        # Group by document_id
        doc_map = {}
        for metadata in all_data['metadatas']:
            doc_id = metadata['document_id']
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    'document_id': doc_id,
                    'filename': metadata['filename'],
                    'chunk_count': 0,
                    'added_date': 'N/A'  # ChromaDB doesn't store timestamps by default
                }
            doc_map[doc_id]['chunk_count'] += 1
        
        documents = list(doc_map.values())
        
        # Format output
        if params.response_format == ResponseFormat.MARKDOWN:
            return format_document_list_markdown(documents)
        else:
            return format_document_list_json(documents)
        
    except Exception as e:
        logger.error(f"Error listing PDFs: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": f"Failed to list PDFs: {str(e)}"
        }, indent=2)


@mcp.tool(
    name="pdf_search_similarity",
    annotations={
        "title": "Search PDFs by Semantic Similarity",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def pdf_search_similarity(params: PdfSearchInput, ctx: Context) -> str:
    """Search PDF documents using semantic similarity (vector search).
    
    This tool performs a semantic search across all PDF chunks using embedding-based
    similarity. It finds chunks that are semantically related to your query, even if
    they don't contain the exact keywords.
    
    Args:
        params (PdfSearchInput): Input parameters containing:
            - query (str): Search query text
            - top_k (Optional[int]): Number of results to return (default: 5)
            - document_filter (Optional[str]): Filter to specific document ID
            - response_format (ResponseFormat): Output format (markdown or json)
    
    Returns:
        str: Search results with text, page numbers, similarity scores (formatted as specified)
        
    Example:
        Input: {"query": "machine learning techniques", "top_k": 3}
        Output: List of 3 most relevant chunks with their content and metadata
    """
    try:
        embedding_model = ctx.request_context.lifespan_context["embedding_model"]
        collection = ctx.request_context.lifespan_context["collection"]
        
        # Generate query embedding
        query_embedding = embedding_model.encode([params.query], show_progress_bar=False)[0].tolist()
        
        # Perform search
        where_filter = {"document_id": params.document_filter} if params.document_filter else None
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=params.top_k,
            where=where_filter
        )
        
        if not results['ids'][0]:
            if params.response_format == ResponseFormat.MARKDOWN:
                return f"No results found for query: '{params.query}'"
            else:
                return json.dumps({"count": 0, "results": []}, indent=2)
        
        # Format results
        formatted_results = []
        for i, doc_id in enumerate(results['ids'][0]):
            formatted_results.append({
                'chunk_id': doc_id,
                'document': results['metadatas'][0][i]['filename'],
                'document_id': results['metadatas'][0][i]['document_id'],
                'page': results['metadatas'][0][i]['page'],
                'chunk_index': results['metadatas'][0][i]['chunk_index'],
                'text': results['documents'][0][i],
                'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        # Check character limit
        if params.response_format == ResponseFormat.MARKDOWN:
            output = format_search_results_markdown(formatted_results, params.query)
        else:
            output = format_search_results_json(formatted_results)
        
        if len(output) > CHARACTER_LIMIT:
            # Truncate results
            truncated_count = max(1, len(formatted_results) // 2)
            formatted_results = formatted_results[:truncated_count]
            
            if params.response_format == ResponseFormat.MARKDOWN:
                output = format_search_results_markdown(formatted_results, params.query)
                output += f"\n\n**Note:** Results truncated to {truncated_count} items to stay within character limit. Use a smaller top_k value or add document_filter for more focused results."
            else:
                output = format_search_results_json(formatted_results)
                result_obj = json.loads(output)
                result_obj['truncated'] = True
                result_obj['truncation_message'] = f"Results truncated to {truncated_count} items. Use smaller top_k or add document_filter."
                output = json.dumps(result_obj, indent=2)
        
        return output
        
    except Exception as e:
        logger.error(f"Error searching PDFs: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": f"Failed to search PDFs: {str(e)}"
        }, indent=2)


@mcp.tool(
    name="pdf_search_keywords",
    annotations={
        "title": "Search PDFs by Keywords",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def pdf_search_keywords(params: PdfKeywordSearchInput, ctx: Context) -> str:
    """Search PDF documents using keyword matching.
    
    This tool performs keyword-based search, looking for exact or partial matches
    of the provided keywords in the document chunks. Use this when you want to find
    specific terms or phrases rather than semantic similarity.
    
    Args:
        params (PdfKeywordSearchInput): Input parameters containing:
            - keywords (str): Keywords to search for (space-separated)
            - top_k (Optional[int]): Number of results to return (default: 5)
            - document_filter (Optional[str]): Filter to specific document ID
            - response_format (ResponseFormat): Output format (markdown or json)
    
    Returns:
        str: Search results with matching chunks and metadata (formatted as specified)
        
    Example:
        Input: {"keywords": "neural network training", "top_k": 5}
        Output: Chunks containing these keywords, ranked by relevance
    """
    try:
        collection = ctx.request_context.lifespan_context["collection"]
        
        # Get all documents (or filtered by document_id)
        where_filter = {"document_id": params.document_filter} if params.document_filter else None
        all_data = collection.get(where=where_filter)
        
        if not all_data['ids']:
            if params.response_format == ResponseFormat.MARKDOWN:
                return f"No results found for keywords: '{params.keywords}'"
            else:
                return json.dumps({"count": 0, "results": []}, indent=2)
        
        # Convert keywords to lowercase for case-insensitive matching
        keywords = params.keywords.lower().split()
        
        # Score each chunk based on keyword matches
        scored_results = []
        for i, doc_id in enumerate(all_data['ids']):
            text = all_data['documents'][i].lower()
            
            # Count keyword occurrences
            score = sum(text.count(keyword) for keyword in keywords)
            
            # Check if any keyword is present
            if score > 0:
                scored_results.append({
                    'chunk_id': doc_id,
                    'document': all_data['metadatas'][i]['filename'],
                    'document_id': all_data['metadatas'][i]['document_id'],
                    'page': all_data['metadatas'][i]['page'],
                    'chunk_index': all_data['metadatas'][i]['chunk_index'],
                    'text': all_data['documents'][i],
                    'similarity': score / len(keywords),  # Normalize score
                    'keyword_matches': score
                })
        
        # Sort by score (descending)
        scored_results.sort(key=lambda x: x['keyword_matches'], reverse=True)
        
        # Take top_k results
        formatted_results = scored_results[:params.top_k]
        
        if not formatted_results:
            if params.response_format == ResponseFormat.MARKDOWN:
                return f"No results found for keywords: '{params.keywords}'"
            else:
                return json.dumps({"count": 0, "results": []}, indent=2)
        
        # Format output
        if params.response_format == ResponseFormat.MARKDOWN:
            output = format_search_results_markdown(formatted_results, params.keywords)
        else:
            output = format_search_results_json(formatted_results)
        
        # Check character limit
        if len(output) > CHARACTER_LIMIT:
            truncated_count = max(1, len(formatted_results) // 2)
            formatted_results = formatted_results[:truncated_count]
            
            if params.response_format == ResponseFormat.MARKDOWN:
                output = format_search_results_markdown(formatted_results, params.keywords)
                output += f"\n\n**Note:** Results truncated to {truncated_count} items to stay within character limit. Use a smaller top_k value or add document_filter for more focused results."
            else:
                output = format_search_results_json(formatted_results)
                result_obj = json.loads(output)
                result_obj['truncated'] = True
                result_obj['truncation_message'] = f"Results truncated to {truncated_count} items. Use smaller top_k or add document_filter."
                output = json.dumps(result_obj, indent=2)
        
        return output
        
    except Exception as e:
        logger.error(f"Error searching PDFs by keywords: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": f"Failed to search PDFs by keywords: {str(e)}"
        }, indent=2)


# Run server
if __name__ == "__main__":
    mcp.run()
