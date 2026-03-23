# ABOUTME: FastMCP server providing RAG capabilities for PDF documents via 5 MCP tools.
# ABOUTME: Orchestrates PDF extraction, chunking, embedding, and database operations.

"""PDF RAG MCP Server - Semantic search and retrieval for PDF documents."""

import os
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

from .pdf import extract_text_from_pdf
from .chunking import (
    create_chunks_from_pages,
    create_paragraph_chunks_from_pages,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
)
from .embeddings import EmbeddingGenerator, DEFAULT_MODEL
from .database import PDFDatabase


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Constants
CHARACTER_LIMIT = 25000
DEFAULT_DB_PATH = "/Users/wgriffin/.dotfiles/files/mcps/pdfrag/chroma_db"


# Enums
class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    MARKDOWN = "markdown"
    JSON = "json"


class ChunkingStrategy(str, Enum):
    """Chunking strategy for PDF ingestion."""
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


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
        from sentence_transformers import CrossEncoder

        device = os.environ.get("PDFRAG_DEVICE")
        database = PDFDatabase(db_path)
        embedding_gen = EmbeddingGenerator(DEFAULT_MODEL, device=device)
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

        yield {
            "database": database,
            "embedding_generator": embedding_gen,
            "reranker": reranker,
            "collection": database.collection  # For backward compatibility
        }

    return app_lifespan


# Helper Functions

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
    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.SENTENCE,
        description="Chunking strategy: 'sentence' (sentence-based) or 'paragraph' (paragraph-aware)"
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
    rerank: bool = Field(
        default=False,
        description="Re-rank results with a cross-encoder for higher precision (slower)"
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


# Parse command line arguments to get database path
def get_db_path_from_args() -> str:
    """Parse command line arguments and return the database path.

    Returns:
        Database path from command line or DEFAULT_DB_PATH
    """
    parser = argparse.ArgumentParser(
        description="PDF RAG MCP Server - Semantic search over PDF documents"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to ChromaDB database directory (default: {DEFAULT_DB_PATH})"
    )
    args, _ = parser.parse_known_args()
    return args.db_path


# Initialize MCP server with database path
_db_path = get_db_path_from_args()
mcp = FastMCP("pdf_rag_mcp", lifespan=create_lifespan(_db_path))


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
        database = ctx.request_context.lifespan_context["database"]
        embedding_gen = ctx.request_context.lifespan_context["embedding_generator"]

        # Generate document ID from file hash
        document_id = get_file_hash(params.pdf_path)
        filename = Path(params.pdf_path).name

        # Check if document already exists
        if database.document_exists(document_id):
            existing_docs = database.list_documents()
            existing_doc = next(
                (d for d in existing_docs if d['document_id'] == document_id), None
            )
            return json.dumps({
                "status": "already_exists",
                "message": f"Document '{filename}' is already in the database",
                "document_id": document_id,
                "existing_chunks": existing_doc['chunk_count'] if existing_doc else 0
            }, indent=2)

        # Extract text from PDF
        ctx.report_progress(0.2, "Extracting text from PDF...")
        pages_text = extract_text_from_pdf(params.pdf_path)

        if not pages_text:
            return json.dumps({
                "status": "error",
                "message": "Could not extract any text from PDF. The file may be empty, corrupted, or OCR failed (ensure tesseract is installed for scanned PDFs)."
            }, indent=2)

        # Create chunks using the requested strategy
        ctx.report_progress(0.4, "Creating semantic chunks...")
        if params.chunking_strategy == ChunkingStrategy.PARAGRAPH:
            chunks = create_paragraph_chunks_from_pages(pages_text)
        else:
            chunks = create_chunks_from_pages(
                pages_text, params.chunk_size or DEFAULT_CHUNK_SIZE, params.overlap or DEFAULT_OVERLAP
            )

        if not chunks:
            return json.dumps({
                "status": "error",
                "message": "Could not create chunks from PDF text."
            }, indent=2)

        # Generate embeddings
        ctx.report_progress(0.6, f"Generating embeddings for {len(chunks)} chunks...")
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = embedding_gen.generate(chunk_texts, show_progress=False)

        # Store in database
        ctx.report_progress(0.8, "Storing in ChromaDB...")
        chunk_count = database.add_document(document_id, filename, chunks, embeddings)

        ctx.report_progress(1.0, "Complete!")

        return json.dumps({
            "status": "success",
            "message": f"Successfully added '{filename}' to the database",
            "document_id": document_id,
            "filename": filename,
            "pages": len(pages_text),
            "chunks": chunk_count,
            "chunking_strategy": params.chunking_strategy.value,
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
        database = ctx.request_context.lifespan_context["database"]

        # Remove document
        result = database.remove_document(params.document_id)

        if not result:
            return json.dumps({
                "status": "not_found",
                "message": f"No document found with ID: {params.document_id}"
            }, indent=2)

        return json.dumps({
            "status": "success",
            "message": f"Successfully removed '{result['filename']}' from the database",
            "document_id": params.document_id,
            "removed_chunks": result['chunk_count']
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
        database = ctx.request_context.lifespan_context["database"]

        # Get all documents
        documents = database.list_documents()

        if not documents:
            if params.response_format == ResponseFormat.MARKDOWN:
                return "No documents in the database."
            else:
                return json.dumps({"count": 0, "documents": []}, indent=2)

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
        database = ctx.request_context.lifespan_context["database"]
        embedding_gen = ctx.request_context.lifespan_context["embedding_generator"]

        # Generate query embedding
        query_embedding = embedding_gen.generate_single(params.query)

        # Perform search (fetch extra candidates when reranking)
        fetch_k = max(params.top_k * 3, 25) if params.rerank else params.top_k
        results = database.search_similarity(
            query_embedding,
            top_k=fetch_k,
            document_filter=params.document_filter
        )

        # Rerank with cross-encoder when requested
        if params.rerank and results:
            reranker = ctx.request_context.lifespan_context["reranker"]
            pairs = [(params.query, r['text']) for r in results]
            scores = reranker.predict(pairs)
            for result, score in zip(results, scores):
                result['similarity'] = float(score)
            results = sorted(results, key=lambda r: r['similarity'], reverse=True)[:params.top_k]

        if not results:
            if params.response_format == ResponseFormat.MARKDOWN:
                return f"No results found for query: '{params.query}'"
            else:
                return json.dumps({"count": 0, "results": []}, indent=2)

        # Format output
        if params.response_format == ResponseFormat.MARKDOWN:
            output = format_search_results_markdown(results, params.query)
        else:
            output = format_search_results_json(results)

        # Check character limit
        if len(output) > CHARACTER_LIMIT:
            # Truncate results
            truncated_count = max(1, len(results) // 2)
            results = results[:truncated_count]

            if params.response_format == ResponseFormat.MARKDOWN:
                output = format_search_results_markdown(results, params.query)
                output += f"\n\n**Note:** Results truncated to {truncated_count} items to stay within character limit. Use a smaller top_k value or add document_filter for more focused results."
            else:
                output = format_search_results_json(results)
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
        database = ctx.request_context.lifespan_context["database"]

        # Convert keywords to list
        keywords = params.keywords.split()

        # Perform search
        results = database.search_keywords(
            keywords,
            top_k=params.top_k,
            document_filter=params.document_filter
        )

        if not results:
            if params.response_format == ResponseFormat.MARKDOWN:
                return f"No results found for keywords: '{params.keywords}'"
            else:
                return json.dumps({"count": 0, "results": []}, indent=2)

        # Format output
        if params.response_format == ResponseFormat.MARKDOWN:
            output = format_search_results_markdown(results, params.keywords)
        else:
            output = format_search_results_json(results)

        # Check character limit
        if len(output) > CHARACTER_LIMIT:
            truncated_count = max(1, len(results) // 2)
            results = results[:truncated_count]

            if params.response_format == ResponseFormat.MARKDOWN:
                output = format_search_results_markdown(results, params.keywords)
                output += f"\n\n**Note:** Results truncated to {truncated_count} items to stay within character limit. Use a smaller top_k value or add document_filter for more focused results."
            else:
                output = format_search_results_json(results)
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


# Entry point
def main():
    """Entry point for pdfrag MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
