# Python Package Reorganization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform pdfrag from flat script directory into proper Python package with src/ layout, pip installation, and CLI entry points.

**Architecture:** Extract monolithic pdf_rag_mcp.py (933 lines) into functional modules (pdf.py, chunking.py, embeddings.py, database.py) while keeping FastMCP server logic cohesive in server.py. Use modern pyproject.toml for packaging with setuptools backend.

**Tech Stack:** Python 3.8+, setuptools, FastMCP, ChromaDB, sentence-transformers, PyMuPDF, NLTK

---

## Task 1: Create Directory Structure

**Files:**
- Create: `src/pdfrag/`
- Create: `tests/`
- Create: `examples/`

**Step 1: Create src/pdfrag directory**

Run:
```bash
mkdir -p src/pdfrag
```

Expected: Directories created successfully

**Step 2: Create tests directory**

Run:
```bash
mkdir -p tests
```

Expected: Directory created successfully

**Step 3: Create examples directory**

Run:
```bash
mkdir -p examples
```

Expected: Directory created successfully

**Step 4: Create __init__.py files**

Run:
```bash
touch src/pdfrag/__init__.py tests/__init__.py
```

Expected: Empty __init__.py files created

**Step 5: Commit**

Run:
```bash
git add src/ tests/ examples/
git commit -m "$(cat <<'EOF'
feat: create Python package directory structure

Add src/pdfrag/, tests/, and examples/ directories for proper Python
package layout following PyPA conventions.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

Expected: Commit created with new directory structure

---

## Task 2: Create pyproject.toml

**Files:**
- Create: `pyproject.toml`

**Step 1: Create pyproject.toml**

Create file at `/Users/wgriffin/! Source/pdfrag/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pdfrag"
version = "1.0.0"
description = "MCP server for RAG capabilities with PDF documents"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}

dependencies = [
    "fastmcp>=0.1.0",
    "chromadb>=0.4.22",
    "sentence-transformers>=2.3.1",
    "pymupdf>=1.23.0",
    "nltk>=3.8.1",
    "pydantic>=2.5.0",
    "httpx>=0.26.0",
    "torch>=2.1.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "ruff>=0.1.0"
]

[project.scripts]
pdfrag = "pdfrag.server:main"
pdfrag-cli = "pdfrag.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

**Step 2: Commit**

Run:
```bash
git add pyproject.toml
git commit -m "$(cat <<'EOF'
feat: add pyproject.toml with packaging configuration

Configure modern Python packaging with setuptools backend, dependencies,
entry points (pdfrag and pdfrag-cli), and pytest configuration.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

Expected: Commit created with pyproject.toml

---

## Task 3: Extract pdf.py Module

**Files:**
- Create: `src/pdfrag/pdf.py`

**Step 1: Create pdf.py with extraction function**

Create file at `/Users/wgriffin/! Source/pdfrag/src/pdfrag/pdf.py`:

```python
# ABOUTME: Extracts text from PDF files using PyMuPDF with OCR fallback.
# ABOUTME: Handles both standard text extraction and scanned document OCR.

"""PDF text extraction with OCR support."""

from typing import List, Dict, Any
import fitz  # PyMuPDF

# Minimum characters to consider a page has text (not scanned)
MIN_TEXT_THRESHOLD = 50


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text from PDF with page numbers, using OCR for scanned pages.

    Tries standard text extraction first. If a page has minimal text (likely scanned),
    falls back to OCR using PyMuPDF's built-in Tesseract integration.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dicts with 'page', 'text', and 'ocr_used' keys

    Example:
        >>> pages = extract_text_from_pdf("/path/to/doc.pdf")
        >>> print(f"Extracted {len(pages)} pages")
        >>> print(f"Page 1: {pages[0]['text'][:100]}")
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
                    text_page = page.get_textpage_ocr()
                    text = page.get_text(textpage=text_page)
                    ocr_used = True
                except Exception:
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
```

**Step 2: Commit**

Run:
```bash
git add src/pdfrag/pdf.py
git commit -m "$(cat <<'EOF'
feat: extract PDF text extraction to pdf.py module

Extract extract_text_from_pdf function from main server file. Handles
standard text extraction and OCR fallback for scanned documents using
PyMuPDF.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

Expected: Commit created with pdf.py module

---

## Task 4: Extract chunking.py Module

**Files:**
- Create: `src/pdfrag/chunking.py`

**Step 1: Create chunking.py with semantic chunking**

Create file at `/Users/wgriffin/! Source/pdfrag/src/pdfrag/chunking.py`:

```python
# ABOUTME: Performs semantic chunking by grouping sentences with configurable overlap.
# ABOUTME: Uses NLTK sentence tokenization to preserve sentence boundaries.

"""Semantic text chunking with sentence boundaries."""

from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 3  # Number of sentences per chunk
DEFAULT_OVERLAP = 1  # Sentence overlap between chunks

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
               overlap: int = DEFAULT_OVERLAP) -> List[str]:
    """Perform semantic chunking by splitting text into sentences and grouping them.

    Splits text at sentence boundaries and groups consecutive sentences into chunks
    with configurable overlap. Preserves context by overlapping sentences between
    adjacent chunks.

    Args:
        text: Text to chunk
        chunk_size: Number of sentences per chunk (default: 3)
        overlap: Number of sentences to overlap between chunks (default: 1)

    Returns:
        List of text chunks

    Example:
        >>> text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        >>> chunks = chunk_text(text, chunk_size=2, overlap=1)
        >>> print(len(chunks))  # 3 chunks with overlap
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


def create_chunks_from_pages(pages_text: List[Dict[str, Any]],
                             chunk_size: int = DEFAULT_CHUNK_SIZE,
                             overlap: int = DEFAULT_OVERLAP) -> List[Dict[str, Any]]:
    """Create semantic chunks from PDF pages with metadata.

    Processes each page's text through semantic chunking and attaches metadata
    including page numbers and chunk indices.

    Args:
        pages_text: List of page dictionaries with 'page' and 'text'
        chunk_size: Number of sentences per chunk (default: 3)
        overlap: Sentence overlap between chunks (default: 1)

    Returns:
        List of chunk dictionaries with text, page number, and chunk index

    Example:
        >>> pages = [{"page": 1, "text": "Content..."}]
        >>> chunks = create_chunks_from_pages(pages)
        >>> print(chunks[0]["page"])  # 1
    """
    all_chunks = []
    global_chunk_idx = 0

    for page_data in pages_text:
        page_num = page_data['page']
        page_text = page_data['text']

        # Create chunks for this page
        chunks = chunk_text(page_text, chunk_size, overlap)

        for chunk_text in chunks:
            all_chunks.append({
                'text': chunk_text,
                'page': page_num,
                'chunk_index': global_chunk_idx
            })
            global_chunk_idx += 1

    return all_chunks
```

**Step 2: Commit**

Run:
```bash
git add src/pdfrag/chunking.py
git commit -m "$(cat <<'EOF'
feat: extract semantic chunking to chunking.py module

Extract semantic_chunking and create_chunks_from_pdf functions (renamed to
chunk_text and create_chunks_from_pages). Uses NLTK for sentence-based
chunking with configurable overlap.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

Expected: Commit created with chunking.py module

---

## Task 5: Extract embeddings.py Module

**Files:**
- Create: `src/pdfrag/embeddings.py`

**Step 1: Create embeddings.py with generator class**

Create file at `/Users/wgriffin/! Source/pdfrag/src/pdfrag/embeddings.py`:

```python
# ABOUTME: Generates text embeddings using sentence-transformers models.
# ABOUTME: Wraps SentenceTransformer for consistent embedding generation interface.

"""Text embedding generation using sentence-transformers."""

from typing import List
from sentence_transformers import SentenceTransformer

# Default embedding model
DEFAULT_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"


class EmbeddingGenerator:
    """Generates embeddings using sentence-transformers.

    Wraps SentenceTransformer model for generating 768-dimensional embeddings
    optimized for question-answering and semantic search tasks.

    Attributes:
        model: SentenceTransformer model instance
        model_name: Name of the loaded model
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """Initialize embedding generator with specified model.

        Args:
            model_name: Name of sentence-transformers model to use
                       (default: multi-qa-mpnet-base-dot-v1)

        Example:
            >>> generator = EmbeddingGenerator()
            >>> embeddings = generator.generate(["Hello world"])
            >>> print(len(embeddings[0]))  # 768
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def generate(self, texts: List[str], show_progress: bool = False) -> List[List[float]]:
        """Generate embeddings for batch of texts.

        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar during encoding

        Returns:
            List of embedding vectors (each 768 dimensions)

        Example:
            >>> generator = EmbeddingGenerator()
            >>> texts = ["First text", "Second text"]
            >>> embeddings = generator.generate(texts)
            >>> print(len(embeddings))  # 2
            >>> print(len(embeddings[0]))  # 768
        """
        embeddings = self.model.encode(texts, show_progress_bar=show_progress)
        return embeddings.tolist()

    def generate_single(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector (768 dimensions)

        Example:
            >>> generator = EmbeddingGenerator()
            >>> embedding = generator.generate_single("Hello world")
            >>> print(len(embedding))  # 768
        """
        embedding = self.model.encode([text], show_progress_bar=False)[0]
        return embedding.tolist()
```

**Step 2: Commit**

Run:
```bash
git add src/pdfrag/embeddings.py
git commit -m "$(cat <<'EOF'
feat: extract embedding generation to embeddings.py module

Create EmbeddingGenerator class wrapping SentenceTransformer for
consistent interface. Generates 768-dimensional embeddings using
multi-qa-mpnet-base-dot-v1 model.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

Expected: Commit created with embeddings.py module

---

## Task 6: Extract database.py Module

**Files:**
- Create: `src/pdfrag/database.py`

**Step 1: Create database.py with ChromaDB wrapper**

Create file at `/Users/wgriffin/! Source/pdfrag/src/pdfrag/database.py`:

```python
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
        results = self.collection.get(where={"document_id": document_id})
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
        all_data = self.collection.get()

        if not all_data['ids']:
            return []

        # Group by document_id
        doc_map = {}
        for metadata in all_data['metadatas']:
            doc_id = metadata['document_id']
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    'document_id': doc_id,
                    'filename': metadata['filename'],
                    'chunk_count': 0,
                    'added_date': 'N/A'
                }
            doc_map[doc_id]['chunk_count'] += 1

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
        all_data = self.collection.get(where=where_filter)

        if not all_data['ids']:
            return []

        # Score each chunk based on keyword matches
        scored_results = []
        for i, doc_id in enumerate(all_data['ids']):
            text = all_data['documents'][i].lower()

            # Count keyword occurrences
            score = sum(text.count(keyword.lower()) for keyword in keywords)

            if score > 0:
                scored_results.append({
                    'chunk_id': doc_id,
                    'document': all_data['metadatas'][i]['filename'],
                    'document_id': all_data['metadatas'][i]['document_id'],
                    'page': all_data['metadatas'][i]['page'],
                    'chunk_index': all_data['metadatas'][i]['chunk_index'],
                    'text': all_data['documents'][i],
                    'similarity': score / len(keywords),
                    'keyword_matches': score
                })

        # Sort by score descending
        scored_results.sort(key=lambda x: x['keyword_matches'], reverse=True)

        return scored_results[:top_k]
```

**Step 2: Commit**

Run:
```bash
git add src/pdfrag/database.py
git commit -m "$(cat <<'EOF'
feat: extract ChromaDB interface to database.py module

Create PDFDatabase class encapsulating all ChromaDB operations:
document add/remove, listing, similarity search, and keyword search.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

Expected: Commit created with database.py module

---

## Task 7: Create server.py with FastMCP Tools

**Files:**
- Create: `src/pdfrag/server.py`

**Step 1: Create server.py (Part 1: Imports and setup)**

Create file at `/Users/wgriffin/! Source/pdfrag/src/pdfrag/server.py` with imports and helper functions. Note: This is a large file, review carefully before creating.

File content starts on next message due to length...

**Step 2: Verify server.py was created correctly**

Run:
```bash
ls -lh src/pdfrag/server.py
```

Expected: File exists and is approximately 20-25KB

**Step 3: Commit**

Run:
```bash
git add src/pdfrag/server.py
git commit -m "$(cat <<'EOF'
feat: create server.py with FastMCP tool definitions

Implement FastMCP server with 5 tools (pdf_add, pdf_remove, pdf_list,
pdf_search_similarity, pdf_search_keywords). Orchestrates calls to
extracted modules. Includes main() entry point for CLI.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

Expected: Commit created with server.py

---

## Task 8: Move and Update cli.py

**Files:**
- Move: `mcp_cli.py` → `src/pdfrag/cli.py`
- Modify: `src/pdfrag/cli.py`

**Step 1: Copy mcp_cli.py to src/pdfrag/cli.py**

Run:
```bash
cp mcp_cli.py src/pdfrag/cli.py
```

Expected: File copied successfully

**Step 2: Add ABOUTME comments to cli.py**

Edit `src/pdfrag/cli.py` - add these two lines at the very top (line 1-2):

```python
# ABOUTME: Command-line interface for discovering and invoking MCP server tools.
# ABOUTME: Supports both interactive and scripting workflows with flexible configuration.
```

**Step 3: Add main() entry point**

Add at the end of `src/pdfrag/cli.py` (replace the existing `if __name__ == "__main__":` block):

```python
def main():
    """Entry point for pdfrag-cli command."""
    asyncio.run(cli_main())


if __name__ == "__main__":
    main()
```

Note: Rename the existing `main()` function to `cli_main()` first.

**Step 4: Commit**

Run:
```bash
git add src/pdfrag/cli.py
git commit -m "$(cat <<'EOF'
feat: move MCP CLI to src/pdfrag/cli.py

Copy mcp_cli.py to package structure and add main() entry point for
pdfrag-cli command. Add ABOUTME comments.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

Expected: Commit created with cli.py

---

## Task 9: Create Package __init__.py

**Files:**
- Modify: `src/pdfrag/__init__.py`

**Step 1: Update __init__.py with exports**

Edit `/Users/wgriffin/! Source/pdfrag/src/pdfrag/__init__.py`:

```python
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
```

**Step 2: Commit**

Run:
```bash
git add src/pdfrag/__init__.py
git commit -m "$(cat <<'EOF'
feat: add package exports to __init__.py

Export PDFDatabase, EmbeddingGenerator, and __version__ for library usage.
Include package docstring with usage examples.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

Expected: Commit created with updated __init__.py

---

## Task 10: Move Test File

**Files:**
- Move: `test_pdf_rag.py` → `tests/test_integration.py`
- Modify: `tests/test_integration.py`

**Step 1: Copy test file**

Run:
```bash
cp test_pdf_rag.py tests/test_integration.py
```

Expected: File copied successfully

**Step 2: Update imports in test file**

Edit `tests/test_integration.py` - replace the imports section (lines 16-21):

```python
from pdfrag.pdf import extract_text_from_pdf
from pdfrag.chunking import chunk_text, create_chunks_from_pages
from pdfrag.database import PDFDatabase
from pdfrag.embeddings import EmbeddingGenerator
```

Remove the sys.path manipulation (lines 13-14).

Update function name references:
- `semantic_chunking` → `chunk_text`
- `create_chunks_from_pdf` → `create_chunks_from_pages`

**Step 3: Commit**

Run:
```bash
git add tests/test_integration.py
git commit -m "$(cat <<'EOF'
feat: move test file to tests/test_integration.py

Copy test_pdf_rag.py to tests directory and update imports to use
new package structure. Update function names to match renamed exports.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

Expected: Commit created with test file

---

## Task 11: Move Documentation Files

**Files:**
- Move: Documentation files to `docs/`
- Move: `claude_desktop_config.json` to `examples/`

**Step 1: Move documentation files**

Run:
```bash
mv GETTING_STARTED.md QUICKSTART.md PROJECT_OVERVIEW.md INDEX.md README-mcp_cli.md docs/
```

Expected: Files moved successfully

**Step 2: Move example config**

Run:
```bash
mv claude_desktop_config.json examples/
```

Expected: File moved successfully

**Step 3: Update examples/claude_desktop_config.json**

Edit `examples/claude_desktop_config.json` to use new command:

```json
{
  "mcpServers": {
    "pdf-rag": {
      "command": "pdfrag",
      "args": ["--db-path", "/Users/wgriffin/.dotfiles/files/mcps/pdfrag/chroma_db"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

**Step 4: Commit**

Run:
```bash
git add docs/ examples/
git commit -m "$(cat <<'EOF'
docs: reorganize documentation and examples

Move all guides to docs/ directory. Move claude_desktop_config.json to
examples/ and update to use new pdfrag command.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

Expected: Commit created with reorganized docs

---

## Task 12: Update README

**Files:**
- Modify: `README.md`

**Step 1: Update installation section in README**

Edit `README.md` - update the "Quick Start" or "Installation" section to include:

```markdown
## Installation

### From Source

1. Clone the repository:
```bash
git clone <repository-url>
cd pdfrag
```

2. Install the package:
```bash
pip install -e .
```

3. Verify installation:
```bash
pdfrag --help
pdfrag-cli --help
```

### Configuration

Configure Claude Desktop to use the MCP server:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "pdf-rag": {
      "command": "pdfrag",
      "args": ["--db-path", "/path/to/your/chroma_db"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

See `examples/claude_desktop_config.json` for a complete example.
```

**Step 2: Add project structure section**

Add a new "Project Structure" section:

```markdown
## Project Structure

```
pdfrag/
├── src/pdfrag/          # Main package
│   ├── server.py        # FastMCP server with 5 tools
│   ├── database.py      # ChromaDB interface
│   ├── embeddings.py    # Embedding generation
│   ├── pdf.py           # PDF text extraction
│   ├── chunking.py      # Semantic chunking
│   └── cli.py           # MCP CLI tool
├── tests/               # Test suite
├── docs/                # Documentation
├── examples/            # Configuration examples
└── pyproject.toml       # Package configuration
```
```

**Step 3: Update documentation links**

Find any links to documentation files and update them:
- `GETTING_STARTED.md` → `docs/GETTING_STARTED.md`
- `QUICKSTART.md` → `docs/QUICKSTART.md`
- `PROJECT_OVERVIEW.md` → `docs/PROJECT_OVERVIEW.md`
- `INDEX.md` → `docs/INDEX.md`

**Step 4: Commit**

Run:
```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: update README for new package structure

Update installation instructions to use pip install. Add project
structure section. Update documentation links to docs/ directory.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

Expected: Commit created with updated README

---

## Task 13: Validate Installation and Remove Old Files

**Files:**
- Delete: `pdf_rag_mcp.py`
- Delete: `mcp_cli.py`
- Delete: `test_pdf_rag.py`
- Delete: `requirements.txt`

**Step 1: Install package in development mode**

Run:
```bash
pip install -e .
```

Expected: Package installs successfully with all dependencies

**Step 2: Verify entry points work**

Run:
```bash
pdfrag --help
```

Expected: Help message displays with --db-path option

Run:
```bash
pdfrag-cli --help
```

Expected: CLI help message displays

**Step 3: Run tests**

Run:
```bash
pytest tests/ -v
```

Expected: Tests pass (or gracefully skip if no PDFs available)

**Step 4: Test imports**

Run:
```bash
python -c "from pdfrag import PDFDatabase, EmbeddingGenerator, __version__; print(f'pdfrag v{__version__}')"
```

Expected: Prints "pdfrag v1.0.0"

**Step 5: Remove old files**

Run:
```bash
git rm pdf_rag_mcp.py mcp_cli.py test_pdf_rag.py requirements.txt
```

Expected: Files staged for deletion

**Step 6: Final commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
refactor: remove old files after validation

Remove pdf_rag_mcp.py, mcp_cli.py, test_pdf_rag.py, and requirements.txt.
All functionality has been migrated to new package structure and validated.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

Expected: Commit created removing old files

**Step 7: Verify clean working directory**

Run:
```bash
git status
```

Expected: "nothing to commit, working tree clean"

---

## Completion Checklist

After completing all tasks, verify:

- [ ] Package installs with `pip install -e .`
- [ ] `pdfrag --help` command works
- [ ] `pdfrag-cli --help` command works
- [ ] Tests pass with `pytest tests/`
- [ ] Imports work: `from pdfrag import PDFDatabase, EmbeddingGenerator`
- [ ] All old files removed (pdf_rag_mcp.py, mcp_cli.py, test_pdf_rag.py, requirements.txt)
- [ ] Documentation updated and links work
- [ ] Git history clean with descriptive commits
- [ ] No uncommitted changes

## Post-Migration

Users can now:
1. Install with `pip install -e .` or `pip install pdfrag`
2. Run MCP server with `pdfrag` command
3. Test with `pdfrag-cli` command
4. Import as library: `from pdfrag import PDFDatabase, EmbeddingGenerator`

## Notes

- Task 7 (server.py) content intentionally truncated here - create the complete file by extracting FastMCP tool definitions from original pdf_rag_mcp.py
- Update imports in server.py to use relative imports from extracted modules
- Preserve all Pydantic models, helper functions, and tool annotations
- Add main() entry point that parses --db-path argument and initializes database
