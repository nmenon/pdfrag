# Python Package Reorganization Design

## Purpose

Transform pdfrag from a flat script directory into a proper Python package following PyPA conventions. Enable pip installation, provide command-line entry points, improve code organization through functional module separation, and maintain backward compatibility.

## Constraints

- Preserve all existing functionality
- Keep FastMCP server logic cohesive (not fragmented across multiple files)
- Maintain working tests throughout migration
- Use src/ layout for modern packaging best practices
- No disruption to existing users during transition

## Success Criteria

Users install with `pip install -e .` and run commands:
- `pdfrag` launches the MCP server
- `pdfrag-cli` launches the CLI tool
- All tests pass
- Code remains maintainable and clear

## Directory Structure

```
pdfrag/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── CLAUDE.md
│
├── docs/
│   ├── GETTING_STARTED.md
│   ├── QUICKSTART.md
│   ├── PROJECT_OVERVIEW.md
│   ├── INDEX.md
│   └── README-mcp_cli.md
│
├── src/
│   └── pdfrag/
│       ├── __init__.py
│       ├── server.py
│       ├── pdf.py
│       ├── chunking.py
│       ├── embeddings.py
│       ├── database.py
│       └── cli.py
│
├── tests/
│   ├── __init__.py
│   ├── test_pdf.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_database.py
│   ├── test_server.py
│   └── test_integration.py
│
└── examples/
    └── claude_desktop_config.json
```

## Module Design

### src/pdfrag/__init__.py

Exports package version and key classes for library usage:

```python
"""PDF RAG MCP Server - Semantic search and retrieval for PDF documents."""

__version__ = "1.0.0"

from .database import PDFDatabase
from .embeddings import EmbeddingGenerator

__all__ = ["PDFDatabase", "EmbeddingGenerator", "__version__"]
```

### src/pdfrag/pdf.py

Extracts text from PDFs with OCR support:

```python
def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from PDF with page numbers.

    Returns: [{"page": 1, "text": "..."}, ...]
    """
```

Responsibilities:
- PyMuPDF text extraction
- OCR detection and fallback
- Page number tracking
- Error handling for corrupted PDFs

### src/pdfrag/chunking.py

Performs semantic chunking with sentence boundaries:

```python
def chunk_text(text: str, chunk_size: int = 3, overlap: int = 1) -> list[str]:
    """Chunk text by sentences with overlap.

    Args:
        text: Input text to chunk
        chunk_size: Sentences per chunk
        overlap: Sentences to overlap between chunks

    Returns: List of text chunks
    """
```

Responsibilities:
- NLTK sentence tokenization
- Configurable chunk size and overlap
- Preserve sentence boundaries
- Handle edge cases (short documents, etc.)

### src/pdfrag/embeddings.py

Generates embeddings using sentence-transformers:

```python
class EmbeddingGenerator:
    """Generates embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "multi-qa-mpnet-base-dot-v1"):
        """Initialize with specified model."""

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for batch of texts."""
```

Responsibilities:
- Model initialization and caching
- Batch embedding generation
- Handle model download on first use
- Return 768-dimensional vectors

### src/pdfrag/database.py

Manages ChromaDB persistence and search:

```python
class PDFDatabase:
    """ChromaDB interface for PDF document storage and retrieval."""

    def __init__(self, db_path: str):
        """Initialize database at specified path."""

    def add_document(self, doc_name: str, chunks: list[str],
                     embeddings: list[list[float]], metadata: list[dict]) -> None:
        """Add document chunks to database."""

    def remove_document(self, doc_name: str) -> bool:
        """Remove all chunks for document."""

    def list_documents(self) -> list[dict]:
        """List all documents with metadata."""

    def search_similarity(self, query_embedding: list[float],
                         top_k: int = 5) -> list[dict]:
        """Search by vector similarity."""

    def search_keywords(self, keywords: list[str],
                       top_k: int = 5) -> list[dict]:
        """Search by keyword matching."""
```

Responsibilities:
- ChromaDB collection management
- Document metadata tracking
- Similarity search with cosine distance
- Keyword search with frequency scoring
- Handle database initialization and persistence

### src/pdfrag/server.py

Provides FastMCP server with five tools:

```python
from fastmcp import FastMCP
from .pdf import extract_text_from_pdf
from .chunking import chunk_text
from .embeddings import EmbeddingGenerator
from .database import PDFDatabase

mcp = FastMCP("pdf-rag")

@mcp.tool()
def pdf_add(pdf_path: str, chunk_size: int = 3, overlap: int = 1) -> str:
    """Add PDF to database with semantic chunking."""

# Additional tools: pdf_remove, pdf_list, pdf_search_similarity, pdf_search_keywords

def main():
    """Entry point for pdfrag command."""
    import argparse
    parser = argparse.ArgumentParser(description="PDF RAG MCP Server")
    parser.add_argument("--db-path",
                       default="~/.dotfiles/files/mcps/pdfrag/chroma_db")
    args = parser.parse_args()

    # Initialize global database with db_path
    mcp.run()

if __name__ == "__main__":
    main()
```

Responsibilities:
- FastMCP tool registration
- Tool orchestration (combine pdf → chunking → embeddings → database)
- Command-line argument parsing
- Context and progress reporting
- Error handling and user-friendly messages

### src/pdfrag/cli.py

MCP CLI tool for testing servers:

```python
def main():
    """Entry point for pdfrag-cli command."""
    # Existing mcp_cli.py logic

if __name__ == "__main__":
    main()
```

Responsibilities:
- MCP server discovery
- Tool invocation with parameters
- Output formatting (human-readable and JSON)

## Packaging Configuration

File: `pyproject.toml`

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

## Migration Strategy

### Phase 1: Setup Structure
- Create `src/pdfrag/`, `tests/`, `docs/`, `examples/` directories
- Add empty `__init__.py` files
- Create `pyproject.toml`
- Commit: "Add Python package structure"

### Phase 2: Extract Modules
Read `pdf_rag_mcp.py` and extract code:
- PDF extraction → `src/pdfrag/pdf.py`
- Chunking logic → `src/pdfrag/chunking.py`
- Embedding generation → `src/pdfrag/embeddings.py`
- ChromaDB interface → `src/pdfrag/database.py`
- FastMCP tools → `src/pdfrag/server.py`

Move `mcp_cli.py` → `src/pdfrag/cli.py`

Commit after each module: "Extract [module] from main server"

### Phase 3: Wire Modules Together
- Update imports in `server.py` to use new modules
- Add `main()` function to `server.py`
- Add `main()` function to `cli.py`
- Update `__init__.py` with exports
- Commit: "Wire modules together with proper imports"

### Phase 4: Create Tests
- Move `test_pdf_rag.py` → `tests/test_integration.py`
- Create unit tests for each module
- Run tests: `pytest tests/`
- Commit: "Add unit tests for all modules"

### Phase 5: Move Documentation
- Move guides to `docs/`
- Move `claude_desktop_config.json` to `examples/`
- Update all path references in documentation
- Commit: "Reorganize documentation"

### Phase 6: Update Installation Instructions
- Update README.md with new installation steps
- Test installation: `pip install -e .`
- Test commands: `pdfrag --help`, `pdfrag-cli --help`
- Update Claude Desktop config example
- Commit: "Update documentation for new package structure"

### Phase 7: Validate and Cleanup
- Run full test suite
- Test actual usage with Claude Desktop
- Remove old files: `pdf_rag_mcp.py`, `mcp_cli.py`, `test_pdf_rag.py`
- Remove `requirements.txt` (replaced by pyproject.toml)
- Commit: "Remove old files after validation"

## Safety Measures

- Work in git worktree (isolated workspace)
- Commit after each phase
- Keep old files until new structure validated
- All tests must pass before removing old files
- Document rollback procedure if needed

## Rollback Procedure

If problems occur:
1. Return to main worktree
2. Delete reorganization worktree
3. Repository remains unchanged

## Post-Migration

Users update their Claude Desktop config:
```json
{
  "mcpServers": {
    "pdf-rag": {
      "command": "pdfrag",
      "args": ["--db-path", "/custom/path"],
      "env": {"PYTHONUNBUFFERED": "1"}
    }
  }
}
```

Installation becomes:
```bash
pip install -e .
pdfrag --help
pdfrag-cli --help
```

## Benefits

- Standard Python package structure
- Easy installation with pip
- Better code organization
- Improved testability
- Professional distribution
- Cleaner imports
- Separation of concerns
