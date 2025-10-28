# PDF RAG MCP Server

A Model Context Protocol (MCP) server that provides powerful RAG (Retrieval-Augmented Generation) capabilities for PDF
documents. This server uses ChromaDB for vector storage, sentence-transformers for embeddings, and semantic chunking for
intelligent text segmentation.

## Features

- ✅ **Semantic Chunking**: Intelligently groups sentences together instead of splitting at arbitrary character limits
- ✅ **Vector Search**: Find semantically similar content using embeddings
- ✅ **Keyword Search**: Traditional keyword-based search for exact terms
- ✅ **OCR Support**: Automatic detection and OCR processing for scanned/image-based PDFs
- ✅ **Source Tracking**: Maintains document names and page numbers for all chunks
- ✅ **Add/Remove PDFs**: Easily manage your document collection
- ✅ **Persistent Storage**: ChromaDB persists your embeddings to disk
- ✅ **Multiple Output Formats**: Get results in Markdown or JSON format
- ✅ **Progress Reporting**: Real-time feedback during long operations

## Architecture

- **Embedding Model**: `multi-qa-mpnet-base-dot-v1` (optimized for question-answering)
- **Vector Database**: ChromaDB with cosine similarity
- **Chunking Strategy**: Semantic chunking with configurable sentence grouping and overlap
- **PDF Extraction**: PyMuPDF for text extraction with OCR fallback for scanned PDFs

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK Data (Automatic)

The server automatically downloads required NLTK punkt tokenizer data on first run.

### 3. Install Tesseract (Optional - for OCR)

For scanned PDF support, install Tesseract:

- **macOS:** `brew install tesseract`
- **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr`
- **Windows:** Download from https://github.com/UB-Mannheim/tesseract/wiki

The server automatically detects scanned pages and uses OCR when Tesseract is available.

### 4. Test the Server

```bash
python pdf_rag_mcp.py --help
```

## Configuration

### Database Location

The server stores its ChromaDB database in a configurable location. You can specify the database path using the `--db-path` command line argument:

```bash
# Use default location (~/.dotfiles/files/mcps/pdfrag/chroma_db)
python pdf_rag_mcp.py

# Use custom database location
python pdf_rag_mcp.py --db-path /path/to/your/database
```

### Chunking Parameters

Default chunking settings:
- **Chunk Size**: 3 sentences per chunk
- **Overlap**: 1 sentence overlap between chunks

These can be customized when adding PDFs:

```python
{
  "pdf_path": "/path/to/document.pdf",
  "chunk_size": 5,      # Use 5 sentences per chunk
  "overlap": 2          # 2 sentences overlap
}
```

### Character Limit

Responses are limited to 25,000 characters by default. If exceeded, results are automatically truncated with a warning
message.

## MCP Tools

### 1. pdf_add

Add a PDF document to the RAG database.

**Input:**
```json
{
  "pdf_path": "/absolute/path/to/document.pdf",
  "chunk_size": 3,  // optional, default: 3
  "overlap": 1      // optional, default: 1
}
```

**Output:**
```json
{
  "status": "success",
  "message": "Successfully added 'document.pdf' to the database",
  "document_id": "a1b2c3d4...",
  "filename": "document.pdf",
  "pages": 15,
  "chunks": 127,
  "chunk_size": 3,
  "overlap": 1
}
```

**Example Use Cases:**
- Adding research papers for reference
- Indexing documentation
- Building a searchable knowledge base

### 2. pdf_remove

Remove a PDF document from the database.

**Input:**
```json
{
  "document_id": "a1b2c3d4..."  // Get from pdf_list
}
```

**Output:**
```json
{
  "status": "success",
  "message": "Successfully removed 'document.pdf' from the database",
  "document_id": "a1b2c3d4...",
  "removed_chunks": 127
}
```

### 3. pdf_list

List all PDF documents in the database.

**Input:**
```json
{
  "response_format": "markdown"  // or "json"
}
```

**Output (Markdown):**
```markdown
# PDF Documents (2 total)

## research_paper.pdf
**Document ID:** a1b2c3d4...
**Chunks:** 127
**Added:** N/A

## documentation.pdf
**Document ID:** e5f6g7h8...
**Chunks:** 89
**Added:** N/A
```

**Output (JSON):**
```json
{
  "count": 2,
  "documents": [
    {
      "document_id": "a1b2c3d4...",
      "filename": "research_paper.pdf",
      "chunk_count": 127
    },
    {
      "document_id": "e5f6g7h8...",
      "filename": "documentation.pdf",
      "chunk_count": 89
    }
  ]
}
```

### 4. pdf_search_similarity

Search using semantic similarity (vector search).

**Input:**
```json
{
  "query": "machine learning techniques for text classification",
  "top_k": 5,                    // optional, default: 5
  "document_filter": null,       // optional, search specific doc
  "response_format": "markdown"  // optional, default: markdown
}
```

**Output (Markdown):**
```markdown
# Search Results for: 'machine learning techniques for text classification'

Found 5 relevant chunks:

## Result 1
**Document:** research_paper.pdf
**Page:** 7
**Similarity Score:** 0.8754

**Content:**
Machine learning approaches to text classification have evolved significantly...

---
```

**Use Cases:**
- Finding relevant information without exact keywords
- Discovering related concepts
- Question answering over documents

### 5. pdf_search_keywords

Search using keyword matching.

**Input:**
```json
{
  "keywords": "neural network backpropagation",
  "top_k": 5,                    // optional, default: 5
  "document_filter": null,       // optional
  "response_format": "markdown"  // optional, default: markdown
}
```

**Output:**
Similar to `pdf_search_similarity`, but ranked by keyword occurrence count.

**Use Cases:**
- Finding specific technical terms
- Locating exact phrases or terminology
- Verifying presence of keywords in documents

## Usage with Claude Desktop

### 1. Add to Claude Desktop Configuration

Edit your Claude Desktop config file:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Add the server:

```json
{
  "mcpServers": {
    "pdf-rag": {
      "command": "python",
      "args": [
        "/absolute/path/to/pdf_rag_mcp.py"
      ]
    }
  }
}
```

**Custom Database Location:**

To use a custom database path with Claude Desktop:

```json
{
  "mcpServers": {
    "pdf-rag": {
      "command": "python",
      "args": [
        "/absolute/path/to/pdf_rag_mcp.py",
        "--db-path",
        "/custom/path/to/database"
      ]
    }
  }
}
```

### 2. Restart Claude Desktop

After adding the configuration, restart Claude Desktop to load the MCP server.

### 3. Test the Connection

In Claude Desktop, try:

```
Can you list the PDFs in the RAG database?
```

Claude will use the `pdf_list` tool to show available documents.

## Example Workflows

### Building a Research Database

```
1. Add documents:
   "Add these PDFs to the database: /research/paper1.pdf, /research/paper2.pdf"

2. Search for concepts:
   "Search for information about 'gradient descent optimization' in the database"

3. Find specific terms:
   "Search for the keyword 'convolutional neural network' and show me the pages"
```

### Document Q&A

```
1. Add documentation:
   "Add this user manual: /docs/product_manual.pdf"

2. Ask questions:
   "How do I configure the network settings according to the manual?"

3. Find references:
   "Which page discusses troubleshooting connection errors?"
```

### Knowledge Base Management

```
1. List documents:
   "Show me all documents in the RAG database"

2. Remove outdated docs:
   "Remove the document with ID a1b2c3d4..."

3. Search across all:
   "Search all documents for information about API authentication"
```

## Advanced Configuration

### Custom Chunk Sizes

For different document types:

**Technical Documents** (code, APIs):
- Smaller chunks (2-3 sentences)
- Minimal overlap (0-1 sentences)
- Preserves code structure

**Narrative Documents** (articles, books):
- Larger chunks (5-7 sentences)
- More overlap (2-3 sentences)
- Maintains context flow

**Scientific Papers**:
- Medium chunks (3-5 sentences)
- Moderate overlap (1-2 sentences)
- Balances detail and context

### Document Filtering

Search within specific documents:

```json
{
  "query": "data preprocessing",
  "document_filter": "a1b2c3d4..."  // Only search this doc
}
```

### Output Format Selection

Choose format based on use case:

**Markdown**: Best for human reading, Claude's analysis
**JSON**: Best for programmatic processing, data extraction

## Troubleshooting

### "File not found" Error

Ensure you're using absolute paths:
```python
"/home/user/documents/paper.pdf"  ✅
"~/documents/paper.pdf"           ❌ (needs expansion)
"./paper.pdf"                     ❌ (relative path)
```

### Empty PDF Results / Scanned PDFs

The server automatically detects and processes scanned PDFs using OCR. If you get an error about no text being extracted:

1. **Install Tesseract** (if not already installed):
   - macOS: `brew install tesseract`
   - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

2. **Retry adding the PDF** - the server will automatically use OCR for pages with minimal text

The error message will indicate if OCR is needed: "ensure tesseract is installed for scanned PDFs"

### Out of Memory

If processing large PDFs causes memory issues:
1. Reduce `chunk_size` to create more, smaller chunks
2. Process documents one at a time
3. Increase system swap space

### ChromaDB Errors

If ChromaDB complains about existing collections:
```bash
# Remove the database directory
rm -rf ./chroma_db
# Restart the server
```

## Performance Considerations

### Embedding Generation

The first time you add a document, the model will be downloaded (~400MB). Subsequent operations are faster.

**Typical Times:**
- 10-page PDF: ~5-10 seconds
- 100-page PDF: ~30-60 seconds
- 1000-page PDF: ~5-10 minutes

### Search Performance

- **Similarity Search**: Fast (< 1 second for most queries)
- **Keyword Search**: Slower for large collections (scales with document count)

### Storage

- **Embeddings**: ~1.5KB per chunk (768-dimensional vectors)
- **Text Storage**: Depends on chunk size
- **Example**: 1000 chunks ≈ 1.5MB in ChromaDB

## Best Practices

### 1. Organize Documents

Use descriptive filenames:
```
research_ml_2024.pdf          ✅
document (1).pdf              ❌
```

### 2. Test Chunk Sizes

Different documents benefit from different chunking:
```python
# Try multiple chunk sizes for the same document
pdf_add(path="doc.pdf", chunk_size=3, overlap=1)  # Test 1
pdf_remove(document_id="...")                      # Remove
pdf_add(path="doc.pdf", chunk_size=5, overlap=2)  # Test 2
```

### 3. Use Document Filters

When searching specific documents:
```python
# More focused, faster results
pdf_search_similarity(
    query="...",
    document_filter="specific_doc_id"
)
```

### 4. Combine Search Types

Use both search methods for comprehensive results:
1. Semantic search for concepts
2. Keyword search for exact terms

## Security Notes

- **File Access**: Server can read any PDF the Python process can access
- **Storage**: Embeddings and text stored unencrypted in ChromaDB
- **No Authentication**: MCP servers trust the client (Claude Desktop)

For production use:
- Restrict file system permissions
- Use dedicated database directories
- Consider encryption for sensitive documents

## Contributing

To extend this server:

1. **Add New Tools**: Follow the `@mcp.tool()` decorator pattern
2. **Custom Chunking**: Implement in `semantic_chunking()` function
3. **Additional Embeddings**: Swap models in initialization
4. **Metadata**: Extend `metadatas` dict in `pdf_add()`

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **Anthropic**: MCP Protocol and SDK
- **ChromaDB**: Vector database
- **Sentence Transformers**: Embedding models
- **PyMuPDF**: PDF text extraction and OCR support

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review MCP documentation: https://modelcontextprotocol.io
3. Check ChromaDB docs: https://docs.trychroma.com

---

**Built with ❤️ using the Model Context Protocol**
