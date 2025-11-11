# Quick Start Guide

Get your PDF RAG MCP server running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Claude Desktop app (optional, for MCP integration)

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- MCP SDK
- ChromaDB (vector database)
- Sentence Transformers (embeddings)
- PyMuPDF (PDF processing with OCR support)
- NLTK (text processing)

**Note:** First run will download the embedding model (~400MB) and NLTK data (~1MB).

**Optional - For OCR Support:** Install Tesseract for scanned PDF support:
- **macOS:** `brew install tesseract`
- **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr`
- **Windows:** Download from https://github.com/UB-Mannheim/tesseract/wiki

### Step 2: Verify Installation

```bash
python pdf_rag_mcp.py --help
```

You should see:
```
Usage: pdf_rag_mcp.py [OPTIONS]
...
```

### Step 3: Test with Sample PDF (Optional)

```bash
python test_pdf_rag.py /path/to/your/test.pdf
```

This will test PDF extraction and chunking without starting the server.

## Usage with Claude Desktop

### Step 1: Locate Your Config File

**macOS:**
```bash
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**Linux:**
```bash
~/.config/Claude/claude_desktop_config.json
```

### Step 2: Edit Configuration

Open the file and add:

```json
{
  "mcpServers": {
    "pdf-rag": {
      "command": "python",
      "args": [
        "/ABSOLUTE/PATH/TO/pdf_rag_mcp.py"
      ]
    }
  }
}
```

**Important:** Replace `/ABSOLUTE/PATH/TO/` with the actual path!

**Example (macOS/Linux):**
```json
"/home/username/projects/pdf-rag/pdf_rag_mcp.py"
```

**Example (Windows):**
```json
"C:\\Users\\YourName\\Documents\\pdf-rag\\pdf_rag_mcp.py"
```

### Step 3: Restart Claude Desktop

Quit and restart Claude Desktop to load the MCP server.

### Step 4: Test the Integration

In Claude Desktop, ask:

```
"Can you list the PDFs in the RAG database?"
```

Claude should respond using the `pdf_list` tool (it will be empty initially).

## First PDF Workflow

### 1. Add a PDF

In Claude Desktop:
```
"Add this PDF to the RAG database: /path/to/document.pdf"
```

Claude will use the `pdf_add` tool and report:
- Document ID
- Number of pages
- Number of chunks created

### 2. Search the PDF

**Semantic Search:**
```
"Search the database for information about machine learning"
```

**Keyword Search:**
```
"Find all mentions of 'neural network' in the database"
```

### 3. View Your Documents

```
"Show me all PDFs in the database"
```

### 4. Remove a Document

```
"Remove the document with ID abc123..."
```

## Common Commands

### Check ChromaDB Status

```bash
# View database directory
ls -la chroma_db/
```

### Reset Database

```bash
# Remove all documents
rm -rf chroma_db/

# Restart the server (it will create a new database)
```

### View Server Logs

When running manually (for debugging):

```bash
python pdf_rag_mcp.py 2>&1 | tee server.log
```

## Troubleshooting

### Problem: "Command not found: python"

**Solution:** Try `python3` instead:
```json
{
  "mcpServers": {
    "pdf-rag": {
      "command": "python3",
      ...
    }
  }
}
```

### Problem: "Module not found"

**Solution:** Reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

### Problem: "File not found" when adding PDFs

**Solution:** Use absolute paths:
```
❌ "~/documents/file.pdf"
✅ "/home/username/documents/file.pdf"
```

### Problem: Server not responding in Claude Desktop

**Solution:** Check the MCP logs:

**macOS:**
```bash
tail -f ~/Library/Logs/Claude/mcp*.log
```

**Windows:**
```
# Check Event Viewer or Claude logs directory
```

### Problem: Out of memory with large PDFs

**Solution:** Use smaller chunk sizes:
```
"Add the PDF with chunk_size=2 and overlap=0"
```

### Problem: "Could not extract any text" from scanned PDFs

**Solution:** Install Tesseract for OCR support:
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Then retry adding the PDF
```

The server automatically detects scanned pages and uses OCR when needed.

## Performance Tips

### 1. Optimize Chunk Size

- **Technical docs**: `chunk_size=2-3`
- **Research papers**: `chunk_size=3-5`
- **Books**: `chunk_size=5-7`

### 2. Use Document Filters

When searching specific documents:
```
"Search for 'API authentication' in document abc123..."
```

### 3. Adjust top_k

For broader searches:
```
"Search for 'machine learning' and show me top 10 results"
```

For focused searches:
```
"Search for 'specific term' and show top 3 results"
```

## Next Steps

1. **Read the full README.md** for detailed documentation
2. **Experiment with different chunk sizes** for your document types
3. **Try both search methods** (similarity and keywords) to compare results
4. **Build your knowledge base** by adding relevant PDFs

## Example Use Cases

### Research Assistant
```
1. Add papers: "Add these research papers: /papers/*.pdf"
2. Ask questions: "What are the main approaches to text classification?"
3. Get citations: "Which paper discusses BERT architecture?"
```

### Documentation Helper
```
1. Add docs: "Add the user manual: /docs/manual.pdf"
2. Search: "How do I reset the password?"
3. Verify: "Does the manual mention SSL certificates?"
```

### Knowledge Base
```
1. Build library: "Add all PDFs from /knowledge_base/"
2. Search: "Find information about data privacy regulations"
3. Cross-reference: "Compare what different documents say about encryption"
```

## Support

- **MCP Documentation**: https://modelcontextprotocol.io
- **ChromaDB Docs**: https://docs.trychroma.com
- **Issues**: Check README.md troubleshooting section

---

**Happy searching! 🔍✨**
