# 🚀 Getting Started with Your PDF RAG MCP Server

## Welcome!

You now have a complete, production-ready PDF RAG MCP server! This guide will get you up and running in minutes.

## 📁 What's Included

Your project contains 8 files:

1. **pdf_rag_mcp.py** - The main MCP server (850+ lines)
2. **requirements.txt** - Python dependencies
3. **README.md** - Full documentation
4. **QUICKSTART.md** - Quick setup guide
5. **PROJECT_OVERVIEW.md** - Architecture and design details
6. **test_pdf_rag.py** - Test script
7. **claude_desktop_config.json** - Configuration example
8. **LICENSE** - MIT license

## ⚡ Quick Setup (5 Minutes)

### Step 1: Install Dependencies
```bash
cd /path/to/project
pip install -r requirements.txt
```

### Step 2: Test the Server
```bash
python pdf_rag_mcp.py --help
```

You should see the MCP server help output.

### Step 3: Configure Claude Desktop

**Find your config file:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

**Add this configuration:**
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

Replace `/absolute/path/to/` with your actual path!

### Step 4: Restart Claude Desktop

Quit and restart Claude Desktop completely.

### Step 5: Test It!

In Claude Desktop, ask:
```
"Can you list the PDFs in my RAG database?"
```

If Claude responds using the `pdf_list` tool, you're all set! 🎉

## 🎯 First Use

### Add Your First PDF

In Claude Desktop:
```
"Add this PDF to the RAG database: /path/to/your/document.pdf"
```

Claude will:
- Extract text from the PDF
- Create semantic chunks
- Generate embeddings
- Store everything in ChromaDB

### Search Your PDF

**Semantic Search (recommended):**
```
"Search the database for information about [topic]"
```

**Keyword Search:**
```
"Find all mentions of '[keyword]' in my PDFs"
```

### View Your Documents
```
"Show me all PDFs in the database"
```

### Remove a Document
```
"Remove document [document_id] from the database"
```

## 📚 Available Tools

Your MCP server provides 5 powerful tools:

1. **pdf_add** - Add PDFs to the database
   - Supports custom chunk sizes
   - Progress reporting
   - Automatic deduplication

2. **pdf_remove** - Remove PDFs from the database
   - Safe deletion with confirmation
   - Removes all associated chunks

3. **pdf_list** - List all PDFs
   - Shows document IDs
   - Chunk counts
   - Multiple formats (Markdown/JSON)

4. **pdf_search_similarity** - Semantic search
   - Uses AI embeddings
   - Finds related concepts
   - Configurable result count

5. **pdf_search_keywords** - Keyword search
   - Traditional text matching
   - Case-insensitive
   - Ranked by frequency

## 🔍 Example Workflows

### Research Assistant
```
1. "Add these research papers: /papers/paper1.pdf, /papers/paper2.pdf"
2. "What are the main findings about [topic] in my papers?"
3. "Which paper discusses [specific concept]?"
```

### Documentation Helper
```
1. "Add the product manual: /docs/manual.pdf"
2. "How do I configure the network settings?"
3. "Does the manual mention troubleshooting?"
```

### Knowledge Base
```
1. "Add all PDFs from my knowledge folder"
2. "Search for information about [topic]"
3. "Compare what different documents say about [concept]"
```

## 🎨 Customization

### Adjust Chunk Size

Different documents work better with different chunk sizes:

**Technical Documentation** (code, APIs):
```
"Add the PDF with chunk_size=2 and overlap=0"
```

**Research Papers**:
```
"Add the PDF with chunk_size=4 and overlap=1"
```

**Books/Long-form**:
```
"Add the PDF with chunk_size=7 and overlap=2"
```

### Change Output Format

Get structured data:
```
"Search for [query] and give me JSON output"
```

### Filter by Document

Search specific documents:
```
"Search for [query] in document [document_id]"
```

## 🐛 Troubleshooting

### Issue: Server not responding
**Solution:** Check Claude Desktop logs and restart

### Issue: Can't find PDF file
**Solution:** Use absolute paths, not relative or ~/ paths

### Issue: No text extracted from PDF
**Solution:** PDF might be image-based, needs OCR preprocessing

### Issue: Out of memory
**Solution:** Use smaller chunk_size or process fewer documents

### Issue: Python command not found
**Solution:** Try `python3` instead of `python` in config

## 📖 Documentation Map

**For quick setup:**
→ Read QUICKSTART.md

**For full documentation:**
→ Read README.md (features, API, troubleshooting)

**For technical details:**
→ Read PROJECT_OVERVIEW.md (architecture, design decisions)

**For testing:**
→ Run test_pdf_rag.py

## 🎓 Key Concepts

### Semantic Chunking
Instead of splitting text at arbitrary points, the server:
- Splits by sentences (using NLTK)
- Groups sentences together (configurable)
- Adds overlap for context
- Preserves meaning

### Embeddings
Each chunk is converted to a 768-dimensional vector that represents its meaning. Similar concepts have similar vectors, enabling semantic search.

### Vector Search
The database finds chunks whose embeddings are closest to your query's embedding. This finds semantically similar content even without exact keywords.

## 💡 Pro Tips

1. **Start with default settings** - The defaults work well for most documents

2. **Test chunk sizes** - Different documents benefit from different chunking

3. **Use both search types** - Semantic for concepts, keywords for exact terms

4. **Add document filters** - Faster, more focused results

5. **Watch the chunk count** - More chunks = more detail but slower searches

6. **Use descriptive filenames** - Makes it easier to track documents

7. **Read the full README** - Lots of advanced features and options

## 🚀 Advanced Features

### Progress Reporting
The server reports progress during long operations:
- "Extracting text from PDF..."
- "Creating semantic chunks..."
- "Generating embeddings..."

### Error Recovery
Smart error handling with helpful messages:
- Suggests fixes for common problems
- Validates inputs before processing
- Graceful failures with clear explanations

### Character Limits
Automatic truncation if responses get too long:
- Truncates to fit context window
- Suggests ways to refine query
- Maintains result quality

## 🔐 Security Notes

- Server can access any PDF the Python process can read
- No built-in authentication (trusts Claude Desktop)
- Data stored unencrypted in ChromaDB
- Document IDs are SHA256 hashes

For sensitive documents, consider additional security measures.

## 🎉 You're Ready!

You now have:
✅ A working PDF RAG MCP server
✅ Complete documentation
✅ Example configurations
✅ Test scripts
✅ Best practices

**Next steps:**
1. Add your first PDF
2. Try searching
3. Experiment with settings
4. Build your knowledge base

## 📞 Need Help?

1. **Quick questions:** Check QUICKSTART.md
2. **Detailed info:** Read README.md
3. **Technical details:** See PROJECT_OVERVIEW.md
4. **Testing issues:** Run test_pdf_rag.py

## 🌟 Features at a Glance

- ✅ Semantic chunking (sentence-based)
- ✅ AI-powered embeddings (multi-qa-mpnet)
- ✅ Vector similarity search
- ✅ Keyword search
- ✅ Source tracking (doc + page)
- ✅ Multiple output formats
- ✅ Progress reporting
- ✅ Smart error handling
- ✅ Persistent storage
- ✅ Production-ready code

## 🎊 Happy Searching!

Your PDF RAG server is ready to help you unlock insights from your documents. Start adding PDFs and exploring!

---

**Questions?** Check the README.md for comprehensive documentation.

**Built following Anthropic's MCP best practices**
