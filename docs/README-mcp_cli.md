# MCP CLI

A command-line interface for interacting with Model Context Protocol (MCP) servers.

## Features

- **Tool Discovery**: List and describe available tools from MCP servers
- **Tool Invocation**: Call tools with parameters from the command line
- **Flexible Configuration**: JSON config files with CLI overrides
- **Parameter Support**: Simple values, JSON files, and file references
- **Output Formats**: Human-readable and JSON output modes

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Create a `server-config.json` file in your working directory:

```json
{
  "server_path": "/path/to/mcp_server.py",
  "server_args": ["--db-path", "/path/to/database"],
  "timeout": 30
}
```

You can also specify a different config file with `--server`:

```bash
python mcp_cli.py --server /path/to/config.json list-tools
```

## Usage

### List Available Tools

```bash
python mcp_cli.py list-tools
```

With JSON output:

```bash
python mcp_cli.py --json list-tools
```

### Describe a Tool

```bash
python mcp_cli.py describe <tool-name>
```

Example:

```bash
python mcp_cli.py describe pdf_add
```

### Call a Tool

```bash
python mcp_cli.py call <tool-name> [param=value ...]
```

Examples:

```bash
# Simple parameters
python mcp_cli.py call pdf_search_similarity query="machine learning" top_k=5

# File reference (loads file content)
python mcp_cli.py call pdf_add pdf_path=@/path/to/document.pdf

# JSON file parameter
python mcp_cli.py call some_tool config=@config.json

# JSON output
python mcp_cli.py --json call pdf_list response_format=json
```

### Parameter Syntax

- **Simple values**: `key=value` (strings, numbers, booleans)
  - Booleans: `active=true` or `active=false`
  - Numbers: `age=30` or `price=19.99`
  - Strings: `name=John`

- **File references**: `key=@filename`
  - JSON files are automatically parsed
  - Other files are read as text/binary content
  - Example: `document=@paper.pdf`

## Examples with PDF RAG Server

```bash
# List all tools
python mcp_cli.py list-tools

# Add a PDF to the database
python mcp_cli.py call pdf_add pdf_path=@/path/to/paper.pdf chunk_size=3 overlap=1

# Search for content
python mcp_cli.py call pdf_search_similarity query="neural networks" top_k=5 response_format=markdown

# List all PDFs in database
python mcp_cli.py call pdf_list response_format=markdown

# Get JSON output for scripting
python mcp_cli.py --json call pdf_list response_format=json
```

## Exit Codes

- `0`: Success
- `1`: Error (see stderr for details)

## Design Decisions

- **Stateless**: Each invocation connects fresh to the server
- **Default Config**: Uses `./server-config.json` by default
- **Explicit Parameters**: File references require `@` prefix
- **Type Inference**: Automatically parses booleans and numbers
- **Error Handling**: Clear error messages to stderr with proper exit codes
