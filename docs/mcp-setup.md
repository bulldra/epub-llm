# MCP Server Setup Guide

## Overview
The EPUB LLM application supports MCP (Model Context Protocol) for decoupled service architecture. The MCP server exposes EPUB processing and RAG functionality as tools.

## Setup Instructions

### 1. MCP Server Configuration

The MCP server is implemented in `src/server.py` using FastMCP. Available tools:

- `list_epub_books()` - Get list of available EPUB files
- `get_epub_metadata(book_id)` - Get metadata for specific book
- `search_epub_content(book_id, query, top_k)` - Search within single book
- `get_context_for_books(book_ids, query, top_k)` - Multi-book context search
- `get_chat_histories()` - List chat session IDs
- `get_chat_history(session_id)` - Get specific session history

### 2. Running the MCP Server

```bash
# Start the MCP server on default port
python src/server.py

# Or use the FastMCP app directly
python -m fastmcp src.server:mcp_app
```

### 3. Client Configuration

To connect to the MCP server from external applications:

```python
from fastmcp.client import FastMCPClient

# Connect to MCP server
client = FastMCPClient("http://localhost:8000/mcp")

# Use tools
books = await client.call_tool("list_epub_books")
context = await client.call_tool("get_context_for_books", {
    "book_ids": ["book1.epub", "book2.epub"],
    "query": "search query",
    "top_k": 10
})
```

### 4. Integration with Main Application

The main FastAPI application (`src/app.py`) and MCP server can run simultaneously:

- Main app: EPUB management, chat interface, file upload
- MCP server: RAG tools, search functionality, session management

### 5. Environment Setup

Required dependencies (already in requirements.txt):
- fastmcp
- fastapi
- mlx-lm
- mlx-embeddings
- faiss-cpu

### 6. Development Mode

For development, you can mount both servers:

```python
# In app.py or separate runner
from fastapi import FastAPI
from src.server import mcp_app

app = FastAPI()
app.mount("/mcp", mcp_app)
```

## Architecture Benefits

- **Decoupled Services**: LLM and RAG can be used independently
- **Tool-based Interface**: Standard MCP protocol for interoperability
- **Scalable**: MCP server can be deployed separately
- **Reusable**: EPUB processing tools available to other applications