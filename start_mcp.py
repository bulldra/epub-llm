#!/usr/bin/env python3
"""MCP Server startup script for EPUB LLM application."""

import uvicorn

if __name__ == "__main__":
    # Start the MCP server
    uvicorn.run(
        "src.server:mcp_app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )