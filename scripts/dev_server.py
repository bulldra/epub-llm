#!/usr/bin/env python3
"""
EPUB-LLM Development Server (reload enabled)
Python 3.12 workaround: force 'fork' start method before starting uvicorn reloader.

Usage:
  ./scripts/dev_server.py --host 127.0.0.1 --port 8000

Environment variables:
  LMSTUDIO_BASE_URL   default: http://localhost:1234/v1
  LMSTUDIO_MODEL      required for /chat
  LMSTUDIO_TEMPERATURE default: 0.2
"""

from __future__ import annotations

import argparse
import multiprocessing

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EPUB-LLM dev server (reload)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    # Workaround for Python 3.12 on macOS with reload
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    args = parse_args()

    uvicorn.run(
        "src.app:app",
        host=args.host,
        port=args.port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
