#!/usr/bin/env python3
"""
EPUB-LLM Dev Watcher

File changes -> restart uvicorn (reload disabled).
Avoids Python 3.12 resource_tracker issue on macOS by not using uvicorn's reloader.

Usage:
  ./scripts/dev_watch.py --host 127.0.0.1 --port 8000

Environment variables are passed through (e.g., LMSTUDIO_MODEL, etc.).
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from collections.abc import Sequence

from watchfiles import watch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EPUB-LLM dev watcher")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--dirs",
        nargs="*",
        default=["src", "templates", "static", "config"],
        help="directories to watch",
    )
    return p.parse_args()


def start_server(host: str, port: int) -> subprocess.Popen[bytes]:
    cmd: Sequence[str] = (
        sys.executable,
        "-m",
        "uvicorn",
        "src.app:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "info",
    )
    print("Starting:", " ".join(cmd))
    return subprocess.Popen(cmd)


def main() -> None:
    args = parse_args()

    # Ensure we run from project root
    os.chdir(os.path.dirname(os.path.dirname(__file__)))

    proc = start_server(args.host, args.port)

    print("Watching:", ", ".join(args.dirs))
    try:
        for changes in watch(*args.dirs, debounce=200, step=500, yield_on_timeout=True):
            if changes:
                print("Changes detected -> restarting server...")
                # Terminate current server
                if proc.poll() is None:
                    try:
                        proc.send_signal(signal.SIGINT)
                        proc.wait(timeout=5)
                    except (subprocess.TimeoutExpired, ProcessLookupError):
                        proc.kill()
                # Start new one
                proc = start_server(args.host, args.port)
    except KeyboardInterrupt:
        pass
    finally:
        if proc and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                proc.kill()


if __name__ == "__main__":
    main()
