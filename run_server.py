#!/usr/bin/env python3
"""
EPUB-LLM Server Runner
Python 3.12 compatible server startup script
"""

import argparse
import multiprocessing
import os
import sys
from pathlib import Path

import uvicorn

# Fix for Python 3.12 multiprocessing issue
if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="EPUB-LLM Server - Python 3.12 compatible launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_server.py                    # Use config defaults
  python run_server.py --host 127.0.0.1   # Custom host
  python run_server.py --port 9000        # Custom port
  python run_server.py --prod             # Production mode
  python run_server.py --dev              # Development mode

Environment variables:
  DEV_MODE=true/false    Override dev mode setting
  HOST=0.0.0.0          Override host setting
  PORT=8000             Override port setting
        """,
    )

    parser.add_argument(
        "-H", "--host", help="Server host (default: from config or 0.0.0.0)"
    )
    parser.add_argument(
        "-p", "--port", type=int, help="Server port (default: from config or 8000)"
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dev", action="store_true", help="Enable development mode"
    )
    mode_group.add_argument(
        "--prod", action="store_true", help="Enable production mode"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the server."""
    # Parse command line arguments
    args = parse_args()

    # Import config manager to read configuration
    from src.config_manager import AppConfig  # pylint: disable=import-outside-toplevel

    # Load configuration
    config = AppConfig()

    # Determine dev mode: CLI args > env vars > config file
    if args.dev:
        dev_mode = True
    elif args.prod:
        dev_mode = False
    else:
        dev_mode = (
            os.getenv("DEV_MODE", str(config.get("llm.dev_mode", True))).lower()
            == "true"
        )

    # Get host and port: CLI args > env vars > config file
    host = args.host or os.getenv("HOST") or config.get("server.host", "0.0.0.0")
    port = args.port or int(os.getenv("PORT", config.get("server.port", 8000)))

    # Set environment variables from final values
    os.environ["DEV_MODE"] = str(dev_mode).lower()

    # Import the FastAPI app after setting environment
    from src.app import app  # pylint: disable=import-outside-toplevel

    # Log configuration
    print("🚀 Starting EPUB-LLM Server")
    print(f"📍 Host: {host}")
    print(f"🔢 Port: {port}")
    print(f"🔧 Dev Mode: {dev_mode}")
    print(f"📁 Config: {config.config_path}")
    print(f"🌐 Access URL: http://localhost:{port}")
    print("⏹️  Press CTRL+C to stop")

    # Run the server without reload to avoid multiprocessing issues
    uvicorn.run(
        app,
        host=host,
        port=port,
        # reload=False to avoid Python 3.12 multiprocessing issues
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
