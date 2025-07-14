#!/usr/bin/env python3
"""
安定版 MCP Server 起動スクリプト

このスクリプトは Claude Desktop などでの接続問題を回避するために
最適化された起動方法を提供します。
"""

import logging
import os
import signal
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# ログ設定
log_dir = project_root / "log"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),  # stderrにログ出力
        logging.FileHandler(log_dir / "mcp_server.log", encoding="utf-8"),
    ],
)

logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    """シグナルハンドラ"""
    logger.info(f"シグナル {signum} を受信。MCP Server を終了中...")
    sys.exit(0)


def ensure_dependencies():
    """依存関係の確認"""
    try:
        import fastmcp
        import uvicorn

        logger.info("依存関係確認: OK")
        return True
    except ImportError as e:
        logger.error(f"依存関係エラー: {e}")
        return False


def start_mcp_server():
    """MCP Server を起動"""
    try:
        # 依存関係確認
        if not ensure_dependencies():
            return False

        logger.info("EPUB-LLM MCP Server を起動中...")

        # MCPアプリケーションをインポート
        from src.mcp_server import mcp_app

        # サーバー情報をログ出力
        logger.info("=" * 50)
        logger.info("EPUB-LLM MCP Server")
        logger.info(f"プロジェクトルート: {project_root}")
        logger.info(f"PythonPath: {sys.path[0]}")
        logger.info("=" * 50)

        # FastMCP の実行
        mcp_app.run()

        return True

    except KeyboardInterrupt:
        logger.info("ユーザーによる中断")
        return False
    except Exception as e:
        logger.error(f"MCP Server 起動エラー: {e}", exc_info=True)
        return False


def main():
    """メイン関数"""
    # シグナルハンドラ登録
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 開発モード設定
    os.environ.setdefault("DEV_MODE", "true")

    logger.info("MCP Server 起動スクリプト開始")

    success = start_mcp_server()

    if not success:
        logger.error("MCP Server の起動に失敗しました")
        sys.exit(1)

    logger.info("MCP Server が正常に終了しました")


if __name__ == "__main__":
    main()
