#!/usr/bin/env python3
"""
安定版 MCP Server 起動スクリプト

このスクリプトは Claude Desktop などでの接続問題を回避するために
最適化された起動方法を提供します。
"""

import importlib
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

import fastmcp
import uvicorn

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


def signal_handler(signum: int, frame: Any) -> None:  # pylint: disable=unused-argument
    """シグナルハンドラ"""
    logger.info("シグナル %d を受信。MCP Server を終了中...", signum)
    sys.exit(0)


def ensure_dependencies() -> bool:
    """依存関係の確認"""
    try:
        # Use variables to avoid unused-variable warnings
        _ = fastmcp
        _ = uvicorn

        logger.info("依存関係確認: OK")
        return True
    except NameError as e:
        logger.error("依存関係エラー: %s", e)
        return False


def start_mcp_server() -> bool:
    """MCP Server を起動"""
    try:
        # 依存関係確認
        if not ensure_dependencies():
            return False

        logger.info("本棚 MCP Server を起動中...")

        # サーバー情報をログ出力
        logger.info("=" * 50)
        logger.info("本棚 MCP Server")
        logger.info("プロジェクトルート: %s", project_root)
        logger.info("PythonPath: %s", sys.path[0])
        logger.info("=" * 50)

        # MCPアプリケーションをインポート（動的インポート）
        mcp_module = importlib.import_module("src.mcp_server")
        mcp_app = mcp_module.mcp_app

        # FastMCP の実行
        mcp_app.run()

        return True

    except KeyboardInterrupt:
        logger.info("ユーザーによる中断")
        return False
    except (ImportError, AttributeError, OSError, RuntimeError) as e:
        logger.error("MCP Server 起動エラー: %s", e, exc_info=True)
        return False


def main() -> None:
    """メイン関数"""
    # シグナルハンドラ登録
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 開発モード設定
    os.environ.setdefault("DEV_MODE", "true")

    # マルチプロセッシング関連の設定
    # MCP環境でのtqdmエラーを回避
    os.environ["NO_PROXY"] = "*"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Hugging Faceのプログレスバーを無効化（tqdmエラー回避）
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    # 警告メッセージの抑制
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

    # macOSでのfork安全性を確保
    if sys.platform == "darwin":
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

    logger.info("MCP Server 起動スクリプト開始")

    success = start_mcp_server()

    if not success:
        logger.error("MCP Server の起動に失敗しました")
        sys.exit(1)

    logger.info("MCP Server が正常に終了しました")


if __name__ == "__main__":
    main()
