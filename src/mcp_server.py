"""FastMCP server integration for EPUB application.

This module provides FastMCP tool integrations for the EPUB application,
exposing key functionality as tools for external consumption.
"""

import json
import logging
import os
import sys
import threading
from typing import Any, cast

from fastmcp import FastMCP

from src.config_manager import AppConfig
from src.simple_epub_service import SimpleEPUBService

# ログ設定
logger = logging.getLogger(__name__)

# FastMCP アプリケーション
mcp_app: FastMCP = FastMCP("本棚")

# Configuration and services initialization
config: AppConfig | None = None
epub_service: SimpleEPUBService | None = None
_init_lock = threading.Lock()

# macOSのfork safety問題を回避
if sys.platform == "darwin":
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


def get_epub_service() -> SimpleEPUBService:
    """EPUBサービスを遅延初期化して返す"""
    with _init_lock:
        if globals()["epub_service"] is None:
            if globals()["config"] is None:
                globals()["config"] = AppConfig()

            epub_dir = globals()["config"].get("directories.epub_dir", "epub")
            if not os.path.isabs(epub_dir):
                epub_dir = os.path.join(os.path.dirname(__file__), "..", epub_dir)
            embedding_model = globals()["config"].get("mlx.embedding_model")
            if not isinstance(embedding_model, str) or not embedding_model:
                raise RuntimeError(
                    "app_config.yaml の mlx.embedding_model が未設定です。必ずモデルIDを設定してください。"
                )
            globals()["epub_service"] = SimpleEPUBService(
                epub_dir, embedding_model=embedding_model
            )

        return cast(SimpleEPUBService, globals()["epub_service"])


def validate_json_response(data: Any) -> Any:
    """JSONレスポンスを検証し、問題のある文字を修正する"""
    try:
        json.dumps(data, ensure_ascii=False)
        return data
    except (TypeError, ValueError) as e:
        logger.warning("JSON serialization failed: %s", e)

        if isinstance(data, str):
            cleaned = "".join(
                char for char in data if char.isprintable() or char.isspace()
            )
            return cleaned.replace("�", "?")

        if isinstance(data, list):
            return [validate_json_response(item) for item in data]

        if isinstance(data, dict):
            return {key: validate_json_response(value) for key, value in data.items()}

        return str(data)


@mcp_app.tool()
def list_epub_books() -> list[dict[str, Any]]:
    """EPUB書籍一覧をタイトルと目次情報付きで取得

    【超重要】このツールは最初に1回だけ実行してください！
    - 書籍リストは変わらないため、繰り返し実行する必要はありません
    - 一度取得した書籍リストと目次を使い回して、複数の検索を実行してください
    - 検索のたびにこのツールを呼ぶのは非効率です

    【効率的な検索の流れ】
    1. 最初に1回だけこのツールで全書籍の一覧と目次（toc）を取得
    2. 取得した目次を分析し、検索したい内容がありそうな書籍を複数選択
    3. 選択した書籍それぞれに対して search_epub_content を実行
    4. 複数の書籍から横断的に情報を収集

    【複数書籍の検索例】
    - 「機械学習」について調べる場合:
      → 目次に「機械学習」「AI」「データサイエンス」がある書籍を全て選択
      → 各書籍で search_epub_content(query="機械学習", book_id="選択した書籍ID") を実行
    - 「Python」の実装例を探す場合:
      → 目次に「Python」「プログラミング」「実装」がある書籍を複数選択
      → 各書籍で検索を実行し、幅広い実装例を収集

    【注意】
    - このツールの実行は1回で十分です
    - 取得した書籍リストは会話全体で再利用してください

    Returns:
        書籍情報のリスト。各書籍は以下の情報を含む:
        - id: ファイル名（検索時のbook_idパラメータに使用）
        - title: 書籍タイトル
        - author: 著者名
        - year: 出版年
        - toc: 章レベルの目次情報（複数書籍選択の判断材料として重要）
    """
    try:
        result = get_epub_service().get_bookshelf()
        logger.debug("list_epub_books: %d冊の書籍を返却", len(result))
        return cast(list[dict[str, Any]], validate_json_response(result))
    except (OSError, ValueError, RuntimeError) as e:
        logger.error("list_epub_books エラー: %s", e)
        return []


@mcp_app.tool()
def get_epub_metadata(book_id: str) -> dict[str, Any]:
    """指定されたEPUBの詳細メタデータと目次を取得

    【推奨】特定の書籍を検索する前に、このツールで目次（toc）を確認して
    どの章に目的の内容があるか把握してから検索することを推奨します。

    Args:
        book_id: EPUB書籍のファイル名（list_epub_booksで取得したid）

    Returns:
        書籍のメタデータ（タイトル、著者、出版年、目次等）
    """
    try:
        result = get_epub_service().get_book_metadata(book_id)
        logger.debug("get_epub_metadata: %s のメタデータを取得", book_id)
        return cast(dict[str, Any], validate_json_response(result))
    except (OSError, ValueError, RuntimeError, KeyError) as e:
        logger.error("get_epub_metadata エラー (%s): %s", book_id, e)
        return cast(dict[str, Any], validate_json_response({"error": str(e)}))


@mcp_app.tool()
def search_epub_content(
    query: str,
    book_id: str | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """EPUB書籍からFAISS検索でコンテンツを検索

    【推奨検索フロー】
    1. 最初に1回だけ list_epub_books で書籍一覧と目次（toc）を取得（重複実行不要！）
    2. 取得済みの目次から関連する書籍を複数選択（1冊だけでなく関連書籍全て）
    3. 選択した各書籍に対してこのツールを実行（book_id を指定）
    4. 複数書籍の検索結果を統合して包括的な情報を取得

    【複数書籍検索の実行例】
    ```
    # ステップ1: 書籍一覧を最初に1回だけ取得（これ以降は再実行不要）
    books = list_epub_books()  # ← この結果を会話全体で使い回す

    # ステップ2: 取得済みの目次から「Python」関連の書籍を選択
    python_books = ["python_basics.epub", "advanced_python.epub", "python_ml.epub"]

    # ステップ3: 各書籍で検索を実行（list_epub_booksの再実行は不要）
    for book in python_books:
        results = search_epub_content("クラス継承", book_id=book, top_k=3)

    # ステップ4: 別のトピックを検索する場合も、既存のbooksリストを使用
    # list_epub_booksを再実行せず、最初に取得したリストから選択
    ml_books = ["ml_intro.epub", "deep_learning.epub"]  # 既存リストから選択
    for book in ml_books:
        results = search_epub_content("ニューラルネットワーク", book_id=book, top_k=3)
    ```

    【検索テクニック】
    - 単語検索: "Python" → Pythonを含む内容
    - AND検索: "Python class" → 両方の単語を含む内容（スペース区切り）
    - 書籍指定: book_id="specific_book.epub" → 特定書籍のみ検索（推奨）
    - 全書籍検索: book_id=None → 全書籍から検索（非推奨、精度が低い）

    【重要】
    - book_idを指定した検索の方が精度が高く高速です
    - 複数の関連書籍を個別に検索することで網羅的な情報収集が可能
    - 全書籍検索（book_id=None）は最終手段として使用

    Args:
        query: 検索クエリ（スペース区切りでAND検索）
        book_id: EPUB書籍のファイル名（list_epub_booksで取得したid）
        top_k: 返却する結果の最大件数（デフォルト: 5）

    Returns:
        検索結果のリスト。各結果は以下の情報を含む:
        - content: マッチしたテキスト内容
        - score: 関連度スコア（FAISS類似度）
        - chunk_id: テキストチャンクのID
        - book_id: 書籍のファイル名
        - book_title: 書籍タイトル（全書籍検索時）
    """
    try:
        service = get_epub_service()

        if book_id:
            # 特定の書籍から検索
            result = service.search_book_content(book_id, query, top_k)
            logger.debug(
                "FAISS search_epub_content: %s で '%s...' を検索、%d件の結果",
                book_id,
                query[:50],
                len(result),
            )
        else:
            # 全書籍から検索
            result = service.search_all_books(query, top_k)
            logger.debug(
                "FAISS search_epub_content: 全書籍で '%s...' を検索、%d件の結果",
                query[:50],
                len(result),
            )

        return cast(list[dict[str, Any]], validate_json_response(result))
    except (OSError, ValueError, RuntimeError) as e:
        search_target = book_id if book_id else "全書籍"
        logger.error(
            "FAISS search_epub_content エラー (%s, %s): %s", search_target, query, e
        )
        return cast(list[dict[str, Any]], validate_json_response([{"error": str(e)}]))


@mcp_app.tool()
def build_faiss_index() -> dict[str, Any]:
    """全てのEPUB書籍からFAISSインデックスを構築

    Returns:
        インデックス構築の結果情報
        - status: 構築状況（success/error）
        - message: 詳細メッセージ
        - stats: インデックス統計情報（成功時）
    """
    try:
        service = get_epub_service()
        # MLXベースの埋め込みサービスでインデックスを準備（存在しなければ構築）
        # SimpleEPUBService が全書籍を走査して必要に応じて構築します
        service.ensure_index_loaded()

        # 統計情報を取得
        stats = service.embedding_service.get_stats()

        logger.debug("FAISSインデックス構築完了: %s", stats)

        return cast(
            dict[str, Any],
            validate_json_response(
                {
                    "status": "success",
                    "message": "FAISSインデックスの構築が完了しました",
                    "stats": stats,
                }
            ),
        )

    except (OSError, ValueError, RuntimeError) as e:
        logger.error("FAISSインデックス構築エラー: %s", e)
        return cast(
            dict[str, Any],
            validate_json_response(
                {
                    "status": "error",
                    "message": f"インデックス構築に失敗しました: {str(e)}",
                }
            ),
        )


@mcp_app.tool()
def get_faiss_index_stats() -> dict[str, Any]:
    """FAISSインデックスの統計情報を取得

    Returns:
        インデックスの統計情報
        - total_chunks: 総チャンク数
        - total_books: 書籍数
        - embedding_dimension: 埋め込みベクトルの次元数
        - model_name: 使用モデル名
        - books: 書籍別チャンク数
    """
    try:
        service = get_epub_service()
        # 必要ならロードしてから統計を返す
        service.ensure_index_loaded()
        stats = service.embedding_service.get_stats()

        logger.debug("FAISSインデックス統計情報を取得: %s", stats)

        return cast(dict[str, Any], validate_json_response(stats))

    except (OSError, ValueError, RuntimeError) as e:
        logger.error("FAISSインデックス統計情報取得エラー: %s", e)
        return cast(
            dict[str, Any],
            validate_json_response({"error": f"統計情報取得に失敗しました: {str(e)}"}),
        )


if __name__ == "__main__":
    # MCP server用のエントリーポイント
    logger.info("MCP Server を直接起動中...")
    try:
        mcp_app.run()
    except (OSError, ValueError, RuntimeError) as e:
        logger.error("MCP Server 直接起動エラー: %s", e)
        sys.exit(1)
