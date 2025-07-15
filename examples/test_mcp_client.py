#!/usr/bin/env python3
"""
MCP Server テストクライアント

このスクリプトは MCP Server の動作を確認するためのテストクライアントです。
"""

import asyncio
import json
import sys
from typing import Any

import requests


class MCPTester:
    """MCP Server テスター"""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.mcp_url = f"{base_url}/mcp"

    def test_health_check(self) -> bool:
        """ヘルスチェック"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            print(f"✅ MCP Server 起動確認: {response.status_code}")
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"❌ MCP Server 接続エラー: {e}")
            return False

    def call_tool(self, tool_name: str, params: dict[str, Any] = None) -> Any:
        """MCPツール呼び出し"""
        if params is None:
            params = {}

        url = f"{self.mcp_url}/tools/{tool_name}"

        try:
            response = requests.post(
                url,
                json=params,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ {tool_name}: 成功")
                return result
            else:
                print(f"❌ {tool_name}: エラー {response.status_code}")
                print(f"   レスポンス: {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"❌ {tool_name}: 接続エラー {e}")
            return None

    def test_list_books(self) -> list[dict[str, Any]]:
        """書籍一覧テスト"""
        print("\n📚 書籍一覧テスト")
        result = self.call_tool("list_epub_books")

        if result:
            print(f"   書籍数: {len(result)}")
            for i, book in enumerate(result[:3]):  # 最初の3冊のみ表示
                title = book.get("title", "不明")
                book_id = book.get("id", "不明")
                print(f"   {i+1}. {title} (ID: {book_id})")

        return result if result else []

    def test_metadata(self, books: list[dict[str, Any]]) -> None:
        """メタデータ取得テスト"""
        if not books:
            print("\n📖 メタデータテスト: スキップ（書籍なし）")
            return

        print("\n📖 メタデータテスト")
        book_id = books[0].get("id")

        if book_id:
            result = self.call_tool("get_epub_metadata", {"book_id": book_id})

            if result:
                title = result.get("title", "不明")
                author = result.get("author", "不明")
                print(f"   タイトル: {title}")
                print(f"   著者: {author}")

    def test_search(self, books: list[dict[str, Any]]) -> None:
        """検索テスト"""
        if not books:
            print("\n🔍 検索テスト: スキップ（書籍なし）")
            return

        print("\n🔍 検索テスト")
        book_id = books[0].get("id")

        if book_id:
            # 単一書籍検索
            result = self.call_tool(
                "search_epub_content",
                {"book_id": book_id, "query": "Python", "top_k": 3},
            )

            if result:
                print(f"   検索結果数: {len(result)}")
                for i, item in enumerate(result[:2]):  # 最初の2件のみ表示
                    text = item.get("text", "")[:100] + "..."
                    score = item.get("score", 0)
                    print(f"   {i+1}. スコア: {score:.3f}")
                    print(f"      テキスト: {text}")

    def test_context_search(self, books: list[dict[str, Any]]) -> None:
        """コンテキスト検索テスト"""
        if not books:
            print("\n🎯 コンテキスト検索テスト: スキップ（書籍なし）")
            return

        print("\n🎯 コンテキスト検索テスト")
        book_ids = [book.get("id") for book in books[:2]]  # 最初の2冊
        book_ids = [bid for bid in book_ids if bid]  # None を除外

        if book_ids:
            result = self.call_tool(
                "get_context_for_books",
                {"book_ids": book_ids, "query": "機械学習", "top_k": 5},
            )

            if result:
                context_length = len(result)
                preview = result[:200] + "..." if len(result) > 200 else result
                print(f"   コンテキスト長: {context_length}文字")
                print(f"   プレビュー: {preview}")

    def test_smart_search(self, books: list[dict[str, Any]]) -> None:
        """スマート検索テスト"""
        if not books:
            print("\n🧠 スマート検索テスト: スキップ（書籍なし）")
            return

        print("\n🧠 スマート検索テスト")
        book_ids = [book.get("id") for book in books[:2]]
        book_ids = [bid for bid in book_ids if bid]

        if book_ids:
            result = self.call_tool(
                "smart_search_books",
                {"book_ids": book_ids, "query": "データ分析", "top_k": 3},
            )

            if result:
                result_length = len(result)
                preview = result[:200] + "..." if len(result) > 200 else result
                print(f"   結果長: {result_length}文字")
                print(f"   プレビュー: {preview}")

    def test_chat_history(self) -> None:
        """チャット履歴テスト"""
        print("\n💬 チャット履歴テスト")

        # 履歴一覧取得
        histories = self.call_tool("get_chat_histories")

        if histories:
            print(f"   履歴数: {len(histories)}")

            # 最初の履歴の詳細取得
            if histories:
                session_id = histories[0]
                history = self.call_tool("get_chat_history", {"session_id": session_id})

                if history:
                    print(f"   セッション {session_id}: {len(history)}メッセージ")
        else:
            print("   履歴なし")

    def run_all_tests(self) -> None:
        """全テスト実行"""
        print("🧪 MCP Server 機能テスト開始")
        print("=" * 50)

        # ヘルスチェック
        if not self.test_health_check():
            print("❌ MCP Server に接続できません")
            return

        # 書籍一覧取得
        books = self.test_list_books()

        # 各種テスト実行
        self.test_metadata(books)
        self.test_search(books)
        self.test_context_search(books)
        self.test_smart_search(books)
        self.test_chat_history()

        print("\n" + "=" * 50)
        print("🎉 テスト完了")


def main():
    """メイン関数"""
    # コマンドライン引数でURL指定可能
    base_url = "http://localhost:8001"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]

    print(f"🌐 MCP Server URL: {base_url}")

    tester = MCPTester(base_url)

    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⏹️ テスト中断")
    except Exception as e:
        print(f"\n\n❌ テスト実行エラー: {e}")


if __name__ == "__main__":
    main()
