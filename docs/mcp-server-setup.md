# MCP Server 設定ガイド

## 概要

EPUB-LLM アプリケーションは MCP (Model Context Protocol) をサポートしており、Claude Desktop や他の MCP 対応アプリケーションから EPUB 処理と RAG 機能を利用できます。

## 🛠️ MCP Server の設定

### 1. 基本構成

MCP Server は `src/server.py` で実装されており、FastMCP を使用してツールを提供します。

```python
from fastmcp import FastMCP

# MCP アプリケーション
mcp_app = FastMCP("/mcp")

@mcp_app.tool()
def list_epub_books() -> list[dict[str, Any]]:
    """EPUBファイル一覧を取得"""
    # 実装...
```

### 2. 利用可能なツール

| ツール名 | 説明 | パラメータ | 戻り値 |
|---------|------|-----------|--------|
| `list_epub_books()` | EPUB書籍一覧取得 | なし | 書籍リスト |
| `get_epub_metadata(book_id)` | 書籍メタデータ取得 | book_id: str | メタデータ辞書 |
| `search_epub_content(book_id, query, top_k)` | 単一書籍検索 | book_id: str, query: str, top_k: int | 検索結果リスト |
| `get_context_for_books(book_ids, query, top_k)` | 複数書籍コンテキスト取得 | book_ids: list[str], query: str, top_k: int | コンテキスト文字列 |
| `smart_search_books(book_ids, query, top_k)` | スマート検索 | book_ids: list[str], query: str, top_k: int | 検索結果文字列 |
| `get_chat_histories()` | チャット履歴一覧 | なし | セッションIDリスト |
| `get_chat_history(session_id)` | チャット履歴取得 | session_id: str | メッセージリスト |

## 🚀 MCP Server の起動

### 方法1: 専用スクリプトを使用
```bash
# MCP Server を単独で起動（ポート8001）
python start_mcp.py
```

### 方法2: FastMCP 直接実行
```bash
# FastMCP モジュールで起動
python -m fastmcp src.server:mcp_app --host 0.0.0.0 --port 8001
```

### 方法3: メインアプリケーションと同時起動
```bash
# メインアプリ（ポート8000）とMCP Server（ポート8001）を同時起動
python run_server.py  # メインアプリ
python start_mcp.py   # 別ターミナルでMCP Server
```

## 🔧 Claude Desktop との統合

### 1. Claude Desktop 設定ファイル

Claude Desktop の設定ファイル（`claude_desktop_config.json`）に以下を追加：

```json
{
  "mcpServers": {
    "epub-llm": {
      "command": "python",
      "args": ["/path/to/epub-llm/start_mcp.py"],
      "env": {
        "PYTHONPATH": "/path/to/epub-llm"
      }
    }
  }
}
```

### 2. 設定ファイルの場所

**macOS:**
```bash
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Windows:**
```bash
%APPDATA%\Claude\claude_desktop_config.json
```

**Linux:**
```bash
~/.config/Claude/claude_desktop_config.json
```

### 3. 設定例（完全版）

```json
{
  "mcpServers": {
    "epub-llm": {
      "command": "python",
      "args": ["/Users/your-username/epub-llm/start_mcp.py"],
      "env": {
        "PYTHONPATH": "/Users/your-username/epub-llm",
        "DEV_MODE": "true"
      }
    }
  }
}
```

## 🐍 プログラマチック使用

### Python クライアント例

```python
import asyncio
from fastmcp.client import FastMCPClient

async def use_epub_mcp():
    # MCP Server に接続
    client = FastMCPClient("http://localhost:8001/mcp")
    
    try:
        # 書籍一覧を取得
        books = await client.call_tool("list_epub_books")
        print(f"利用可能な書籍: {len(books)}冊")
        
        if books:
            book_id = books[0]["id"]
            
            # メタデータ取得
            metadata = await client.call_tool("get_epub_metadata", {
                "book_id": book_id
            })
            print(f"書籍情報: {metadata}")
            
            # 検索実行
            results = await client.call_tool("search_epub_content", {
                "book_id": book_id,
                "query": "Python",
                "top_k": 3
            })
            print(f"検索結果: {len(results)}件")
            
            # 複数書籍からコンテキスト取得
            context = await client.call_tool("get_context_for_books", {
                "book_ids": [book_id],
                "query": "プログラミング",
                "top_k": 5
            })
            print(f"コンテキスト: {context[:200]}...")
            
    except Exception as e:
        print(f"エラー: {e}")
    finally:
        await client.close()

# 実行
asyncio.run(use_epub_mcp())
```

### cURL でのテスト

```bash
# 書籍一覧取得
curl -X POST http://localhost:8001/mcp/tools/list_epub_books \
  -H "Content-Type: application/json" \
  -d '{}'

# 書籍検索
curl -X POST http://localhost:8001/mcp/tools/search_epub_content \
  -H "Content-Type: application/json" \
  -d '{
    "book_id": "sample.epub",
    "query": "machine learning",
    "top_k": 5
  }'
```

## 🔍 MCP Server の動作確認

### 1. ヘルスチェック
```bash
# MCP Server が起動しているか確認
curl http://localhost:8001/mcp/health

# または
python -c "
import requests
response = requests.get('http://localhost:8001/mcp')
print(f'Status: {response.status_code}')
"
```

### 2. ツール一覧の確認
```bash
curl http://localhost:8001/mcp/tools
```

### 3. ログによる確認
```bash
# ログファイル確認
tail -f log/epub-llm.log

# または標準出力を確認
python start_mcp.py
```

## ⚠️ トラブルシューティング

### よくある問題と解決策

#### 1. MCP Server が起動しない
```bash
# 依存関係確認
pip install fastmcp uvicorn

# ポート確認
lsof -i :8001

# 設定ファイル確認
python -c "from src.config_manager import AppConfig; print(AppConfig())"
```

#### 2. Claude Desktop で認識されない
```bash
# 設定ファイルの構文確認
python -m json.tool ~/Library/Application\ Support/Claude/claude_desktop_config.json

# パス確認
ls -la /path/to/epub-llm/start_mcp.py

# 権限確認
chmod +x /path/to/epub-llm/start_mcp.py
```

#### 3. ツール実行エラー
```bash
# メインアプリケーションが起動しているか確認
curl http://localhost:8000/bookshelf

# EPUB ファイルが存在するか確認
ls -la epub/

# キャッシュのクリア
rm -rf cache/*
```

## 🛡️ セキュリティ考慮事項

### 1. ネットワーク設定
```yaml
# config/app_config.yaml
server:
  host: '127.0.0.1'  # ローカルのみアクセス許可
  port: 8001
```

### 2. アクセス制御
```python
# 必要に応じてAPIキー認証を追加
@mcp_app.middleware("http")
async def add_auth_header(request: Request, call_next):
    # 認証ロジック
    return await call_next(request)
```

## 📊 パフォーマンス最適化

### 1. キャッシュ設定
```yaml
# config/app_config.yaml
cache:
  embeddings_cache: true
  text_cache: true
```

### 2. 並列処理設定
```yaml
# config/smart_rag_config.yaml
performance:
  parallel_book_processing: true
  cache_bm25_index: true
```

### 3. ログレベル調整
```yaml
# config/app_config.yaml
logging:
  level: 'WARNING'  # 本番環境では WARNING 以上
```

## 🔄 開発とデバッグ

### 1. 開発モード設定
```bash
# 環境変数で開発モード有効化
export DEV_MODE=true
python start_mcp.py
```

### 2. デバッグログ有効化
```python
# start_mcp.py に追加
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 3. ホットリロード
```bash
# ホットリロード有効
uvicorn src.server:mcp_app --reload --host 0.0.0.0 --port 8001
```

---

このガイドに従って設定することで、Claude Desktop や他の MCP 対応アプリケーションから EPUB-LLM の機能を活用できます。