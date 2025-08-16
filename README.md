# EPUB App

EPUB ファイルの管理・閲覧に特化したシンプルなアプリケーションです。EPUB ファイルのアップロード、メタデータ表示、基本的な管理機能を提供し、Model Context Protocol (MCP) を通じて外部ツールからアクセス可能にします。

## ✨ 主要機能

### 📚 書籍管理

-   **EPUB ファイル処理**: ドラッグ&ドロップでの簡単アップロード
-   **メタデータ抽出**: タイトル、著者、出版情報の自動抽出
-   **書棚 UI**: 視覚的な書籍管理インターフェース
-   **カバー画像表示**: 自動カバー抽出と表示

### 🔧 MCP 統合

-   **MCP Server**: EPUB ライブラリを外部ツールに公開
-   **書籍一覧**: `list_epub_books()` でライブラリ一覧取得
-   **メタデータ取得**: `get_epub_metadata()` で詳細情報取得
-   **Claude Desktop 対応**: 設定ファイルでの簡単統合

## ⚡️ 技術スタック

### Backend

-   **FastAPI**: 高性能 Web フレームワーク
-   **Python 3.12**: 最新 Python 環境
-   **FastMCP**: Model Context Protocol 実装

### Frontend

-   **Vanilla JavaScript**: 軽量フロントエンド
-   **CSS3**: モダン UI

### データ処理

-   **EbookLib**: EPUB ファイル処理
-   **BeautifulSoup4**: HTML 解析
-   **PyYAML**: 設定管理
-   **MLX + FAISS（MLX一本化）**: 埋め込み生成と意味検索は MLX ベースで一貫実装

## 📋 システム要件

-   **Python**: 3.12 以上
-   **OS**: macOS、Linux、Windows
-   **メモリ**: 4GB 以上の RAM
-   **ストレージ**: 5GB 以上の空き容量

## ⚡ クイックスタート

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd epub-llm
```

### 2. 依存関係のインストール

```bash
# 開発依存関係含む
pip install -r requirements-dev.txt

# 本番環境のみ
pip install -r requirements.txt
```

### 3. アプリケーションの起動

```bash
# 開発モード（推奨）
python run_server.py

# 直接起動
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Web アプリケーションへのアクセス

-   **書棚 UI**: http://localhost:8000/bookshelf_ui
-   **API Documentation**: http://localhost:8000/docs

## 🔎 意味検索（MLX）

- 埋め込みは Apple MLX（`mlx-lm`）で生成し、類似度検索は FAISS の内積（ベクトルは L2 正規化済み＝コサイン相当）を使用。
- 検索は FastAPI エンドポイントから利用可能で、アプリ機能もすべてこの経路に統一しています。

### モデル設定（必須）

- `config/app_config.yaml` → `mlx.embedding_model` に MLX 互換モデルIDを指定。
  - 例: `mlx-community/multilingual-e5-large-mlx`（多言語）
- 環境変数でも上書き可能: `MLX_EMBEDDING_MODEL`

### キャッシュとモデル整合性

- インデックスは `cache/` に保存されます（例: `mlx_faiss_index.bin`, `mlx_chunks_metadata.pkl`, `mlx_index_info.pkl`）。
- 起動時/検索時に、キャッシュに保存されたモデルIDと現在の設定が異なる場合は、自動的にキャッシュを無効化し再生成します（モデル変更時の不整合を解消）。

### 主な API（/mlx プレフィックス）

- `POST /mlx/index/rebuild`
  - EPUB ディレクトリ配下の全 `.epub` を再インデックス化
- `POST /mlx/index/book`
  - ボディ: `{ "book_id": "<ID>", "epub_path": "<相対 or 絶対パス>" }`
  - 単一書籍をインデックスに追加して保存
- `POST /mlx/search`
  - ボディ: `{ "query": "質問", "top_k": 5, "book_id": "任意" }`
  - AND 検索: 半角スペース区切りで全語を含むチャンクに絞り込み
- `GET /mlx/stats`
  - インデックス統計（チャンク数・本数・次元・モデル名）
- `DELETE /mlx/index`
  - キャッシュとインデックスを削除

### アプリ統合 API

- `POST /search`: 全書籍横断の意味検索
- `POST /book/{book_id}/search`: 特定書籍内の意味検索

## 🏗️ アーキテクチャ

### ディレクトリ構造

```
epub-llm/
├── src/                        # メインソースコード
│   ├── app.py                  # FastAPIアプリケーション
│   ├── mcp_server.py           # MCP サーバー
│   ├── config_manager.py       # 設定管理
│   ├── simple_epub_service.py  # EPUB サービス
│   └── common_util.py          # 共通ユーティリティ
├── config/                     # 設定ファイル
│   └── app_config.yaml         # アプリケーション設定
├── test/                       # テストスイート
├── docs/                       # ドキュメント
├── epub/                       # EPUBファイル保存
├── static/                     # 静的ファイル
├── templates/                  # HTMLテンプレート
└── log/                        # ログファイル
```

### サービス層アーキテクチャ

#### Core Services

-   **SimpleEPUBService**: 基本 EPUB 管理機能
-   **MLXEmbeddingService**: MLX 埋め込み生成＋FAISS 検索（本番経路）
-   **MCPServer**: Model Context Protocol サーバー

#### Infrastructure

-   **ConfigManager**: 設定管理システム

## 🔧 設定

### アプリケーション設定 (`config/app_config.yaml`)

```yaml
server:
    host: '0.0.0.0'
    port: 8000

directories:
    epub_dir: 'epub'
    cache_dir: 'cache'
    log_dir: 'log'

logging:
    level: 'INFO'
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    files:
        app_log: 'epub-app.log'

mlx:
    embedding_model: 'mlx-community/multilingual-e5-large-mlx'
```

## 🧪 開発

### コード品質管理

```bash
# 全品質チェック実行
python scripts/quality_check.py

# 個別ツール実行
mypy src/                       # 型チェック
python scripts/quality_check.py --tool pylint   # コード品質
black src/                      # フォーマッタ
ruff check src/                 # リンタ
```

### テスト実行

```bash
# 全テスト実行
pytest

# 詳細出力
pytest -v

# カバレッジ付き
pytest --cov=src

# 特定テストファイル
pytest test/test_sample_rag_system.py
```

### Pre-commit 設定

```bash
pre-commit install
pre-commit run --all-files
```

## 📡 API エンドポイント

### 書籍管理 API

-   `GET /bookshelf` - 書棚データ取得
-   `POST /upload_epub` - EPUB アップロード
-   `DELETE /delete_epub` - EPUB 削除
-   `GET /book_metadata/{book_id}` - メタデータ取得

### UI エンドポイント

-   `GET /bookshelf_ui` - 書棚 UI 表示
-   `GET /docs` - API ドキュメント

### MCP Server

詳細な設定方法は [MCP Server 設定ガイド](docs/mcp-setup.md) を参照

### MCP クイック設定（1分で開始）

1) 基本セットアップ

```
# アプリ起動（別ターミナルでOK）
python run_server.py

# MCP Server 起動
python start_mcp.py

# 動作確認（サンプルクライアント）
python examples/test_mcp_client.py
```

2) Claude Desktop 設定

- 設定ファイルの場所を開く

```
# macOS
open ~/Library/Application\ Support/Claude/

# Windows
explorer %APPDATA%\Claude\

# Linux
nautilus ~/.config/Claude/
```

- `claude_desktop_config.json` を編集（絶対パスに置き換え）

```json
{
  "mcpServers": {
    "epub-app": {
      "command": "python",
      "args": ["/absolute/path/to/epub-llm/start_mcp.py"],
      "env": { "PYTHONPATH": "/absolute/path/to/epub-llm" }
    }
  }
}
```

- Claude Desktop を再起動

3) 動作確認の例

```
EPUBファイル一覧を取得してください
```

```
「sample.epub」から「Python」について検索してください
```

4) トラブルシューティング（抜粋）

- 依存確認: `pip install fastmcp uvicorn`
- ポート確認: `lsof -i :8001`
- 反映されない: パスの再確認、JSON構文、Claude 再起動

#### MCP Server 起動

```bash
# MCP Server 単独起動
python start_mcp.py
```

#### Claude Desktop 設定

```json
{
	"mcpServers": {
		"epub-app": {
			"command": "python",
			"args": ["/path/to/epub-llm/start_mcp.py"],
			"env": {
				"PYTHONPATH": "/path/to/epub-llm"
			}
		}
	}
}
```

#### 利用可能な MCP Tools

-   `list_epub_books()` - 書籍一覧取得
-   `get_epub_metadata(book_id)` - 書籍メタデータ取得

### 検索 API（MLX 埋め込み）

-   `POST /search` - 全書籍横断検索（JSON: `{ "query": "...", "top_k": 10 }`）
-   `POST /book/{book_id}/search` - 特定書籍内検索（クエリ `?query=...`）

インデックスは初回検索時に自動構築され、以降はキャッシュを使用します。埋め込みは MLX モデル（既定: `mlx-community/bge-small-en-v1.5`）で生成します。

## 使用例

### 基本的な使用フロー

1. アプリケーションを起動
2. EPUB ファイルをアップロード
3. 書棚 UI で書籍を管理
4. MCP を通じて外部ツールからアクセス

### MCP 統合例

```bash
# Claude Desktopで利用可能
# 書籍一覧を確認
list_epub_books()

# 特定の書籍の詳細を取得
get_epub_metadata("book-id")
```

## 🐛 トラブルシューティング

### 一般的な問題

#### EPUB ファイル処理

```bash
# EPUB ライブラリの確認
python -c "import ebooklib; print('EbookLib OK')"

# ファイル形式確認
file your_book.epub
```

#### MCP 接続問題

```bash
# MCP サーバー起動確認
python start_mcp.py

# Claude Desktop 設定確認
cat ~/.config/claude_desktop_config.json
```

#### 設定問題

```bash
# 設定ファイル確認
python -c "from src.config_manager import AppConfig; print(AppConfig().get_config())"
```

### ログレベル設定

```yaml
# config/app_config.yaml
logging:
    level: 'DEBUG' # INFO, WARNING, ERROR
```

## 🔍 監視とデバッグ

### ログ監視

```bash
# ログファイル監視
tail -f log/epub-app.log

# アプリケーション起動ログ
python run_server.py
```

### パフォーマンス監視

-   EPUB ファイル処理時間
-   API レスポンス時間
-   メタデータ抽出パフォーマンス

## 🤝 開発ガイドライン

### コーディング規約

-   Python 3.12 使用
-   型ヒント必須
-   88 文字行長制限
-   t-wada 流 TDD 採用
-   日本語コメント推奨

### コミット前チェック

```bash
# 必須実行項目
mypy src/
black src/
pylint src/
ruff check src/
pytest
```

## 📚 ドキュメント

-   [MCP 設定ガイド](docs/mcp-setup.md) - Model Context Protocol 設定
-   [クイック MCP 設定](QUICK_MCP_SETUP.md) - 簡単な MCP 設定手順
-   [開発ルール](docs/rule.md) - 開発規約とベストプラクティス

## 📄 ライセンス

MIT License - 詳細は[LICENSE](LICENSE)ファイルを参照

## 🙏 謝辞

-   FastAPI コミュニティ
-   Model Context Protocol 開発チーム
-   オープンソースコントリビューター

---

**注意**: このアプリケーションは EPUB ファイル管理に特化したシンプルな実装です。Model Context Protocol を通じて外部ツールとの統合が可能です。
