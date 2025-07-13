# EPUB LLM

EPUBファイルを使用したRAG（Retrieval-Augmented Generation）システムです。EPUBファイルをアップロードし、その内容を検索して関連する情報を取得できるWebアプリケーションです。

## 特徴

- **EPUBファイル管理**: EPUBファイルのアップロード、削除、メタデータ表示
- **RAG検索機能**: EPUBの内容から関連する文章を検索
- **チャット機能**: 複数のEPUBファイルを参照してチャット対話
- **履歴管理**: チャット履歴の保存と管理
- **MCP Server**: FastMCPを使用したMCP（Model Context Protocol）サーバー機能

## 技術スタック

- **Backend**: FastAPI, Python 3.8+
- **ML/AI**: MLX-LM, MLX-Embeddings, FAISS
- **Frontend**: HTML, CSS, JavaScript
- **EPUB処理**: EbookLib
- **MCP**: FastMCP

## システム要件

- Python 3.8以上
- MLX（Apple Silicon推奨）
- 8GB以上のRAM

## セットアップ

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd epub-llm
```

### 2. 仮想環境の作成

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# または
.venv\Scripts\activate  # Windows
```

### 3. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 4. 初期設定とモデルダウンロード

```bash
python start.py --setup
```

## 使用方法

### アプリケーションの起動

#### 開発モード
```bash
python start.py
# または
python start.py --dev
```

#### 本番モード
```bash
python start.py --prod
```

#### 直接起動
```bash
python run_server.py
```

### Webアプリケーションへのアクセス

ブラウザで以下のURLにアクセスしてください：
- チャットUI: http://localhost:8000
- 書棚UI: http://localhost:8000/bookshelf_ui

### MCP サーバーの使用

MCP サーバーはClaude等のAIアシスタントから利用できます：
- MCP Server: http://localhost:8000/mcp

利用可能なMCP ツール：
- `list_epub_books()` - EPUB書籍一覧の取得
- `get_epub_metadata(book_id)` - 書籍メタデータの取得
- `search_epub_content(book_id, query, top_k)` - 書籍内容の検索
- `get_chat_histories()` - チャット履歴一覧の取得
- `get_chat_history(session_id)` - チャット履歴の取得

## アーキテクチャ

### ディレクトリ構造

```
epub-llm/
├── src/
│   ├── app.py              # メインアプリケーション
│   ├── server.py           # MCP サーバー
│   ├── epub_service.py     # EPUB関連サービス層
│   ├── epub_util.py        # EPUB処理ユーティリティ
│   ├── embedding_util.py   # 埋め込みベクトル処理
│   ├── history_util.py     # チャット履歴処理
│   ├── llm_util.py         # LLM処理ユーティリティ
│   └── rag_util.py         # RAG処理ユーティリティ
├── epub/                   # EPUBファイル保存ディレクトリ
├── cache/                  # 埋め込みベクトルキャッシュ
├── static/                 # 静的ファイル
├── templates/              # HTMLテンプレート
├── docs/                   # ドキュメント
├── requirements.txt        # 依存関係
├── pyproject.toml         # プロジェクト設定
└── start.py               # 起動スクリプト
```

### サービス層アーキテクチャ

このアプリケーションは以下のサービス層に分かれています：

- **EPUBService**: EPUB関連の業務ロジック（書籍管理、検索等）
- **ChatService**: チャット関連の業務ロジック（履歴管理、セッション管理等）
- **RAGManager**: RAG処理の管理（埋め込み生成、類似度検索等）
- **LLMManager**: LLM処理の管理（テキスト生成、プロンプト処理等）

## 開発

### コード品質チェック

```bash
python quality_check.py
```

### テストの実行

```bash
pytest
```

### 開発用依存関係のインストール

```bash
pip install -r requirements-dev.txt
pre-commit install
```

## API エンドポイント

### 主要なエンドポイント

- `GET /` - チャットUI
- `GET /bookshelf` - 書棚データAPI
- `POST /chat` - チャットストリーミング
- `POST /upload_epub` - EPUBファイルアップロード
- `DELETE /delete_epub` - EPUBファイル削除
- `GET /history/{session_id}` - チャット履歴取得
- `POST /history/{session_id}` - チャット履歴保存

詳細は[API仕様書](docs/api.md)を参照してください。

## 設定

### 環境変数

- `HOST`: サーバーのホスト（デフォルト: 0.0.0.0）
- `PORT`: サーバーのポート（デフォルト: 8000）
- `MODE`: 実行モード（dev/prod）

### 設定ファイル

アプリケーションの設定は`config.json`で行うことができます。

## トラブルシューティング

### よくある問題

1. **MLXの依存関係エラー**
   - Apple Siliconを使用していることを確認
   - 最新のMLXライブラリを確認

2. **EPUBファイルの処理エラー**
   - EPUBファイルが破損していないか確認
   - ファイルサイズが100MB以下であることを確認

3. **メモリ不足**
   - cache/とstatic/cache/ディレクトリの容量を確認

### ログの確認

```bash
tail -f epub-llm.log
```

## 貢献

1. プロジェクトをフォーク
2. 新しい機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## ライセンス

MIT License - 詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 追加ドキュメント

- [設計書](docs/design.md)
- [要件定義書](docs/requirements.md)
- [開発ルール](docs/rule.md)
- [TODO リスト](docs/todo.md)