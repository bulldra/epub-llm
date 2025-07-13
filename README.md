# EPUB-LLM

EPUBファイルを使用したアドバンスドRAG（Retrieval-Augmented Generation）システムです。EPUBファイルから知識を抽出し、インテリジェントな検索とチャット機能を提供するWebアプリケーションです。

## ✨ 主要機能

### 📚 書籍管理
- **EPUBファイル処理**: ドラッグ&ドロップでの簡単アップロード
- **メタデータ抽出**: タイトル、著者、出版情報の自動抽出
- **書棚UI**: 視覚的な書籍管理インターフェース
- **カバー画像表示**: 自動カバー抽出と表示

### 🔍 高度なRAG検索システム
- **ハイブリッド検索**: 意味検索とキーワード検索の組み合わせ
- **スマートRAG**: 適応的検索戦略とクエリ拡張
- **BM25 + 意味検索**: 最適化された検索アルゴリズム
- **リランキング**: 検索結果の品質向上
- **コンテキスト圧縮**: 効率的な情報抽出

### 💬 インテリジェントチャット
- **マルチブック対応**: 複数書籍を同時参照
- **ストリーミング応答**: リアルタイムレスポンス
- **履歴管理**: セッション管理と履歴保存
- **コンテキスト継続**: 会話の文脈保持

### 🛠️ 開発者向け機能
- **MCP Server**: Model Context Protocol対応
- **REST API**: 完全なRESTful API
- **設定管理**: YAML設定ファイル
- **ログ機能**: 詳細なクエリ・プロンプトログ
- **開発モード**: モデル無効化での高速開発

### 📊 サンプルRAGシステム
- **デモコーパス**: 10,000トークンのサンプル技術文書
- **インタラクティブデモ**: 検索機能のテスト環境
- **パフォーマンス測定**: システム統計情報

## 🛠️ 技術スタック

### Backend
- **FastAPI**: 高性能Webフレームワーク
- **Python 3.12**: 最新Python環境
- **MLX**: Apple Silicon最適化ML

### AI/ML
- **MLX-LM**: 言語モデル処理
- **MLX-Embeddings**: 埋め込みベクトル生成
- **FAISS**: 高速類似度検索
- **Rank-BM25**: 統計的検索

### Frontend
- **Vanilla JavaScript**: 軽量フロントエンド
- **CSS3**: モダンUI
- **WebSockets**: リアルタイム通信

### データ処理
- **EbookLib**: EPUB処理
- **BeautifulSoup4**: HTML解析
- **PyYAML**: 設定管理

## 📋 システム要件

- **Python**: 3.12以上
- **OS**: macOS (Apple Silicon推奨)
- **メモリ**: 8GB以上のRAM
- **ストレージ**: 10GB以上の空き容量

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

### 4. Webアプリケーションへのアクセス
- **チャットUI**: http://localhost:8000
- **書棚UI**: http://localhost:8000/bookshelf_ui
- **API Documentation**: http://localhost:8000/docs

## 🏗️ アーキテクチャ

### ディレクトリ構造
```
epub-llm/
├── src/                        # メインソースコード
│   ├── app.py                  # FastAPIアプリケーション
│   ├── server.py               # MCP サーバー
│   ├── config_manager.py       # 設定管理
│   ├── enhanced_epub_service.py # 拡張EPUB サービス
│   ├── epub_service.py         # 基本EPUB サービス
│   ├── smart_rag_util.py       # スマートRAG機能
│   ├── rag_util.py             # 基本RAG機能
│   ├── llm_util.py             # LLM処理
│   ├── embedding_util.py       # 埋め込み処理
│   ├── query_expansion_util.py # クエリ拡張
│   ├── rerank_util.py          # リランキング
│   ├── sample_rag_system.py    # サンプルRAGシステム
│   └── common_util.py          # 共通ユーティリティ
├── config/                     # 設定ファイル
│   ├── app_config.yaml         # アプリケーション設定
│   └── smart_rag_config.yaml   # スマートRAG設定
├── test/                       # テストスイート
├── examples/                   # サンプルとデモ
├── docs/                       # ドキュメント
├── epub/                       # EPUBファイル保存
├── cache/                      # 埋め込みベクトルキャッシュ
├── static/                     # 静的ファイル
├── templates/                  # HTMLテンプレート
└── log/                        # ログファイル
```

### サービス層アーキテクチャ

#### Core Services
- **EnhancedEPUBService**: 拡張EPUB処理とスマート検索
- **EPUBService**: 基本EPUB管理機能
- **EnhancedChatService**: 高度なチャット機能
- **ChatService**: 基本チャット機能

#### RAG Components
- **SmartRAGManager**: 適応的検索戦略
- **RAGManager**: 基本RAG機能
- **QueryExpander**: クエリ拡張とLLM強化
- **RerankingUtility**: 検索結果の最適化

#### Infrastructure
- **LLMManager**: 言語モデル管理
- **ConfigManager**: 設定管理システム
- **EmbeddingManager**: 埋め込みベクトル処理

## 🔧 設定

### アプリケーション設定 (`config/app_config.yaml`)
```yaml
llm:
  dev_mode: true                    # 開発モード
  model_name: 'llama-model'         # LLMモデル
  embedding_model_name: 'e5-large' # 埋め込みモデル

server:
  host: '0.0.0.0'
  port: 8000

directories:
  epub_dir: 'epub'
  cache_dir: 'cache'
  log_dir: 'log'
```

### スマートRAG設定 (`config/smart_rag_config.yaml`)
```yaml
hybrid_search:
  semantic_weight: 0.7      # 意味検索の重み
  keyword_weight: 0.3       # キーワード検索の重み

chunking:
  chunk_size: 4000          # チャンクサイズ
  overlap: 500              # オーバーラップ

reranking:
  diversity_weight: 0.2     # 多様性重み
  quality_weight: 0.1       # 品質重み
```

## 🧪 開発

### コード品質管理
```bash
# 全品質チェック実行
python quality_check.py

# 個別ツール実行
mypy src/           # 型チェック
pylint src/         # コード品質
black src/          # フォーマッタ
ruff check src/     # リンタ
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

### Pre-commit設定
```bash
pre-commit install
pre-commit run --all-files
```

## 📡 API エンドポイント

### 書籍管理API
- `GET /bookshelf` - 書棚データ取得
- `POST /upload_epub` - EPUBアップロード
- `DELETE /delete_epub` - EPUB削除
- `GET /book_metadata/{book_id}` - メタデータ取得

### チャットAPI
- `POST /chat` - ストリーミングチャット
- `GET /history/{session_id}` - 履歴取得
- `POST /history/{session_id}` - 履歴保存
- `DELETE /history/{session_id}` - 履歴削除

### 検索API
- `POST /search_single_book` - 単一書籍検索
- `POST /search_books` - マルチ書籍検索
- `POST /smart_search` - スマート検索

### MCP Tools
- `list_epub_books()` - 書籍一覧
- `get_epub_metadata(book_id)` - メタデータ
- `search_epub_content(book_id, query, top_k)` - 検索
- `get_context_for_books(book_ids, query, top_k)` - コンテキスト取得
- `smart_search_books(book_ids, query, top_k)` - スマート検索

## 📈 使用例

### サンプルRAGシステムのデモ
```bash
# デモ実行
python examples/sample_rag_demo.py

# インタラクティブモード
python examples/sample_rag_demo.py --interactive
```

### 基本的な使用フロー
1. EPUBファイルをアップロード
2. 自動で埋め込みベクトル生成
3. チャットで質問
4. 関連情報を検索・回答

## 🐛 トラブルシューティング

### 一般的な問題

#### MLX関連
```bash
# MLXインストール確認
python -c "import mlx.core; print('MLX OK')"

# Apple Siliconチェック
uname -m  # arm64 であることを確認
```

#### メモリ不足
```bash
# キャッシュクリア
rm -rf cache/
rm -rf static/cache/

# ログ確認
tail -f log/epub-llm.log
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
  level: 'DEBUG'  # INFO, WARNING, ERROR
```

## 🔍 監視とデバッグ

### 使用量監視（ccusage）
```bash
# 日次レポート
npx ccusage@latest daily

# リアルタイム監視
npx ccusage@latest blocks --live
```

### パフォーマンス監視
- クエリ生成ログ
- プロンプト詳細ログ
- 検索パフォーマンス
- レスポンス時間

## 🤝 開発ガイドライン

### コーディング規約
- Python 3.12使用
- 型ヒント必須
- 88文字行長制限
- t-wada流TDD採用
- 日本語コメント推奨

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

- [設計書](docs/design.md) - アーキテクチャと設計思想
- [要件定義](docs/requirements.md) - 機能要件と非機能要件
- [Smart RAG機能](docs/smart-rag-features.md) - 高度検索機能の詳細
- [MCP設定](docs/mcp-setup.md) - Model Context Protocol設定
- [開発ルール](docs/rule.md) - 開発規約とベストプラクティス

## 📄 ライセンス

MIT License - 詳細は[LICENSE](LICENSE)ファイルを参照

## 🙏 謝辞

- Apple MLX チーム
- FastAPI コミュニティ
- オープンソースコントリビューター

---

**注意**: このプロジェクトはApple Silicon環境での最適化を重視しています。他の環境での動作は保証されていません。