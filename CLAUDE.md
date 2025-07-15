# Claude Code Memory

## プロジェクト概要

EPUB-LLM プロジェクトは、EPUB ファイルの読み込み、テキスト抽出、埋め込み処理、そして RAG（Retrieval-Augmented Generation）機能を提供する Python アプリケーションです。

## 開発ガイドライン

プロジェクトの開発ガイドラインは `.github/copilot-instructions.md` に定義されています。

### 主要な開発原則

-   Python 3.12 を使用
-   型ヒント必須
-   返却されるのは日本語で
-   コードフォーマットは black、ruff 使用
-   88 文字行長制限
-   テストは pytest 使用
-   t-wada 流 TDD（Test-Driven Development）を採用
-   コード修正したら mypy black pylint ruff を実行して全ての警告を解消すること
-   import-outside-toplevel を禁止
-   pylint: disable を使用しない

### テスト実行コマンド

```bash
# テスト実行
pytest

# テスト実行（詳細出力）
pytest -v

# カバレッジ測定
pytest --cov=src

# 型チェック
mypy src/

# リンティング
ruff check src/
flake8 src/

# コードフォーマット
black src/
ruff format src/
```

## プロジェクト構造

-   `src/`: メインソースコード
-   `test/`: テストファイル
-   `epub/`: EPUB ファイル格納
-   `cache/`: キャッシュファイル
-   `static/`: 静的ファイル（CSS、JS）
-   `templates/`: HTML テンプレート
-   `config/`: 設定ファイル

## 重要なモジュール

-   `epub_util.py`: EPUB 処理ユーティリティ
-   `embedding_util.py`: 埋め込み処理
-   `rag_util.py`: RAG 機能
-   `history_util.py`: 履歴管理
-   `server.py`: Web サーバー
-   `config_manager.py`: 設定管理（AppConfig, SmartRAGConfig）

## 設定管理

### アプリケーション設定 (`config/app_config.yaml`)

-   LLM モデル設定（開発モード、モデル名、埋め込みモデル名）
-   サーバー設定（ホスト、ポート）
-   ディレクトリ設定（epub, cache, log, config）
-   ログ設定（レベル、フォーマット、ファイル名）
-   環境変数オーバーライド設定

### Smart RAG 設定 (`config/smart_rag_config.yaml`)

-   ハイブリッド検索設定
-   BM25 設定
-   チャンク設定
-   リランキング設定

## t-wada 流 TDD 実装方針

1. テストファーストの原則を厳守
2. Red-Green-Refactor サイクルを遵守
3. まず失敗するテストを書き、次に最小限のコードで通し、最後にリファクタリング
4. テストは可読性と保守性を重視
5. テストケースは具体的で明確な名前を付ける
6. モックやスタブは必要最小限に留める
7. 統合テストよりも単体テストを優先
8. テストの実行速度を重視
9. テストコードも本番コードと同じ品質基準を適用

## Claude Code 使用量監視 (ccusage)

### 導入済みツール

ccusage は、Claude Code の使用量を監視するための CLI ツールです。

### 主要コマンド

```bash
# 日次レポート
npx ccusage@latest daily

# 月次レポート
npx ccusage@latest monthly

# セッション別レポート
npx ccusage@latest session

# ブロック別レポート（5時間課金単位）
npx ccusage@latest blocks

# ライブ監視（リアルタイムダッシュボード）
npx ccusage@latest blocks --live
```

### 機能

-   トークン使用量とコスト分析
-   リアルタイム監視とバーンレート計算
-   モデル別コスト内訳
-   セッション進捗と予測
-   プライバシー重視（ローカルファイル分析）

### 重要な運用ルール

**Claude Code での作業完了後は必ず ccusage を実行してコストを確認すること**

```bash
# 作業完了後の必須チェック
npx ccusage@latest daily
```
