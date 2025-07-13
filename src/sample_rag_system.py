"""
サンプルRAGシステム - 10,000トークン程度のコーパスでのRAG実装
"""

import json
import logging
import os
from typing import Any, Dict, List

import numpy as np
from rank_bm25 import BM25Okapi

from src.common_util import create_text_chunks
from src.embedding_util import (
    ModelPair,
    build_faiss_index,
    create_embeddings_from_texts,
    search_similar,
)


class SampleRAGSystem:  # pylint: disable=too-many-instance-attributes
    """10,000トークン程度のサンプルコーパスを使用するRAGシステム"""

    def __init__(
        self, embed_model: Any, embed_tokenizer: Any, cache_dir: str = "cache"
    ):
        """
        初期化

        Args:
            embed_model: 埋め込みモデル
            embed_tokenizer: 埋め込みトークナイザー
            cache_dir: キャッシュディレクトリ
        """
        self.embed_model = embed_model
        self.embed_tokenizer = embed_tokenizer
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        self.model_pair = ModelPair(model=embed_model, tokenizer=embed_tokenizer)

        # サンプルコーパス
        self.corpus_data = self._load_sample_corpus()
        self.text_chunks: List[str] = []
        self.embeddings: np.ndarray = None
        self.bm25_index: BM25Okapi = None
        self.faiss_index = None

        # 設定読み込み（デフォルト値で初期化）
        self.config = self._get_default_config()

    def _load_sample_corpus(self) -> List[Dict[str, str]]:
        """サンプルコーパスデータを読み込み"""
        return [
            {
                "title": "Python基礎",
                "content": """
Pythonは汎用プログラミング言語として広く利用されています。シンプルで読みやすい構文が特徴で、
初心者から専門家まで幅広く愛用されています。

基本的なデータ型には、文字列型（str）、整数型（int）、浮動小数点型（float）、真偽値型（bool）があります。
制御構造には条件分岐のif文、繰り返し処理のforループやwhileループがあります。

関数は def キーワードで定義し、引数と戻り値を指定できます。
クラスは class キーワードで定義し、オブジェクト指向プログラミングを実現します。
""",
            },
            {
                "title": "Web開発とHTTP",
                "content": """
Web開発におけるHTTPプロトコルの理解は不可欠です。HTTPはクライアントとサーバー間の通信プロトコルで、
リクエストとレスポンスで構成されます。

主要なHTTPメソッドには GET、POST、PUT、DELETE、PATCH があります。
ステータスコードには 200 OK、201 Created、400 Bad Request、401 Unauthorized、404 Not Found、
500 Internal Server Error などがあります。

RESTful API設計では、リソースをURIで表現し、HTTPメソッドで操作を定義します。
例えば、ユーザー管理APIでは GET /api/users でユーザー一覧を取得し、
POST /api/users で新規ユーザーを作成します。
""",
            },
            {
                "title": "データベース設計とSQL",
                "content": """
リレーショナルデータベースは構造化データを効率的に管理するためのシステムです。
SQLを使用してデータの操作を行います。

基本的なSQL操作には、データ挿入（INSERT）、データ取得（SELECT）、
データ更新（UPDATE）、データ削除（DELETE）があります。

正規化は第一正規形（1NF）、第二正規形（2NF）、第三正規形（3NF）に分けられます。
インデックスはクエリのパフォーマンスを向上させるために使用されます。

トランザクションはACID特性（原子性、一貫性、独立性、持続性）を満たす必要があります。
""",
            },
            {
                "title": "機械学習基礎",
                "content": """
機械学習は大量のデータからパターンを学習し、予測や分類を行う技術です。

教師あり学習には回帰と分類があります。回帰は連続値を予測し、分類はカテゴリを予測します。
主要なアルゴリズムには線形回帰、ロジスティック回帰、決定木、ランダムフォレスト、
サポートベクターマシンがあります。

教師なし学習にはクラスタリングと次元削減があります。
K-meansは重心ベースのクラスタリング、PCAは主成分分析による次元削減手法です。

データ前処理では欠損値処理、外れ値検出、特徴量スケーリング、
カテゴリ変数のエンコーディングが重要です。
""",
            },
            {
                "title": "ソフトウェアアーキテクチャ",
                "content": """
マイクロサービスアーキテクチャはアプリケーションを小さな独立したサービスに分割する設計パターンです。
独立したデプロイとスケーリングが可能ですが、分散システムの複雑性があります。

レイヤードアーキテクチャではアプリケーションを層に分けて設計します。
プレゼンテーション層、ビジネス層、データアクセス層に分かれます。

ドメイン駆動設計（DDD）はビジネスドメインを中心とした設計手法です。
エンティティ、値オブジェクト、集約、ドメインサービスなどの概念があります。

SOLID原則は単一責任原則、開放閉鎖原則、リスコフ置換原則、
インターフェース分離原則、依存性逆転原則から構成されます。
""",
            },
            {
                "title": "セキュリティとベストプラクティス",
                "content": """
セキュリティにおいて認証と認可は重要な概念です。認証はユーザーの身元確認、
認可はリソースへのアクセス権限確認を行います。

一般的なセキュリティ脅威にはSQLインジェクション、XSS（Cross-Site Scripting）、
CSRF（Cross-Site Request Forgery）、セッションハイジャックがあります。

セキュリティ対策として入力検証、出力エスケープ、HTTPS使用、
定期的なセキュリティ更新が必要です。

認証方式にはBasic認証、Token認証、OAuth 2.0があります。
パスワードハッシュ化にはbcrypt、scrypt、Argon2などのアルゴリズムを使用します。
""",
            },
            {
                "title": "パフォーマンス最適化",
                "content": """
データベース最適化ではインデックス最適化、クエリ最適化、
正規化と非正規化のバランスが重要です。

キャッシング戦略にはメモリキャッシュ（Redis、Memcached）、CDN、
アプリケーションレベルキャッシュがあります。

非同期処理にはメッセージキュー（RabbitMQ、Apache Kafka）、
バックグラウンドジョブ（Celery、Sidekiq）、
イベント駆動アーキテクチャがあります。

プロファイリングツールを使用してボトルネックを特定し、
メモリ使用量とCPU使用量を監視します。
ロードバランシングとスケーリング戦略も重要です。
""",
            },
        ]

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を取得"""
        return {
            "chunking": {
                "chunk_size": 1000,
                "overlap": 200,
                "sentence_boundary_search": 100,
            },
            "hybrid_search": {
                "semantic_weight": 0.7,
                "keyword_weight": 0.3,
                "default_top_k": 10,
            },
        }

    def initialize_system(self) -> bool:
        """
        RAGシステムを初期化

        Returns:
            bool: 初期化成功の可否
        """
        try:
            self.logger.info("サンプルRAGシステムを初期化中...")

            # テキストチャンクを作成
            self._create_text_chunks()

            # 埋め込みとインデックスを作成
            self._create_embeddings()
            self._create_bm25_index()
            self._create_faiss_index()

            self.logger.info("サンプルRAGシステムの初期化完了")
            return True

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error("RAGシステムの初期化に失敗: %s", e)
            return False

    def _create_text_chunks(self) -> None:
        """テキストをチャンクに分割"""
        chunk_size = self.config.get("chunking", {}).get("chunk_size", 1000)
        overlap = self.config.get("chunking", {}).get("overlap", 200)

        for doc in self.corpus_data:
            title = doc["title"]
            content = doc["content"]

            # タイトルを含むコンテンツでチャンク作成
            full_content = f"# {title}\n\n{content}"
            chunks = create_text_chunks(full_content, chunk_size, overlap)

            for chunk in chunks:
                self.text_chunks.append(chunk)

        self.logger.info("テキストチャンク作成完了: %d個", len(self.text_chunks))

    def _create_embeddings(self) -> None:
        """テキストチャンクの埋め込みを作成"""
        if not self.text_chunks:
            raise ValueError("テキストチャンクが空です")

        self.embeddings = create_embeddings_from_texts(
            self.text_chunks, self.embed_model, self.embed_tokenizer
        )

        self.logger.info("埋め込み作成完了: %s", self.embeddings.shape)

    def _create_bm25_index(self) -> None:
        """BM25インデックスを作成"""
        if not self.text_chunks:
            raise ValueError("テキストチャンクが空です")

        # テキストをトークン化
        tokenized_chunks = []
        for chunk in self.text_chunks:
            # 簡易的なトークン化（実際の実装では形態素解析を使用）
            tokens = chunk.lower().split()
            tokenized_chunks.append(tokens)

        # BM25インデックス作成
        self.bm25_index = BM25Okapi(tokenized_chunks)

        self.logger.info("BM25インデックス作成完了")

    def _create_faiss_index(self) -> None:
        """FAISSインデックスを作成"""
        if self.embeddings is None:
            raise ValueError("埋め込みが作成されていません")

        self.faiss_index = build_faiss_index(self.embeddings)

        self.logger.info("FAISSインデックス作成完了")

    def search(  # pylint: disable=too-many-positional-arguments
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        ハイブリッド検索を実行

        Args:
            query: 検索クエリ
            top_k: 返す結果数
            use_hybrid: ハイブリッド検索を使用するか
            semantic_weight: 意味検索の重み
            keyword_weight: キーワード検索の重み

        Returns:
            List[Dict[str, Any]]: 検索結果
        """
        if not self._is_initialized():
            raise ValueError("RAGシステムが初期化されていません")

        if use_hybrid:
            return self._hybrid_search(query, top_k, semantic_weight, keyword_weight)
        return self._semantic_search(query, top_k)

    def _is_initialized(self) -> bool:
        """システムが初期化されているかチェック"""
        return (
            self.embeddings is not None
            and self.bm25_index is not None
            and self.faiss_index is not None
            and len(self.text_chunks) > 0
        )

    def _semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """意味検索を実行"""
        results = search_similar(
            query=query,
            model_pair=self.model_pair,
            index=self.faiss_index,
            texts=self.text_chunks,
            top_k=top_k,
        )

        formatted_results = []
        for idx, score, text in results:
            formatted_results.append(
                {
                    "index": idx,
                    "score": float(score),
                    "text": text,
                    "search_type": "semantic",
                }
            )

        return formatted_results

    def _hybrid_search(
        self, query: str, top_k: int, semantic_weight: float, keyword_weight: float
    ) -> List[Dict[str, Any]]:
        """ハイブリッド検索を実行"""
        # 意味検索
        semantic_results = self._semantic_search(query, top_k * 2)

        # キーワード検索
        keyword_results = self._keyword_search(query, top_k * 2)

        # スコアを正規化して結合
        combined_scores = {}

        # 意味検索スコアを正規化
        if semantic_results:
            max_semantic_score = max(r["score"] for r in semantic_results)
            for result in semantic_results:
                idx = result["index"]
                normalized_score = result["score"] / max_semantic_score
                combined_scores[idx] = (
                    combined_scores.get(idx, 0) + normalized_score * semantic_weight
                )

        # キーワード検索スコアを正規化
        if keyword_results:
            max_keyword_score = max(r["score"] for r in keyword_results)
            for result in keyword_results:
                idx = result["index"]
                normalized_score = result["score"] / max_keyword_score
                combined_scores[idx] = (
                    combined_scores.get(idx, 0) + normalized_score * keyword_weight
                )

        # 結果をスコア順にソート
        sorted_indices = sorted(
            combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True
        )

        # 上位top_k個を選択
        final_results = []
        for idx in sorted_indices[:top_k]:
            final_results.append(
                {
                    "index": idx,
                    "score": combined_scores[idx],
                    "text": self.text_chunks[idx],
                    "search_type": "hybrid",
                }
            )

        return final_results

    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """キーワード検索を実行"""
        query_tokens = query.lower().split()
        scores = self.bm25_index.get_scores(query_tokens)

        # スコア順にソート
        sorted_indices = np.argsort(scores)[::-1]

        results = []
        for idx in sorted_indices[:top_k]:
            if scores[idx] > 0:  # スコアが0より大きいもののみ
                results.append(
                    {
                        "index": int(idx),
                        "score": float(scores[idx]),
                        "text": self.text_chunks[idx],
                        "search_type": "keyword",
                    }
                )

        return results

    def get_context(self, query: str, max_length: int = 2000) -> str:
        """
        クエリに対する関連コンテキストを取得

        Args:
            query: 検索クエリ
            max_length: 最大文字数

        Returns:
            str: 結合されたコンテキスト
        """
        results = self.search(query, top_k=5)

        context_parts = []
        current_length = 0

        for result in results:
            text = result["text"]
            if current_length + len(text) <= max_length:
                context_parts.append(text)
                current_length += len(text)
            else:
                # 残り文字数分だけ追加
                remaining = max_length - current_length
                if remaining > 100:  # 最小100文字は確保
                    context_parts.append(text[:remaining] + "...")
                break

        return "\n\n---\n\n".join(context_parts)

    def save_system(self, filepath: str) -> bool:
        """
        システム状態を保存

        Args:
            filepath: 保存先ファイルパス

        Returns:
            bool: 保存成功の可否
        """
        try:
            system_data = {
                "text_chunks": self.text_chunks,
                "corpus_data": self.corpus_data,
                "embeddings_shape": (
                    self.embeddings.shape if self.embeddings is not None else None
                ),
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(system_data, f, ensure_ascii=False, indent=2)

            # 埋め込みは別ファイルに保存
            if self.embeddings is not None:
                embeddings_file = filepath.replace(".json", "_embeddings.npy")
                np.save(embeddings_file, self.embeddings)

            self.logger.info("システム状態を保存: %s", filepath)
            return True

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error("システム状態の保存に失敗: %s", e)
            return False

    def load_system(self, filepath: str) -> bool:
        """
        システム状態を読み込み

        Args:
            filepath: 読み込み元ファイルパス

        Returns:
            bool: 読み込み成功の可否
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                system_data = json.load(f)

            self.text_chunks = system_data["text_chunks"]
            self.corpus_data = system_data["corpus_data"]

            # 埋め込みを読み込み
            embeddings_file = filepath.replace(".json", "_embeddings.npy")
            if os.path.exists(embeddings_file):
                self.embeddings = np.load(embeddings_file)

            # インデックスを再作成
            if self.embeddings is not None:
                self._create_bm25_index()
                self._create_faiss_index()

            self.logger.info("システム状態を読み込み: %s", filepath)
            return True

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error("システム状態の読み込みに失敗: %s", e)
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """システム統計情報を取得"""
        return {
            "total_chunks": len(self.text_chunks),
            "total_documents": len(self.corpus_data),
            "embeddings_shape": (
                self.embeddings.shape if self.embeddings is not None else None
            ),
            "is_initialized": self._is_initialized(),
            "corpus_topics": [doc["title"] for doc in self.corpus_data],
        }
