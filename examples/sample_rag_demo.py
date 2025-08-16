#!/usr/bin/env python3
"""
サンプルRAGシステムのデモンストレーション

このスクリプトは10,000トークン程度のサンプルコーパスを使用して
RAG（Retrieval-Augmented Generation）システムの動作を実演します。
"""

import logging
import os
import sys

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from src.config_manager import get_config
from src.sample_rag_system import SampleRAGSystem


def setup_logging():
    """ログ設定を初期化"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_mock_models():
    """
    モックモデルを作成（開発モード用）
    実際の使用時はMLXモデルをロードしてください
    """

    class MockModel:
        def __init__(self, embed_dim=128):
            self.embed_dim = embed_dim

        def __call__(self, texts, attention_mask=None, **kwargs):
            # 疑似的な埋め込みベクトルを返す
            import numpy as np

            if isinstance(texts, str):
                texts = [texts]
            elif hasattr(texts, "__getitem__") and "input_ids" in texts:
                # HuggingFace形式の入力の場合
                texts = texts["input_ids"]

            batch_size = len(texts) if hasattr(texts, "__len__") else 1
            embeddings = np.random.rand(batch_size, self.embed_dim)

            # MLXモデル互換の戻り値
            class MockOutput:
                def __init__(self, embeds):
                    self.text_embeds = embeds

            return MockOutput(embeddings)

    class MockTokenizer:
        def __call__(self, texts):
            return texts

        def encode(self, text):
            return text.split()

        def batch_encode_plus(self, texts, **kwargs):
            """HuggingFaceトークナイザー互換のメソッド"""
            return {
                "input_ids": [list(range(len(text.split()))) for text in texts],
                "attention_mask": [[1] * len(text.split()) for text in texts],
            }

    return MockModel(), MockTokenizer()


def demonstrate_rag_system():
    """RAGシステムのデモンストレーション"""
    logger = logging.getLogger(__name__)

    # 設定読み込み（簡略化）
    dev_mode = True  # デモ用にモックモデルを使用

    # モデル初期化
    if dev_mode:
        logger.info("開発モード: モックモデルを使用")
        embed_model, embed_tokenizer = create_mock_models()
    else:
        logger.info("本番モード: 実際のMLXモデルをロード")
        # 実際のMLXモデルロード処理をここに実装
        # from mlx_lm import load
        # embed_model, embed_tokenizer = load("multilingual-e5-large-mlx")
        embed_model, embed_tokenizer = create_mock_models()

    # RAGシステム初期化
    cache_dir = "cache"
    rag_system = SampleRAGSystem(embed_model, embed_tokenizer, cache_dir)

    logger.info("RAGシステムを初期化中...")
    success = rag_system.initialize_system()

    if not success:
        logger.error("RAGシステムの初期化に失敗しました")
        return

    # システム統計情報表示
    stats = rag_system.get_statistics()
    logger.info("=== システム統計情報 ===")
    logger.info(f"総チャンク数: {stats['total_chunks']}")
    logger.info(f"総文書数: {stats['total_documents']}")
    logger.info(f"埋め込み形状: {stats['embeddings_shape']}")
    logger.info(f"コーパストピック: {', '.join(stats['corpus_topics'])}")

    # デモクエリの実行
    demo_queries = [
        "Pythonのデータ型について教えて",
        "SQLの基本操作は何ですか？",
        "機械学習のアルゴリズムにはどんなものがありますか？",
        "セキュリティの脅威について",
        "パフォーマンス最適化の方法",
    ]

    logger.info("\n=== 検索デモンストレーション ===")

    for i, query in enumerate(demo_queries, 1):
        logger.info(f"\n--- クエリ {i}: {query} ---")

        # 意味検索
        semantic_results = rag_system.search(query, top_k=2, use_hybrid=False)
        logger.info("意味検索結果:")
        for j, result in enumerate(semantic_results[:2], 1):
            logger.info(f"  {j}. スコア: {result['score']:.3f}")
            logger.info(f"     テキスト: {result['text'][:100]}...")

        # ハイブリッド検索
        hybrid_results = rag_system.search(
            query, top_k=2, use_hybrid=True, semantic_weight=0.7, keyword_weight=0.3
        )
        logger.info("ハイブリッド検索結果:")
        for j, result in enumerate(hybrid_results[:2], 1):
            logger.info(f"  {j}. スコア: {result['score']:.3f}")
            logger.info(f"     テキスト: {result['text'][:100]}...")

        # コンテキスト取得
        context = rag_system.get_context(query, max_length=500)
        logger.info(f"統合コンテキスト（{len(context)}文字）:")
        logger.info(f"  {context[:200]}...")

    # システム保存のデモ
    save_path = os.path.join(cache_dir, "sample_rag_system.json")
    logger.info(f"\nシステム状態を保存: {save_path}")
    rag_system.save_system(save_path)

    logger.info("デモンストレーション完了!")


def interactive_mode():
    """インタラクティブモード"""
    logger = logging.getLogger(__name__)

    # モデル初期化
    embed_model, embed_tokenizer = create_mock_models()
    rag_system = SampleRAGSystem(embed_model, embed_tokenizer, "cache")

    # 既存のシステム状態があれば読み込み
    save_path = os.path.join("cache", "sample_rag_system.json")
    if os.path.exists(save_path):
        logger.info("既存のシステム状態を読み込み中...")
        rag_system.load_system(save_path)
    else:
        logger.info("RAGシステムを初期化中...")
        rag_system.initialize_system()

    print("\n=== サンプルRAGシステム - インタラクティブモード ===")
    print("質問を入力してください（'quit'で終了）")
    print(
        "利用可能なトピック:", ", ".join(rag_system.get_statistics()["corpus_topics"])
    )

    while True:
        try:
            query = input("\n質問: ").strip()

            if query.lower() in ["quit", "exit", "終了"]:
                break

            if not query:
                continue

            # 検索実行
            results = rag_system.search(query, top_k=3, use_hybrid=True)

            print(f"\n検索結果（{len(results)}件）:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. スコア: {result['score']:.3f}")
                print(f"   テキスト: {result['text'][:300]}...")

            # コンテキスト表示
            context = rag_system.get_context(query, max_length=1000)
            print("\n関連コンテキスト:")
            print(context[:500] + "..." if len(context) > 500 else context)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"エラーが発生しました: {e}")

    print("\nインタラクティブモードを終了しました。")


def main():
    """メイン関数"""
    setup_logging()

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        demonstrate_rag_system()


if __name__ == "__main__":
    main()
