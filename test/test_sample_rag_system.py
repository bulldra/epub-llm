"""
SampleRAGSystemのテストケース
"""
# pylint: disable=protected-access

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.sample_rag_system import SampleRAGSystem


class TestSampleRAGSystem:
    """SampleRAGSystemのテストクラス"""

    @pytest.fixture
    def mock_models(self):
        """モックモデルとトークナイザーを作成"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        return mock_model, mock_tokenizer

    @pytest.fixture
    def temp_cache_dir(self):
        """一時キャッシュディレクトリを作成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def sample_rag_system(self, mock_models, temp_cache_dir):
        """SampleRAGSystemインスタンスを作成"""
        mock_model, mock_tokenizer = mock_models
        return SampleRAGSystem(mock_model, mock_tokenizer, temp_cache_dir)

    def test_initialization(self, sample_rag_system):
        """初期化のテスト"""
        assert sample_rag_system.embed_model is not None
        assert sample_rag_system.embed_tokenizer is not None
        assert len(sample_rag_system.corpus_data) > 0
        assert sample_rag_system.text_chunks == []
        assert sample_rag_system.embeddings is None

    def test_load_sample_corpus(self, sample_rag_system):
        """サンプルコーパス読み込みのテスト"""
        corpus = sample_rag_system._load_sample_corpus()

        assert isinstance(corpus, list)
        assert len(corpus) > 0

        for doc in corpus:
            assert "title" in doc
            assert "content" in doc
            assert isinstance(doc["title"], str)
            assert isinstance(doc["content"], str)
            assert len(doc["title"]) > 0
            assert len(doc["content"]) > 0

    def test_create_text_chunks(self, sample_rag_system):
        """テキストチャンク作成のテスト"""
        sample_rag_system._create_text_chunks()

        assert len(sample_rag_system.text_chunks) > 0
        assert all(isinstance(chunk, str) for chunk in sample_rag_system.text_chunks)
        assert all(len(chunk) > 0 for chunk in sample_rag_system.text_chunks)

    @patch("src.sample_rag_system.create_embeddings_from_texts")
    def test_create_embeddings(self, mock_create_embeddings, sample_rag_system):
        """埋め込み作成のテスト"""
        # テストデータ準備
        sample_rag_system.text_chunks = ["chunk1", "chunk2", "chunk3"]
        mock_embeddings = np.random.rand(3, 128)
        mock_create_embeddings.return_value = mock_embeddings

        sample_rag_system._create_embeddings()

        assert sample_rag_system.embeddings is not None
        assert sample_rag_system.embeddings.shape == (3, 128)
        mock_create_embeddings.assert_called_once()

    def test_create_bm25_index(self, sample_rag_system):
        """BM25インデックス作成のテスト"""
        sample_rag_system.text_chunks = [
            "Python programming language",
            "Web development with HTTP",
            "Database design and SQL",
        ]

        sample_rag_system._create_bm25_index()

        assert sample_rag_system.bm25_index is not None

    @patch("src.sample_rag_system.build_faiss_index")
    def test_create_faiss_index(self, mock_build_faiss, sample_rag_system):
        """FAISSインデックス作成のテスト"""
        sample_rag_system.embeddings = np.random.rand(3, 128)
        mock_index = MagicMock()
        mock_build_faiss.return_value = mock_index

        sample_rag_system._create_faiss_index()

        assert sample_rag_system.faiss_index is not None
        mock_build_faiss.assert_called_once_with(sample_rag_system.embeddings)

    def test_is_initialized(self, sample_rag_system):
        """初期化チェックのテスト"""
        # 初期状態では未初期化
        assert not sample_rag_system._is_initialized()

        # 必要な要素を設定
        sample_rag_system.embeddings = np.random.rand(3, 128)
        sample_rag_system.bm25_index = MagicMock()
        sample_rag_system.faiss_index = MagicMock()
        sample_rag_system.text_chunks = ["chunk1", "chunk2", "chunk3"]

        # 初期化完了
        assert sample_rag_system._is_initialized()

    def test_keyword_search(self, sample_rag_system):
        """キーワード検索のテスト"""
        sample_rag_system.text_chunks = [
            "Python programming language basics",
            "Web development with HTTP protocol",
            "Database design and SQL queries",
        ]
        sample_rag_system._create_bm25_index()

        results = sample_rag_system._keyword_search("Python programming", top_k=2)

        assert isinstance(results, list)
        assert len(results) <= 2
        for result in results:
            assert "index" in result
            assert "score" in result
            assert "text" in result
            assert "search_type" in result
            assert result["search_type"] == "keyword"

    @patch("src.sample_rag_system.search_similar")
    def test_semantic_search(self, mock_search_similar, sample_rag_system):
        """意味検索のテスト"""
        # テストデータ準備
        sample_rag_system.text_chunks = ["chunk1", "chunk2", "chunk3"]
        sample_rag_system.embeddings = np.random.rand(3, 128)
        sample_rag_system.faiss_index = MagicMock()

        mock_search_similar.return_value = [
            (0, 0.9, "chunk1"),
            (1, 0.8, "chunk2"),
            (2, 0.7, "chunk3"),
        ]

        results = sample_rag_system._semantic_search("test query", top_k=2)

        assert isinstance(results, list)
        assert len(results) == 3  # mock_search_similarの戻り値数
        for result in results:
            assert "index" in result
            assert "score" in result
            assert "text" in result
            assert "search_type" in result
            assert result["search_type"] == "semantic"

    @patch("src.sample_rag_system.search_similar")
    def test_hybrid_search(self, mock_search_similar, sample_rag_system):
        """ハイブリッド検索のテスト"""
        # テストデータ準備
        sample_rag_system.text_chunks = [
            "Python programming language",
            "Web development HTTP",
            "Database SQL queries",
        ]
        sample_rag_system.embeddings = np.random.rand(3, 128)
        sample_rag_system.faiss_index = MagicMock()
        sample_rag_system._create_bm25_index()

        mock_search_similar.return_value = [
            (0, 0.9, "Python programming language"),
            (1, 0.8, "Web development HTTP"),
        ]

        results = sample_rag_system._hybrid_search(
            "Python programming", top_k=2, semantic_weight=0.7, keyword_weight=0.3
        )

        assert isinstance(results, list)
        assert len(results) <= 2
        for result in results:
            assert "index" in result
            assert "score" in result
            assert "text" in result
            assert "search_type" in result
            assert result["search_type"] == "hybrid"

    def test_get_context(self, sample_rag_system):
        """コンテキスト取得のテスト"""
        # システムを初期化状態に設定
        sample_rag_system.text_chunks = ["chunk1", "chunk2", "chunk3"]
        sample_rag_system.embeddings = np.random.rand(3, 128)
        sample_rag_system.bm25_index = MagicMock()
        sample_rag_system.faiss_index = MagicMock()

        with patch.object(sample_rag_system, "search") as mock_search:
            mock_search.return_value = [
                {"index": 0, "score": 0.9, "text": "short chunk"},
                {"index": 1, "score": 0.8, "text": "another chunk"},
            ]

            context = sample_rag_system.get_context("test query", max_length=100)

            assert isinstance(context, str)
            assert len(context) > 0
            mock_search.assert_called_once_with("test query", top_k=5)

    def test_save_and_load_system(self, sample_rag_system, temp_cache_dir):
        """システム保存・読み込みのテスト"""
        # テストデータ準備
        sample_rag_system.text_chunks = ["chunk1", "chunk2"]
        sample_rag_system.embeddings = np.random.rand(2, 128)

        # 保存
        filepath = os.path.join(temp_cache_dir, "test_system.json")
        success = sample_rag_system.save_system(filepath)
        assert success
        assert os.path.exists(filepath)

        # 新しいインスタンスで読み込み
        new_system = SampleRAGSystem(
            sample_rag_system.embed_model,
            sample_rag_system.embed_tokenizer,
            temp_cache_dir,
        )
        success = new_system.load_system(filepath)
        assert success
        assert new_system.text_chunks == sample_rag_system.text_chunks
        assert np.array_equal(new_system.embeddings, sample_rag_system.embeddings)

    def test_get_statistics(self, sample_rag_system):
        """統計情報取得のテスト"""
        stats = sample_rag_system.get_statistics()

        assert isinstance(stats, dict)
        assert "total_chunks" in stats
        assert "total_documents" in stats
        assert "embeddings_shape" in stats
        assert "is_initialized" in stats
        assert "corpus_topics" in stats

        assert stats["total_chunks"] == len(sample_rag_system.text_chunks)
        assert stats["total_documents"] == len(sample_rag_system.corpus_data)
        assert stats["is_initialized"] == sample_rag_system._is_initialized()

    def test_search_uninitialized_system(self, sample_rag_system):
        """未初期化システムでの検索エラーテスト"""
        with pytest.raises(ValueError, match="RAGシステムが初期化されていません"):
            sample_rag_system.search("test query")

    def test_empty_text_chunks_error(self, sample_rag_system):
        """空のテキストチャンクでのエラーテスト"""
        sample_rag_system.text_chunks = []

        with pytest.raises(ValueError, match="テキストチャンクが空です"):
            sample_rag_system._create_embeddings()

        with pytest.raises(ValueError, match="テキストチャンクが空です"):
            sample_rag_system._create_bm25_index()

    @patch("src.sample_rag_system.create_embeddings_from_texts")
    @patch("src.sample_rag_system.build_faiss_index")
    def test_initialize_system_success(
        self, mock_build_faiss, mock_create_embeddings, sample_rag_system
    ):
        """システム初期化成功のテスト"""
        mock_embeddings = np.random.rand(5, 128)
        mock_create_embeddings.return_value = mock_embeddings
        mock_build_faiss.return_value = MagicMock()

        success = sample_rag_system.initialize_system()

        assert success
        assert sample_rag_system._is_initialized()
        assert len(sample_rag_system.text_chunks) > 0

    @patch("src.sample_rag_system.create_embeddings_from_texts")
    def test_initialize_system_failure(self, mock_create_embeddings, sample_rag_system):
        """システム初期化失敗のテスト"""
        mock_create_embeddings.side_effect = Exception("Embedding creation failed")

        success = sample_rag_system.initialize_system()

        assert not success
        assert not sample_rag_system._is_initialized()


if __name__ == "__main__":
    pytest.main([__file__])
