"""
クエリ生成ログのテスト
"""

import asyncio
import logging
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from src.llm_util import LLMManager
from src.query_expansion_util import QueryExpander


class TestQueryLogging:
    """クエリ生成とログ出力のテスト"""

    @pytest.fixture
    def mock_models(self):
        """モックモデルとトークナイザーを作成"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        return mock_model, mock_tokenizer

    @pytest.fixture
    def log_capture(self):
        """ログ出力をキャプチャ"""
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
        handler.setFormatter(formatter)
        
        # 既存のハンドラを保存
        original_handlers = {}
        loggers = [
            logging.getLogger('src.llm_util'),
            logging.getLogger('src.query_expansion_util')
        ]
        
        for logger in loggers:
            original_handlers[logger.name] = logger.handlers.copy()
            logger.handlers.clear()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        yield log_stream
        
        # ハンドラを復元
        for logger in loggers:
            logger.handlers.clear()
            for handler in original_handlers[logger.name]:
                logger.addHandler(handler)

    def test_llm_generate_rag_query_logging_dev_mode(self, mock_models, log_capture):
        """開発モードでのRAGクエリ生成ログテスト"""
        mock_model, mock_tokenizer = mock_models
        llm_manager = LLMManager(mock_model, mock_tokenizer)
        llm_manager.dev_mode = True
        
        messages = [
            {"role": "user", "content": "Pythonのデータ型について教えてください"}
        ]
        
        query = llm_manager.generate_rag_query(messages)
        
        # ログ出力を確認
        log_output = log_capture.getvalue()
        assert "[RAG] Generating query from context" in log_output
        assert "[RAG] Generated query (dev mode):" in log_output
        assert "検索クエリ: Pythonのデータ型について教えてください" in log_output
        assert query == "検索クエリ: Pythonのデータ型について教えてください"

    def test_llm_generate_rag_query_logging_empty_messages(self, mock_models, log_capture):
        """空メッセージでのRAGクエリ生成ログテスト"""
        mock_model, mock_tokenizer = mock_models
        llm_manager = LLMManager(mock_model, mock_tokenizer)
        llm_manager.dev_mode = True
        
        messages = []
        
        query = llm_manager.generate_rag_query(messages)
        
        # ログ出力を確認
        log_output = log_capture.getvalue()
        assert "[RAG] Generated query (dev mode): 開発モード検索" in log_output
        assert query == "開発モード検索"

    @pytest.mark.asyncio
    async def test_query_expansion_logging(self, mock_models, log_capture):
        """クエリ拡張のログテスト"""
        mock_model, mock_tokenizer = mock_models
        llm_manager = MagicMock()
        llm_manager.dev_mode = True
        
        query_expander = QueryExpander(llm_manager)
        
        # LLMの応答をモック
        async def mock_stream_generate(*args, **kwargs):
            yield "検索用語"
        
        llm_manager.stream_generate = mock_stream_generate
        
        result = await query_expander.expand_query("Pythonの使い方")
        
        # ログ出力を確認
        log_output = log_capture.getvalue()
        assert "[QueryExpander] Expanding query: Pythonの使い方" in log_output
        assert "[QueryExpander] Generated" in log_output
        assert "search queries" in log_output
        
        # 結果の確認
        assert result["original_query"] == "Pythonの使い方"
        assert isinstance(result["search_queries"], list)
        assert len(result["search_queries"]) > 0

    @pytest.mark.asyncio
    async def test_query_expansion_with_synonyms_logging(self, mock_models, log_capture):
        """同義語展開のログテスト"""
        mock_model, mock_tokenizer = mock_models
        llm_manager = MagicMock()
        llm_manager.dev_mode = True
        
        query_expander = QueryExpander(llm_manager)
        
        # LLMの応答をモック
        async def mock_stream_generate(*args, **kwargs):
            yield "Pythonの利用方法"
        
        llm_manager.stream_generate = mock_stream_generate
        
        # 同義語辞書に登録されているクエリでテスト
        result = await query_expander.expand_query("Pythonの使い方")
        
        # ログ出力を確認
        log_output = log_capture.getvalue()
        
        # 同義語が生成された場合のログ
        if result["synonym_queries"]:
            assert "[QueryExpander] Generated" in log_output
            assert "synonym queries" in log_output

    @pytest.mark.asyncio
    async def test_llm_expansion_logging(self, mock_models, log_capture):
        """LLMベースの拡張ログテスト"""
        mock_model, mock_tokenizer = mock_models
        llm_manager = MagicMock()
        llm_manager.dev_mode = False  # 実際のLLM拡張をテスト
        
        query_expander = QueryExpander(llm_manager)
        
        # LLMの応答をモック
        async def mock_stream_generate(*args, **kwargs):
            # 異なる拡張を生成
            yield "Python プログラミング言語の使用方法"
        
        llm_manager.stream_generate = mock_stream_generate
        
        expanded = await query_expander._llm_expand_query("Pythonの使い方")
        
        # ログ出力を確認
        log_output = log_capture.getvalue()
        if expanded:
            assert "[QueryExpander] LLM generated expansion:" in log_output
            assert "Python プログラミング言語の使用方法" in log_output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])