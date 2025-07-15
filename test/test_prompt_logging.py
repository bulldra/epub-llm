"""
プロンプト生成ログのテスト
"""

import logging
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from src.llm_util import LLMManager
from src.query_expansion_util import QueryExpander


class TestPromptLogging:
    """プロンプト生成とログ出力のテスト"""

    @pytest.fixture
    def mock_models(self):
        """モックモデルとトークナイザーを作成"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # apply_chat_templateのモック
        def mock_apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False  # pylint: disable=unused-argument
        ):
            formatted = ""
            for msg in messages:
                formatted += f"<{msg['role']}>{msg['content']}</{msg['role']}>\n"
            if add_generation_prompt:
                formatted += "<assistant>"
            return formatted

        mock_tokenizer.apply_chat_template = mock_apply_chat_template
        return mock_model, mock_tokenizer

    @pytest.fixture
    def log_capture(self):
        """ログ出力をキャプチャ（DEBUGレベル含む）"""
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)

        # 既存のハンドラを保存
        original_handlers = {}
        loggers = [
            logging.getLogger("src.llm_util"),
            logging.getLogger("src.query_expansion_util"),
        ]

        for logger in loggers:
            original_handlers[logger.name] = logger.handlers.copy()
            logger.handlers.clear()
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)  # DEBUGレベルに設定

        yield log_stream

        # ハンドラを復元
        for logger in loggers:
            logger.handlers.clear()
            for handler in original_handlers[logger.name]:
                logger.addHandler(handler)

    @pytest.mark.asyncio
    async def test_stream_generate_prompt_logging(self, mock_models, log_capture):
        """stream_generateでのプロンプトログテスト"""
        mock_model, mock_tokenizer = mock_models
        llm_manager = LLMManager(mock_model, mock_tokenizer)
        llm_manager.dev_mode = True

        test_prompt = (
            "これはテスト用のプロンプトです。少し長いプロンプトをテストします。"
        )

        # ストリーミング生成を実行
        response = "".join([token async for token in llm_manager.generate_stream(test_prompt)])

        # ログ出力を確認
        log_output = log_capture.getvalue()
        assert "[LLM] Input prompt:" in log_output
        assert "これはテスト用のプロンプトです" in log_output

    def test_format_messages_as_prompt_logging(self, mock_models, log_capture):
        """format_messages_as_promptでのログテスト"""
        mock_model, mock_tokenizer = mock_models
        llm_manager = LLMManager(mock_model, mock_tokenizer)

        messages = [
            {"role": "system", "content": "あなたはアシスタントです。"},
            {"role": "user", "content": "こんにちは"},
            {"role": "assistant", "content": "こんにちは！何をお手伝いしましょうか？"},
            {"role": "user", "content": "天気について教えて"},
        ]

        prompt = llm_manager.format_messages_as_prompt(messages)

        # ログ出力を確認
        log_output = log_capture.getvalue()
        assert "[LLM] Formatted prompt (4 messages" in log_output
        assert "chars)" in log_output
        assert "[LLM] Full prompt:" in log_output

        # プロンプトの内容確認
        assert "system: あなたはアシスタントです。" in prompt
        assert "user: こんにちは" in prompt
        assert "assistant: こんにちは！何をお手伝いしましょうか？" in prompt
        assert "user: 天気について教えて" in prompt

    def test_generate_rag_query_prompt_logging(self, mock_models, log_capture):
        """generate_rag_queryでのプロンプトログテスト"""
        mock_model, mock_tokenizer = mock_models
        llm_manager = LLMManager(mock_model, mock_tokenizer)
        llm_manager.dev_mode = False  # 実際のプロンプト生成をテスト

        messages = [
            {
                "role": "user",
                "content": "Pythonのデータ型について詳しく説明してください",
            }
        ]

        # MLX生成をモック
        with patch("src.llm_util.mlx_stream_generate") as mock_generate:
            mock_token = MagicMock()
            mock_token.text = "Python データ型"
            mock_generate.return_value = [mock_token]

            _ = llm_manager.generate_rag_query(messages)

        # ログ出力を確認
        log_output = log_capture.getvalue()
        assert "[RAG] Generated prompt:" in log_output
        assert "<system>" in log_output  # apply_chat_templateの出力を確認

    @pytest.mark.asyncio
    async def test_query_expansion_prompt_logging(self, mock_models, log_capture):
        """QueryExpanderでのプロンプトログテスト"""
        _, _ = mock_models
        llm_manager = MagicMock()
        llm_manager.dev_mode = False

        query_expander = QueryExpander(llm_manager)

        # LLMの応答をモック
        async def mock_stream_generate(prompt):  # pylint: disable=unused-argument
            yield "Python プログラミング言語の利用方法"

        llm_manager.generate_stream = mock_stream_generate

        _ = await query_expander._llm_expand_query("Pythonの使い方")  # pylint: disable=protected-access

        # ログ出力を確認
        log_output = log_capture.getvalue()
        assert "[QueryExpander] LLM expansion prompt:" in log_output
        assert "以下の質問を、より検索しやすい形に言い換えてください" in log_output
        assert "元の質問: Pythonの使い方" in log_output

    def test_long_prompt_truncation(self, mock_models, log_capture):
        """長いプロンプトの切り詰めテスト"""
        mock_model, mock_tokenizer = mock_models
        llm_manager = LLMManager(mock_model, mock_tokenizer)

        # 非常に長いメッセージを作成
        long_content = "これは非常に長いメッセージです。" * 50  # 約1500文字
        messages = [{"role": "user", "content": long_content}]

        prompt = llm_manager.format_messages_as_prompt(messages)

        # ログ出力を確認
        log_output = log_capture.getvalue()
        assert "[LLM] Full prompt:" in log_output

        # 切り詰めが行われていることを確認
        if len(prompt) > 500:
            assert "..." in log_output

    @pytest.mark.asyncio
    async def test_debug_vs_info_logging(self, mock_models, log_capture):
        """DEBUGとINFOレベルのログ分けテスト"""
        mock_model, mock_tokenizer = mock_models
        llm_manager = LLMManager(mock_model, mock_tokenizer)
        llm_manager.dev_mode = True

        messages = [{"role": "user", "content": "テスト"}]

        # プロンプトフォーマット（INFOログ）
        _ = llm_manager.format_messages_as_prompt(messages)

        # ストリーミング生成（DEBUGログ）
        async for _ in llm_manager.generate_stream("テストプロンプト"):
            break

        log_output = log_capture.getvalue()

        # INFOレベルのログ確認
        assert "[INFO]" in log_output
        assert "Formatted prompt" in log_output

        # DEBUGレベルのログ確認
        assert "[DEBUG]" in log_output
        assert "Input prompt:" in log_output or "Full prompt:" in log_output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
