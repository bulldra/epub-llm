"""
LLM utility functions for text generation and query processing.
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from mlx_lm import stream_generate as mlx_stream_generate  # type: ignore
from mlx_lm.sample_utils import make_sampler


class LLMManager:
    """Manages LLM operations including text generation and query processing."""

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
        self.dev_mode = model is None or tokenizer is None

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 128000,
        *,
        temperature: float = 0.2,
        top_p: float = 1.0,
        top_k: int = 40,
    ) -> AsyncGenerator[str, None]:
        """Generate text stream from prompt."""
        # Log prompt for debugging
        truncated_prompt = prompt[:500] + "..." if len(prompt) > 500 else prompt
        self.logger.debug("[LLM] Input prompt: %s", truncated_prompt)
        # Development mode: return mock response
        if self.dev_mode:
            self.logger.info("[LLM] Development mode: generating mock response")
            mock_response = "開発モードで動作中です。実際のLLMモデルは無効化されています。この画面では基本的な機能をテストできます。"

            # Simulate streaming by yielding character by character
            for char in mock_response:
                yield char
                # Small delay to simulate streaming
                await asyncio.sleep(0.01)
            return

        try:
            token_count = 0
            start_time: float | None = None

            for token in mlx_stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=make_sampler(temp=temperature, top_p=top_p, top_k=top_k),
            ):
                if token.text:
                    if start_time is None:
                        start_time = time.time()
                        self.logger.info("[LLM] First token generated")

                    token_count += 1
                    yield token.text

            if start_time is not None:
                elapsed = time.time() - start_time
                tps = token_count / elapsed if elapsed > 0 else 0.0
                self.logger.info(
                    "[LLM] Generation complete. tokens=%d elapsed=%.2fs tps=%.2f",
                    token_count,
                    elapsed,
                    tps,
                )
        except Exception as e:
            self.logger.error("[LLM] Generation error: %s", e, exc_info=True)
            raise

    def generate_rag_query(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 48,
        temperature: float = 0.2,
    ) -> str:
        """Generate RAG search query from conversation context."""
        self.logger.info("[RAG] Generating query from context")

        # Development mode: return simple query
        if self.dev_mode:
            if messages:
                last_message = messages[-1].get("content", "検索")
                query = f"検索クエリ: {last_message[:50]}"
                self.logger.info("[RAG] Generated query (dev mode): %s", query)
                return query
            query = "開発モード検索"
            self.logger.info("[RAG] Generated query (dev mode): %s", query)
            return query

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "あなたは専門検索エージェントです。\n"
                    "## 出力仕様\n"
                    "- 日本語で1文のみ\n"
                    "- 箇条書き・記号は禁止\n"
                    "- 冗長な説明やJSONは出力しない\n"
                    "## 例\n"
                    "入力: 19世紀のチェスオートマトンの実態は？ → "
                    "出力: チェス 自動人形 ターク 仕組み 真相\n"
                ),
            },
            *messages,
            {
                "role": "user",
                "content": (
                    "書籍のRAG検索用に意図抽出したクエリ文を1文だけ日本語で出力してください。"
                    "一覧性が必要な場合は目次の章タイトルも活用してください。"
                    "出力はクエリ文のみで、他の語は含めないでください。"
                ),
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=False
        )
        truncated = prompt[:200] + "..." if len(prompt) > 200 else prompt
        self.logger.debug("[RAG] Generated prompt: %s", truncated)

        try:
            llm_query = ""
            for token in mlx_stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=make_sampler(temp=temperature, top_p=1.0, top_k=40),
            ):
                if token.text:
                    llm_query += token.text

            result = llm_query.strip()
            self.logger.info("[RAG] Generated query: %s", result)
            return result

        except (RuntimeError, ValueError, OSError) as e:
            self.logger.error("[RAG] Query generation error: %s", e, exc_info=True)
            # Fallback to last user message
            fallback = messages[-1].get("content", "") if messages else ""
            self.logger.info("[RAG] Using fallback query: %s", fallback)
            return fallback

    def format_messages_as_prompt(self, messages: list[dict[str, str]]) -> str:
        """Format message list as a single prompt string."""
        prompt = ""
        for message in messages:
            prompt += f"{message['role']}: {message['content']}\n"
        self.logger.info(
            "[LLM] Formatted prompt (%d messages, %d chars)", len(messages), len(prompt)
        )
        truncated_prompt = prompt[:500] + "..." if len(prompt) > 500 else prompt
        self.logger.debug("[LLM] Full prompt: %s", truncated_prompt)
        return prompt
