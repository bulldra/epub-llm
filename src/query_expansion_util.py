"""
Query expansion and understanding utilities for improved RAG performance.
"""

import logging
import re
from typing import Any


class QueryExpander:
    """Expand and improve queries for better search results."""

    def __init__(self, llm_manager: Any):
        self.llm_manager = llm_manager
        self.logger = logging.getLogger(__name__)

        # Japanese synonyms and related terms
        self.synonym_map = {
            "方法": ["やり方", "手段", "手法", "アプローチ"],
            "理由": ["原因", "根拠", "要因", "背景"],
            "効果": ["結果", "影響", "作用", "メリット"],
            "問題": ["課題", "困った", "トラブル", "悩み"],
            "特徴": ["特色", "性質", "性格", "傾向"],
            "違い": ["差", "相違", "区別", "比較"],
            "使い方": ["利用法", "活用法", "操作法", "使用法"],
            "意味": ["定義", "概念", "内容", "説明"],
        }

    def _generate_synonyms(self, query: str) -> list[str]:
        """Generate synonyms for query terms."""
        synonyms = []

        for term, synonym_list in self.synonym_map.items():
            if term in query:
                for synonym in synonym_list:
                    expanded_query = query.replace(term, synonym)
                    if expanded_query != query:
                        synonyms.append(expanded_query)

        return synonyms

    def _extract_entities_and_concepts(self, query: str) -> dict[str, list[str]]:
        """Extract entities and concepts from query."""
        # Simple pattern-based extraction for Japanese
        entities = {
            "nouns": re.findall(r"[一-龯]+", query),  # Kanji compounds
            "actions": re.findall(
                r"[ぁ-ん]+る|[ぁ-ん]+す|[ぁ-ん]+た", query
            ),  # Verb patterns
            "adjectives": re.findall(
                r"[ぁ-ん]+い|[ぁ-ん]+な", query
            ),  # Adjective patterns
        }

        # Filter out single characters and common words
        for key in entities:
            entities[key] = [term for term in entities[key] if len(term) > 1]

        return entities

    async def expand_query(self, original_query: str) -> dict[str, Any]:
        """Expand query using multiple strategies."""
        self.logger.info("[QueryExpander] Expanding query: %s", original_query[:50])

        # Generate synonym variations
        synonym_queries = self._generate_synonyms(original_query)
        if synonym_queries:
            self.logger.info(
                "[QueryExpander] Generated %d synonym queries: %s", 
                len(synonym_queries),
                ", ".join(synonym_queries[:3])
            )

        # Extract entities and concepts
        entities = self._extract_entities_and_concepts(original_query)

        # Generate LLM-based expansion
        llm_expanded = await self._llm_expand_query(original_query)
        if llm_expanded:
            self.logger.info("[QueryExpander] LLM expanded query: %s", llm_expanded)

        search_queries: list[str] = [original_query] + synonym_queries[:2]
        expansion_result = {
            "original_query": original_query,
            "synonym_queries": synonym_queries[:3],  # Limit to top 3
            "entities": entities,
            "llm_expanded": llm_expanded,
            "search_queries": search_queries,  # For search
        }

        if llm_expanded:
            search_queries.append(llm_expanded)

        self.logger.info(
            "[QueryExpander] Generated %d search queries: %s",
            len(search_queries),
            "; ".join(search_queries)
        )

        return expansion_result

    async def _llm_expand_query(self, query: str) -> str:
        """Use LLM to expand and rephrase query."""
        try:
            expansion_prompt = (
                f"以下の質問を、より検索しやすい形に言い換えてください。"
                f"類義語や関連語を含めて、同じ意味の別の表現を1つ提案してください。\n\n"
                f"元の質問: {query}\n\n言い換えた質問:"
            )
            self.logger.debug(
                "[QueryExpander] LLM expansion prompt: %s", expansion_prompt
            )

            # Generate response using streaming
            full_response = ""
            async for token in self.llm_manager.generate_stream(expansion_prompt):
                full_response += token
                if len(full_response) > 200:  # Limit response length
                    break

            # Clean up response
            expanded = full_response.strip()
            if expanded and expanded != query:
                result = expanded[:100]  # Limit length
                self.logger.info("[QueryExpander] LLM generated expansion: %s", result)
                return result

        except Exception as e:
            self.logger.error("[QueryExpander] LLM expansion failed: %s", e)

        return ""

    def analyze_query_intent(self, query: str) -> dict[str, Any]:
        """Analyze query intent and type."""
        intent_analysis = {
            "query_type": "general",
            "intent": "information",
            "specificity": "medium",
            "temporal": False,
            "comparison": False,
            "procedural": False,
            "factual": False,
        }

        query_lower = query.lower()

        # Detect query type
        if any(word in query_lower for word in ["どう", "なぜ", "何故", "理由"]):
            intent_analysis["query_type"] = "explanatory"
            intent_analysis["intent"] = "explanation"

        if any(
            word in query_lower for word in ["方法", "やり方", "手順", "どうやって"]
        ):
            intent_analysis["query_type"] = "procedural"
            intent_analysis["procedural"] = True
            intent_analysis["intent"] = "instruction"

        if any(word in query_lower for word in ["何", "どれ", "誰", "いつ", "どこ"]):
            intent_analysis["query_type"] = "factual"
            intent_analysis["factual"] = True
            intent_analysis["intent"] = "fact"

        if any(word in query_lower for word in ["違い", "比較", "差", "対比"]):
            intent_analysis["comparison"] = True
            intent_analysis["intent"] = "comparison"

        # Detect temporal aspects
        if any(
            word in query_lower
            for word in ["いつ", "時期", "期間", "前", "後", "昔", "今"]
        ):
            intent_analysis["temporal"] = True

        # Assess specificity
        specific_indicators = len(
            re.findall(r"[一-龯]{2,}", query)
        )  # Multi-character kanji terms
        if specific_indicators > 3:
            intent_analysis["specificity"] = "high"
        elif specific_indicators < 1:
            intent_analysis["specificity"] = "low"

        self.logger.info(
            "[QueryAnalysis] Query type: %s, Intent: %s, Specificity: %s",
            intent_analysis["query_type"],
            intent_analysis["intent"],
            intent_analysis["specificity"],
        )

        return intent_analysis


class AdaptiveRAGStrategy:
    """Adaptive RAG strategy based on query analysis."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def determine_search_strategy(
        self, query_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Determine optimal search strategy based on query analysis."""
        strategy = {
            "top_k": 10,
            "semantic_weight": 0.7,
            "keyword_weight": 0.3,
            "diversity_weight": 0.2,
            "use_reranking": True,
            "context_compression": True,
            "max_context_length": 8000,
        }

        query_type = query_analysis.get("query_type", "general")
        specificity = query_analysis.get("specificity", "medium")

        # Adjust strategy based on query type
        if query_type == "factual":
            # For factual queries, prioritize keyword matching
            strategy["semantic_weight"] = 0.5
            strategy["keyword_weight"] = 0.5
            strategy["top_k"] = 5
            strategy["diversity_weight"] = 0.1

        elif query_type == "procedural":
            # For procedural queries, need comprehensive context
            strategy["top_k"] = 15
            strategy["max_context_length"] = 10000
            strategy["diversity_weight"] = 0.3

        elif query_type == "explanatory":
            # For explanatory queries, prioritize semantic understanding
            strategy["semantic_weight"] = 0.8
            strategy["keyword_weight"] = 0.2
            strategy["diversity_weight"] = 0.2

        # Adjust based on specificity
        if specificity == "high":
            # High specificity: fewer but more relevant results
            strategy["top_k"] = min(strategy["top_k"], 8)
            strategy["keyword_weight"] += 0.1
            strategy["semantic_weight"] -= 0.1

        elif specificity == "low":
            # Low specificity: more diverse results
            strategy["top_k"] = max(strategy["top_k"], 12)
            strategy["diversity_weight"] += 0.1

        # Special handling for comparison queries
        if query_analysis.get("comparison", False):
            strategy["diversity_weight"] = 0.4  # High diversity for comparisons
            strategy["top_k"] = 15

        self.logger.info(
            "[Strategy] Determined strategy - top_k: %d, semantic: %.1f, keyword: %.1f",
            strategy["top_k"],
            strategy["semantic_weight"],
            strategy["keyword_weight"],
        )

        return strategy
