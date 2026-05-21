"""
agents/router.py
LLM-powered query router.
Classifies the query → picks the right retrieval strategy.
"""
import json
import re
from loguru import logger

from config import QUERY_CATEGORIES, LLM_PROVIDER
from utils.llm_client import get_llm_response


ROUTER_PROMPT = """You are a query router for an enterprise knowledge base.

Classify the user's query into EXACTLY ONE category from this list:
- factual       -> asks for a specific fact, number, name, date, definition
- analytical    -> asks for analysis, insights, trends, root cause
- comparison    -> asks to compare two or more things
- procedural    -> asks how to do something step-by-step
- conversational -> small talk, greetings, follow-up without a topic
- out_of_scope  -> completely unrelated to company knowledge

Return ONLY valid JSON, nothing else. No explanation, no markdown, just JSON:
{"category": "factual", "reasoning": "one sentence", "needs_retrieval": true, "sub_queries": []}

User query: {query}
"""


class QueryRouter:
    def route(self, query: str) -> dict:
        try:
            raw    = get_llm_response(ROUTER_PROMPT.format(query=query),
                                      max_tokens=200, temperature=0.0)
            result = self._parse(raw)
        except Exception as e:
            logger.warning(f"Router failed ({type(e).__name__}). Using default.")
            result = self._default()

        result["strategy"] = self._pick_strategy(result["category"])
        logger.info(f"Router: '{result['category']}' → '{result['strategy']}'")
        return result

    @staticmethod
    def _parse(raw: str) -> dict:
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        # Try direct parse first
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try extracting the first JSON object
            match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON object found in: {raw[:80]!r}")
            data = json.loads(match.group())

        # Validate category
        valid = {"factual","analytical","comparison","procedural",
                 "conversational","out_of_scope"}
        category = data.get("category", "factual")
        if category not in valid:
            category = "factual"

        return {
            "category":        category,
            "reasoning":       data.get("reasoning", ""),
            "needs_retrieval": bool(data.get("needs_retrieval", True)),
            "sub_queries":     data.get("sub_queries", []),
        }

    @staticmethod
    def _default() -> dict:
        return {
            "category":        "factual",
            "reasoning":       "Default fallback",
            "needs_retrieval": True,
            "sub_queries":     [],
        }

    @staticmethod
    def _pick_strategy(category: str) -> str:
        return {
            "factual":        "single_retrieval",
            "analytical":     "retrieval_with_synthesis",
            "comparison":     "multi_retrieval",
            "procedural":     "retrieval_with_steps",
            "conversational": "memory_only",
            "out_of_scope":   "refuse",
        }.get(category, "single_retrieval")
