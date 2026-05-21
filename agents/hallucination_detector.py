"""
agents/hallucination_detector.py

Two-stage detection:
  1. NLI-style prompt — asks the LLM to score faithfulness.
  2. Token overlap heuristic as a fast sanity check.

Returns a confidence score [0.0, 1.0] and a binary verdict.
"""
import re
import json
import math
from loguru import logger

from config import HALLUCINATION_THRESHOLD
from utils.llm_client import get_llm_response


FAITHFULNESS_PROMPT = """You are an expert fact-checker for an enterprise AI system.

Given the CONTEXT (retrieved documents) and the AI ANSWER, determine whether
every claim in the answer is fully supported by the context.

Return ONLY a JSON object:
{{
  "faithfulness_score": <float 0.0 to 1.0>,
  "unsupported_claims": ["<any claims NOT found in context>"],
  "verdict": "faithful" | "hallucinated" | "partial"
}}

Rules:
- 1.0 = every single claim is directly supported by context
- 0.0 = answer contradicts or fabricates information
- Partial = some claims supported, some not

CONTEXT:
{context}

AI ANSWER:
{answer}
"""


class HallucinationDetector:
    """
    Detects whether an LLM answer is grounded in the retrieved context.
    """

    def detect(
        self,
        answer: str,
        context_docs: list[dict],
    ) -> dict:
        """
        Returns:
        {
            faithfulness_score: float,   # 0–1
            confidence: float,           # composite score
            verdict: str,                # faithful | hallucinated | partial
            unsupported_claims: list,
            token_overlap: float,        # fast heuristic
            is_hallucinated: bool,
        }
        """
        if not context_docs or not answer.strip():
            return self._unknown()

        context_text = "\n\n---\n\n".join(
            d["text"] for d in context_docs[:5]
        )

        # Stage 1: LLM faithfulness check
        llm_result = self._llm_check(answer, context_text)

        # Stage 2: Token overlap heuristic
        overlap = self._token_overlap(answer, context_text)

        # Composite confidence
        confidence = round(
            0.7 * llm_result["faithfulness_score"] + 0.3 * overlap, 3
        )

        is_hallucinated = (
            llm_result["verdict"] == "hallucinated"
            or confidence < HALLUCINATION_THRESHOLD
        )

        return {
            "faithfulness_score": llm_result["faithfulness_score"],
            "confidence":         confidence,
            "verdict":            llm_result["verdict"],
            "unsupported_claims": llm_result["unsupported_claims"],
            "token_overlap":      round(overlap, 3),
            "is_hallucinated":    is_hallucinated,
        }

    # ── LLM Check ─────────────────────────────────────────────────────────────

    def _llm_check(self, answer: str, context: str) -> dict:
        prompt = FAITHFULNESS_PROMPT.format(
            context=context[:4000],
            answer=answer[:2000],
        )
        try:
            raw    = get_llm_response(prompt, max_tokens=512, temperature=0.0)
            match  = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("No JSON in response")
            data = json.loads(match.group())
            return {
                "faithfulness_score": float(data.get("faithfulness_score", 0.5)),
                "unsupported_claims": data.get("unsupported_claims", []),
                "verdict":            data.get("verdict", "partial"),
            }
        except Exception as e:
            logger.warning(f"Hallucination LLM check failed: {e}")
            return {
                "faithfulness_score": 0.5,
                "unsupported_claims": [],
                "verdict": "partial",
            }

    # ── Token Overlap Heuristic ───────────────────────────────────────────────

    @staticmethod
    def _token_overlap(answer: str, context: str) -> float:
        """
        Jaccard-like token overlap between answer and context.
        Fast and language-model-free.
        """
        def tokens(text: str) -> set:
            words = re.findall(r"\b[a-z0-9]{3,}\b", text.lower())
            # Remove stopwords
            stops = {"the","and","for","are","was","with","that","this",
                     "have","from","they","will","been","its","has"}
            return set(w for w in words if w not in stops)

        ans_tokens = tokens(answer)
        ctx_tokens = tokens(context)

        if not ans_tokens:
            return 0.5

        overlap = ans_tokens & ctx_tokens
        return len(overlap) / len(ans_tokens)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _unknown() -> dict:
        return {
            "faithfulness_score": 1.0,
            "confidence":         1.0,
            "verdict":            "faithful",
            "unsupported_claims": [],
            "token_overlap":      1.0,
            "is_hallucinated":    False,
        }
