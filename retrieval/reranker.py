"""
retrieval/reranker.py
Cross-encoder reranking via BAAI/bge-reranker-base.
Dramatically improves precision over the pre-filtered candidates.
"""
from loguru import logger
from config import RERANK_MODEL, TOP_K_RERANK


class Reranker:
    """
    FlagEmbedding's FlagReranker (cross-encoder).
    Scores every (query, passage) pair and re-orders.
    """

    _instance: "Reranker | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def _load(self):
        if not self._loaded:
            try:
                from FlagEmbedding import FlagReranker
                logger.info(f"Loading reranker: {RERANK_MODEL}")
                self.model = FlagReranker(RERANK_MODEL, use_fp16=False)
                self._loaded = True
                logger.info("Reranker ready.")
            except ImportError:
                logger.warning(
                    "FlagEmbedding not installed. Falling back to score passthrough."
                )
                self.model = None
                self._loaded = True

    def rerank(
        self,
        query: str,
        docs: list[dict],
        top_k: int = TOP_K_RERANK,
    ) -> list[dict]:
        """
        Rerank docs using cross-encoder scores.
        Returns top_k docs sorted by reranker score (high → low).
        Each doc gets a `rerank_score` field added.
        """
        self._load()

        if not docs:
            return []

        if self.model is None:
            # Graceful degradation: return by RRF score
            for d in docs:
                d["rerank_score"] = d.get("rrf_score", d.get("score", 0.0))
            return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

        pairs  = [[query, d["text"]] for d in docs]
        scores = self.model.compute_score(pairs)

        if isinstance(scores, float):
            scores = [scores]

        for doc, sc in zip(docs, scores):
            doc["rerank_score"] = float(sc)

        ranked = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]
