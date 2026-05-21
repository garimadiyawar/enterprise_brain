"""
retrieval/hybrid_retriever.py
Reciprocal Rank Fusion (RRF) of dense + BM25, then BGE-reranker.
This is the secret sauce — beats vanilla RAG significantly.
"""
from collections import defaultdict

from loguru import logger

from config import (
    TOP_K_DENSE, TOP_K_BM25, TOP_K_RERANK,
    DENSE_WEIGHT, BM25_WEIGHT,
)
from retrieval.embedder       import Embedder
from retrieval.bm25_retriever import BM25Retriever
from retrieval.reranker        import Reranker
from database.vector_store     import VectorStore


class HybridRetriever:
    """
    Pipeline:
        query
          ├─ dense retrieval (ChromaDB)        → top-K dense candidates
          ├─ sparse retrieval (BM25)            → top-K sparse candidates
          └─ RRF fusion + BGE-reranker          → top-K final results
    """

    def __init__(self, collection_name: str | None = None):
        self.embedder     = Embedder()
        self.bm25         = BM25Retriever()
        self.reranker     = Reranker()
        self.vector_store = VectorStore(collection_name) if collection_name \
                            else VectorStore()

    # ── Public ────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RERANK,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Returns top_k reranked chunks as list of:
        {text, metadata, score, id, dense_rank, bm25_rank, rrf_score}
        """
        # 1. Dense retrieval
        q_emb   = self.embedder.embed_query(query)
        dense   = self.vector_store.query(q_emb, top_k=TOP_K_DENSE, where=filters)
        logger.debug(f"Dense: {len(dense)} results")

        # 2. BM25 retrieval
        sparse  = self.bm25.query(query, top_k=TOP_K_BM25)
        logger.debug(f"BM25:  {len(sparse)} results")

        # 3. RRF fusion
        fused   = self._rrf_fusion(dense, sparse)
        logger.debug(f"Fused: {len(fused)} results")

        if not fused:
            return []

        # 4. Rerank
        reranked = self.reranker.rerank(query, fused, top_k=top_k)
        return reranked

    def rebuild_bm25(self):
        """Call after ingesting new documents."""
        all_docs = self.vector_store.get_all_texts()
        self.bm25.build(all_docs)

    # ── RRF ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _rrf_fusion(
        dense: list[dict],
        sparse: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion.
        score = Σ weight_i / (k + rank_i)
        """
        scores: dict[str, float]  = defaultdict(float)
        doc_map: dict[str, dict]  = {}

        for rank, doc in enumerate(dense, 1):
            doc_id = doc["id"]
            scores[doc_id]  += DENSE_WEIGHT / (k + rank)
            doc_map[doc_id]  = {**doc, "dense_rank": rank}

        for rank, doc in enumerate(sparse, 1):
            doc_id = doc["id"]
            scores[doc_id]  += BM25_WEIGHT / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = {**doc, "dense_rank": None}
            doc_map[doc_id]["bm25_rank"] = rank

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        fused = []
        for _id in sorted_ids:
            d = doc_map[_id]
            d["rrf_score"] = scores[_id]
            fused.append(d)

        return fused
