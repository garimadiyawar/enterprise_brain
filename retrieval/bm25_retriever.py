"""
retrieval/bm25_retriever.py
In-memory BM25 index built from ChromaDB corpus.
Rebuilt automatically when new documents are ingested.
"""
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from loguru import logger

from config import BASE_DIR


class BM25Retriever:
    """
    BM25 sparse retriever.
    Persists the index to disk so it survives restarts.
    """

    INDEX_PATH = BASE_DIR / "bm25_index.pkl"

    def __init__(self):
        self.corpus_docs: list[dict] = []   # [{text, metadata, id}]
        self.bm25: BM25Okapi | None  = None
        self._load_from_disk()

    # ── Build / Rebuild ───────────────────────────────────────────────────────

    def build(self, docs: list[dict]):
        """
        (Re)build the BM25 index from a list of {text, metadata, id} dicts.
        Call this after every ingestion batch.
        """
        if not docs:
            logger.warning("BM25: empty corpus, skipping build.")
            return

        self.corpus_docs = docs
        tokenized = [self._tokenize(d["text"]) for d in docs]
        self.bm25  = BM25Okapi(tokenized)
        self._save_to_disk()
        logger.info(f"BM25 index built: {len(docs)} documents")

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Returns list of {text, metadata, score, id}
        Scores are normalised to [0, 1].
        """
        if self.bm25 is None or not self.corpus_docs:
            logger.warning("BM25 index not built yet. Returning empty results.")
            return []

        tokens = self._tokenize(query)
        raw_scores = self.bm25.get_scores(tokens)

        # Normalize
        max_score = raw_scores.max()
        if max_score > 0:
            norm_scores = raw_scores / max_score
        else:
            norm_scores = raw_scores

        top_indices = np.argsort(norm_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if norm_scores[idx] > 0:
                results.append({
                    "text":     self.corpus_docs[idx]["text"],
                    "metadata": self.corpus_docs[idx]["metadata"],
                    "score":    float(norm_scores[idx]),
                    "id":       self.corpus_docs[idx]["id"],
                })

        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_to_disk(self):
        with open(self.INDEX_PATH, "wb") as f:
            pickle.dump({"corpus": self.corpus_docs, "bm25": self.bm25}, f)

    def _load_from_disk(self):
        if self.INDEX_PATH.exists():
            try:
                with open(self.INDEX_PATH, "rb") as f:
                    data = pickle.load(f)
                self.corpus_docs = data["corpus"]
                self.bm25        = data["bm25"]
                logger.info(f"BM25 index loaded from disk: {len(self.corpus_docs)} docs")
            except Exception as e:
                logger.warning(f"BM25 load failed: {e}. Will rebuild on next ingest.")

    # ── Utils ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()
