"""
retrieval/embedder.py
Dense embeddings via BAAI/bge-small-en-v1.5 (runs on CPU).
Falls back to TF-IDF when HuggingFace is unreachable (air-gapped / sandboxed env).
"""
from typing import Union
from loguru import logger
from config import EMBED_MODEL


class Embedder:
    """
    Singleton embedding wrapper.
    Primary  : BAAI/bge-small-en-v1.5 via SentenceTransformers
    Fallback : TF-IDF (sklearn) — fully offline, lower quality
    """

    _instance: "Embedder | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    # ── Load ──────────────────────────────────────────────────────────────────

    def _load(self):
        if self._loaded:
            return
        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        try:
            from sentence_transformers import SentenceTransformer
            self.model      = SentenceTransformer(EMBED_MODEL)
            self.dim        = self.model.get_sentence_embedding_dimension()
            self._use_tfidf = False
            logger.info(f"SentenceTransformer ready. dim={self.dim}")
        except Exception as e:
            logger.warning(
                f"Cannot load SentenceTransformer ({type(e).__name__}: {e}). "
                "Falling back to TF-IDF (offline-safe, lower accuracy)."
            )
            self._init_tfidf()
            self._use_tfidf = True
        self._loaded = True

    def _init_tfidf(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf         = TfidfVectorizer(max_features=384, sublinear_tf=True)
        self._tfidf_fitted  = False
        self._corpus: list[str] = []
        self.dim            = 384
        logger.info("TF-IDF fallback embedder ready. dim=384")

    # ── Embed ─────────────────────────────────────────────────────────────────

    def embed(self, texts: Union[str, list[str]]) -> list[list[float]]:
        self._load()
        if isinstance(texts, str):
            texts = [texts]
        return self._tfidf_embed(texts) if self._use_tfidf else self._st_embed(texts)

    def embed_query(self, query: str) -> list[float]:
        self._load()
        if self._use_tfidf:
            return self._tfidf_embed([query])[0]
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        import numpy as np
        vec = self.model.encode([prefixed], normalize_embeddings=True,
                                show_progress_bar=False)
        return vec[0].tolist()

    def embed_documents(self, docs: list[str]) -> list[list[float]]:
        return self.embed(docs)

    # ── Backends ──────────────────────────────────────────────────────────────

    def _st_embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True,
                                 show_progress_bar=False, batch_size=64)
        return vecs.tolist()

    def _tfidf_embed(self, texts: list[str]) -> list[list[float]]:
        import numpy as np
        self._corpus.extend(texts)
        try:
            if not self._tfidf_fitted:
                raise ValueError("not fitted")
            mat = self._tfidf.transform(texts).toarray().astype("float32")
        except Exception:
            self._tfidf.fit(self._corpus)
            self._tfidf_fitted = True
            mat = self._tfidf.transform(texts).toarray().astype("float32")
        # Zero-pad to always output exactly max_features=384 dims (ChromaDB needs consistent dim)
        if mat.shape[1] < 384:
            pad = np.zeros((mat.shape[0], 384 - mat.shape[1]), dtype="float32")
            mat = np.hstack([mat, pad])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (mat / norms).tolist()
