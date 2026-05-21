"""
database/vector_store.py
ChromaDB wrapper — persistent, multi-collection, metadata-rich.
"""
import json
import hashlib
from typing import Any
from pathlib import Path

import chromadb
from chromadb.config import Settings
from loguru import logger

from config import CHROMA_DIR, CHROMA_COLLECTION


class VectorStore:
    """
    Thin wrapper over ChromaDB.
    Supports multiple named collections (for RBAC) and
    stores rich metadata alongside every chunk.
    """

    def __init__(self, collection_name: str = CHROMA_COLLECTION):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"VectorStore ready: collection='{collection_name}', "
                    f"docs={self.collection.count()}")

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        ids: list[str] | None = None,
    ) -> int:
        """Upsert chunks. Returns number of new chunks added."""
        if ids is None:
            ids = [self._make_id(t, m) for t, m in zip(texts, metadatas)]

        # Chroma expects metadata values to be str/int/float/bool
        clean_meta = [self._clean_metadata(m) for m in metadatas]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=clean_meta,
        )
        logger.debug(f"Upserted {len(texts)} chunks into '{self.collection_name}'")
        return len(texts)

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Dense retrieval.
        Returns list of dicts: {text, metadata, score, id}
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, max(self.collection.count(), 1)),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        docs = []
        for doc, meta, dist, _id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            docs.append({
                "text":     doc,
                "metadata": meta,
                "score":    float(1 - dist),   # cosine distance → similarity
                "id":       _id,
            })
        return docs

    def get_all_texts(self, where: dict | None = None) -> list[dict]:
        """Fetch all stored texts (for BM25 index rebuild)."""
        kwargs: dict[str, Any] = {
            "include": ["documents", "metadatas"],
            "limit": 100_000,
        }
        if where:
            kwargs["where"] = where

        results = self.collection.get(**kwargs)
        return [
            {"text": doc, "metadata": meta, "id": _id}
            for doc, meta, _id in zip(
                results["documents"],
                results["metadatas"],
                results["ids"],
            )
        ]

    def count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> list[str]:
        """Return unique source filenames in the collection."""
        all_docs = self.collection.get(include=["metadatas"])
        sources = {m.get("source", "unknown") for m in all_docs["metadatas"]}
        return sorted(sources)

    def delete_source(self, source: str):
        """Delete all chunks belonging to a source document."""
        self.collection.delete(where={"source": source})
        logger.info(f"Deleted chunks for source: {source}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_id(text: str, metadata: dict) -> str:
        key = text[:128] + str(metadata.get("source", "")) + str(metadata.get("chunk_index", 0))
        return hashlib.md5(key.encode()).hexdigest()

    @staticmethod
    def _clean_metadata(meta: dict) -> dict:
        """ChromaDB only accepts primitive types as metadata values."""
        clean = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                clean[k] = json.dumps(v)
        return clean
