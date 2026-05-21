"""
utils/citation_builder.py
Extracts [1], [2]… references from the LLM's answer
and maps them back to the retrieved source documents.
"""
import re


def build_citations(answer: str, docs: list[dict]) -> list[dict]:
    """
    Scans the answer for [N] reference markers and builds a
    structured citations list.

    Returns: [{index, source, page, text_snippet, score}]
    """
    if not docs:
        return []

    # Find all unique [N] references in the answer
    refs = set(int(m) for m in re.findall(r"\[(\d+)\]", answer))

    citations = []
    for ref in sorted(refs):
        idx = ref - 1
        if 0 <= idx < len(docs):
            doc  = docs[idx]
            meta = doc.get("metadata", {})
            citations.append({
                "index":    ref,
                "source":   meta.get("source", "unknown"),
                "page":     meta.get("page", ""),
                "channel":  meta.get("channel", ""),
                "type":     meta.get("type", ""),
                "snippet":  doc["text"][:300] + ("…" if len(doc["text"]) > 300 else ""),
                "score":    round(doc.get("rerank_score", doc.get("score", 0.0)), 3),
            })

    # If no explicit references but docs exist, cite top-3 implicitly
    if not citations and docs:
        for i, doc in enumerate(docs[:3], 1):
            meta = doc.get("metadata", {})
            citations.append({
                "index":   i,
                "source":  meta.get("source", "unknown"),
                "page":    meta.get("page", ""),
                "channel": meta.get("channel", ""),
                "type":    meta.get("type", ""),
                "snippet": doc["text"][:300] + "…",
                "score":   round(doc.get("rerank_score", doc.get("score", 0.0)), 3),
            })

    return citations
