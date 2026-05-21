"""
evaluation/evaluator.py
Tracks every query+answer and computes RAG quality metrics.
Stores results in TinyDB. Dashboard reads from here.
"""
import json
from datetime import datetime
from pathlib import Path

from tinydb import TinyDB, Query
from loguru import logger

from config import BASE_DIR, EVAL_DIR

EVAL_DB_PATH = BASE_DIR / "eval_log.json"


class EvaluationLogger:
    """
    Logs every RAG interaction with auto-computed metrics.
    Also accepts user thumbs-up / thumbs-down feedback.
    """

    def __init__(self):
        self.db    = TinyDB(EVAL_DB_PATH)
        self.table = self.db.table("interactions")

    # ── Logging ───────────────────────────────────────────────────────────────

    def log(self, rag_result: dict) -> int:
        """
        Log a completed RAG interaction.
        Returns the record ID for later feedback updates.
        """
        record = {
            "ts":            datetime.utcnow().isoformat(),
            "session_id":    rag_result.get("session_id", ""),
            "user_id":       rag_result.get("user_id", ""),
            "query":         rag_result.get("query", ""),
            "answer":        rag_result.get("final_answer", ""),
            "confidence":    rag_result.get("confidence", 0.0),
            "hallucination": rag_result.get("hallucination", {}),
            "route":         rag_result.get("route_result", {}),
            "n_docs":        len(rag_result.get("retrieved_docs", [])),
            "elapsed_ms":    rag_result.get("elapsed_ms", 0),
            "citations":     rag_result.get("citations", []),
            "feedback":      None,   # filled by user later
            "feedback_text": "",
            "metrics":       self._compute_metrics(rag_result),
        }
        record_id = self.table.insert(record)
        return record_id

    # ── Feedback ──────────────────────────────────────────────────────────────

    def add_feedback(
        self,
        record_id: int,
        rating: int,       # 1 = thumbs up, -1 = thumbs down
        comment: str = "",
    ):
        self.table.update(
            {"feedback": rating, "feedback_text": comment},
            doc_ids=[record_id],
        )
        logger.info(f"Feedback recorded for record {record_id}: {rating}")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_all(self, limit: int = 500) -> list[dict]:
        all_records = self.table.all()
        return all_records[-limit:]

    def aggregate_stats(self) -> dict:
        records = self.table.all()
        if not records:
            return {}

        n            = len(records)
        confidences  = [r["confidence"] for r in records if r.get("confidence")]
        positives    = [r for r in records if r.get("feedback") == 1]
        negatives    = [r for r in records if r.get("feedback") == -1]
        hallucinated = [r for r in records if r.get("hallucination", {}).get("is_hallucinated")]
        latencies    = [r["elapsed_ms"] for r in records if r.get("elapsed_ms")]

        route_dist: dict[str, int] = {}
        for r in records:
            cat = r.get("route", {}).get("category", "unknown")
            route_dist[cat] = route_dist.get(cat, 0) + 1

        return {
            "total_queries":      n,
            "avg_confidence":     round(sum(confidences) / len(confidences), 3) if confidences else 0,
            "positive_feedback":  len(positives),
            "negative_feedback":  len(negatives),
            "feedback_rate":      round((len(positives) + len(negatives)) / n, 3),
            "thumbs_up_rate":     round(len(positives) / max(len(positives)+len(negatives), 1), 3),
            "hallucination_rate": round(len(hallucinated) / n, 3),
            "avg_latency_ms":     round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "route_distribution": route_dist,
        }

    def daily_stats(self) -> list[dict]:
        """Group stats by day for time-series charts."""
        records = self.table.all()
        by_day: dict[str, list] = {}
        for r in records:
            day = r["ts"][:10]
            by_day.setdefault(day, []).append(r)

        result = []
        for day in sorted(by_day.keys()):
            recs = by_day[day]
            confs = [r["confidence"] for r in recs if r.get("confidence")]
            result.append({
                "date":         day,
                "queries":      len(recs),
                "avg_conf":     round(sum(confs)/len(confs), 3) if confs else 0,
                "hallucinated": sum(1 for r in recs if r.get("hallucination",{}).get("is_hallucinated")),
            })
        return result

    def export_csv(self) -> str:
        """Export eval log as CSV string."""
        import csv
        import io
        records = self.table.all()
        if not records:
            return ""
        fields = ["ts","user_id","query","confidence","elapsed_ms",
                  "n_docs","feedback","feedback_text"]
        buf = io.StringIO()
        w   = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
        return buf.getvalue()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(result: dict) -> dict:
        docs    = result.get("retrieved_docs", [])
        h       = result.get("hallucination", {})
        answer  = result.get("final_answer", "")

        # Answer length score (penalise very short/very long)
        words   = len(answer.split())
        len_score = min(1.0, words / 80) if words < 80 else max(0.5, 1 - (words - 300)/1000)

        return {
            "faithfulness":     h.get("faithfulness_score", 0),
            "token_overlap":    h.get("token_overlap", 0),
            "n_sources":        len({d["metadata"].get("source") for d in docs}),
            "avg_rerank_score": round(
                sum(d.get("rerank_score", 0) for d in docs) / max(len(docs), 1), 3
            ),
            "answer_length":    words,
            "length_score":     round(len_score, 3),
        }
