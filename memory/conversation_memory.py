"""
memory/conversation_memory.py
Per-session sliding window memory stored in TinyDB.
Survives restarts (disk-backed).
"""
import json
from datetime import datetime
from pathlib import Path

from tinydb import TinyDB, Query
from loguru import logger

from config import BASE_DIR, MEMORY_MAX_TURNS

MEMORY_DB_PATH = BASE_DIR / "memory.json"


class ConversationMemory:
    """
    Session-scoped conversation memory.
    Each session_id gets its own turn history.
    Max MEMORY_MAX_TURNS retained (sliding window).
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.db         = TinyDB(MEMORY_DB_PATH)
        self.table      = self.db.table("sessions")

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_turn(self, user_msg: str, assistant_msg: str):
        Session = Query()
        existing = self.table.search(Session.session_id == self.session_id)

        turn = {
            "user":      user_msg,
            "assistant": assistant_msg,
            "ts":        datetime.utcnow().isoformat(),
        }

        if existing:
            record = existing[0]
            turns  = record.get("turns", [])
            turns.append(turn)
            # Sliding window
            turns = turns[-MEMORY_MAX_TURNS:]
            self.table.update(
                {"turns": turns},
                Session.session_id == self.session_id,
            )
        else:
            self.table.insert({
                "session_id": self.session_id,
                "turns":      [turn],
                "created_at": datetime.utcnow().isoformat(),
            })

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_turns(self) -> list[dict]:
        Session  = Query()
        existing = self.table.search(Session.session_id == self.session_id)
        if not existing:
            return []
        return existing[0].get("turns", [])

    def get_history_text(self) -> str:
        """Format conversation history as a plain-text block for LLM prompts."""
        turns = self.get_turns()
        if not turns:
            return ""
        lines = []
        for t in turns[-6:]:   # last 6 turns max in prompt
            lines.append(f"User: {t['user']}")
            lines.append(f"Assistant: {t['assistant']}")
        return "\n".join(lines)

    def get_history_messages(self) -> list[dict]:
        """OpenAI-style message list (for models that take message arrays)."""
        turns = self.get_turns()
        msgs  = []
        for t in turns:
            msgs.append({"role": "user",      "content": t["user"]})
            msgs.append({"role": "assistant", "content": t["assistant"]})
        return msgs

    # ── Clear ─────────────────────────────────────────────────────────────────

    def clear(self):
        Session = Query()
        self.table.remove(Session.session_id == self.session_id)
        logger.info(f"Memory cleared for session: {self.session_id}")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        turns = self.get_turns()
        return {
            "session_id": self.session_id,
            "turn_count": len(turns),
            "first_ts":   turns[0]["ts"]  if turns else None,
            "last_ts":    turns[-1]["ts"] if turns else None,
        }
