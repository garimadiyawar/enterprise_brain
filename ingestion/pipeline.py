"""
ingestion/pipeline.py
Unified ingestion for PDF / Slack / Email.
Handles chunking, embedding, and upsert in one shot.
"""
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config import CHUNK_SIZE, CHUNK_OVERLAP
from retrieval.embedder    import Embedder
from database.vector_store import VectorStore


Source = Literal["pdf", "slack", "email", "text"]

# ── Text Splitter ─────────────────────────────────────────────────────────────
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_pdf(path: str | Path) -> list[dict]:
    """Extract text pages from a PDF. Returns [{text, page, source}]."""
    import pdfplumber
    path = Path(path)
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({
                        "text":   text,
                        "page":   i + 1,
                        "source": path.name,
                        "type":   "pdf",
                    })
        logger.info(f"PDF loaded: {path.name}, {len(pages)} pages")
    except Exception as e:
        logger.error(f"PDF load error {path}: {e}")
    return pages


def load_slack_export(export_dir: str | Path) -> list[dict]:
    """
    Parse a Slack export directory (JSON files per channel).
    Returns [{text, channel, ts, user, source}]
    """
    import json
    export_dir = Path(export_dir)
    messages = []
    for json_file in export_dir.rglob("*.json"):
        channel = json_file.parent.name
        try:
            data = json.loads(json_file.read_text())
            for msg in data:
                if msg.get("type") == "message" and msg.get("text"):
                    messages.append({
                        "text":    _clean_slack(msg["text"]),
                        "channel": channel,
                        "ts":      msg.get("ts", ""),
                        "user":    msg.get("user", "unknown"),
                        "source":  f"slack/{channel}/{json_file.name}",
                        "type":    "slack",
                    })
        except Exception as e:
            logger.warning(f"Slack parse error {json_file}: {e}")

    logger.info(f"Slack export loaded: {len(messages)} messages")
    return messages


def load_slack_live(channels: list[str]) -> list[dict]:
    """Fetch messages live from Slack API."""
    from slack_sdk import WebClient
    from config import SLACK_BOT_TOKEN

    client   = WebClient(token=SLACK_BOT_TOKEN)
    messages = []
    for channel in channels:
        try:
            resp = client.conversations_history(channel=channel.strip(), limit=200)
            for msg in resp.get("messages", []):
                if msg.get("text"):
                    messages.append({
                        "text":    _clean_slack(msg["text"]),
                        "channel": channel,
                        "ts":      msg.get("ts", ""),
                        "user":    msg.get("user", "unknown"),
                        "source":  f"slack/{channel}",
                        "type":    "slack",
                    })
        except Exception as e:
            logger.warning(f"Slack live error for {channel}: {e}")
    return messages


def load_emails(max_emails: int = 200) -> list[dict]:
    """Fetch emails via IMAP. Returns [{subject, from, date, body, source}]"""
    import imapclient
    import email as email_lib
    from config import EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_FOLDER

    if not EMAIL_USER or not EMAIL_PASS:
        logger.warning("Email credentials not configured.")
        return []

    records = []
    try:
        with imapclient.IMAPClient(EMAIL_HOST, port=EMAIL_PORT, ssl=True) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.select_folder(EMAIL_FOLDER)
            uids = server.search(["ALL"])[-max_emails:]
            raw  = server.fetch(uids, ["RFC822"])
            for uid, data in raw.items():
                msg = email_lib.message_from_bytes(data[b"RFC822"])
                body = _extract_email_body(msg)
                if body.strip():
                    records.append({
                        "text":    f"Subject: {msg['subject']}\nFrom: {msg['from']}\n\n{body}",
                        "subject": msg.get("subject", ""),
                        "from":    msg.get("from", ""),
                        "date":    msg.get("date", ""),
                        "source":  f"email/{uid}",
                        "type":    "email",
                    })
    except Exception as e:
        logger.error(f"Email ingest error: {e}")

    logger.info(f"Emails loaded: {len(records)}")
    return records


# ── Chunker ───────────────────────────────────────────────────────────────────

def chunk_documents(docs: list[dict]) -> list[dict]:
    """
    Split raw document dicts into smaller chunks.
    Preserves all metadata; adds chunk_index.
    """
    chunks = []
    for doc in docs:
        text   = doc.get("text", "")
        splits = SPLITTER.split_text(text)
        meta   = {k: v for k, v in doc.items() if k != "text"}
        for i, split in enumerate(splits):
            chunks.append({
                "text":        split,
                "chunk_index": i,
                **meta,
            })
    return chunks


# ── Master Ingestion ──────────────────────────────────────────────────────────

class IngestionPipeline:
    def __init__(self, collection_name: str | None = None):
        self.embedder     = Embedder()
        self.vector_store = VectorStore(collection_name) if collection_name \
                            else VectorStore()

    def ingest(self, docs: list[dict], source_type: Source = "text") -> dict:
        """
        Chunk → embed → upsert.
        Returns ingestion stats.
        """
        t0     = time.time()
        chunks = chunk_documents(docs)
        if not chunks:
            return {"chunks": 0, "elapsed": 0}

        texts     = [c["text"] for c in chunks]
        metadatas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]

        logger.info(f"Embedding {len(texts)} chunks…")
        embeddings = self.embedder.embed_documents(texts)

        n = self.vector_store.add_documents(texts, embeddings, metadatas)

        elapsed = round(time.time() - t0, 2)
        logger.info(f"Ingested {n} chunks in {elapsed}s")
        return {"chunks": n, "elapsed": elapsed, "source_type": source_type}

    def ingest_pdf(self, path: str | Path) -> dict:
        docs = load_pdf(path)
        return self.ingest(docs, "pdf")

    def ingest_slack_export(self, export_dir: str | Path) -> dict:
        docs = load_slack_export(export_dir)
        return self.ingest(docs, "slack")

    def ingest_slack_live(self, channels: list[str]) -> dict:
        docs = load_slack_live(channels)
        return self.ingest(docs, "slack")

    def ingest_emails(self) -> dict:
        docs = load_emails()
        return self.ingest(docs, "email")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_slack(text: str) -> str:
    text = re.sub(r"<@[A-Z0-9]+>", "[user]", text)
    text = re.sub(r"<#[A-Z0-9]+\|([^>]+)>", r"#\1", text)
    text = re.sub(r"<https?://[^>]+>", "[link]", text)
    return text.strip()


def _extract_email_body(msg) -> str:
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                try:
                    body += part.get_payload(decode=True).decode("utf-8", errors="replace")
                except Exception:
                    pass
    else:
        try:
            body = msg.get_payload(decode=True).decode("utf-8", errors="replace")
        except Exception:
            pass
    return body
