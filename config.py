"""
Enterprise Brain — Central Configuration
All tunable knobs in one place.
"""
import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
DATA_DIR       = BASE_DIR / "data"
CHROMA_DIR     = BASE_DIR / "chroma_db"
LOGS_DIR       = BASE_DIR / "logs"
UPLOADS_DIR    = DATA_DIR / "uploads"
EVAL_DIR       = BASE_DIR / "evaluation" / "results"

for d in [DATA_DIR, CHROMA_DIR, LOGS_DIR, UPLOADS_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "ollama")   # "ollama" | "anthropic"
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "llama3")    # llama3 / qwen2 / mistral
OLLAMA_BASE_URL= os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBED_MODEL    = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL   = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

# ── Retrieval ────────────────────────────────────────────────────────────────
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "64"))
TOP_K_DENSE    = int(os.getenv("TOP_K_DENSE", "20"))
TOP_K_BM25     = int(os.getenv("TOP_K_BM25", "20"))
TOP_K_RERANK   = int(os.getenv("TOP_K_RERANK", "5"))
DENSE_WEIGHT   = float(os.getenv("DENSE_WEIGHT", "0.6"))
BM25_WEIGHT    = float(os.getenv("BM25_WEIGHT", "0.4"))

# ── Hallucination Detection ───────────────────────────────────────────────────
HALLUCINATION_THRESHOLD = float(os.getenv("HALLUCINATION_THRESHOLD", "0.75"))
CONFIDENCE_THRESHOLD    = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

# ── Slack ─────────────────────────────────────────────────────────────────────
SLACK_BOT_TOKEN  = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_CHANNELS   = os.getenv("SLACK_CHANNELS", "").split(",")

# ── Email (IMAP) ──────────────────────────────────────────────────────────────
EMAIL_HOST     = os.getenv("EMAIL_HOST", "imap.gmail.com")
EMAIL_PORT     = int(os.getenv("EMAIL_PORT", "993"))
EMAIL_USER     = os.getenv("EMAIL_USER", "")
EMAIL_PASS     = os.getenv("EMAIL_PASS", "")
EMAIL_FOLDER   = os.getenv("EMAIL_FOLDER", "INBOX")
EMAIL_LIMIT    = int(os.getenv("EMAIL_LIMIT", "200"))

# ── Auth ──────────────────────────────────────────────────────────────────────
JWT_SECRET     = os.getenv("JWT_SECRET", "change-me-in-production-please")
JWT_EXPIRE_MIN = int(os.getenv("JWT_EXPIRE_MIN", "480"))

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "enterprise_brain"

# ── Roles & Permissions ───────────────────────────────────────────────────────
ROLES = {
    "admin":     {"can_ingest": True,  "can_delete": True,  "can_eval": True,  "collections": ["*"]},
    "manager":   {"can_ingest": True,  "can_delete": False, "can_eval": True,  "collections": ["*"]},
    "analyst":   {"can_ingest": False, "can_delete": False, "can_eval": False, "collections": ["public", "reports"]},
    "viewer":    {"can_ingest": False, "can_delete": False, "can_eval": False, "collections": ["public"]},
}

# ── Memory ───────────────────────────────────────────────────────────────────
MEMORY_MAX_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "10"))

# ── Query Router Categories ───────────────────────────────────────────────────
QUERY_CATEGORIES = [
    "factual",          # → RAG
    "analytical",       # → RAG + synthesis
    "comparison",       # → multi-doc RAG
    "procedural",       # → RAG + step formatting
    "conversational",   # → memory only
    "out_of_scope",     # → polite refusal
]
