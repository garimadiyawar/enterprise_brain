# 🧠 Enterprise Brain

> **Multi-Agent RAG Knowledge Base** — PDF + Slack + Email ingestion,  
> hybrid BM25+dense retrieval, hallucination detection, RBAC, memory, eval dashboard.  
> **Cost: ₹0** — runs entirely on free, local tooling.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                       │
│   Chat │ Ingest │ Dashboard │ Admin                         │
└────────────────┬────────────────────────────────────────────┘
                 │
         ┌───────▼────────┐
         │   RAG Agent    │  ← LangGraph orchestration
         │   (LangGraph)  │
         └─┬──────────┬───┘
           │          │
   ┌───────▼──┐  ┌────▼──────────┐
   │  Router  │  │ HybridRetriever│
   │(classify)│  │  BM25 + Dense  │
   └──────────┘  │  + Reranker    │
                 └────────┬───────┘
                          │
                 ┌────────▼───────┐
                 │  ChromaDB      │  ← persistent vector store
                 │  + BM25 index  │
                 └────────────────┘
                          │
         ┌────────────────▼────────────┐
         │  Hallucination Detector     │
         │  LLM NLI + token overlap    │
         └─────────────────────────────┘
```

## Features

| Feature | Implementation |
|---|---|
| PDF ingestion | pdfplumber → chunked → BAAI/bge-small embeddings |
| Slack ingestion | Export JSON parser + live Slack SDK |
| Email ingestion | IMAP (imapclient) → text extraction |
| Dense retrieval | ChromaDB cosine similarity |
| Sparse retrieval | BM25Okapi (rank-bm25) |
| Hybrid fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | BAAI/bge-reranker-base cross-encoder |
| LLM | Ollama (Llama 3 / Qwen2 / Mistral) — free, local |
| Hallucination detection | LLM NLI check + Jaccard token overlap |
| Query routing | LLM classifier → 6 strategies |
| Memory | TinyDB-backed sliding window per session |
| RBAC | bcrypt + JWT, 4 roles (admin/manager/analyst/viewer) |
| Citations | [N] reference extraction → source mapping |
| Evaluation | Per-query metrics, feedback, Plotly dashboard |

## Quickstart

### 1. Prerequisites

```bash
# Python 3.11+
python -m venv venv && source venv/bin/activate

# Install Ollama (free local LLM)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3        # ~4GB — or try qwen2:7b, mistral
```

### 2. Install

```bash
git clone <this-repo>
cd enterprise_brain

pip install -r requirements.txt

cp .env.example .env
# Edit .env — at minimum: LLM_PROVIDER=ollama, OLLAMA_MODEL=llama3
```

### 3. Run

```bash
# Make sure Ollama is running
ollama serve &

# Launch the app
streamlit run app.py
```

Open `http://localhost:8501`  
Default login: **admin / admin123** ← change immediately!

---

## Roles & Permissions

| Role | Ingest | Delete | Eval Dashboard | Collections |
|---|---|---|---|---|
| `admin` | ✅ | ✅ | ✅ | All |
| `manager` | ✅ | ❌ | ✅ | All |
| `analyst` | ❌ | ❌ | ❌ | public, reports |
| `viewer` | ❌ | ❌ | ❌ | public |

---

## Ingestion Examples

```python
from ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline()

# PDF
pipeline.ingest_pdf("docs/employee_handbook.pdf")

# Slack export
pipeline.ingest_slack_export("/exports/slack_export_2024")

# Email
pipeline.ingest_emails()   # reads creds from .env
```

---

## Query the Agent Directly

```python
from agents.rag_agent import RAGAgent

agent = RAGAgent()
result = agent.ask(
    query      = "What is our parental leave policy?",
    user_id    = "alice",
    session_id = "session-abc",
    role       = "analyst",
)

print(result["final_answer"])
print(f"Confidence: {result['confidence']:.0%}")
print(f"Hallucinated: {result['hallucination']['is_hallucinated']}")
for c in result["citations"]:
    print(f"  [{c['index']}] {c['source']} — {c['snippet'][:80]}")
```

---

## Swap the LLM

```bash
# Free alternatives (set in .env):
OLLAMA_MODEL=qwen2:7b        # Better reasoning
OLLAMA_MODEL=mistral         # Fast and compact
OLLAMA_MODEL=llama3:70b      # Highest quality (needs 40GB RAM)
OLLAMA_MODEL=phi3:mini       # Runs on 8GB RAM

# Cloud fallback:
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

---

## File Structure

```
enterprise_brain/
├── app.py                   # Streamlit frontend (4 pages)
├── config.py                # All tunable parameters
├── requirements.txt
├── .env.example
│
├── agents/
│   ├── rag_agent.py         # LangGraph pipeline
│   ├── router.py            # Query classifier (6 categories)
│   └── hallucination_detector.py
│
├── retrieval/
│   ├── embedder.py          # BAAI/bge-small
│   ├── bm25_retriever.py    # BM25Okapi
│   ├── hybrid_retriever.py  # RRF fusion
│   └── reranker.py          # BAAI/bge-reranker-base
│
├── ingestion/
│   └── pipeline.py          # PDF + Slack + Email → chunks
│
├── database/
│   └── vector_store.py      # ChromaDB wrapper
│
├── memory/
│   └── conversation_memory.py  # TinyDB sliding window
│
├── auth/
│   └── rbac.py              # bcrypt + JWT + 4 roles
│
├── evaluation/
│   └── evaluator.py         # Metrics, feedback, export
│
└── utils/
    ├── llm_client.py        # Ollama + Anthropic unified
    └── citation_builder.py  # [N] ref → source mapping
```

---

## Production Checklist

- [ ] Change `admin123` password immediately
- [ ] Set a strong `JWT_SECRET` in `.env`
- [ ] Enable HTTPS (nginx + certbot)
- [ ] Switch to PostgreSQL/Redis for multi-instance scale
- [ ] Add rate limiting per user
- [ ] Set up automated nightly email/Slack re-sync
- [ ] Configure backup for `chroma_db/` and `*.json` stores

---

*Built with: ChromaDB · LangGraph · Ollama · Streamlit · BAAI/bge · BM25*
