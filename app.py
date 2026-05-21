"""
app.py
Enterprise Brain — Streamlit Frontend

Pages:
  1. 💬 Chat       — Ask questions, see citations, confidence badge
  2. 📥 Ingest     — Upload PDFs / Slack exports / trigger email sync
  3. 📊 Dashboard  — Eval metrics, latency charts, feedback trends
  4. 👥 Admin      — User management, RBAC, system stats
"""

import uuid
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title  = "Enterprise Brain",
    page_icon   = "🧠",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Lazy imports (avoid loading heavy models on every rerun) ──────────────────
@st.cache_resource
def get_auth():
    from auth.rbac import AuthManager
    return AuthManager()

@st.cache_resource
def get_agent():
    from agents.rag_agent import RAGAgent
    return RAGAgent()

@st.cache_resource
def get_ingestion(collection=None):
    from ingestion.pipeline import IngestionPipeline
    return IngestionPipeline(collection)

@st.cache_resource
def get_evaluator():
    from evaluation.evaluator import EvaluationLogger
    return EvaluationLogger()


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stSidebar"] { background: #0f1117; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.main { background: #0f1117; }

/* ── Chat bubbles ── */
.user-bubble {
    background: #1e40af; color: white;
    padding: 12px 16px; border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 20%; max-width: 80%;
    float: right; clear: both; font-size: 15px;
}
.bot-bubble {
    background: #1e293b; color: #e2e8f0;
    padding: 12px 16px; border-radius: 18px 18px 18px 4px;
    margin: 8px 20% 8px 0; max-width: 80%;
    float: left; clear: both; font-size: 15px;
    border: 1px solid #334155;
}
.clearfix { clear: both; }

/* ── Confidence badge ── */
.badge-high   { background:#166534; color:#bbf7d0; padding:3px 10px;
                border-radius:999px; font-size:12px; font-weight:600; }
.badge-medium { background:#854d0e; color:#fef3c7; padding:3px 10px;
                border-radius:999px; font-size:12px; font-weight:600; }
.badge-low    { background:#7f1d1d; color:#fee2e2; padding:3px 10px;
                border-radius:999px; font-size:12px; font-weight:600; }

/* ── Citation card ── */
.citation-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 8px; padding: 10px 14px; margin: 4px 0;
    font-size: 13px; color: #94a3b8;
}
.citation-card strong { color: #38bdf8; }

/* ── Metric card ── */
.metric-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 12px; padding: 20px; text-align: center;
}
.metric-value { font-size: 32px; font-weight: 700; color: #38bdf8; }
.metric-label { font-size: 13px; color: #64748b; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Auth Gate
# ═══════════════════════════════════════════════════════════════════════════════

def login_page():
    st.markdown("## 🧠 Enterprise Brain")
    st.markdown("#### Sign in to your knowledge base")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In", use_container_width=True)

    if submitted:
        auth  = get_auth()
        token = auth.authenticate(username, password)
        if token:
            user_info = auth.verify_token(token)
            st.session_state["token"]    = token
            st.session_state["username"] = user_info["username"]
            st.session_state["role"]     = user_info["role"]
            st.session_state["session_id"] = str(uuid.uuid4())
            st.success("Signed in!")
            st.rerun()
        else:
            st.error("Invalid username or password.")


def is_logged_in():
    if "token" not in st.session_state:
        return False
    auth   = get_auth()
    result = auth.verify_token(st.session_state["token"])
    return result is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar Navigation
# ═══════════════════════════════════════════════════════════════════════════════

def sidebar():
    with st.sidebar:
        st.markdown("### 🧠 Enterprise Brain")
        st.divider()

        role     = st.session_state.get("role", "viewer")
        username = st.session_state.get("username", "")
        st.markdown(f"👤 **{username}**  \n🔑 `{role}`")
        st.divider()

        pages = {
            "💬 Chat":       "chat",
            "📥 Ingest":     "ingest",
            "📊 Dashboard":  "dashboard",
        }
        if role == "admin":
            pages["👥 Admin"] = "admin"

        page = st.radio("Navigate", list(pages.keys()), label_visibility="collapsed")

        st.divider()
        if st.button("Sign Out", use_container_width=True):
            for k in ["token","username","role","session_id","messages"]:
                st.session_state.pop(k, None)
            st.rerun()

        # System status
        st.markdown("#### System Status")
        try:
            from database.vector_store import VectorStore
            vs    = VectorStore()
            count = vs.count()
            st.success(f"✅ Vector DB  ({count:,} chunks)")
        except Exception:
            st.error("❌ Vector DB offline")

        return pages[page]


# ═══════════════════════════════════════════════════════════════════════════════
# Page: Chat
# ═══════════════════════════════════════════════════════════════════════════════

def page_chat():
    st.markdown("## 💬 Chat with Your Knowledge Base")

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "eval_ids" not in st.session_state:
        st.session_state["eval_ids"] = []

    # Render existing messages
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">{msg["content"]}</div>'
                '<div class="clearfix"></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="bot-bubble">{msg["content"]}</div>'
                '<div class="clearfix"></div>',
                unsafe_allow_html=True,
            )
            # Show meta below bot message
            if "meta" in msg:
                _render_answer_meta(msg["meta"])

    # Input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            query = st.text_input(
                "Ask anything…",
                placeholder="e.g. What is our refund policy for enterprise contracts?",
                label_visibility="collapsed",
            )
        with col2:
            send = st.form_submit_button("Send ➤", use_container_width=True)

    if send and query.strip():
        _handle_query(query.strip())
        st.rerun()

    # Feedback for last answer
    if st.session_state["eval_ids"]:
        with st.expander("📝 Rate the last answer"):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Helpful", use_container_width=True):
                    _submit_feedback(1)
                    st.success("Thanks for the feedback!")
            with col2:
                if st.button("👎 Not helpful", use_container_width=True):
                    _submit_feedback(-1)
                    st.warning("Noted. We'll use this to improve.")
            comment = st.text_input("Optional comment")
            if comment and st.button("Submit comment"):
                _submit_feedback(0, comment)


def _handle_query(query: str):
    agent     = get_agent()
    evaluator = get_evaluator()

    st.session_state["messages"].append({"role": "user", "content": query})

    with st.spinner("Thinking…"):
        result = agent.ask(
            query      = query,
            user_id    = st.session_state.get("username", "anon"),
            session_id = st.session_state.get("session_id", "default"),
            role       = st.session_state.get("role", "viewer"),
        )

    answer = result.get("final_answer", "I couldn't find an answer.")
    meta   = {
        "confidence":    result.get("confidence", 0.0),
        "hallucination": result.get("hallucination", {}),
        "citations":     result.get("citations", []),
        "route":         result.get("route_result", {}),
        "elapsed_ms":    result.get("elapsed_ms", 0),
        "n_docs":        len(result.get("retrieved_docs", [])),
    }

    st.session_state["messages"].append({
        "role":    "assistant",
        "content": answer,
        "meta":    meta,
    })

    record_id = evaluator.log(result)
    st.session_state["eval_ids"].append(record_id)


def _render_answer_meta(meta: dict):
    conf = meta.get("confidence", 0)
    badge_class = (
        "badge-high"   if conf >= 0.75 else
        "badge-medium" if conf >= 0.50 else
        "badge-low"
    )
    label = "High" if conf >= 0.75 else "Medium" if conf >= 0.50 else "Low"

    h = meta.get("hallucination", {})
    verdict_emoji = "✅" if not h.get("is_hallucinated") else "⚠️"

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f'<span class="{badge_class}">Confidence: {label} ({conf:.0%})</span>',
        unsafe_allow_html=True,
    )
    col2.caption(f"{verdict_emoji} {h.get('verdict','').capitalize()}")
    col3.caption(f"🗂 {meta.get('n_docs', 0)} sources | ⚡ {meta.get('elapsed_ms',0):.0f}ms")
    route_cat = meta.get("route", {}).get("category", "")
    col4.caption(f"🧭 {route_cat}")

    citations = meta.get("citations", [])
    if citations:
        with st.expander(f"📎 {len(citations)} Citation(s)"):
            for c in citations:
                src   = c.get("source", "")
                page  = f" · p.{c['page']}" if c.get("page") else ""
                score = c.get("score", 0)
                st.markdown(
                    f'<div class="citation-card">'
                    f'<strong>[{c["index"]}] {src}{page}</strong> '
                    f'<em>(score: {score:.3f})</em><br/>'
                    f'{c.get("snippet","")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def _submit_feedback(rating: int, comment: str = ""):
    evaluator = get_evaluator()
    if st.session_state["eval_ids"]:
        evaluator.add_feedback(st.session_state["eval_ids"][-1], rating, comment)


# ═══════════════════════════════════════════════════════════════════════════════
# Page: Ingest
# ═══════════════════════════════════════════════════════════════════════════════

def page_ingest():
    st.markdown("## 📥 Ingest Documents")
    role = st.session_state.get("role", "viewer")
    auth = get_auth()

    if not auth.can(role, "can_ingest"):
        st.error("🚫 Your role does not have ingestion permissions.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["📄 PDF", "💬 Slack", "📧 Email", "🗂 Manage"])

    # ── PDF Upload ─────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Upload PDF Documents")
        st.caption("Supported: PDF files. Multiple files allowed.")
        files = st.file_uploader(
            "Drop PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        collection = st.text_input("Collection tag", value="public",
                                   help="Used for RBAC filtering")

        if st.button("🚀 Ingest PDFs", disabled=not files):
            pipeline = get_ingestion()
            total = 0
            with st.spinner("Ingesting…"):
                progress = st.progress(0)
                for i, f in enumerate(files):
                    tmp = Path(tempfile.gettempdir()) / f.name
                    tmp.write_bytes(f.read())
                    result = pipeline.ingest_pdf(tmp)
                    total += result.get("chunks", 0)
                    progress.progress((i + 1) / len(files))

                # Rebuild BM25
                from retrieval.hybrid_retriever import HybridRetriever
                HybridRetriever().rebuild_bm25()

            st.success(f"✅ Ingested {len(files)} file(s) → {total:,} chunks")

    # ── Slack ──────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Slack Ingestion")
        mode = st.radio("Mode", ["Export ZIP/Directory", "Live API"], horizontal=True)

        if mode == "Export ZIP/Directory":
            export_path = st.text_input(
                "Path to Slack export directory",
                placeholder="/path/to/slack_export"
            )
            if st.button("🚀 Ingest Slack Export", disabled=not export_path):
                pipeline = get_ingestion()
                with st.spinner("Parsing Slack export…"):
                    result = pipeline.ingest_slack_export(export_path)
                    HybridRetriever().rebuild_bm25()
                st.success(f"✅ {result['chunks']:,} chunks from Slack export")

        else:
            st.info("Requires SLACK_BOT_TOKEN in .env")
            channels = st.text_input("Channel IDs (comma-separated)")
            if st.button("🚀 Fetch from Slack"):
                from ingestion.pipeline import load_slack_live
                pipeline = get_ingestion()
                with st.spinner("Fetching from Slack…"):
                    result = pipeline.ingest_slack_live(channels.split(","))
                    HybridRetriever().rebuild_bm25()
                st.success(f"✅ {result['chunks']:,} chunks from Slack")

    # ── Email ──────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Email Ingestion")
        st.info("Configure EMAIL_HOST / EMAIL_USER / EMAIL_PASS in .env")
        limit = st.slider("Max emails to fetch", 50, 1000, 200)
        if st.button("🚀 Fetch Emails"):
            pipeline = get_ingestion()
            with st.spinner("Connecting to IMAP…"):
                result = pipeline.ingest_emails()
                HybridRetriever().rebuild_bm25()
            st.success(f"✅ {result['chunks']:,} chunks from email")

    # ── Manage ────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### Document Library")
        try:
            from database.vector_store import VectorStore
            vs      = VectorStore()
            sources = vs.list_sources()
            count   = vs.count()
            st.caption(f"Total: **{count:,} chunks** across **{len(sources)} sources**")

            if sources:
                df = pd.DataFrame({"Source": sources})
                st.dataframe(df, use_container_width=True, height=300)

                if auth.can(role, "can_delete"):
                    to_delete = st.selectbox("Delete a source", [""] + sources)
                    if to_delete and st.button(f"🗑 Delete '{to_delete}'", type="primary"):
                        vs.delete_source(to_delete)
                        HybridRetriever().rebuild_bm25()
                        st.success(f"Deleted: {to_delete}")
                        st.rerun()
            else:
                st.info("No documents ingested yet.")
        except Exception as e:
            st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Page: Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def page_dashboard():
    st.markdown("## 📊 Evaluation Dashboard")
    evaluator = get_evaluator()
    stats     = evaluator.aggregate_stats()

    if not stats:
        st.info("No queries logged yet. Start chatting!")
        return

    # ── KPI Row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Queries",      f"{stats['total_queries']:,}")
    c2.metric("Avg Confidence",     f"{stats['avg_confidence']:.0%}")
    c3.metric("Hallucination Rate", f"{stats['hallucination_rate']:.0%}")
    c4.metric("Thumbs Up Rate",     f"{stats['thumbs_up_rate']:.0%}")
    c5.metric("Avg Latency",        f"{stats['avg_latency_ms']:.0f}ms")

    st.divider()

    col_l, col_r = st.columns(2)

    # ── Daily Trend ────────────────────────────────────────────────────────────
    with col_l:
        daily = evaluator.daily_stats()
        if daily:
            df_daily = pd.DataFrame(daily)
            fig = px.line(
                df_daily, x="date", y=["queries","hallucinated"],
                title="Daily Query Volume & Hallucinations",
                color_discrete_sequence=["#38bdf8","#f87171"],
                template="plotly_dark",
            )
            fig.update_layout(paper_bgcolor="#1e293b", plot_bgcolor="#0f1117")
            st.plotly_chart(fig, use_container_width=True)

    # ── Route Distribution ────────────────────────────────────────────────────
    with col_r:
        route_dist = stats.get("route_distribution", {})
        if route_dist:
            fig2 = px.pie(
                names  = list(route_dist.keys()),
                values = list(route_dist.values()),
                title  = "Query Category Distribution",
                color_discrete_sequence = px.colors.sequential.Blues_r,
                template="plotly_dark",
            )
            fig2.update_layout(paper_bgcolor="#1e293b")
            st.plotly_chart(fig2, use_container_width=True)

    # ── Recent Logs ───────────────────────────────────────────────────────────
    st.markdown("### Recent Interactions")
    records = evaluator.get_all(limit=100)[::-1]   # newest first

    if records:
        rows = []
        for r in records[:50]:
            h = r.get("hallucination", {})
            rows.append({
                "Time":       r["ts"][:19].replace("T"," "),
                "User":       r.get("user_id",""),
                "Query":      r.get("query","")[:80],
                "Confidence": f"{r.get('confidence',0):.0%}",
                "Verdict":    h.get("verdict",""),
                "Latency ms": r.get("elapsed_ms",""),
                "Feedback":   "👍" if r.get("feedback")==1 else "👎" if r.get("feedback")==-1 else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=400)

    # ── Export ────────────────────────────────────────────────────────────────
    csv = evaluator.export_csv()
    if csv:
        st.download_button(
            "⬇️ Export CSV",
            data     = csv,
            file_name= "enterprise_brain_eval.csv",
            mime     = "text/csv",
        )

    # ── Confidence Histogram ──────────────────────────────────────────────────
    confs = [r.get("confidence", 0) for r in records if r.get("confidence") is not None]
    if confs:
        fig3 = px.histogram(
            x          = confs,
            nbins      = 20,
            title      = "Confidence Score Distribution",
            labels     = {"x": "Confidence"},
            template   = "plotly_dark",
            color_discrete_sequence=["#38bdf8"],
        )
        fig3.update_layout(paper_bgcolor="#1e293b", plot_bgcolor="#0f1117")
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Page: Admin
# ═══════════════════════════════════════════════════════════════════════════════

def page_admin():
    st.markdown("## 👥 Admin Panel")
    auth = get_auth()
    role = st.session_state.get("role", "viewer")

    if role != "admin":
        st.error("🚫 Admin access only.")
        return

    tab1, tab2 = st.tabs(["👤 User Management", "⚙️ System"])

    with tab1:
        st.markdown("### All Users")
        users = auth.list_users()
        if users:
            df = pd.DataFrame(users)[["username","role","full_name","created_at","active"]]
            st.dataframe(df, use_container_width=True)

        st.divider()
        st.markdown("### Create User")
        with st.form("create_user"):
            c1, c2 = st.columns(2)
            new_user = c1.text_input("Username")
            new_pass = c2.text_input("Password", type="password")
            c3, c4   = st.columns(2)
            new_role = c3.selectbox("Role", ["viewer","analyst","manager","admin"])
            full_nm  = c4.text_input("Full Name")
            if st.form_submit_button("Create User"):
                try:
                    auth.create_user(new_user, new_pass, new_role, full_nm)
                    st.success(f"User '{new_user}' created as {new_role}")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        st.divider()
        st.markdown("### Update Role")
        user_list = [u["username"] for u in users]
        if user_list:
            c1, c2, c3 = st.columns(3)
            upd_user = c1.selectbox("User", user_list, key="upd_u")
            upd_role = c2.selectbox("New Role",
                                    ["viewer","analyst","manager","admin"], key="upd_r")
            if c3.button("Update"):
                auth.update_role(upd_user, upd_role)
                st.success(f"Updated {upd_user} → {upd_role}")
                st.rerun()

    with tab2:
        st.markdown("### Vector Store Stats")
        try:
            from database.vector_store import VectorStore
            vs = VectorStore()
            st.metric("Total Chunks", f"{vs.count():,}")
            sources = vs.list_sources()
            st.markdown(f"**{len(sources)} ingested sources:**")
            for s in sources:
                st.caption(f"• {s}")
        except Exception as e:
            st.error(f"Vector store error: {e}")

        st.divider()
        st.markdown("### Clear Session Memory")
        sid = st.text_input("Session ID to clear")
        if st.button("Clear Memory") and sid:
            from memory.conversation_memory import ConversationMemory
            ConversationMemory(sid).clear()
            st.success(f"Memory cleared for: {sid}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if not is_logged_in():
        login_page()
        return

    page = sidebar()

    if   page == "chat":       page_chat()
    elif page == "ingest":     page_ingest()
    elif page == "dashboard":  page_dashboard()
    elif page == "admin":      page_admin()


if __name__ == "__main__":
    main()
