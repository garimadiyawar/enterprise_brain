"""
agents/rag_agent.py

LangGraph-powered multi-agent RAG pipeline.

Graph nodes:
  route → retrieve → rerank → generate → detect_hallucination → respond

State flows through all nodes; each node enriches it.
"""
from __future__ import annotations
import time
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, END
from loguru import logger

from config import TOP_K_RERANK, CONFIDENCE_THRESHOLD
from agents.router               import QueryRouter
from agents.hallucination_detector import HallucinationDetector
from retrieval.hybrid_retriever  import HybridRetriever
from memory.conversation_memory  import ConversationMemory
from utils.llm_client            import get_llm_response
from utils.citation_builder      import build_citations


# ── State ─────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    # Input
    query:         str
    user_id:       str
    session_id:    str
    role:          str
    collection:    str | None

    # Router output
    route_result:  dict

    # Retrieval output
    retrieved_docs: list[dict]

    # Generation output
    raw_answer:    str
    citations:     list[dict]

    # Hallucination detection output
    hallucination: dict

    # Final response
    final_answer:  str
    confidence:    float
    elapsed_ms:    float
    error:         str | None


# ── Node functions ────────────────────────────────────────────────────────────

def node_route(state: RAGState) -> RAGState:
    router = QueryRouter()
    result = router.route(state["query"])
    return {**state, "route_result": result}


def node_retrieve(state: RAGState) -> RAGState:
    route   = state["route_result"]
    filters = _build_filters(state["role"], state.get("collection"))

    retriever = HybridRetriever(state.get("collection"))

    if route["strategy"] == "memory_only":
        return {**state, "retrieved_docs": []}

    if route["strategy"] == "refuse":
        return {**state, "retrieved_docs": []}

    # For comparison / multi-query: retrieve for each sub-query and merge
    queries   = [state["query"]] + route.get("sub_queries", [])[:2]
    all_docs  = []
    seen_ids  = set()

    for q in queries:
        docs = retriever.retrieve(q, top_k=TOP_K_RERANK, filters=filters)
        for d in docs:
            if d["id"] not in seen_ids:
                all_docs.append(d)
                seen_ids.add(d["id"])

    return {**state, "retrieved_docs": all_docs}


def node_generate(state: RAGState) -> RAGState:
    route    = state["route_result"]
    docs     = state["retrieved_docs"]
    memory   = ConversationMemory(state["session_id"])
    history  = memory.get_history_text()

    strategy = route["strategy"]

    if strategy == "refuse":
        answer = (
            "I'm sorry, this question falls outside the scope of the company "
            "knowledge base. Please consult the relevant team directly."
        )
        return {
            **state,
            "raw_answer": answer,
            "citations":  [],
        }

    if strategy == "memory_only":
        prompt = _build_conversational_prompt(state["query"], history)
    else:
        context_text = _format_context(docs)
        prompt = _build_rag_prompt(
            query        = state["query"],
            context      = context_text,
            history      = history,
            strategy     = strategy,
        )

    try:
        answer = get_llm_response(prompt, max_tokens=1024, temperature=0.2)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        answer = (
            "\u26a0\ufe0f **LLM Unavailable** — Could not reach the language model.\n\n"
            "**Using Ollama?** Run `ollama serve` and `ollama pull llama3`.\n\n"
            "**Using Anthropic?** Check `ANTHROPIC_API_KEY` in your `.env` file."
        )

    citations = build_citations(answer, docs)
    memory.add_turn(state["query"], answer)
    return {**state, "raw_answer": answer, "citations": citations}


def node_detect_hallucination(state: RAGState) -> RAGState:
    detector = HallucinationDetector()
    result   = detector.detect(state["raw_answer"], state["retrieved_docs"])
    return {**state, "hallucination": result}


def node_respond(state: RAGState) -> RAGState:
    h          = state["hallucination"]
    confidence = h["confidence"]
    answer     = state["raw_answer"]

    # Attach warning if low confidence
    if h["is_hallucinated"]:
        warning = (
            "\n\n⚠️ **Confidence Warning**: This answer may contain information "
            "not fully supported by the source documents. Please verify with "
            "the original sources before acting on it."
        )
        answer += warning

    return {
        **state,
        "final_answer": answer,
        "confidence":   confidence,
    }


# ── Graph Assembly ────────────────────────────────────────────────────────────

def build_rag_graph() -> StateGraph:
    g = StateGraph(RAGState)

    g.add_node("route",               node_route)
    g.add_node("retrieve",            node_retrieve)
    g.add_node("generate",            node_generate)
    g.add_node("detect_hallucination",node_detect_hallucination)
    g.add_node("respond",             node_respond)

    g.set_entry_point("route")
    g.add_edge("route",                "retrieve")
    g.add_edge("retrieve",             "generate")
    g.add_edge("generate",             "detect_hallucination")
    g.add_edge("detect_hallucination", "respond")
    g.add_edge("respond",              END)

    return g.compile()


# ── Public API ────────────────────────────────────────────────────────────────

class RAGAgent:
    def __init__(self):
        self.graph = build_rag_graph()

    def ask(
        self,
        query:      str,
        user_id:    str    = "anonymous",
        session_id: str    = "default",
        role:       str    = "viewer",
        collection: str | None = None,
    ) -> dict:
        """
        Run the full RAG pipeline.
        Returns the enriched state dict (final_answer, citations, confidence, …)
        """
        t0    = time.time()
        state = RAGState(
            query          = query,
            user_id        = user_id,
            session_id     = session_id,
            role           = role,
            collection     = collection,
            route_result   = {},
            retrieved_docs = [],
            raw_answer     = "",
            citations      = [],
            hallucination  = {},
            final_answer   = "",
            confidence     = 0.0,
            elapsed_ms     = 0.0,
            error          = None,
        )

        try:
            result = self.graph.invoke(state)
        except Exception as e:
            logger.error(f"RAG graph error: {e}")
            result = {
                **state,
                "final_answer": f"An error occurred: {e}",
                "confidence":   0.0,
                "error":        str(e),
            }

        result["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
        return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_filters(role: str, collection: str | None) -> dict | None:
    from config import ROLES
    perms = ROLES.get(role, ROLES["viewer"])
    allowed = perms["collections"]
    if "*" in allowed or not collection:
        return None
    if collection in allowed:
        return {"collection": collection}
    return {"collection": "__none__"}   # forces empty results


def _format_context(docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        src   = doc["metadata"].get("source", "unknown")
        page  = doc["metadata"].get("page", "")
        label = f"[{i}] Source: {src}" + (f", page {page}" if page else "")
        parts.append(f"{label}\n{doc['text']}")
    return "\n\n---\n\n".join(parts)


def _build_rag_prompt(
    query: str,
    context: str,
    history: str,
    strategy: str,
) -> str:
    strategy_instructions = {
        "single_retrieval":         "Answer concisely and factually.",
        "retrieval_with_synthesis": "Provide a thorough analytical response with insights.",
        "multi_retrieval":          "Compare and contrast the relevant information clearly.",
        "retrieval_with_steps":     "Provide clear, numbered step-by-step instructions.",
    }.get(strategy, "Answer the question.")

    return f"""You are an enterprise AI assistant with access to company documents.
{strategy_instructions}

RULES:
- Answer ONLY from the provided context. Do NOT use prior knowledge.
- If the context doesn't contain enough information, say so explicitly.
- Reference source numbers like [1], [2] when citing specific facts.
- Be concise but complete.

CONVERSATION HISTORY:
{history if history else "None"}

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


def _build_conversational_prompt(query: str, history: str) -> str:
    return f"""You are a helpful enterprise assistant.

CONVERSATION HISTORY:
{history if history else "None"}

User: {query}
Assistant:"""
