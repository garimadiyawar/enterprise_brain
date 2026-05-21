"""
utils/llm_client.py
Unified LLM interface.
  - Primary:  Ollama (Llama 3, Qwen2, Mistral) — free, local
  - Fallback: Anthropic Claude via API
"""
import httpx
import json
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_BASE_URL,
    ANTHROPIC_KEY,
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def get_llm_response(
    prompt:       str,
    system:       str  = "You are a helpful enterprise assistant.",
    max_tokens:   int  = 1024,
    temperature:  float = 0.2,
) -> str:
    """
    Single entry-point for all LLM calls.
    Tries Ollama first; falls back to Anthropic if configured.
    """
    if LLM_PROVIDER == "ollama":
        try:
            return _ollama(prompt, system, max_tokens, temperature)
        except Exception as e:
            logger.warning(f"Ollama failed: {e}. Trying Anthropic fallback.")
            if ANTHROPIC_KEY:
                return _anthropic(prompt, system, max_tokens, temperature)
            raise
    elif LLM_PROVIDER == "anthropic":
        return _anthropic(prompt, system, max_tokens, temperature)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


# ── Ollama ────────────────────────────────────────────────────────────────────

def _ollama(
    prompt:      str,
    system:      str,
    max_tokens:  int,
    temperature: float,
) -> str:
    url     = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system",  "content": system},
            {"role": "user",    "content": prompt},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
        "stream": False,
    }

    response = httpx.post(url, json=payload, timeout=120.0)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"].strip()


# ── Anthropic ─────────────────────────────────────────────────────────────────

def _anthropic(
    prompt:      str,
    system:      str,
    max_tokens:  int,
    temperature: float,
) -> str:
    if not ANTHROPIC_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set.")

    url     = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key":         ANTHROPIC_KEY,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }
    payload = {
        "model":      "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "system":     system,
        "messages":   [{"role": "user", "content": prompt}],
    }

    response = httpx.post(url, headers=headers, json=payload, timeout=60.0)
    response.raise_for_status()
    return response.json()["content"][0]["text"].strip()


# ── Streaming (for Streamlit) ─────────────────────────────────────────────────

def stream_ollama(prompt: str, system: str = "You are a helpful assistant."):
    """Generator that yields tokens from Ollama's streaming API."""
    url     = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "stream": True,
    }
    try:
        with httpx.stream("POST", url, json=payload, timeout=120.0) as r:
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "message" in chunk:
                        yield chunk["message"].get("content", "")
                    if chunk.get("done"):
                        break
    except Exception as e:
        yield f"\n[Stream error: {e}]"
