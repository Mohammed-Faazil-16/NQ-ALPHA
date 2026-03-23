from __future__ import annotations

from time import perf_counter
from typing import Any

import requests

from backend.config import settings


FALLBACK_RESPONSE = """Strategy: Maintain current allocation
Risk Level: Medium
Reasoning: The local LLM is temporarily unavailable, so the advisor is preserving the existing system portfolio until live model reasoning resumes."""
LLM_TIMEOUT_SECONDS = min(settings.request_timeout_seconds, 25)
STATUS_TIMEOUT_SECONDS = 5


def _post_ollama(endpoint: str, payload: dict[str, Any], timeout_seconds: int | float = LLM_TIMEOUT_SECONDS) -> dict[str, Any]:
    response = requests.post(
        f"{settings.ollama_base_url}{endpoint}",
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def get_ollama_status() -> dict[str, object]:
    try:
        response = requests.get(f"{settings.ollama_base_url}/tags", timeout=STATUS_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models") or []
        available = [model.get("name") for model in models if model.get("name")]
        return {
            "connected": True,
            "configured_model": settings.ollama_chat_model,
            "embedding_model": settings.ollama_embedding_model,
            "available_models": available,
            "using_live_model": any(str(name).startswith(settings.ollama_chat_model) for name in available),
        }
    except Exception as exc:
        return {
            "connected": False,
            "configured_model": settings.ollama_chat_model,
            "embedding_model": settings.ollama_embedding_model,
            "available_models": [],
            "using_live_model": False,
            "error": str(exc),
        }


def generate_response_with_metadata(prompt: str) -> dict[str, object]:
    payload = {
        "model": settings.ollama_chat_model,
        "prompt": prompt,
        "stream": False,
    }
    started_at = perf_counter()
    try:
        data = _post_ollama("/generate", payload)
        text = data.get("response")
        if not text:
            raise RuntimeError("Ollama generate response did not contain 'response'")
        return {
            "text": text.strip(),
            "source": "ollama",
            "model": settings.ollama_chat_model,
            "latency_seconds": round(perf_counter() - started_at, 2),
        }
    except Exception as exc:
        return {
            "text": FALLBACK_RESPONSE,
            "source": "fallback",
            "model": settings.ollama_chat_model,
            "latency_seconds": round(perf_counter() - started_at, 2),
            "error": str(exc),
        }


def generate_response(prompt: str) -> str:
    return str(generate_response_with_metadata(prompt)["text"])


def chat_response(messages: list[dict[str, str]]) -> str:
    payload = {
        "model": settings.ollama_chat_model,
        "messages": messages,
        "stream": False,
    }
    try:
        data = _post_ollama("/chat", payload)
        text = data.get("response")
        if text:
            return text.strip()

        message_payload = data.get("message") or {}
        fallback = message_payload.get("content")
        if fallback:
            return fallback.strip()

        raise RuntimeError("Ollama chat response did not contain usable text")
    except Exception:
        return FALLBACK_RESPONSE
