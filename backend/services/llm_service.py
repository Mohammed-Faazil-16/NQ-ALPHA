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
ADVISOR_TIMEOUT_SECONDS = min(settings.request_timeout_seconds, 24)
DEFAULT_GENERATION_OPTIONS: dict[str, Any] = {
    "temperature": 0.2,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
}
ADVISOR_GENERATION_OPTIONS: dict[str, Any] = {
    **DEFAULT_GENERATION_OPTIONS,
    "num_ctx": 6144,
    "num_predict": 280,
}
FORECAST_GENERATION_OPTIONS: dict[str, Any] = {
    **DEFAULT_GENERATION_OPTIONS,
    "num_ctx": 4096,
    "num_predict": 180,
}


def _post_ollama(endpoint: str, payload: dict[str, Any], timeout_seconds: int | float = LLM_TIMEOUT_SECONDS) -> dict[str, Any]:
    response = requests.post(
        f"{settings.ollama_base_url}{endpoint}",
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def _fetch_available_models() -> list[str]:
    response = requests.get(f"{settings.ollama_base_url}/tags", timeout=STATUS_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    return [model.get("name") for model in (payload.get("models") or []) if model.get("name")]


def choose_advisor_model(available_models: list[str] | None = None) -> str:
    models = available_models
    if models is None:
        try:
            models = _fetch_available_models()
        except Exception:
            models = []

    preferred = settings.ollama_advisor_model
    if any(str(name) == preferred or str(name).startswith(preferred) for name in models):
        return preferred

    for name in models:
        text = str(name)
        if text.startswith("llama3.2-vision"):
            return text

    return settings.ollama_chat_model


def get_ollama_status() -> dict[str, object]:
    try:
        available = _fetch_available_models()
        preferred_advisor_model = choose_advisor_model(available)
        return {
            "connected": True,
            "configured_model": settings.ollama_chat_model,
            "configured_advisor_model": settings.ollama_advisor_model,
            "preferred_advisor_model": preferred_advisor_model,
            "embedding_model": settings.ollama_embedding_model,
            "available_models": available,
            "using_live_model": any(str(name).startswith(settings.ollama_chat_model) for name in available),
            "using_live_advisor_model": any(
                str(name) == preferred_advisor_model or str(name).startswith(preferred_advisor_model)
                for name in available
            ),
        }
    except Exception as exc:
        return {
            "connected": False,
            "configured_model": settings.ollama_chat_model,
            "configured_advisor_model": settings.ollama_advisor_model,
            "preferred_advisor_model": settings.ollama_chat_model,
            "embedding_model": settings.ollama_embedding_model,
            "available_models": [],
            "using_live_model": False,
            "using_live_advisor_model": False,
            "error": str(exc),
        }


def generate_response_with_metadata(
    prompt: str,
    model_name: str | None = None,
    timeout_seconds: int | float | None = None,
    allow_model_fallback: bool = True,
    options: dict[str, Any] | None = None,
) -> dict[str, object]:
    requested_model = model_name or settings.ollama_chat_model
    timeout_value = timeout_seconds if timeout_seconds is not None else LLM_TIMEOUT_SECONDS
    payload: dict[str, Any] = {
        "model": requested_model,
        "prompt": prompt,
        "stream": False,
    }
    if options:
        payload["options"] = options

    started_at = perf_counter()
    try:
        data = _post_ollama("/generate", payload, timeout_seconds=timeout_value)
        text = data.get("response")
        if not text:
            raise RuntimeError("Ollama generate response did not contain 'response'")
        return {
            "text": text.strip(),
            "source": "ollama",
            "model": requested_model,
            "requested_model": requested_model,
            "latency_seconds": round(perf_counter() - started_at, 2),
        }
    except Exception as exc:
        if allow_model_fallback and requested_model != settings.ollama_chat_model:
            fallback_result = generate_response_with_metadata(
                prompt,
                model_name=settings.ollama_chat_model,
                timeout_seconds=min(timeout_value, LLM_TIMEOUT_SECONDS),
                allow_model_fallback=False,
                options=options,
            )
            fallback_result.setdefault("requested_model", requested_model)
            fallback_result["fallback_error"] = str(exc)
            return fallback_result

        return {
            "text": FALLBACK_RESPONSE,
            "source": "fallback",
            "model": requested_model,
            "requested_model": requested_model,
            "latency_seconds": round(perf_counter() - started_at, 2),
            "error": str(exc),
        }


def generate_response(prompt: str) -> str:
    return str(generate_response_with_metadata(prompt)["text"])


def generate_advisor_response_with_metadata(
    prompt: str,
    timeout_seconds: int | float | None = None,
    forecast_mode: bool = False,
) -> dict[str, object]:
    advisor_model = choose_advisor_model()
    return generate_response_with_metadata(
        prompt,
        model_name=advisor_model,
        timeout_seconds=timeout_seconds if timeout_seconds is not None else ADVISOR_TIMEOUT_SECONDS,
        allow_model_fallback=True,
        options=FORECAST_GENERATION_OPTIONS if forecast_mode else ADVISOR_GENERATION_OPTIONS,
    )


def chat_response(messages: list[dict[str, str]]) -> str:
    payload = {
        "model": settings.ollama_chat_model,
        "messages": messages,
        "stream": False,
        "options": ADVISOR_GENERATION_OPTIONS,
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
