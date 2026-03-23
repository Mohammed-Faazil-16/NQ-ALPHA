from typing import Any

import requests

from backend.config import settings


EMBED_TIMEOUT_SECONDS = min(settings.request_timeout_seconds, 3)


def get_embedding(text: str) -> list[float]:
    payload = {
        "model": settings.ollama_embedding_model,
        "input": text,
    }
    try:
        response = requests.post(
            f"{settings.ollama_base_url}/embed",
            json=payload,
            timeout=EMBED_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        embeddings = data.get("embeddings")
        if not embeddings or not isinstance(embeddings, list):
            raise RuntimeError("Ollama embedding response did not contain 'embeddings'")

        vector = embeddings[0]
        if not isinstance(vector, list) or not vector:
            raise RuntimeError("Ollama embedding vector is empty")

        return [float(value) for value in vector]
    except Exception:
        return []
