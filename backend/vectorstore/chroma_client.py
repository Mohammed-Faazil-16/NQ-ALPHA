from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from backend.config import settings


_COLLECTION: Collection | None = None


def get_collection() -> Collection:
    global _COLLECTION
    if _COLLECTION is not None:
        return _COLLECTION

    chroma_path = Path(settings.chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    _COLLECTION = client.get_or_create_collection(name=settings.chroma_collection)
    return _COLLECTION
