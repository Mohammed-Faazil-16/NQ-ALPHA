from __future__ import annotations

from threading import Thread

from backend.services.model_registry import warm_model_registry
from backend.services.precompute_service import start_precompute_scheduler


def _warm_runtime_services() -> None:
    try:
        warm_model_registry()
    except Exception:
        pass

    try:
        start_precompute_scheduler()
    except Exception:
        pass


def start_runtime_warmup() -> None:
    Thread(target=_warm_runtime_services, daemon=True, name="neuroquant-runtime-warmup").start()
