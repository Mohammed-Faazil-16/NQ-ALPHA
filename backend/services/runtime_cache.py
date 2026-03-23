from __future__ import annotations

from copy import deepcopy
from threading import Lock
from time import monotonic


class TTLCache:
    def __init__(self):
        self._values: dict[object, tuple[float, object]] = {}
        self._lock = Lock()

    def get(self, key):
        with self._lock:
            payload = self._values.get(key)
            if payload is None:
                return None
            expires_at, value = payload
            if expires_at < monotonic():
                self._values.pop(key, None)
                return None
            return deepcopy(value)

    def set(self, key, value, ttl_seconds: float):
        with self._lock:
            self._values[key] = (monotonic() + max(float(ttl_seconds), 0.0), deepcopy(value))

    def clear(self):
        with self._lock:
            self._values.clear()


runtime_cache = TTLCache()
