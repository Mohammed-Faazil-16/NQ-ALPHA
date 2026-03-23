from __future__ import annotations

from typing import Any

import requests

from backend.config import settings
from backend.services.runtime_cache import runtime_cache


NEWSDATA_BASE_URL = "https://newsdata.io/api/1/news"
NEWS_CACHE_TTL_SECONDS = 6 * 60 * 60
NEWS_TIMEOUT_SECONDS = 6
MAX_NEWS_ITEMS = 3


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper().lstrip("$")


def _build_news_query(symbol: str, asset_name: str | None = None, asset_type: str | None = None) -> str:
    normalized_symbol = _normalize_symbol(symbol)
    cleaned_symbol = normalized_symbol.replace(".NS", "").replace(".BO", "").replace("-USD", "")
    cleaned_name = " ".join(str(asset_name or "").replace("Ltd", "").replace("Limited", "").split()).strip()
    normalized_type = str(asset_type or "stock").strip().lower()

    if normalized_type == "crypto":
        return cleaned_name or cleaned_symbol
    if normalized_type == "commodity":
        return cleaned_name or cleaned_symbol
    if cleaned_name:
        return cleaned_name
    return cleaned_symbol


def get_asset_news(symbol: str, asset_name: str | None = None, asset_type: str | None = None, limit: int = MAX_NEWS_ITEMS) -> dict[str, Any]:
    normalized_symbol = _normalize_symbol(symbol)
    normalized_limit = max(1, min(int(limit), MAX_NEWS_ITEMS))
    query = _build_news_query(normalized_symbol, asset_name=asset_name, asset_type=asset_type)
    cache_key = ("newsdata", normalized_symbol, query.lower(), str(asset_type or "stock").lower(), normalized_limit)
    cached = runtime_cache.get(cache_key)
    if cached is not None:
        return cached

    if not settings.newsdata_api_key:
        payload = {
            "enabled": False,
            "symbol": normalized_symbol,
            "query": query,
            "articles": [],
            "error": "NewsData.io API key is not configured.",
        }
        runtime_cache.set(cache_key, payload, ttl_seconds=60)
        return payload

    params = {
        "apikey": settings.newsdata_api_key,
        "q": query,
        "language": "en",
        "category": "business",
        "size": normalized_limit,
    }

    try:
        response = requests.get(NEWSDATA_BASE_URL, params=params, timeout=NEWS_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload_json = response.json()
        raw_articles = payload_json.get("results") or []
        articles = []
        for item in raw_articles[:normalized_limit]:
            articles.append(
                {
                    "title": str(item.get("title") or "").strip(),
                    "summary": str(item.get("description") or item.get("content") or "").strip(),
                    "source": str(item.get("source_id") or item.get("source_name") or "newsdata.io").strip(),
                    "published_at": str(item.get("pubDate") or "").strip(),
                    "link": str(item.get("link") or "").strip(),
                }
            )

        payload = {
            "enabled": True,
            "symbol": normalized_symbol,
            "query": query,
            "articles": articles,
            "article_count": len(articles),
        }
        runtime_cache.set(cache_key, payload, ttl_seconds=NEWS_CACHE_TTL_SECONDS)
        return payload
    except Exception as exc:
        payload = {
            "enabled": True,
            "symbol": normalized_symbol,
            "query": query,
            "articles": [],
            "article_count": 0,
            "error": str(exc),
        }
        runtime_cache.set(cache_key, payload, ttl_seconds=15 * 60)
        return payload
