from __future__ import annotations

from difflib import get_close_matches
from functools import lru_cache
import re

from sqlalchemy import func, or_
import yfinance as yf

from backend.database.models.all_assets import AllAssets
from backend.database.models.asset_universe import AssetUniverse
from backend.database.postgres import SessionLocal
from backend.services.runtime_cache import runtime_cache


STATIC_SYMBOL_MAP = {
    "accenture": "ACN",
    "acn": "ACN",
    "apple": "AAPL",
    "aapl": "AAPL",
    "microsoft": "MSFT",
    "msft": "MSFT",
    "tesla": "TSLA",
    "tsla": "TSLA",
    "amazon": "AMZN",
    "amzn": "AMZN",
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "bank of baroda": "BANKBARODA.NS",
    "bankbaroda": "BANKBARODA.NS",
    "baroda bank": "BANKBARODA.NS",
    "state bank of india": "SBIN.NS",
    "sbi": "SBIN.NS",
    "hdfc bank": "HDFCBANK.NS",
    "hdfc": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "icici": "ICICIBANK.NS",
    "reliance": "RELIANCE.NS",
    "reliance industries": "RELIANCE.NS",
    "reliance industries ltd": "RELIANCE.NS",
    "ril": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "maruti": "MARUTI.NS",
    "maruti suzuki": "MARUTI.NS",
    "hpcl": "HINDPETRO.NS",
    "hindustan petroleum": "HINDPETRO.NS",
    "hindustan petroleum corporation": "HINDPETRO.NS",
    "hindpetro": "HINDPETRO.NS",
    "bpcl": "BPCL.NS",
    "bharat petroleum": "BPCL.NS",
    "ioc": "IOC.NS",
    "indian oil": "IOC.NS",
    "ongc": "ONGC.NS",
    "bitcoin": "BTC-USD",
    "btc": "BTC-USD",
    "btc-usd": "BTC-USD",
    "ethereum": "ETH-USD",
    "eth": "ETH-USD",
    "eth-usd": "ETH-USD",
    "solana": "SOL-USD",
    "sol": "SOL-USD",
    "gold": "GC=F",
    "silver": "SI=F",
    "crude oil": "CL=F",
}
NSE_HINTS = {
    "reliance",
    "reilance",
    "reaiance",
    "tcs",
    "infosys",
    "hdfc",
    "icici",
    "sbi",
    "sbank",
    "axis",
    "kotak",
    "wipro",
    "baroda",
    "bankbaroda",
    "hindustan",
    "petroleum",
    "hpcl",
    "maruti",
    "ongc",
    "bpcl",
    "ioc",
}
CRYPTO_HINTS = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA"}
SEARCH_CACHE_TTL_SECONDS = 3600
DIRECT_PROBE_TIMEOUT_SECONDS = 4
ASSET_QUERY_STOPWORDS = {
    "a",
    "an",
    "advice",
    "analyze",
    "analyse",
    "asset",
    "buy",
    "check",
    "current",
    "do",
    "for",
    "give",
    "hold",
    "how",
    "i",
    "in",
    "invest",
    "is",
    "it",
    "look",
    "me",
    "of",
    "on",
    "price",
    "query",
    "quote",
    "share",
    "should",
    "stock",
    "take",
    "tell",
    "the",
    "this",
    "to",
    "today",
    "view",
    "what",
    "which",
    "will",
    "with",
    "provide",
    "return",
    "returns",
    "year",
    "years",
    "next",
    "confident",
    "good",
    "about",
    "are",
}
ASSET_QUERY_PATTERNS = (
    r"(?:stock|share|price|quote)\s+of\s+(.+)$",
    r"(?:take|view|opinion|thoughts|outlook)\s+on\s+(.+)$",
    r"(?:analyze|analyse|review|check)\s+(.+)$",
    r"(?:buy|invest\s+in|avoid|hold)\s+(.+)$",
    r"(?:about)\s+(.+)$",
)


def _normalize_query(query: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\-\.\/\s]", " ", query or "")
    return " ".join(cleaned.lower().strip().split())


def _compact_symbol(query: str) -> str:
    return query.replace(" ", "").upper().lstrip("$")


def _looks_like_market_symbol(value: str) -> bool:
    compact = _compact_symbol(value)
    if not compact or len(compact) > 18:
        return False
    return bool(re.fullmatch(r"[A-Z0-9]{1,12}([.=/-][A-Z0-9]{1,6})?", compact))


def _infer_asset_type(symbol: str) -> str:
    upper = _compact_symbol(symbol)
    if upper.endswith("-USD") or "/" in upper:
        return "crypto"
    if upper.endswith("=F"):
        return "commodity"
    return "stock"


def _persist_all_asset(symbol: str, name: str | None = None, asset_type: str | None = None) -> None:
    symbol_value = _compact_symbol(symbol)
    if not symbol_value:
        return

    db = SessionLocal()
    try:
        row = db.query(AllAssets).filter(func.upper(AllAssets.symbol) == symbol_value).first()
        resolved_asset_type = (asset_type or _infer_asset_type(symbol_value)).strip().lower()
        resolved_name = (name or symbol_value).strip()
        if row:
            row.name = resolved_name or row.name
            row.asset_type = resolved_asset_type or row.asset_type
        else:
            db.add(
                AllAssets(
                    symbol=symbol_value,
                    name=resolved_name,
                    asset_type=resolved_asset_type,
                )
            )
        db.commit()
        _load_all_assets_records.cache_clear()
    except Exception:
        db.rollback()
    finally:
        db.close()


def _log_resolution(query: str, symbol: str, source: str) -> None:
    print(f"Input query: {query}")
    print(f"Resolved symbol: {symbol}")
    print(f"Source: {source}")


def extract_asset_candidates(query: str) -> list[str]:
    normalized_query = _normalize_query(query)
    if not normalized_query:
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(value: str) -> None:
        normalized = _normalize_query(value)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        candidates.append(normalized)

    add_candidate(normalized_query)

    for pattern in ASSET_QUERY_PATTERNS:
        match = re.search(pattern, normalized_query)
        if match:
            add_candidate(match.group(1))

    tokens = [token for token in normalized_query.split() if token not in ASSET_QUERY_STOPWORDS]
    if tokens:
        filtered_phrase = " ".join(tokens)
        add_candidate(filtered_phrase)
        max_ngram = min(4, len(tokens))
        for width in range(max_ngram, 0, -1):
            for idx in range(0, len(tokens) - width + 1):
                add_candidate(" ".join(tokens[idx : idx + width]))

    return candidates


@lru_cache(maxsize=1)
def _load_all_assets_records() -> tuple[tuple[str, str], ...]:
    db = SessionLocal()
    try:
        rows = db.query(AllAssets.symbol, AllAssets.name).order_by(AllAssets.symbol.asc()).all()
        records: list[tuple[str, str]] = []
        for symbol, name in rows:
            symbol_value = str(symbol or "").strip().upper()
            if not symbol_value:
                continue
            records.append((symbol_value, str(name or symbol_value).strip().lower()))
        return tuple(records)
    except Exception:
        return tuple()
    finally:
        db.close()


@lru_cache(maxsize=1)
def _load_asset_universe_candidates() -> tuple[str, ...]:
    db = SessionLocal()
    try:
        rows = db.query(AssetUniverse.symbol).order_by(AssetUniverse.symbol.asc()).all()
        return tuple(symbol.upper() for (symbol,) in rows if isinstance(symbol, str) and symbol.strip())
    except Exception:
        return tuple()
    finally:
        db.close()


def _resolve_from_all_assets(normalized_query: str) -> tuple[str | None, str | None]:
    db = SessionLocal()
    try:
        query_upper = _compact_symbol(normalized_query)

        exact_match = db.query(AllAssets).filter(func.upper(AllAssets.symbol) == query_upper).first()
        if exact_match and exact_match.symbol:
            return exact_match.symbol.upper(), "all-assets-exact"

        name_exact_match = db.query(AllAssets).filter(func.lower(AllAssets.name) == normalized_query).first()
        if name_exact_match and name_exact_match.symbol:
            return name_exact_match.symbol.upper(), "all-assets-name"

        partial_match = (
            db.query(AllAssets)
            .filter(
                or_(
                    AllAssets.symbol.ilike(f"%{query_upper}%"),
                    AllAssets.name.ilike(f"%{normalized_query}%"),
                )
            )
            .order_by(AllAssets.symbol.asc())
            .first()
        )
        if partial_match and partial_match.symbol:
            return partial_match.symbol.upper(), "all-assets-partial"
        return None, None
    except Exception:
        return None, None
    finally:
        db.close()


def _resolve_from_asset_universe(normalized_query: str) -> tuple[str | None, str | None]:
    db = SessionLocal()
    try:
        query_upper = _compact_symbol(normalized_query)
        exact_match = db.query(AssetUniverse).filter(func.upper(AssetUniverse.symbol) == query_upper).first()
        if exact_match and exact_match.symbol:
            return exact_match.symbol.upper(), "asset-universe-exact"

        partial_match = (
            db.query(AssetUniverse)
            .filter(AssetUniverse.symbol.ilike(f"%{query_upper}%"))
            .order_by(AssetUniverse.symbol.asc())
            .first()
        )
        if partial_match and partial_match.symbol:
            return partial_match.symbol.upper(), "asset-universe-partial"
        return None, None
    except Exception:
        return None, None
    finally:
        db.close()


def _score_quote(normalized_query: str, quote: dict, variant: str) -> int:
    compact = _compact_symbol(normalized_query)
    symbol = str(quote.get("symbol") or "").upper().strip().lstrip("$")
    name = str(quote.get("shortname") or quote.get("longname") or "").lower().strip()
    exchange = str(quote.get("exchange") or quote.get("fullExchangeName") or "").lower().strip()
    quote_type = str(quote.get("quoteType") or quote.get("typeDisp") or "").lower().strip()

    score = 0
    if symbol == compact:
        score += 120
    if name == normalized_query:
        score += 100
    if normalized_query in name:
        score += 80
    if compact and compact in symbol.replace(".NS", "").replace(".BO", ""):
        score += 60
    if symbol.endswith(".NS"):
        score += 30
    if symbol.endswith(".BO"):
        score += 22
    if "nse" in exchange or "bse" in exchange or "india" in exchange:
        score += 25
    if any(hint in normalized_query for hint in NSE_HINTS) and (
        symbol.endswith(".NS") or symbol.endswith(".BO") or "india" in exchange or "nse" in exchange or "bse" in exchange
    ):
        score += 40
    if variant.endswith(" nse") or variant.endswith(" india"):
        if symbol.endswith(".NS") or symbol.endswith(".BO"):
            score += 20
    if quote_type in {"cryptocurrency", "crypto"} and symbol.endswith("-USD"):
        score += 15
    if quote_type in {"equity", "etf"}:
        score += 5
    return score


def _resolve_from_fuzzy(normalized_query: str) -> tuple[str | None, str | None]:
    static_match = get_close_matches(normalized_query, list(STATIC_SYMBOL_MAP.keys()), n=1, cutoff=0.58)
    if static_match:
        return STATIC_SYMBOL_MAP[static_match[0]], "fuzzy-static"

    query_upper = _compact_symbol(normalized_query)
    records = list(_load_all_assets_records())
    name_candidates = [name for _, name in records if name]
    name_match = get_close_matches(normalized_query, name_candidates, n=1, cutoff=0.62)
    if name_match:
        matched_name = name_match[0]
        for symbol, name in records:
            if name == matched_name:
                return symbol, "fuzzy-all-assets-name"

    symbol_candidates = [symbol for symbol, _ in records]
    symbol_match = get_close_matches(query_upper, symbol_candidates, n=1, cutoff=0.72)
    if symbol_match:
        return symbol_match[0].upper(), "fuzzy-all-assets-symbol"

    universe_candidates = list(_load_asset_universe_candidates())
    universe_match = get_close_matches(query_upper, universe_candidates, n=1, cutoff=0.72)
    if universe_match:
        return universe_match[0].upper(), "fuzzy-asset-universe"

    return None, None


def _resolve_from_yfinance_search(normalized_query: str) -> tuple[str | None, str | None]:
    cache_key = ("yf-search", normalized_query)
    cached = runtime_cache.get(cache_key)
    if cached is not None:
        return cached.get("symbol"), cached.get("source")

    search_variants = [normalized_query]
    if normalized_query:
        search_variants.extend(
            [
                f"{normalized_query} stock",
                f"{normalized_query} india",
                f"{normalized_query} nse",
            ]
        )

    best_symbol: str | None = None
    best_name: str | None = None
    best_score = -1

    for variant in search_variants:
        try:
            search = yf.Search(variant, max_results=10)
            quotes = getattr(search, "quotes", None) or []
        except Exception:
            continue

        for quote in quotes:
            raw_symbol = str(quote.get("symbol") or "").upper().strip().lstrip("$")
            if not raw_symbol:
                continue

            symbol = raw_symbol
            exchange = str(quote.get("exchange") or quote.get("fullExchangeName") or "")
            if exchange.lower().startswith("nse") and not symbol.endswith(".NS"):
                symbol = f"{symbol}.NS"
            elif exchange.lower().startswith("bse") and not symbol.endswith(".BO"):
                symbol = f"{symbol}.BO"

            score = _score_quote(normalized_query, {**quote, "symbol": symbol}, variant)
            if score <= best_score:
                continue

            best_score = score
            best_symbol = symbol
            best_name = str(quote.get("shortname") or quote.get("longname") or normalized_query.title())

    if best_symbol:
        _persist_all_asset(best_symbol, name=best_name, asset_type=_infer_asset_type(best_symbol))
        payload = {"symbol": best_symbol, "source": "yfinance-search"}
        runtime_cache.set(cache_key, payload, ttl_seconds=SEARCH_CACHE_TTL_SECONDS)
        return best_symbol, "yfinance-search"

    return None, None


def _resolve_from_direct_probe(normalized_query: str) -> tuple[str | None, str | None]:
    explicit_symbol = _compact_symbol(normalized_query)
    candidates = [explicit_symbol]
    if explicit_symbol and "." not in explicit_symbol and not explicit_symbol.endswith(".NS"):
        candidates.append(f"{explicit_symbol}.NS")
        candidates.append(f"{explicit_symbol}.BO")
    if explicit_symbol and "-" not in explicit_symbol and explicit_symbol not in {"GC=F", "SI=F", "CL=F"}:
        candidates.append(f"{explicit_symbol}-USD")
    if explicit_symbol and "=" not in explicit_symbol:
        candidates.append(f"{explicit_symbol}=F")

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            history = yf.Ticker(candidate).history(
                period="5d",
                interval="1d",
                auto_adjust=False,
                timeout=DIRECT_PROBE_TIMEOUT_SECONDS,
            )
            if history is None or history.empty:
                continue
            _persist_all_asset(candidate, name=normalized_query.title(), asset_type=_infer_asset_type(candidate))
            return candidate, "direct-probe"
        except Exception:
            continue

    return None, None


def resolve_symbol(query: str) -> str:
    candidate_queries = extract_asset_candidates(query)
    if not candidate_queries:
        raise ValueError("Query must not be empty")

    for normalized_query in candidate_queries:
        if normalized_query in STATIC_SYMBOL_MAP:
            symbol = STATIC_SYMBOL_MAP[normalized_query]
            _persist_all_asset(symbol, name=normalized_query.title(), asset_type=_infer_asset_type(symbol))
            _log_resolution(query, symbol, "static")
            return symbol

        explicit_symbol = _compact_symbol(normalized_query)
        if explicit_symbol.endswith("-USD") and explicit_symbol.split("-", 1)[0] in CRYPTO_HINTS:
            _persist_all_asset(explicit_symbol, name=normalized_query.title(), asset_type="crypto")
            _log_resolution(query, explicit_symbol, "explicit-crypto")
            return explicit_symbol

        all_assets_symbol, all_assets_source = _resolve_from_all_assets(normalized_query)
        if all_assets_symbol:
            _log_resolution(query, all_assets_symbol, all_assets_source or "all-assets")
            return all_assets_symbol

        universe_symbol, universe_source = _resolve_from_asset_universe(normalized_query)
        if universe_symbol:
            _log_resolution(query, universe_symbol, universe_source or "asset-universe")
            return universe_symbol

        search_symbol, search_source = _resolve_from_yfinance_search(normalized_query)
        if search_symbol:
            _log_resolution(query, search_symbol, search_source or "yfinance-search")
            return search_symbol

        probe_symbol, probe_source = _resolve_from_direct_probe(normalized_query)
        if probe_symbol:
            _log_resolution(query, probe_symbol, probe_source or "direct-probe")
            return probe_symbol

        fuzzy_symbol, fuzzy_source = _resolve_from_fuzzy(normalized_query)
        if fuzzy_symbol:
            _log_resolution(query, fuzzy_symbol, fuzzy_source or "fuzzy")
            return fuzzy_symbol

        if _looks_like_market_symbol(explicit_symbol):
            fallback_symbol = explicit_symbol
            if any(hint in normalized_query for hint in NSE_HINTS) and not (
                fallback_symbol.endswith(".NS") or fallback_symbol.endswith(".BO")
            ):
                fallback_symbol = f"{fallback_symbol}.NS"
            _log_resolution(query, fallback_symbol, "fallback")
            return fallback_symbol

    raise ValueError(f"Could not resolve an asset symbol from query: {query}")
