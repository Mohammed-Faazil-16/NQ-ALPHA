from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.asset_universe_agent.load_asset_universe import (
    load_crypto_assets,
    load_nasdaq_assets,
    load_sp500_assets,
)
from backend.database.models.all_assets import AllAssets
from backend.database.postgres import SessionLocal


def _normalize_payloads(payloads):
    deduped = {}
    for asset in payloads:
        symbol = str(asset.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        deduped[symbol] = {
            "symbol": symbol,
            "name": asset.get("name") or symbol,
            "asset_type": asset.get("asset_type") or "stock",
        }
    return list(deduped.values())


def _upsert_assets(db, payloads):
    if not payloads:
        return 0, 0

    symbols = [payload["symbol"] for payload in payloads]
    existing_rows = db.query(AllAssets).filter(AllAssets.symbol.in_(symbols)).all()
    existing_by_symbol = {row.symbol.upper(): row for row in existing_rows}

    inserted = 0
    updated = 0
    for payload in payloads:
        symbol = payload["symbol"]
        existing = existing_by_symbol.get(symbol)
        if existing is None:
            db.add(AllAssets(**payload))
            inserted += 1
            continue

        changed = False
        if (existing.name or "") != payload["name"]:
            existing.name = payload["name"]
            changed = True
        if (existing.asset_type or "") != payload["asset_type"]:
            existing.asset_type = payload["asset_type"]
            changed = True
        if changed:
            updated += 1

    db.commit()
    return inserted, updated


def load_full_assets():
    print("Loading full asset universe into all_assets...")

    loaders = [
        ("S&P500", load_sp500_assets),
        ("NASDAQ", load_nasdaq_assets),
        ("Crypto", load_crypto_assets),
    ]

    all_payloads = []
    for label, loader in loaders:
        try:
            payloads = loader()
            all_payloads.extend(payloads)
            print(f"{label} payload count: {len(payloads)}")
        except Exception as exc:
            print(f"Skipping {label} loader: {exc}")

    normalized_payloads = _normalize_payloads(all_payloads)

    db = SessionLocal()
    try:
        inserted, updated = _upsert_assets(db, normalized_payloads)
        total = db.query(AllAssets).count()
        print(f"Full assets processed: {len(normalized_payloads)}")
        print(f"Inserted: {inserted}")
        print(f"Updated: {updated}")
        print(f"Total all_assets rows: {total}")
    finally:
        db.close()


if __name__ == "__main__":
    load_full_assets()
