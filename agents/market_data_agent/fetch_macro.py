from pathlib import Path
import sys

from fredapi import Fred

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.config import FRED_API_KEY
from backend.database.models.asset import Asset
from backend.database.models.market_data import MarketData
from backend.database.postgres import SessionLocal


MACRO_SYMBOLS = ("FEDFUNDS", "CPIAUCSL", "UNRATE")


def fetch_macro_data(symbol):
    db = SessionLocal()
    fred = Fred(api_key=FRED_API_KEY)
    inserted = 0

    try:
        print(f"Fetching macro data for {symbol}")
        series = fred.get_series(symbol)
        if series.empty:
            print(f"No data returned for {symbol}")
            return 0

        for timestamp, value in series.items():
            if hasattr(timestamp, "to_pydatetime"):
                timestamp = timestamp.to_pydatetime()

            existing = (
                db.query(MarketData)
                .filter(
                    MarketData.symbol == symbol,
                    MarketData.timestamp == timestamp,
                )
                .first()
            )
            if existing:
                continue

            market_row = MarketData(
                symbol=symbol,
                timestamp=timestamp,
                open=float(value),
                high=float(value),
                low=float(value),
                close=float(value),
                volume=0.0,
                source="fred",
            )
            db.add(market_row)
            inserted += 1

        db.commit()
        print(f"Inserted {inserted} rows")
        return inserted
    except Exception as exc:
        db.rollback()
        print(f"Error fetching macro data for {symbol}: {exc}")
        return 0
    finally:
        db.close()


def fetch_macro():
    db = SessionLocal()
    try:
        macro_assets = db.query(Asset).filter(Asset.asset_type == "macro", Asset.active.is_(True)).all()
        for asset in macro_assets:
            fetch_macro_data(asset.symbol)
    finally:
        db.close()


if __name__ == "__main__":
    fetch_macro()
