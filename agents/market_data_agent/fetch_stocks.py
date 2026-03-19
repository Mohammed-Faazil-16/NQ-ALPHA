from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database.models.asset import Asset
from backend.database.models.market_data import MarketData
from backend.database.postgres import SessionLocal


def scalar_to_float(value):
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if pd.isna(value):
        return None
    return float(value)


def fetch_stock_data(symbol, start_date, end_date):
    db = SessionLocal()
    inserted = 0

    try:
        print(f"Fetching stock data for {symbol}")
        data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=False,
            threads=False,
        )

        if data.empty:
            print(f"No data returned for {symbol}")
            return 0

        data = data.reset_index()
        timestamp_column = data.columns[0]

        for _, row in data.iterrows():
            timestamp = row[timestamp_column]
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
                open=scalar_to_float(row["Open"]),
                high=scalar_to_float(row["High"]),
                low=scalar_to_float(row["Low"]),
                close=scalar_to_float(row["Close"]),
                volume=scalar_to_float(row["Volume"]),
                source="yfinance",
            )
            db.add(market_row)
            inserted += 1

        db.commit()
        print(f"Inserted {inserted} rows")
        return inserted
    except Exception as exc:
        db.rollback()
        print(f"Error fetching stock data for {symbol}: {exc}")
        return 0
    finally:
        db.close()


def fetch_stocks():
    db = SessionLocal()
    try:
        stock_assets = db.query(Asset).filter(Asset.asset_type == "stock", Asset.active.is_(True)).all()
        end_date = datetime.utcnow().date().isoformat()
        start_date = "2015-01-01"

        for asset in stock_assets:
            fetch_stock_data(asset.symbol, start_date, end_date)
    finally:
        db.close()


if __name__ == "__main__":
    fetch_stocks()
