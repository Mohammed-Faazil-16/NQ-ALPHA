from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import ccxt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database.models.asset import Asset
from backend.database.models.market_data import MarketData
from backend.database.postgres import SessionLocal


DEFAULT_CRYPTO_START = datetime(2017, 1, 1, tzinfo=timezone.utc)
ONE_DAY_MS = 24 * 60 * 60 * 1000


def fetch_crypto_data(symbol):
    db = SessionLocal()
    exchange = ccxt.binance({"enableRateLimit": True})
    inserted = 0

    try:
        print(f"Fetching crypto data for {symbol}")
        last_row = (
            db.query(MarketData)
            .filter(MarketData.symbol == symbol)
            .order_by(MarketData.timestamp.desc())
            .first()
        )

        if last_row:
            since_dt = last_row.timestamp + timedelta(days=1)
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=timezone.utc)
        else:
            since_dt = DEFAULT_CRYPTO_START

        since_ms = int(since_dt.timestamp() * 1000)

        while True:
            candles = exchange.fetch_ohlcv(symbol, timeframe="1d", since=since_ms, limit=1000)
            if not candles:
                break

            batch_inserted = 0
            for candle in candles:
                timestamp = datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc).replace(tzinfo=None)
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
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5]),
                    source="binance",
                )
                db.add(market_row)
                inserted += 1
                batch_inserted += 1

            db.commit()
            since_ms = candles[-1][0] + ONE_DAY_MS

            if len(candles) < 1000:
                break

        print(f"Inserted {inserted} rows")
        return inserted
    except Exception as exc:
        db.rollback()
        print(f"Error fetching crypto data for {symbol}: {exc}")
        return 0
    finally:
        db.close()


def fetch_crypto():
    db = SessionLocal()
    try:
        crypto_assets = db.query(Asset).filter(Asset.asset_type == "crypto", Asset.active.is_(True)).all()
        for asset in crypto_assets:
            fetch_crypto_data(asset.symbol)
    finally:
        db.close()


if __name__ == "__main__":
    fetch_crypto()
