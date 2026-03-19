from pathlib import Path
import sys

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database.models.features import Features
from backend.database.models.market_data import MarketData
from backend.database.postgres import SessionLocal


def to_float_or_none(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or pd.isna(value):
        return None
    return float(value)


def generate_features():
    db = SessionLocal()
    try:
        symbols = [row[0] for row in db.query(MarketData.symbol).distinct().all()]

        for symbol in symbols:
            print(f"Generating features for {symbol}")

            market_rows = (
                db.query(MarketData)
                .filter(MarketData.symbol == symbol)
                .order_by(MarketData.timestamp)
                .all()
            )

            if not market_rows:
                continue

            df = pd.DataFrame(
                [
                    {
                        "timestamp": row.timestamp,
                        "open": row.open,
                        "high": row.high,
                        "low": row.low,
                        "close": row.close,
                        "volume": row.volume,
                    }
                    for row in market_rows
                ]
            )

            df = df.sort_values("timestamp").reset_index(drop=True)
            df["return_1d"] = df["close"].pct_change()
            df["sma_20"] = df["close"].rolling(20).mean()
            df["sma_50"] = df["close"].rolling(50).mean()
            df["volatility_20"] = df["close"].pct_change().rolling(20).std()
            df["rsi_14"] = RSIIndicator(close=df["close"], window=14).rsi()

            existing_timestamps = {
                row[0]
                for row in db.query(Features.timestamp)
                .filter(Features.symbol == symbol)
                .all()
            }

            inserted = 0
            for _, feature_row in df.iterrows():
                timestamp = feature_row["timestamp"]
                if timestamp in existing_timestamps:
                    continue

                feature = Features(
                    symbol=symbol,
                    timestamp=timestamp,
                    return_1d=to_float_or_none(feature_row["return_1d"]),
                    sma_20=to_float_or_none(feature_row["sma_20"]),
                    sma_50=to_float_or_none(feature_row["sma_50"]),
                    volatility_20=to_float_or_none(feature_row["volatility_20"]),
                    rsi_14=to_float_or_none(feature_row["rsi_14"]),
                )
                db.add(feature)
                existing_timestamps.add(timestamp)
                inserted += 1

            db.commit()
            print(f"Inserted features for {symbol}")
    finally:
        db.close()


if __name__ == "__main__":
    generate_features()
