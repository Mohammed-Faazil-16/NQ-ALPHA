from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, select

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database.models.asset import Asset
from backend.database.models.market_data import MarketData
from backend.database.postgres import SessionLocal


FEATURE_COLUMNS = [
    "log_return_1",
    "log_return_5",
    "log_return_20",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "volatility_10",
    "volatility_20",
    "volatility_60",
    "sma_20",
    "sma_50",
    "sma_200",
    "price_vs_sma20",
    "price_vs_sma50",
    "price_vs_sma200",
    "RSI_14",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "volume_change",
    "volume_zscore",
]


def reflect_features_table(bind):
    metadata = MetaData()
    features_table = Table("features", metadata, autoload_with=bind)
    columns = {column.name for column in features_table.columns}

    missing_columns = {"timestamp", *FEATURE_COLUMNS} - columns
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RuntimeError(f"features table is missing required columns: {missing}")

    if "asset_id" not in columns and "symbol" not in columns:
        raise RuntimeError("features table must contain either asset_id or symbol for asset linkage")

    return features_table, columns


def safe_float(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    return float(value)



def compute_features(df):
    df = df.sort_values("timestamp").reset_index(drop=True).copy()

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    log_close = np.log(close.replace(0, np.nan))
    returns = close.pct_change()

    df["log_return_1"] = log_close.diff(1)
    df["log_return_5"] = log_close.diff(5)
    df["log_return_20"] = log_close.diff(20)

    df["momentum_5"] = close.pct_change(5)
    df["momentum_20"] = close.pct_change(20)
    df["momentum_60"] = close.pct_change(60)

    df["volatility_10"] = returns.rolling(10).std()
    df["volatility_20"] = returns.rolling(20).std()
    df["volatility_60"] = returns.rolling(60).std()

    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_200"] = close.rolling(200).mean()

    df["price_vs_sma20"] = (close / df["sma_20"]) - 1
    df["price_vs_sma50"] = (close / df["sma_50"]) - 1
    df["price_vs_sma200"] = (close / df["sma_200"]) - 1

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    df["volume_change"] = volume.pct_change()
    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std()
    df["volume_zscore"] = (volume - volume_mean) / volume_std.replace(0, np.nan)

    return df



def prepare_insert_row(asset, row, column_names):
    insert_row = {
        "timestamp": row.timestamp,
        "log_return_1": safe_float(row.log_return_1),
        "log_return_5": safe_float(row.log_return_5),
        "log_return_20": safe_float(row.log_return_20),
        "momentum_5": safe_float(row.momentum_5),
        "momentum_20": safe_float(row.momentum_20),
        "momentum_60": safe_float(row.momentum_60),
        "volatility_10": safe_float(row.volatility_10),
        "volatility_20": safe_float(row.volatility_20),
        "volatility_60": safe_float(row.volatility_60),
        "sma_20": safe_float(row.sma_20),
        "sma_50": safe_float(row.sma_50),
        "sma_200": safe_float(row.sma_200),
        "price_vs_sma20": safe_float(row.price_vs_sma20),
        "price_vs_sma50": safe_float(row.price_vs_sma50),
        "price_vs_sma200": safe_float(row.price_vs_sma200),
        "RSI_14": safe_float(row.RSI_14),
        "MACD": safe_float(row.MACD),
        "MACD_signal": safe_float(row.MACD_signal),
        "MACD_hist": safe_float(row.MACD_hist),
        "volume_change": safe_float(row.volume_change),
        "volume_zscore": safe_float(row.volume_zscore),
    }

    if "asset_id" in column_names:
        insert_row["asset_id"] = asset.id
    if "symbol" in column_names:
        insert_row["symbol"] = asset.symbol

    return {key: value for key, value in insert_row.items() if key in column_names}



def generate_features():
    print("Generating features...")

    db = SessionLocal()
    try:
        features_table, column_names = reflect_features_table(db.bind)
        assets = db.query(Asset).filter(Asset.active.is_(True)).order_by(Asset.symbol).all()

        total_inserted = 0

        for asset in assets:
            print(f"Processing asset: {asset.symbol}")
            market_rows = (
                db.query(MarketData)
                .filter(MarketData.symbol == asset.symbol)
                .order_by(MarketData.timestamp)
                .all()
            )

            if not market_rows:
                print(f"Inserted 0 feature rows")
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

            feature_df = compute_features(df)
            print(f"Computed features: {len(feature_df)} rows")

            if "asset_id" in column_names:
                existing_query = select(features_table.c.timestamp).where(features_table.c.asset_id == asset.id)
            else:
                existing_query = select(features_table.c.timestamp).where(features_table.c.symbol == asset.symbol)

            existing_timestamps = {row[0] for row in db.execute(existing_query).all()}

            rows_to_insert = []
            for row in feature_df.itertuples(index=False):
                timestamp = row.timestamp
                if hasattr(timestamp, "to_pydatetime"):
                    timestamp = timestamp.to_pydatetime()
                if timestamp in existing_timestamps:
                    continue

                row_dict = prepare_insert_row(asset, row, column_names)
                row_dict["timestamp"] = timestamp
                rows_to_insert.append(row_dict)
                existing_timestamps.add(timestamp)

            if rows_to_insert:
                db.execute(features_table.insert(), rows_to_insert)
                db.commit()

            inserted = len(rows_to_insert)
            total_inserted += inserted
            print(f"Inserted {inserted} feature rows")

        print(f"Total feature rows inserted: {total_inserted}")
    finally:
        db.close()


if __name__ == "__main__":
    generate_features()
