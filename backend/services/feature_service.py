from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "log_return_1",
    "log_return_5",
    "momentum_20",
    "volatility_20",
    "sma_20",
    "sma_50",
    "price_vs_sma20",
    "price_vs_sma50",
    "RSI_14",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "volume_change",
    "volume_zscore",
]

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
FEATURE_CLIP_VALUE = 10.0
CLIPPED_FEATURE_COLUMNS = [
    "log_return_1",
    "log_return_5",
    "momentum_20",
    "volatility_20",
    "price_vs_sma20",
    "price_vs_sma50",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "volume_change",
    "volume_zscore",
]


def _validate_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    working_df = df.loc[:, REQUIRED_COLUMNS].copy()
    working_df["timestamp"] = pd.to_datetime(working_df["timestamp"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        working_df[column] = pd.to_numeric(working_df[column], errors="coerce")

    working_df = working_df.dropna(subset=["timestamp", "close", "volume"])
    working_df = working_df.sort_values("timestamp").reset_index(drop=True)
    if working_df.empty:
        raise ValueError("No valid OHLCV rows available after input validation")

    return working_df


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def generate_feature_history_for_asset(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = _validate_input_frame(df)

    close = feature_df["close"].astype(float)
    volume = feature_df["volume"].astype(float)

    feature_df["log_return_1"] = np.log(close / close.shift(1))
    feature_df["log_return_5"] = np.log(close / close.shift(5))
    feature_df["momentum_20"] = close / close.shift(20) - 1.0
    feature_df["volatility_20"] = feature_df["log_return_1"].rolling(20).std()

    feature_df["sma_20"] = close.rolling(20).mean()
    feature_df["sma_50"] = close.rolling(50).mean()
    feature_df["price_vs_sma20"] = close / feature_df["sma_20"] - 1.0
    feature_df["price_vs_sma50"] = close / feature_df["sma_50"] - 1.0

    feature_df["RSI_14"] = _compute_rsi(close, window=14)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    feature_df["MACD"] = ema_12 - ema_26
    feature_df["MACD_signal"] = feature_df["MACD"].ewm(span=9, adjust=False).mean()
    feature_df["MACD_hist"] = feature_df["MACD"] - feature_df["MACD_signal"]

    feature_df["volume_change"] = volume.pct_change()
    rolling_volume_mean = volume.rolling(20).mean()
    rolling_volume_std = volume.rolling(20).std()
    feature_df["volume_zscore"] = (volume - rolling_volume_mean) / (rolling_volume_std + 1e-6)

    feature_df = feature_df.replace([np.inf, -np.inf], 0.0)
    feature_df = feature_df.dropna(subset=FEATURE_COLUMNS)
    if feature_df.empty:
        raise ValueError("Not enough history to compute live features")

    feature_df[CLIPPED_FEATURE_COLUMNS] = feature_df[CLIPPED_FEATURE_COLUMNS].clip(-FEATURE_CLIP_VALUE, FEATURE_CLIP_VALUE)
    feature_df["RSI_14"] = feature_df["RSI_14"].clip(0.0, 100.0)
    return feature_df.reset_index(drop=True)


def generate_features_for_asset(df: pd.DataFrame) -> pd.DataFrame:
    return generate_feature_history_for_asset(df).tail(1).reset_index(drop=True)
