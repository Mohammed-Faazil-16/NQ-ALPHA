from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
from sqlalchemy import and_
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database.models.asset_universe import AssetUniverse
from backend.database.models.features import Features
from backend.database.models.market_data import MarketData
from backend.database.models.market_regime import MarketRegime
from backend.database.postgres import SessionLocal


BASE_FEATURES = [
    "log_return_1",
    "sma_50",
    "price_vs_sma20",
    "price_vs_sma50",
    "RSI_14",
    "MACD_hist",
    "volume_zscore",
]

CROSS_ASSET_FEATURES = [
    "market_return",
    "market_volatility",
    "rank_return_1",
    "relative_return_1",
    "relative_strength_20",
    "rolling_rank_mean_10",
    "rolling_rank_std_10",
    "cross_sectional_volatility",
    "top_bottom_spread",
]

FACTOR_FEATURES = [
    "momentum_20",
    "volatility_20",
    "vol_adjusted_return_1",
    "return_zscore_5",
    "price_deviation_20",
    "volume_acceleration",
    "price_volume_divergence",
]

INTERACTION_FEATURES = [
    "momentum_x_volatility",
    "momentum_x_volume",
    "momentum_x_regime",
    "volatility_x_regime",
]

MARKET_BREADTH_FEATURES = [
    "breadth_up_pct",
    "breadth_above_sma50_pct",
    "breadth_dispersion",
]

REGIME_TRANSITION_FEATURES = [
    "regime_change_flag",
    "time_since_regime_change",
    "previous_regime_state",
]

CORRELATION_FEATURES = [
    "asset_market_corr_20",
    "correlation_cluster_strength",
    "corr_x_volatility",
]

VOLUME_SHOCK_FEATURES = [
    "volume_spike",
    "abnormal_volume_indicator",
]

VOLATILITY_REGIME_FEATURES = [
    "high_vol_regime_flag",
]

LAGGED_SIGNAL_FEATURES = [
    "lag_rank_return_1",
]

GLOBAL_CONTEXT_FEATURES = [
    "market_return",
    "market_volatility",
    "breadth_up_pct",
    "breadth_above_sma50_pct",
    "breadth_dispersion",
    "cross_sectional_volatility",
    "top_bottom_spread",
]

FEATURE_COLUMNS = (
    BASE_FEATURES
    + CROSS_ASSET_FEATURES
    + FACTOR_FEATURES
    + INTERACTION_FEATURES
    + MARKET_BREADTH_FEATURES
    + REGIME_TRANSITION_FEATURES
    + CORRELATION_FEATURES
    + VOLUME_SHOCK_FEATURES
    + VOLATILITY_REGIME_FEATURES
    + LAGGED_SIGNAL_FEATURES
)
DEFAULT_ACTIVE_FEATURES = [
    "momentum_20",
    "volatility_20",
    "vol_adjusted_return_1",
    "rank_return_1",
    "relative_strength_20",
    "rolling_rank_mean_10",
    "rolling_rank_std_10",
    "cross_sectional_volatility",
    "top_bottom_spread",
    "return_zscore_5",
    "price_deviation_20",
    "volume_acceleration",
    "price_volume_divergence",
    "asset_market_corr_20",
    "correlation_cluster_strength",
    "corr_x_volatility",
    "momentum_x_volatility",
    "momentum_x_volume",
    "momentum_x_regime",
    "volatility_x_regime",
]
STORED_FEATURE_COLUMNS = [
    "log_return_1",
    "log_return_5",
    "log_return_20",
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

REGIME_TO_ID = {
    "bull": 0,
    "normal": 1,
    "volatile": 2,
    "crisis": 3,
}

SEQUENCE_LENGTH = 30
MIN_ASSETS = 10
FEATURE_CLIP_VALUE = 5.0
EXTREME_VALUE_THRESHOLD = 1_000.0
TARGET_CLIP_VALUE = 1.0
SPLIT_DATE = pd.Timestamp("2022-01-01")
TARGET_HORIZON = 5
LOGS_DIR = PROJECT_ROOT / "logs"
ACTIVE_FEATURES_PATH = LOGS_DIR / "selected_features.json"


def get_feature_groups():
    return {
        "base": list(BASE_FEATURES),
        "cross_asset": list(CROSS_ASSET_FEATURES),
        "factor": list(FACTOR_FEATURES),
        "interaction": list(INTERACTION_FEATURES),
        "market_breadth": list(MARKET_BREADTH_FEATURES),
        "regime_transition": list(REGIME_TRANSITION_FEATURES),
        "correlation": list(CORRELATION_FEATURES),
        "volume_shock": list(VOLUME_SHOCK_FEATURES),
        "volatility_regime": list(VOLATILITY_REGIME_FEATURES),
        "lagged": list(LAGGED_SIGNAL_FEATURES),
    }


def _validate_feature_list(features):
    cleaned = []
    for feature in features:
        if feature in FEATURE_COLUMNS and feature not in cleaned:
            cleaned.append(feature)
    return cleaned


def _load_active_features_from_disk():
    if not ACTIVE_FEATURES_PATH.exists():
        return None

    try:
        data = json.loads(ACTIVE_FEATURES_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    if isinstance(data, dict):
        features = data.get("active_features", [])
    elif isinstance(data, list):
        features = data
    else:
        return None

    validated = _validate_feature_list(features)
    return validated or None


ACTIVE_FEATURES = _load_active_features_from_disk() or list(DEFAULT_ACTIVE_FEATURES)


def get_active_features():
    return list(ACTIVE_FEATURES)


def set_active_features(features, persist=True):
    global ACTIVE_FEATURES

    validated = _validate_feature_list(features)
    if not validated:
        raise ValueError("No valid active features were provided")

    ACTIVE_FEATURES = validated

    if persist:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        ACTIVE_FEATURES_PATH.write_text(
            json.dumps({"active_features": ACTIVE_FEATURES}, indent=2),
            encoding="utf-8",
        )

    return get_active_features()


def reset_active_features(persist=True):
    return set_active_features(DEFAULT_ACTIVE_FEATURES, persist=persist)


class AlphaDataset(Dataset):
    def __init__(self, features_tensor, regimes_tensor, targets_tensor, masks_tensor, timestamps_tensor, symbols):
        self.features_tensor = features_tensor
        self.regimes_tensor = regimes_tensor
        self.targets_tensor = targets_tensor
        self.masks_tensor = masks_tensor
        self.timestamps_tensor = timestamps_tensor
        self.symbols = symbols

    def __len__(self):
        return len(self.timestamps_tensor)

    def __getitem__(self, idx):
        return (
            self.features_tensor[idx],
            self.regimes_tensor[idx],
            self.targets_tensor[idx],
            self.masks_tensor[idx],
        )


def _empty_dataset(sequence_length=SEQUENCE_LENGTH):
    return AlphaDataset(
        torch.empty((0, 0, sequence_length, len(get_active_features())), dtype=torch.float32),
        torch.empty((0, 0), dtype=torch.long),
        torch.empty((0, 0), dtype=torch.float32),
        torch.empty((0, 0), dtype=torch.float32),
        torch.empty((0,), dtype=torch.long),
        [],
    )


def load_selected_symbols(db):
    symbols = [
        row[0]
        for row in db.query(AssetUniverse.symbol)
        .order_by(AssetUniverse.score.desc(), AssetUniverse.symbol.asc())
        .all()
    ]

    if not symbols:
        raise RuntimeError("No selected assets found in asset_universe. Run select_clean_assets.py first.")

    print(f"Using {len(symbols)} high-quality assets for training")
    return symbols


def _compute_future_return(series, horizon=TARGET_HORIZON):
    future_return = pd.Series(0.0, index=series.index, dtype=float)
    for step in range(1, horizon + 1):
        future_return = future_return.add(series.shift(-step), fill_value=0.0)

    if len(future_return) > 0:
        future_return.iloc[-min(horizon, len(future_return)) :] = np.nan
    return future_return


def _normalize_cross_sectional_rank(series):
    normalized = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.notna()
    count = int(valid.sum())

    if count == 0:
        return normalized
    if count == 1:
        normalized.loc[valid] = 0.0
        return normalized

    ranked = series[valid].rank(method="average")
    normalized.loc[valid] = 2.0 * ((ranked - 1.0) / (count - 1.0)) - 1.0
    return normalized


def _compute_rank_based_target(df):
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True).copy()
    df["future_return"] = df.groupby("symbol")["log_return_1"].transform(_compute_future_return)
    df["target_rank"] = df.groupby("timestamp")["future_return"].transform(_normalize_cross_sectional_rank)
    df["target_return"] = df["target_rank"].clip(-1.0, 1.0)
    df["target_return"] = df["target_return"].replace([np.inf, -np.inf], np.nan)
    return df


def _compute_cross_sectional_rank(series_by_timestamp):
    return series_by_timestamp.rank(pct=True)


def _rolling_corr(left, right, window):
    return left.rolling(window, min_periods=max(5, window // 2)).corr(right)


def _compute_time_since_change(flags):
    flags = flags.fillna(0).astype(int)
    steps = []
    since = 0
    for idx, flag in enumerate(flags):
        if idx == 0 or flag == 1:
            since = 0
        else:
            since += 1
        steps.append(float(since))
    return pd.Series(steps, index=flags.index, dtype=float)


def _compute_research_features(df):
    grouped = df.groupby("symbol", sort=False)

    df["momentum_5"] = grouped["log_return_1"].transform(lambda series: series.rolling(5).sum())
    df["momentum_10"] = grouped["log_return_1"].transform(lambda series: series.rolling(10).sum())
    df["momentum_20"] = grouped["log_return_1"].transform(lambda series: series.rolling(20).sum())

    df["reversion_5"] = -df["momentum_5"]
    df["reversion_10"] = -df["momentum_10"]

    df["volatility_10"] = grouped["log_return_1"].transform(lambda series: series.rolling(10).std())
    df["volatility_20"] = grouped["log_return_1"].transform(lambda series: series.rolling(20).std())
    df["volume_trend"] = grouped["volume"].transform(lambda series: series.rolling(10).mean())

    df["market_return"] = df.groupby("timestamp")["log_return_1"].transform("mean")
    df["market_volatility"] = df.groupby("timestamp")["log_return_1"].transform("std").fillna(0.0)
    df["relative_strength"] = df["log_return_1"] - df["market_return"]
    df["market_momentum_5"] = df.groupby("timestamp")["momentum_5"].transform("mean")
    df["zscore_return"] = (
        df["log_return_1"] - df.groupby("timestamp")["log_return_1"].transform("mean")
    ) / (df.groupby("timestamp")["log_return_1"].transform("std") + 1e-6)

    df["rank_return_1"] = df.groupby("timestamp")["log_return_1"].transform(_compute_cross_sectional_rank)
    df["rank_return_5"] = df.groupby("timestamp")["log_return_5"].transform(_compute_cross_sectional_rank)
    df["rank_volume"] = df.groupby("timestamp")["volume"].transform(_compute_cross_sectional_rank)

    df["relative_return_1"] = df["log_return_1"] - df["market_return"]
    df["relative_momentum_5"] = df["momentum_5"] - df["market_momentum_5"]
    df["relative_strength_20"] = df["momentum_20"] - df.groupby("timestamp")["momentum_20"].transform("mean")
    df["rolling_rank_mean_10"] = grouped["rank_return_1"].transform(lambda series: series.rolling(10).mean())
    df["rolling_rank_std_10"] = grouped["rank_return_1"].transform(lambda series: series.rolling(10).std())

    df["cross_sectional_volatility"] = df.groupby("timestamp")["log_return_1"].transform("std")
    df["top_bottom_spread"] = df.groupby("timestamp")["log_return_1"].transform(
        lambda series: series.quantile(0.9) - series.quantile(0.1)
    )

    df["vol_adjusted_return_1"] = df["log_return_1"] / (df["volatility_20"] + 1e-6)
    df["liquidity_ratio"] = df["volume"] / (df["volume_trend"] + 1e-6)
    rolling_return_mean_5 = grouped["log_return_1"].transform(lambda series: series.rolling(5).mean())
    rolling_return_std_5 = grouped["log_return_1"].transform(lambda series: series.rolling(5).std())
    df["return_zscore_5"] = (df["log_return_1"] - rolling_return_mean_5) / (rolling_return_std_5 + 1e-6)
    df["price_deviation_20"] = df["price_vs_sma20"]

    rolling_volume_mean_10 = grouped["volume"].transform(lambda series: series.rolling(10).mean())
    df["volume_acceleration"] = df["volume"] / (rolling_volume_mean_10 + 1e-6)
    df["price_volume_divergence"] = df["log_return_1"] * df["volume_zscore"]

    regime_ids = df["regime_label"].map(REGIME_TO_ID).fillna(0).astype(float)

    df["breadth_up_pct"] = df.groupby("timestamp")["log_return_1"].transform(lambda series: (series > 0).mean())
    df["breadth_above_sma50_pct"] = df.groupby("timestamp")["price_vs_sma50"].transform(lambda series: (series > 0).mean())
    df["breadth_dispersion"] = df.groupby("timestamp")["log_return_1"].transform(
        lambda series: series.quantile(0.75) - series.quantile(0.25)
    )

    df["previous_regime_state"] = grouped["regime_label"].shift(1).map(REGIME_TO_ID).fillna(0.0)
    df["regime_change_flag"] = grouped["regime_label"].transform(
        lambda series: series.ne(series.shift(1)).fillna(False).astype(float)
    )
    df["time_since_regime_change"] = grouped["regime_change_flag"].transform(_compute_time_since_change)

    rolling_market = grouped.apply(
        lambda group: _rolling_corr(group["log_return_1"], group["market_return"], 20)
    ).reset_index(level=0, drop=True)
    df["asset_market_corr_20"] = rolling_market.sort_index()

    symbol_returns = df.pivot(index="timestamp", columns="symbol", values="log_return_1").sort_index()
    cluster_strength_frames = []
    for symbol in symbol_returns.columns:
        peer_mean = symbol_returns.drop(columns=symbol).mean(axis=1) if symbol_returns.shape[1] > 1 else symbol_returns[symbol]
        cluster_corr = _rolling_corr(symbol_returns[symbol], peer_mean, 20)
        cluster_strength_frames.append(
            pd.DataFrame(
                {
                    "timestamp": cluster_corr.index,
                    "symbol": symbol,
                    "correlation_cluster_strength": cluster_corr.values,
                }
            )
        )
    cluster_strength_df = pd.concat(cluster_strength_frames, ignore_index=True)
    df = df.merge(cluster_strength_df, on=["timestamp", "symbol"], how="left")

    rolling_volume_mean = grouped["volume"].transform(lambda series: series.rolling(20, min_periods=5).mean())
    rolling_volume_std = grouped["volume"].transform(lambda series: series.rolling(20, min_periods=5).std())
    df["volume_spike"] = df["volume"] / (rolling_volume_mean + 1e-6)
    rolling_volume_zscore = (df["volume"] - rolling_volume_mean) / (rolling_volume_std + 1e-6)
    df["abnormal_volume_indicator"] = (rolling_volume_zscore.abs() > 2.0).astype(float)

    df["momentum_x_volatility"] = df["momentum_20"] * df["volatility_20"]
    df["momentum_x_volume"] = df["momentum_20"] * df["volume_zscore"]
    df["corr_x_volatility"] = df["asset_market_corr_20"] * df["volatility_20"]
    df["momentum_x_regime"] = df["momentum_20"] * regime_ids
    df["volatility_x_regime"] = df["volatility_20"] * regime_ids

    rolling_vol_median = grouped["volatility_20"].transform(
        lambda series: series.rolling(60, min_periods=20).median()
    )
    df["high_vol_regime_flag"] = (df["volatility_20"] > rolling_vol_median).astype(float)

    df["lag_rank_return_1"] = grouped["rank_return_1"].shift(1)

    return df


def load_training_dataframe():
    db = SessionLocal()
    try:
        selected_symbols = load_selected_symbols(db)
        query = (
            db.query(
                Features.symbol,
                Features.timestamp,
                Features.log_return_1,
                Features.log_return_5,
                Features.log_return_20,
                Features.momentum_20,
                Features.momentum_60,
                Features.volatility_10,
                Features.volatility_20,
                Features.volatility_60,
                Features.sma_20,
                Features.sma_50,
                Features.sma_200,
                Features.price_vs_sma20,
                Features.price_vs_sma50,
                Features.price_vs_sma200,
                Features.RSI_14,
                Features.MACD,
                Features.MACD_signal,
                Features.MACD_hist,
                Features.volume_change,
                Features.volume_zscore,
                MarketRegime.regime_label,
            )
            .join(
                MarketRegime,
                and_(
                    Features.symbol == MarketRegime.symbol,
                    Features.timestamp == MarketRegime.timestamp,
                ),
            )
            .filter(Features.symbol.in_(selected_symbols))
            .order_by(Features.symbol, Features.timestamp)
        )

        rows = query.all()
        if not rows:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "timestamp",
                    *FEATURE_COLUMNS,
                    "regime_label",
                    "future_return",
                    "target_return",
                    "target_rank",
                ]
            )

        columns = [
            "symbol",
            "timestamp",
            "log_return_1",
            "log_return_5",
            "log_return_20",
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
            "regime_label",
        ]
        df = pd.DataFrame(rows, columns=columns)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        volume_rows = (
            db.query(MarketData.symbol, MarketData.timestamp, MarketData.volume)
            .filter(MarketData.symbol.in_(selected_symbols))
            .all()
        )
        if volume_rows:
            volume_df = pd.DataFrame(volume_rows, columns=["symbol", "timestamp", "volume"])
            volume_df["timestamp"] = pd.to_datetime(volume_df["timestamp"])
            volume_df = (
                volume_df.sort_values(["symbol", "timestamp"])
                .groupby(["symbol", "timestamp"], as_index=False)["volume"]
                .mean()
            )
            df = df.merge(volume_df, on=["symbol", "timestamp"], how="left")
        else:
            df["volume"] = np.nan

        df = _compute_research_features(df)

        print("Using cross-asset enhanced features")
        print("Using financial alpha factors")
        print("Using new information source features")
        print("Using strict cross-sectional normalization inputs")
        df = _compute_rank_based_target(df)
        print("Using cross-sectional rank target")

        return df
    finally:
        db.close()


def sanitize_training_dataframe(df):
    if df.empty:
        return df

    df = df.copy()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df["target_return"] = pd.to_numeric(df["target_return"], errors="coerce")

    raw_feature_array = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    feature_nan_count = int(np.isnan(raw_feature_array).sum())
    feature_inf_count = int(np.isinf(raw_feature_array).sum())

    print(f"Feature NaN count before sanitization: {feature_nan_count}")
    print(f"Feature Inf count before sanitization: {feature_inf_count}")

    feature_frame = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    max_abs_by_feature = feature_frame.abs().max().sort_values(ascending=False)
    extreme_features = max_abs_by_feature[max_abs_by_feature > EXTREME_VALUE_THRESHOLD].head(10)

    if not extreme_features.empty:
        formatted = ", ".join(f"{name}={value:.4f}" for name, value in extreme_features.items())
        print(f"Extreme raw feature ranges detected: {formatted}")
    else:
        top_features = max_abs_by_feature.head(5)
        formatted = ", ".join(f"{name}={value:.4f}" for name, value in top_features.items())
        print(f"Top raw feature magnitudes: {formatted}")

    df[FEATURE_COLUMNS] = feature_frame
    df["target_return"] = df["target_return"].replace([np.inf, -np.inf], np.nan)

    invalid_target_rows = int(df["target_return"].isna().sum())
    if invalid_target_rows:
        print(f"Dropping {invalid_target_rows} rows with invalid target_return")

    invalid_regime_rows = int((~df["regime_label"].isin(REGIME_TO_ID)).sum())
    if invalid_regime_rows:
        print(f"Dropping {invalid_regime_rows} rows with invalid regime labels")

    df = df[df["regime_label"].isin(REGIME_TO_ID)]
    df = df.dropna(subset=["target_return"]).reset_index(drop=True)
    df["target_return"] = df["target_return"].clip(-TARGET_CLIP_VALUE, TARGET_CLIP_VALUE)

    return df


def _recompute_targets_within_split(split_df):
    split_df = _compute_rank_based_target(split_df)
    split_df = split_df.dropna(subset=["target_return"]).reset_index(drop=True)
    split_df["target_return"] = split_df["target_return"].clip(-TARGET_CLIP_VALUE, TARGET_CLIP_VALUE)
    return split_df


def _normalize_per_timestamp(split_df, fill_values, global_means, global_stds):
    feature_frame = split_df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.fillna(fill_values)

    timestamp_index = split_df["timestamp"]
    cross_sectional_features = [feature for feature in FEATURE_COLUMNS if feature not in GLOBAL_CONTEXT_FEATURES]
    normalized = pd.DataFrame(index=feature_frame.index, columns=FEATURE_COLUMNS, dtype=np.float32)

    if cross_sectional_features:
        group_means = feature_frame[cross_sectional_features].groupby(timestamp_index).transform("mean")
        group_stds = feature_frame[cross_sectional_features].groupby(timestamp_index).transform("std")
        normalized[cross_sectional_features] = (
            (feature_frame[cross_sectional_features] - group_means) / (group_stds + 1e-6)
        )

    if GLOBAL_CONTEXT_FEATURES:
        normalized[GLOBAL_CONTEXT_FEATURES] = (
            feature_frame[GLOBAL_CONTEXT_FEATURES] - global_means[GLOBAL_CONTEXT_FEATURES]
        ) / (global_stds[GLOBAL_CONTEXT_FEATURES] + 1e-6)

    normalized = normalized.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    normalized = normalized.clip(-FEATURE_CLIP_VALUE, FEATURE_CLIP_VALUE)
    split_df[FEATURE_COLUMNS] = normalized.astype(np.float32)
    return split_df


def split_and_normalize_dataframe(df, split_date=SPLIT_DATE):
    if df.empty:
        return df.copy(), df.copy()

    train_df = df[df["timestamp"] < split_date].copy()
    test_df = df[df["timestamp"] >= split_date].copy()

    train_df = _recompute_targets_within_split(train_df)
    test_df = _recompute_targets_within_split(test_df)

    if train_df.empty:
        raise RuntimeError(f"No training rows found before {split_date.date()}")
    if test_df.empty:
        raise RuntimeError(f"No test rows found on or after {split_date.date()}")

    train_features = train_df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    train_fill_values = train_features.median(numeric_only=True).fillna(0.0)
    train_feature_frame = train_features.fillna(train_fill_values)
    global_means = train_feature_frame[GLOBAL_CONTEXT_FEATURES].mean().fillna(0.0)
    global_stds = train_feature_frame[GLOBAL_CONTEXT_FEATURES].std().replace(0, np.nan).fillna(1.0)

    print("Applying hybrid normalization: cross-sectional for asset features, temporal z-score for global context")
    train_df = _normalize_per_timestamp(train_df, train_fill_values, global_means, global_stds)
    test_df = _normalize_per_timestamp(test_df, train_fill_values, global_means, global_stds)

    train_feature_array = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)
    test_feature_array = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=False)

    if not np.isfinite(train_feature_array).all():
        raise RuntimeError("Non-finite values remain in training features after normalization")
    if not np.isfinite(test_feature_array).all():
        raise RuntimeError("Non-finite values remain in test features after normalization")

    print(
        "Train normalized feature range: "
        f"min={train_feature_array.min():.4f}, max={train_feature_array.max():.4f}"
    )
    print(
        "Test normalized feature range: "
        f"min={test_feature_array.min():.4f}, max={test_feature_array.max():.4f}"
    )

    return train_df, test_df


def build_modeling_dataframes(split_date=SPLIT_DATE):
    df = sanitize_training_dataframe(load_training_dataframe())
    if df.empty:
        return df.copy(), df.copy()
    return split_and_normalize_dataframe(df, split_date=split_date)


def build_dataset_from_dataframe(df, sequence_length=SEQUENCE_LENGTH):
    if df.empty:
        return _empty_dataset(sequence_length)

    active_features = get_active_features()
    symbol_groups = {}
    for symbol, group in df.groupby("symbol", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < sequence_length:
            continue

        feature_matrix = group[active_features].to_numpy(dtype=np.float32, copy=False)
        regime_series = group["regime_label"].map(REGIME_TO_ID).to_numpy(dtype=np.int64, copy=False)
        target_series = group["target_return"].to_numpy(dtype=np.float32, copy=False)
        timestamp_series = group["timestamp"]

        per_timestamp = {}
        for end_idx in range(sequence_length - 1, len(group)):
            start_idx = end_idx - sequence_length + 1
            timestamp = timestamp_series.iloc[end_idx]
            per_timestamp[timestamp] = {
                "features": feature_matrix[start_idx : end_idx + 1],
                "regime": regime_series[end_idx],
                "target": target_series[end_idx],
            }

        if per_timestamp:
            symbol_groups[symbol] = per_timestamp

    if not symbol_groups:
        return _empty_dataset(sequence_length)

    symbol_order = sorted(symbol_groups)
    all_timestamps = sorted(set().union(*(set(per_timestamp) for per_timestamp in symbol_groups.values())))

    if not all_timestamps:
        print("No timestamps found across selected assets after sequence construction")
        return _empty_dataset(sequence_length)

    feature_batches = []
    regime_batches = []
    target_batches = []
    timestamp_values = []
    asset_counts = []

    for timestamp in all_timestamps:
        feature_rows = []
        regime_rows = []
        target_rows = []

        for symbol in symbol_order:
            payload = symbol_groups[symbol].get(timestamp)
            if payload is None:
                continue
            feature_rows.append(payload["features"])
            regime_rows.append(payload["regime"])
            target_rows.append(payload["target"])

        if len(feature_rows) < MIN_ASSETS:
            continue

        feature_batches.append(feature_rows)
        regime_batches.append(regime_rows)
        target_batches.append(target_rows)
        timestamp_values.append(pd.Timestamp(timestamp).value)
        asset_counts.append(len(feature_rows))

    if not feature_batches:
        print(f"No cross-sectional samples met the minimum asset threshold of {MIN_ASSETS}")
        return _empty_dataset(sequence_length)

    max_assets = max(asset_counts)
    num_samples = len(feature_batches)
    feature_array = np.zeros((num_samples, max_assets, sequence_length, len(active_features)), dtype=np.float32)
    regime_array = np.zeros((num_samples, max_assets), dtype=np.int64)
    target_array = np.zeros((num_samples, max_assets), dtype=np.float32)
    mask_array = np.zeros((num_samples, max_assets), dtype=np.float32)

    for sample_idx, (feature_rows, regime_rows, target_rows) in enumerate(
        zip(feature_batches, regime_batches, target_batches)
    ):
        count = len(feature_rows)
        feature_array[sample_idx, :count] = np.asarray(feature_rows, dtype=np.float32)
        regime_array[sample_idx, :count] = np.asarray(regime_rows, dtype=np.int64)
        target_array[sample_idx, :count] = np.asarray(target_rows, dtype=np.float32)
        mask_array[sample_idx, :count] = 1.0

    timestamp_array = np.asarray(timestamp_values, dtype=np.int64)

    if not np.isfinite(feature_array).all():
        raise RuntimeError("Non-finite values detected in cross-sectional feature batches")
    if not np.isfinite(target_array).all():
        raise RuntimeError("Non-finite values detected in cross-sectional target batches")

    print(f"Cross-sectional samples: {len(timestamp_array)}")
    print(f"Average assets per sample: {float(np.mean(asset_counts)):.2f}")

    return AlphaDataset(
        torch.from_numpy(feature_array),
        torch.from_numpy(regime_array),
        torch.from_numpy(target_array),
        torch.from_numpy(mask_array),
        torch.from_numpy(timestamp_array),
        symbol_order,
    )


def build_alpha_datasets(sequence_length=SEQUENCE_LENGTH, split_date=SPLIT_DATE):
    train_df, test_df = build_modeling_dataframes(split_date=split_date)
    if train_df.empty and test_df.empty:
        empty_dataset = _empty_dataset(sequence_length)
        return empty_dataset, empty_dataset

    train_dataset = build_dataset_from_dataframe(train_df, sequence_length=sequence_length)
    test_dataset = build_dataset_from_dataframe(test_df, sequence_length=sequence_length)

    return train_dataset, test_dataset


def build_alpha_dataset(sequence_length=SEQUENCE_LENGTH, split="train", split_date=SPLIT_DATE):
    train_dataset, test_dataset = build_alpha_datasets(sequence_length=sequence_length, split_date=split_date)

    if split == "train":
        return train_dataset
    if split == "test":
        return test_dataset

    raise ValueError("split must be 'train' or 'test'")
