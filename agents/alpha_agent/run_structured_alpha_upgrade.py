from pathlib import Path
import contextlib
import io
import random
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.alpha_agent.dataset_builder import (
    GLOBAL_CONTEXT_FEATURES,
    REGIME_TO_ID,
    SPLIT_DATE,
    load_training_dataframe,
    set_active_features,
)
from agents.alpha_agent.loss_functions import masked_mse_loss, masked_pairwise_ranking_loss
from agents.alpha_agent.neuroquant_model import NeuroQuantAlphaModel


warnings.filterwarnings("ignore", message="enable_nested_tensor is True")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = PROJECT_ROOT / "models" / "neuroquant_alpha_model.pt"
MODEL_TYPE = "Transformer"
SEED = 42
TARGET_HORIZON = 10
SEQUENCE_LENGTH = 30
EPOCHS = 8
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
TOP_K_FRACTION = 0.10
MAX_FEATURES = 30
MAX_SAMPLE_TIMESTAMPS = 1000

FEATURE_CANDIDATES = [
    "sma_50",
    "RSI_14",
    "momentum_20",
    "volatility_20",
    "vol_adjusted_return_1",
    "rank_return_1",
    "breadth_up_pct",
    "breadth_above_sma50_pct",
    "breadth_dispersion",
    "regime_change_flag",
    "time_since_regime_change",
    "previous_regime_state",
    "asset_market_corr_20",
    "correlation_cluster_strength",
    "volume_spike",
    "abnormal_volume_indicator",
    "high_vol_regime_flag",
    "lag_rank_return_1",
    "momentum_20_lag_1",
    "momentum_20_lag_2",
    "momentum_20_lag_5",
    "volatility_20_lag_1",
    "volatility_20_lag_2",
    "volatility_20_lag_5",
    "rank_return_1_lag_1",
    "rank_return_1_lag_2",
    "rank_return_1_lag_5",
    "volume_spike_lag_1",
    "volume_spike_lag_2",
    "volume_spike_lag_5",
    "momentum_20_zscore_20",
    "vol_adjusted_return_1_zscore_20",
    "asset_market_corr_20_zscore_20",
    "volume_spike_zscore_20",
    "momentum_volatility_interaction",
    "return_regime_interaction",
]

FEATURE_GROUPS = {
    "momentum": {
        "momentum_20",
        "rank_return_1",
        "lag_rank_return_1",
        "momentum_20_lag_1",
        "momentum_20_lag_2",
        "momentum_20_lag_5",
        "rank_return_1_lag_1",
        "rank_return_1_lag_2",
        "rank_return_1_lag_5",
        "momentum_20_zscore_20",
        "vol_adjusted_return_1",
        "vol_adjusted_return_1_zscore_20",
    },
    "volatility": {
        "volatility_20",
        "high_vol_regime_flag",
        "volatility_20_lag_1",
        "volatility_20_lag_2",
        "volatility_20_lag_5",
        "momentum_volatility_interaction",
    },
    "regime": {
        "regime_change_flag",
        "time_since_regime_change",
        "previous_regime_state",
        "return_regime_interaction",
        "breadth_up_pct",
        "breadth_above_sma50_pct",
        "breadth_dispersion",
    },
    "volume": {
        "volume_spike",
        "abnormal_volume_indicator",
        "volume_spike_lag_1",
        "volume_spike_lag_2",
        "volume_spike_lag_5",
        "volume_spike_zscore_20",
    },
    "correlation": {
        "asset_market_corr_20",
        "correlation_cluster_strength",
        "asset_market_corr_20_zscore_20",
    },
}


class AlphaClassificationDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.from_numpy(sample["features"]),
            torch.from_numpy(sample["regimes"]),
            torch.from_numpy(sample["target"]),
            torch.from_numpy(sample["mask"]),
            torch.from_numpy(sample["future_return_10"]),
        )


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _rolling_zscore(series, window=20):
    mean = series.rolling(window, min_periods=5).mean()
    std = series.rolling(window, min_periods=5).std()
    return (series - mean) / (std + 1e-6)


def _centered_rank(series):
    ranked = series.rank(pct=True)
    return ranked - 0.5


def _engineer_research_features(df):
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True).copy()
    grouped = df.groupby("symbol", sort=False)

    lag_sources = {
        "momentum_20": [1, 2, 5],
        "volatility_20": [1, 2, 5],
        "rank_return_1": [1, 2, 5],
        "volume_spike": [1, 2, 5],
    }
    for feature_name, lags in lag_sources.items():
        for lag in lags:
            df[f"{feature_name}_lag_{lag}"] = grouped[feature_name].shift(lag)

    rolling_zscore_sources = [
        "momentum_20",
        "vol_adjusted_return_1",
        "asset_market_corr_20",
        "volume_spike",
    ]
    for feature_name in rolling_zscore_sources:
        df[f"{feature_name}_zscore_20"] = grouped[feature_name].transform(_rolling_zscore)

    regime_id = df["regime_label"].map(REGIME_TO_ID).fillna(0.0).astype(float)
    df["momentum_volatility_interaction"] = df["momentum_20"] * df["volatility_20"]
    df["return_regime_interaction"] = df["log_return_1"] * regime_id

    df["future_return_10"] = grouped["log_return_1"].transform(
        lambda series: series.rolling(TARGET_HORIZON).sum().shift(-TARGET_HORIZON)
    )
    return df


def _add_target_signal(df):
    df = df.copy()
    df["target_signal"] = df.groupby("timestamp")["future_return_10"].transform(_centered_rank)
    df["target_signal"] = df["target_signal"].clip(-0.5, 0.5)
    return df


def _safe_cross_sectional_ic(group, feature_name):
    sample = group[[feature_name, "future_return_10"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < 8:
        return np.nan
    if sample[feature_name].nunique() < 2 or sample["future_return_10"].nunique() < 2:
        return np.nan
    return float(sample[feature_name].corr(sample["future_return_10"], method="spearman"))


def _select_top_features(train_df):
    selection_timestamps = sorted(train_df["timestamp"].unique())[-MAX_SAMPLE_TIMESTAMPS:]
    selection_df = train_df[train_df["timestamp"].isin(selection_timestamps)].copy()
    grouped = list(selection_df.groupby("timestamp", sort=True))
    rows = []
    for feature_name in FEATURE_CANDIDATES:
        ic_values = []
        for _, group in grouped:
            ic = _safe_cross_sectional_ic(group, feature_name)
            if pd.notna(ic):
                ic_values.append(ic)

        if ic_values:
            ic_series = pd.Series(ic_values, dtype=float)
            corr_mean = float(ic_series.mean())
            corr_std = float(ic_series.std(ddof=0)) if len(ic_series) > 1 else 0.0
            stability = corr_std
            score = abs(corr_mean) / (corr_std + 1e-6)
        else:
            corr_mean = 0.0
            stability = float("inf")
            score = 0.0

        group_name = next((name for name, members in FEATURE_GROUPS.items() if feature_name in members), "other")
        rows.append(
            {
                "feature_name": feature_name,
                "correlation": corr_mean,
                "stability": stability,
                "score": score,
                "group": group_name,
            }
        )

    feature_stats = pd.DataFrame(rows).sort_values(["score", "correlation", "stability"], ascending=[False, False, True])
    selected = []
    for group_name in ["momentum", "volatility", "regime", "volume", "correlation"]:
        group_features = feature_stats[feature_stats["group"] == group_name]["feature_name"].tolist()
        for feature_name in group_features[:4]:
            if feature_name not in selected:
                selected.append(feature_name)

    for feature_name in feature_stats["feature_name"]:
        if feature_name not in selected:
            selected.append(feature_name)
        if len(selected) >= MAX_FEATURES:
            break

    return selected[:MAX_FEATURES]


def _normalize_features(train_df, test_df, selected_features):
    selected_global = [feature for feature in selected_features if feature in set(GLOBAL_CONTEXT_FEATURES)]
    train_features = train_df[selected_features].replace([np.inf, -np.inf], np.nan)
    fill_values = train_features.median(numeric_only=True).fillna(0.0)
    train_feature_frame = train_features.fillna(fill_values)

    if selected_global:
        global_means = train_feature_frame[selected_global].mean().fillna(0.0)
        global_stds = train_feature_frame[selected_global].std().replace(0, np.nan).fillna(1.0)
    else:
        global_means = pd.Series(dtype=float)
        global_stds = pd.Series(dtype=float)

    def normalize_split(split_df):
        feature_frame = split_df[selected_features].replace([np.inf, -np.inf], np.nan).fillna(fill_values)
        timestamp_index = split_df["timestamp"]
        normalized = pd.DataFrame(index=feature_frame.index, columns=selected_features, dtype=np.float32)

        cross_sectional = [feature for feature in selected_features if feature not in selected_global]
        if cross_sectional:
            means = feature_frame[cross_sectional].groupby(timestamp_index).transform("mean")
            stds = feature_frame[cross_sectional].groupby(timestamp_index).transform("std")
            normalized[cross_sectional] = (feature_frame[cross_sectional] - means) / (stds + 1e-6)

        if selected_global:
            normalized[selected_global] = (feature_frame[selected_global] - global_means[selected_global]) / (global_stds[selected_global] + 1e-6)

        normalized = normalized.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)
        split_df = split_df.copy()
        split_df[selected_features] = normalized.astype(np.float32)
        return split_df

    return normalize_split(train_df), normalize_split(test_df)


def build_research_frames():
    with contextlib.redirect_stdout(io.StringIO()):
        raw_df = load_training_dataframe()

    df = _engineer_research_features(raw_df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["future_return_10", "regime_label"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    train_df = _add_target_signal(df[df["timestamp"] < SPLIT_DATE].copy())
    test_df = _add_target_signal(df[df["timestamp"] >= SPLIT_DATE].copy())

    selected_features = _select_top_features(train_df)
    train_df, test_df = _normalize_features(train_df, test_df, selected_features)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), selected_features


def _build_samples(split_df, selected_features, max_timestamps=MAX_SAMPLE_TIMESTAMPS):
    symbol_frames = {}
    timestamp_lookup = {}
    for symbol, symbol_df in split_df.groupby("symbol", sort=False):
        ordered = symbol_df.sort_values("timestamp").reset_index(drop=True)
        symbol_frames[symbol] = ordered
        timestamp_lookup[symbol] = {timestamp: idx for idx, timestamp in enumerate(ordered["timestamp"])}

    grouped_by_timestamp = {timestamp: group.sort_values("symbol") for timestamp, group in split_df.groupby("timestamp", sort=True)}
    candidate_timestamps = sorted(grouped_by_timestamp.keys())[-max_timestamps:]
    samples = []
    for timestamp in candidate_timestamps:
        group = grouped_by_timestamp[timestamp]
        ordered_symbols = group["symbol"].tolist()
        if len(ordered_symbols) < 10:
            continue

        asset_payloads = []
        for symbol in ordered_symbols:
            symbol_df = symbol_frames[symbol]
            end_idx = timestamp_lookup[symbol].get(timestamp)
            if end_idx is None or end_idx < SEQUENCE_LENGTH - 1:
                continue

            window = symbol_df.iloc[end_idx - SEQUENCE_LENGTH + 1 : end_idx + 1]
            if len(window) != SEQUENCE_LENGTH:
                continue

            label = window.iloc[-1]["target_signal"]
            mask = 1.0 if pd.notna(label) else 0.0
            label = float(label) if pd.notna(label) else 0.0
            future_return_10 = float(window.iloc[-1]["future_return_10"])

            asset_payloads.append(
                {
                    "features": window[selected_features].to_numpy(dtype=np.float32, copy=False),
                    "regime": np.int64(REGIME_TO_ID.get(window.iloc[-1]["regime_label"], 0)),
                    "target": np.float32(label),
                    "mask": np.float32(mask),
                    "future_return_10": np.float32(future_return_10 if np.isfinite(future_return_10) else 0.0),
                }
            )

        if len(asset_payloads) < 10:
            continue

        samples.append(
            {
                "features": np.stack([payload["features"] for payload in asset_payloads], axis=0).astype(np.float32),
                "regimes": np.asarray([payload["regime"] for payload in asset_payloads], dtype=np.int64),
                "target": np.asarray([payload["target"] for payload in asset_payloads], dtype=np.float32),
                "mask": np.asarray([payload["mask"] for payload in asset_payloads], dtype=np.float32),
                "future_return_10": np.asarray([payload["future_return_10"] for payload in asset_payloads], dtype=np.float32),
            }
        )

    return samples


def _collate_batch(batch):
    max_assets = max(item[0].shape[0] for item in batch)
    batch_size = len(batch)
    seq_len = batch[0][0].shape[1]
    feature_dim = batch[0][0].shape[2]

    features = torch.zeros((batch_size, max_assets, seq_len, feature_dim), dtype=torch.float32)
    regimes = torch.zeros((batch_size, max_assets), dtype=torch.long)
    target = torch.zeros((batch_size, max_assets), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_assets), dtype=torch.float32)
    future_return_10 = torch.zeros((batch_size, max_assets), dtype=torch.float32)

    for idx, item in enumerate(batch):
        asset_count = item[0].shape[0]
        features[idx, :asset_count] = item[0]
        regimes[idx, :asset_count] = item[1]
        target[idx, :asset_count] = item[2]
        mask[idx, :asset_count] = item[3]
        future_return_10[idx, :asset_count] = item[4]

    return features, regimes, target, mask, future_return_10


def train_model(train_dataset, feature_dim):
    set_seed(SEED)
    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_batch,
    )

    model = NeuroQuantAlphaModel(feature_dim=feature_dim, model_type=MODEL_TYPE.lower()).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    for _epoch in range(EPOCHS):
        model.train()
        for features, regimes, target, mask, _future_return_10 in dataloader:
            features = features.to(device)
            regimes = regimes.to(device)
            target = target.to(device)
            mask = mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features, regimes)
            ranking_loss = masked_pairwise_ranking_loss(logits, target, mask)
            mse_loss = masked_mse_loss(logits, target, mask)
            loss = 0.7 * ranking_loss + 0.3 * mse_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return model


def _safe_spearman(pred, target):
    if len(pred) < 5:
        return np.nan
    pred_series = pd.Series(pred)
    target_series = pd.Series(target)
    if pred_series.nunique() < 2 or target_series.nunique() < 2:
        return np.nan
    return float(pred_series.corr(target_series, method="spearman"))


def evaluate_model(model, test_dataset):
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=_collate_batch)
    model.eval()

    ic_values = []
    direction_hits = []
    top_k_excess_returns = []

    with torch.no_grad():
        for features, regimes, _target, mask, future_return_10 in dataloader:
            predictions = model(features.to(device), regimes.to(device)).cpu()
            mask = mask.cpu()

            for batch_idx in range(predictions.shape[0]):
                batch_pred = predictions[batch_idx].numpy()
                batch_future_returns = future_return_10[batch_idx].numpy()
                valid = np.isfinite(batch_future_returns) & (mask[batch_idx].numpy() > 0.5)
                if valid.sum() < 10:
                    continue

                batch_pred = batch_pred[valid]
                batch_future_returns = batch_future_returns[valid]
                ic = _safe_spearman(batch_pred, batch_future_returns)
                if np.isfinite(ic):
                    ic_values.append(ic)

                centered_returns = batch_future_returns - float(batch_future_returns.mean())
                direction_hits.extend((np.sign(batch_pred) == np.sign(centered_returns)).astype(np.float32).tolist())

                k = max(1, int(np.ceil(len(batch_pred) * TOP_K_FRACTION)))
                top_idx = np.argsort(batch_pred)[-k:]
                top_k_excess_returns.append(float(batch_future_returns[top_idx].mean() - batch_future_returns.mean()))

    correlation = float(np.mean(ic_values)) if ic_values else 0.0
    directional_accuracy = float(np.mean(direction_hits)) if direction_hits else 0.0
    top_k_return = float(np.mean(top_k_excess_returns)) if top_k_excess_returns else 0.0

    return {
        "correlation": correlation,
        "directional_accuracy": directional_accuracy,
        "top_k_return": top_k_return,
    }


def main():
    set_seed(SEED)
    train_df, test_df, selected_features = build_research_frames()
    set_active_features(selected_features, persist=True)

    train_samples = _build_samples(train_df, selected_features)
    test_samples = _build_samples(test_df, selected_features)
    train_dataset = AlphaClassificationDataset(train_samples)
    test_dataset = AlphaClassificationDataset(test_samples)

    model = train_model(train_dataset, feature_dim=len(selected_features))
    metrics = evaluate_model(model, test_dataset)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    print(f"CORRELATION={metrics['correlation']:.6f}")
    print(f"DIRECTIONAL_ACCURACY={metrics['directional_accuracy']:.6f}")
    print(f"TOP_K_RETURN={metrics['top_k_return']:.6f}")


if __name__ == "__main__":
    main()
