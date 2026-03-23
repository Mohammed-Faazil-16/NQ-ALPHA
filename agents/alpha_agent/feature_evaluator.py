from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.alpha_agent.dataset_builder import (
    FEATURE_COLUMNS,
    LOGS_DIR,
    build_modeling_dataframes,
    set_active_features,
)


FEATURE_IMPORTANCE_PATH = LOGS_DIR / "feature_importance.csv"


def _safe_cross_sectional_ic(group, feature_name):
    sample = group[[feature_name, "target_return"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < 5:
        return np.nan
    if sample[feature_name].nunique() < 2 or sample["target_return"].nunique() < 2:
        return np.nan
    return sample[feature_name].corr(sample["target_return"], method="spearman")


def _feature_temporal_variance(df, feature_name):
    symbol_variances = df.groupby("symbol", sort=False)[feature_name].var(ddof=0)
    symbol_variances = symbol_variances.replace([np.inf, -np.inf], np.nan).dropna()
    if symbol_variances.empty:
        return 0.0
    return float(symbol_variances.median())


def _feature_distribution_shift(df, feature_name):
    sample = df[["timestamp", feature_name]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < 10:
        return np.inf

    ordered_timestamps = sorted(sample["timestamp"].unique())
    midpoint = len(ordered_timestamps) // 2
    if midpoint == 0 or midpoint >= len(ordered_timestamps):
        return np.inf

    first_half = sample[sample["timestamp"].isin(ordered_timestamps[:midpoint])][feature_name]
    second_half = sample[sample["timestamp"].isin(ordered_timestamps[midpoint:])][feature_name]
    if len(first_half) < 5 or len(second_half) < 5:
        return np.inf

    median_shift = abs(float(first_half.median()) - float(second_half.median()))
    spread_shift = abs(float(first_half.std(ddof=0)) - float(second_half.std(ddof=0)))
    return median_shift + spread_shift


def evaluate_feature_importance(df=None, save_path=FEATURE_IMPORTANCE_PATH):
    if df is None:
        df, _ = build_modeling_dataframes()

    if df.empty:
        raise RuntimeError("Feature evaluation dataframe is empty")

    grouped = list(df.groupby("timestamp", sort=True))
    results = []

    for feature_name in FEATURE_COLUMNS:
        valid_frame = df[[feature_name, "target_return"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_frame) < 5:
            correlation_mean = 0.0
            correlation_std = np.inf
            ic_mean = 0.0
            ic_std = np.inf
            stability = np.inf
        else:
            ic_values = []
            for _, group in grouped:
                ic = _safe_cross_sectional_ic(group, feature_name)
                if pd.notna(ic):
                    ic_values.append(ic)

            if ic_values:
                ic_series = pd.Series(ic_values, dtype=float)
                rolling_ic = ic_series.rolling(30, min_periods=10).mean().dropna()
                correlation_mean = float(ic_series.mean())
                correlation_std = float(ic_series.std(ddof=0)) if len(ic_series) > 1 else 0.0
                ic_mean = correlation_mean
                ic_std = correlation_std
                stability = float(rolling_ic.std(ddof=0)) if len(rolling_ic) > 1 else 0.0
            else:
                correlation_mean = 0.0
                correlation_std = np.inf
                ic_mean = 0.0
                ic_std = np.inf
                stability = np.inf

        global_corr = float(valid_frame[feature_name].corr(valid_frame["target_return"])) if len(valid_frame) >= 5 else 0.0
        temporal_variance = _feature_temporal_variance(df, feature_name)
        distribution_shift = _feature_distribution_shift(df, feature_name)

        results.append(
            {
                "feature_name": feature_name,
                "correlation_mean": correlation_mean,
                "correlation_std": correlation_std,
                "global_correlation": global_corr,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "stability_over_time": stability,
                "temporal_variance": temporal_variance,
                "distribution_shift": distribution_shift,
                "signal_score": abs(correlation_mean),
            }
        )

    feature_stats = pd.DataFrame(results)
    feature_stats = feature_stats.sort_values(["signal_score", "stability_over_time"], ascending=[False, True]).reset_index(drop=True)
    feature_stats["rank"] = np.arange(1, len(feature_stats) + 1)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    feature_stats.to_csv(save_path, index=False)
    return feature_stats


def filter_features(
    feature_stats,
    df,
    correlation_threshold=0.03,
    std_threshold=0.45,
    correlation_ceiling=0.9,
    variance_floor=0.30,
    shift_ceiling=0.25,
):
    if feature_stats.empty:
        return pd.DataFrame(), pd.DataFrame()

    feature_stats = feature_stats.copy()
    candidate_stats = feature_stats[
        (feature_stats["correlation_mean"].abs() > correlation_threshold)
        & (feature_stats["correlation_std"] < std_threshold)
        & (feature_stats["stability_over_time"] < std_threshold)
        & (feature_stats["temporal_variance"] > variance_floor)
        & (feature_stats["distribution_shift"] < shift_ceiling)
    ].copy()

    dropped_rows = []
    for _, row in feature_stats.iterrows():
        feature_name = row["feature_name"]
        if feature_name not in set(candidate_stats["feature_name"]):
            reason = "low_signal"
            if row["correlation_std"] >= std_threshold or row["stability_over_time"] >= std_threshold:
                reason = "unstable_ic"
            elif row["temporal_variance"] <= variance_floor:
                reason = "low_temporal_variance"
            elif row["distribution_shift"] >= shift_ceiling:
                reason = "distribution_shift"
            dropped_rows.append({**row.to_dict(), "drop_reason": reason, "correlated_with": ""})

    if candidate_stats.empty:
        return candidate_stats, pd.DataFrame(dropped_rows)

    candidate_names = candidate_stats["feature_name"].tolist()
    corr_matrix = df[candidate_names].corr().abs().fillna(0.0)

    kept = []
    correlated_drops = []
    for _, row in candidate_stats.sort_values(["signal_score", "distribution_shift", "stability_over_time"], ascending=[False, True, True]).iterrows():
        feature_name = row["feature_name"]
        correlated_with = next((kept_feature for kept_feature in kept if corr_matrix.loc[feature_name, kept_feature] > correlation_ceiling), None)
        if correlated_with is not None:
            correlated_drops.append({**row.to_dict(), "drop_reason": "highly_correlated", "correlated_with": correlated_with})
            continue
        kept.append(feature_name)

    kept_stats = candidate_stats[candidate_stats["feature_name"].isin(kept)].copy()
    dropped_stats = pd.DataFrame(dropped_rows + correlated_drops)
    return kept_stats, dropped_stats


def select_top_features(feature_stats, top_k=10, persist=True):
    if feature_stats.empty:
        raise RuntimeError("No candidate features available for selection")

    selected = (
        feature_stats.sort_values(["signal_score", "stability_over_time"], ascending=[False, True])
        .head(top_k)["feature_name"]
        .tolist()
    )
    set_active_features(selected, persist=persist)
    return selected
