from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sqlalchemy import and_

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.alpha_agent.dataset_builder import REGIME_TO_ID, STORED_FEATURE_COLUMNS
from backend.database.models.asset_universe import AssetUniverse
from backend.database.models.features import Features
from backend.database.models.market_regime import MarketRegime
from backend.database.postgres import SessionLocal


MIN_TIMESTEPS = 500
MAX_MISSING_RATIO = 0.05
DEFAULT_TOP_N = 30


def load_feature_dataframe(db):
    query = (
        db.query(
            Features.symbol,
            Features.timestamp,
            *[getattr(Features, column) for column in STORED_FEATURE_COLUMNS],
            MarketRegime.regime_label,
        )
        .join(
            MarketRegime,
            and_(
                Features.symbol == MarketRegime.symbol,
                Features.timestamp == MarketRegime.timestamp,
            ),
        )
        .order_by(Features.symbol, Features.timestamp)
    )

    rows = query.all()
    columns = ["symbol", "timestamp", *STORED_FEATURE_COLUMNS, "regime_label"]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def score_assets(df, min_timesteps=MIN_TIMESTEPS, max_missing_ratio=MAX_MISSING_RATIO):
    if df.empty:
        return [], 0, 0

    total_assets = int(df["symbol"].nunique())
    qualified_assets = []

    for symbol, group in df.groupby("symbol", sort=False):
        row_count = len(group)
        if row_count < min_timesteps:
            continue

        feature_frame = group[STORED_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        missing_ratio = float(feature_frame.isna().mean().mean())
        if missing_ratio > max_missing_ratio:
            continue

        raw_feature_values = feature_frame.to_numpy(dtype=np.float64, copy=True)
        inf_mask = np.isinf(raw_feature_values)
        nan_mask = np.isnan(raw_feature_values)
        invalid_mask = inf_mask | nan_mask
        invalid_count = int(invalid_mask.sum())
        if invalid_count > int(max(1, row_count * len(STORED_FEATURE_COLUMNS) * max_missing_ratio)):
            continue

        if not group["regime_label"].isin(REGIME_TO_ID).all():
            continue

        finite_feature_frame = feature_frame.fillna(0.0)
        if not np.isfinite(finite_feature_frame.to_numpy(dtype=np.float64, copy=False)).all():
            continue

        volatility_stability = float(pd.to_numeric(group["log_return_1"], errors="coerce").std(ddof=0) or 0.0)
        score = float(row_count - (missing_ratio * 1000.0))

        qualified_assets.append(
            {
                "symbol": symbol,
                "score": score,
                "data_length_score": int(row_count),
                "missing_ratio": missing_ratio,
                "volatility_stability": volatility_stability,
            }
        )

    return qualified_assets, total_assets, len(qualified_assets)


def save_selected_assets(db, selected_assets):
    db.query(AssetUniverse).delete()

    records = [
        AssetUniverse(symbol=asset["symbol"], score=asset["score"])
        for asset in selected_assets
    ]

    if records:
        db.bulk_save_objects(records)
    db.commit()


def select_clean_assets(top_n=DEFAULT_TOP_N):
    db = SessionLocal()
    try:
        df = load_feature_dataframe(db)
        scored_assets, total_assets, filtered_count = score_assets(df)

        scored_assets = sorted(scored_assets, key=lambda item: item["score"], reverse=True)
        selected_assets = scored_assets[:top_n]
        selected_symbols = [asset["symbol"] for asset in selected_assets]

        save_selected_assets(db, selected_assets)

        print(f"Total assets found: {total_assets}")
        print(f"Assets after filtering: {filtered_count}")
        print(f"Final selected assets ({len(selected_symbols)}): {', '.join(selected_symbols)}")

        return selected_symbols
    finally:
        db.close()


if __name__ == "__main__":
    select_clean_assets()
