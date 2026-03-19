from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database.models.asset import Asset
from backend.database.models.features import Features
from backend.database.models.market_regime import MarketRegime
from backend.database.postgres import SessionLocal


WINDOW_SIZE = 60
HALF_WINDOW = 30


def load_returns(symbol, db=None):
    owns_session = db is None
    if owns_session:
        db = SessionLocal()

    try:
        # Use log returns produced by the feature engineering pipeline.
        rows = (
            db.query(Features.timestamp, Features.log_return_1)
            .filter(
                Features.symbol == symbol,
                Features.log_return_1 != None
            )
            .order_by(Features.timestamp)
            .all()
        )
        if not rows:
            return pd.Series(dtype=float)

        returns = pd.Series(
            [row[1] for row in rows],
            index=pd.to_datetime([row[0] for row in rows]),
            dtype=float,
        )
        return returns.dropna().sort_index()
    finally:
        if owns_session:
            db.close()


def compute_regime_scores(returns):
    if returns.empty or len(returns) < WINDOW_SIZE + HALF_WINDOW:
        return pd.DataFrame(columns=["timestamp", "regime_score"])

    values = returns.to_numpy(dtype=float)
    timestamps = returns.index
    raw_scores = []
    score_timestamps = []

    for t in range(WINDOW_SIZE, len(values) - HALF_WINDOW):
        window_1 = values[t - WINDOW_SIZE : t]
        window_2 = values[t - HALF_WINDOW : t + HALF_WINDOW]
        if len(window_1) != WINDOW_SIZE or len(window_2) != WINDOW_SIZE:
            continue

        raw_scores.append(wasserstein_distance(window_1, window_2))
        score_timestamps.append(timestamps[t])

    if not raw_scores:
        return pd.DataFrame(columns=["timestamp", "regime_score"])

    raw_scores = np.asarray(raw_scores, dtype=float)
    max_score = raw_scores.max()
    if max_score > 0:
        normalized_scores = raw_scores / max_score
    else:
        normalized_scores = np.zeros_like(raw_scores)

    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(score_timestamps),
            "regime_score": normalized_scores,
        }
    )


def classify_regime(score):
    if score < 0.2:
        return "bull"
    if score < 0.5:
        return "normal"
    if score < 0.8:
        return "volatile"
    return "crisis"


def store_regime(symbol, timestamp, score, label, db, existing_timestamps):
    if timestamp in existing_timestamps:
        return False

    db.add(
        MarketRegime(
            symbol=symbol,
            timestamp=timestamp,
            regime_score=float(score),
            regime_label=label,
        )
    )
    existing_timestamps.add(timestamp)
    return True


def main():
    db = SessionLocal()
    try:
        symbols = [row[0] for row in db.query(Asset.symbol).distinct().all()]

        for symbol in symbols:
            print(f"Processing symbol: {symbol}")
            returns = load_returns(symbol, db=db)
            regime_scores = compute_regime_scores(returns)
            print(f"Computed {len(regime_scores)} regime scores")

            existing_timestamps = {
                row[0]
                for row in db.query(MarketRegime.timestamp)
                .filter(MarketRegime.symbol == symbol)
                .all()
            }

            inserted = 0
            for row in regime_scores.itertuples(index=False):
                timestamp = row.timestamp
                if hasattr(timestamp, "to_pydatetime"):
                    timestamp = timestamp.to_pydatetime()

                label = classify_regime(row.regime_score)
                if store_regime(symbol, timestamp, row.regime_score, label, db, existing_timestamps):
                    inserted += 1

            db.commit()
            print(f"Inserted {inserted} rows")
    finally:
        db.close()


if __name__ == "__main__":
    main()
