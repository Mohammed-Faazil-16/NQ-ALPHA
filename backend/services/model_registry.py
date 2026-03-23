from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.alpha_agent.neuroquant_model import NeuroQuantAlphaModel


MODEL_PATH = PROJECT_ROOT / "models" / "neuroquant_alpha_model.pt"
SELECTED_FEATURES_PATH = PROJECT_ROOT / "logs" / "selected_features.json"
SEQUENCE_LENGTH = 30
REGIME_TO_ID = {
    "bull": 0,
    "normal": 1,
    "volatile": 2,
    "crisis": 3,
}
REGIME_LABEL_BY_ID = {value: key for key, value in REGIME_TO_ID.items()}
DEFAULT_ACTIVE_FEATURES = [
    "relative_return_1",
    "rank_return_1",
    "vol_adjusted_return_1",
    "return_zscore_5",
    "price_vs_sma20",
    "RSI_14",
    "rolling_rank_mean_10",
    "price_vs_sma50",
    "momentum_x_volatility",
    "momentum_x_regime",
]


@lru_cache(maxsize=1)
def get_active_features() -> tuple[str, ...]:
    if SELECTED_FEATURES_PATH.exists():
        try:
            payload = json.loads(SELECTED_FEATURES_PATH.read_text(encoding="utf-8"))
            features = payload.get("active_features") or []
            normalized = tuple(str(feature) for feature in features if str(feature).strip())
            if normalized:
                return normalized
        except Exception:
            pass
    return tuple(DEFAULT_ACTIVE_FEATURES)


@lru_cache(maxsize=1)
def get_model() -> NeuroQuantAlphaModel:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Alpha model not found at {MODEL_PATH}")

    model = NeuroQuantAlphaModel(feature_dim=len(get_active_features()))
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def warm_model_registry() -> None:
    get_model()
    get_active_features()
