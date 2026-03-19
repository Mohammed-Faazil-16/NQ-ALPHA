from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.alpha_agent.dataset_builder import build_alpha_dataset, get_active_features
from agents.alpha_agent.neuroquant_model import NeuroQuantAlphaModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = PROJECT_ROOT / "models" / "neuroquant_alpha_model.pt"


def _safe_spearman(predictions, targets):
    if len(predictions) < 2:
        return np.nan

    pred_series = pd.Series(predictions)
    target_series = pd.Series(targets)
    if pred_series.nunique() < 2 or target_series.nunique() < 2:
        return np.nan
    return float(pred_series.corr(target_series, method="spearman"))


def evaluate(model_path=MODEL_PATH, return_metrics=False):
    active_features = get_active_features()

    print("Loading dataset...")
    print(f"Active features ({len(active_features)}): {', '.join(active_features)}")
    dataset = build_alpha_dataset(split="test")

    if len(dataset) == 0:
        raise RuntimeError("No test samples found on or after 2022-01-01")

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    model = NeuroQuantAlphaModel(feature_dim=len(active_features))
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    rank_ic_values = []
    direction_hits = []
    sample_count = 0

    print("Evaluating on TEST SET (post-2022)")
    print(f"Test samples: {len(dataset)}")

    with torch.no_grad():
        for features, regime_ids, targets, masks in dataloader:
            features = features.to(device)
            regime_ids = regime_ids.to(device)

            preds = model(features, regime_ids).cpu()
            targets = targets.cpu()
            masks = masks.cpu()

            for batch_idx in range(preds.shape[0]):
                valid_mask = masks[batch_idx] > 0
                valid_count = int(valid_mask.sum().item())
                if valid_count < 2:
                    continue

                sample_preds = preds[batch_idx][valid_mask].numpy()
                sample_targets = targets[batch_idx][valid_mask].numpy()
                sample_count += valid_count

                rank_ic = _safe_spearman(sample_preds, sample_targets)
                if np.isfinite(rank_ic):
                    rank_ic_values.append(rank_ic)

                direction_hits.extend((np.sign(sample_preds) == np.sign(sample_targets)).astype(np.float32).tolist())

    rank_ic = float(np.mean(rank_ic_values)) if rank_ic_values else 0.0
    direction_acc = float(np.mean(direction_hits)) if direction_hits else 0.0

    print("\n===== TEST EVALUATION =====")
    print(f"Rank IC: {rank_ic:.4f}")
    print(f"Test Directional Accuracy: {direction_acc:.4f}")

    metrics = {
        "correlation": rank_ic,
        "rank_ic": rank_ic,
        "directional_accuracy": direction_acc,
        "sample_count": sample_count,
    }
    if return_metrics:
        return metrics
    return metrics


if __name__ == "__main__":
    evaluate()
