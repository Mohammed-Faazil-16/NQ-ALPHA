from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.alpha_agent.dataset_builder import build_alpha_dataset, get_active_features
from agents.alpha_agent.loss_functions import masked_mse_loss
from agents.alpha_agent.neuroquant_model import NeuroQuantAlphaModel


EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 5e-4
MAX_FEATURE_ABS = 20.0

MODEL_PATH = PROJECT_ROOT / "models" / "neuroquant_alpha_model.pt"


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_alpha_model(
    model_path=MODEL_PATH,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    return_metrics=False,
):
    set_seed(42)
    active_features = get_active_features()

    print("Using device:", device)
    print("Loading dataset...")
    print(f"Active features ({len(active_features)}): {', '.join(active_features)}")
    dataset = build_alpha_dataset(split="train")

    print(f"Total samples: {len(dataset)}")

    if len(dataset) == 0:
        raise RuntimeError("Alpha dataset is empty; cannot train model")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = NeuroQuantAlphaModel(feature_dim=len(active_features)).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    last_avg_loss = float("nan")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0

        print(f"\nEpoch {epoch}/{epochs}")

        for batch_idx, (features, regime_ids, targets, masks) in enumerate(dataloader):
            if not torch.isfinite(features).all():
                nan_count = int(torch.isnan(features).sum().item())
                inf_count = int(torch.isinf(features).sum().item())
                print(
                    f"Non-finite features detected at epoch {epoch}, batch {batch_idx}: "
                    f"nan={nan_count}, inf={inf_count}"
                )
                features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            if not torch.isfinite(targets).all():
                nan_count = int(torch.isnan(targets).sum().item())
                inf_count = int(torch.isinf(targets).sum().item())
                print(
                    f"Non-finite targets detected at epoch {epoch}, batch {batch_idx}: "
                    f"nan={nan_count}, inf={inf_count}"
                )
                targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)

            batch_feature_abs_max = float(features.abs().max().item())
            if batch_idx == 0:
                print(
                    "Batch 1 stats -> "
                    f"feature_range=[{features.min().item():.4f}, {features.max().item():.4f}], "
                    f"target_range=[{targets.min().item():.6f}, {targets.max().item():.6f}]"
                )
            elif batch_feature_abs_max > MAX_FEATURE_ABS:
                print(
                    f"Extreme feature range detected at epoch {epoch}, batch {batch_idx}: "
                    f"abs_max={batch_feature_abs_max:.4f}"
                )

            features = features.to(device, non_blocking=True)
            regime_ids = regime_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            predicted_scores = model(features, regime_ids)
            if not torch.isfinite(predicted_scores).all():
                print(f"Non-finite predictions at epoch {epoch}, batch {batch_idx}; skipping batch")
                continue

            loss = masked_mse_loss(predicted_scores, targets, masks)
            if not torch.isfinite(loss):
                print(f"Non-finite loss at epoch {epoch}, batch {batch_idx}; skipping batch")
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if not torch.isfinite(grad_norm):
                print(f"Non-finite gradient norm at epoch {epoch}, batch {batch_idx}; skipping optimizer step")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()

            epoch_loss += float(loss.item())
            valid_batches += 1

        if valid_batches == 0:
            raise RuntimeError(f"No valid batches were processed in epoch {epoch}")

        last_avg_loss = epoch_loss / valid_batches
        print(f"Loss: {last_avg_loss:.6f}")

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    print("\nTraining completed")
    print(f"Model saved to -> {model_path}")

    metrics = {
        "final_loss": last_avg_loss,
        "model_path": str(model_path),
        "feature_count": len(active_features),
    }
    if return_metrics:
        return metrics
    return metrics


if __name__ == "__main__":
    train_alpha_model()
