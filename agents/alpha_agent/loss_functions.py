import torch
import torch.nn.functional as F


def masked_mse_loss(pred, target, mask):
    pred = pred.float()
    target = target.float()
    mask = mask.float()

    squared_error = (pred - target) ** 2
    weighted_error = squared_error * mask
    normalizer = torch.clamp(mask.sum(), min=1.0)
    return weighted_error.sum() / normalizer


def masked_pairwise_ranking_loss(pred, target, mask):
    pred = pred.float()
    target = target.float()
    mask = mask.float()

    batch_losses = []
    for batch_idx in range(pred.size(0)):
        valid = mask[batch_idx] > 0.5
        if valid.sum() < 2:
            continue

        batch_pred = pred[batch_idx][valid]
        batch_target = target[batch_idx][valid]

        target_diff = batch_target[:, None] - batch_target[None, :]
        pair_mask = target_diff > 0
        if not pair_mask.any():
            continue

        pred_diff = batch_pred[:, None] - batch_pred[None, :]
        pair_loss = F.softplus(-pred_diff[pair_mask])
        target_scale = target_diff[pair_mask].abs().clamp_min(1e-6)
        batch_losses.append((pair_loss * target_scale).mean())

    if not batch_losses:
        return torch.zeros((), dtype=pred.dtype, device=pred.device)

    return torch.stack(batch_losses).mean()


def masked_bce_with_logits_loss(logits, target, mask):
    logits = logits.float()
    target = target.float()
    mask = mask.float()

    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted_loss = loss * mask
    normalizer = torch.clamp(mask.sum(), min=1.0)
    return weighted_loss.sum() / normalizer
