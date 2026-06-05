"""Shared metric utilities for baseline implementations."""

from typing import List, Tuple
import torch


def compute_prf(preds_t: torch.Tensor, targets_t: torch.Tensor,
                num_classes: int) -> Tuple[float, float, float]:
    """Macro-averaged precision, recall, F1 (ignores classes with no true samples)."""
    p_sum = r_sum = f_sum = 0.0
    valid = 0
    for c in range(num_classes):
        tp = ((preds_t == c) & (targets_t == c)).sum().item()
        fp = ((preds_t == c) & (targets_t != c)).sum().item()
        fn = ((preds_t != c) & (targets_t == c)).sum().item()
        if (targets_t == c).sum().item() == 0:
            continue
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-8)
        p_sum += p; r_sum += r; f_sum += f
        valid += 1
    n = max(valid, 1)
    return p_sum / n, r_sum / n, f_sum / n


def find_convergence(accuracies: List[float], window_size: int = 10) -> int:
    """Return round index (1-based) when accuracy stabilises, or -1 if not converged."""
    if len(accuracies) < window_size:
        return -1
    for i in range(window_size, len(accuracies)):
        window = accuracies[i - window_size:i]
        if max(window) - min(window) < 0.001:
            return i - window_size + 1
    return -1
