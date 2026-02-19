import numpy as np
import torch
from typing import Optional


def channel_scores(E: torch.Tensor) -> np.ndarray:
    """
    Reduce attribution tensor to per-sample, per-channel scores.

    Args:
        E: attribution tensor (N, C, T)

    Returns:
        numpy array (N, C)
    """
    return E.abs().mean(dim=-1).detach().cpu().numpy()


def per_class_acc(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    n_classes: Optional[int] = None,
) -> list:
    """
    Per-class accuracy.

    Args:
        y_true:    ground truth labels (N,)
        y_pred:    predicted labels (N,)
        n_classes: number of classes; inferred from data if None

    Returns:
        list of per-class accuracy floats (nan if class absent)
    """
    yt = y_true.detach().cpu().numpy()
    yp = y_pred.detach().cpu().numpy()
    K = int(n_classes) if n_classes is not None else int(max(yp.max(), yt.max()) + 1)
    out = []
    for c in range(K):
        idx = (yt == c)
        out.append(float((yp[idx] == c).mean()) if idx.any() else float("nan"))
    return out


def l2_norms(x: torch.Tensor, x_adv: torch.Tensor) -> np.ndarray:
    """Per-sample L2 norm of perturbation."""
    return (x_adv - x).flatten(1).norm(p=2, dim=1).detach().cpu().numpy()


def linf_norms(x: torch.Tensor, x_adv: torch.Tensor) -> np.ndarray:
    """Per-sample Linf norm of perturbation."""
    return (x_adv - x).abs().flatten(1).max(dim=1).values.detach().cpu().numpy()


def boundary_fraction(
    x_adv: torch.Tensor,
    train_min_t: torch.Tensor,
    train_max_t: torch.Tensor,
) -> float:
    """Fraction of adversarial values sitting at the clamp boundary."""
    bmask = (
        (x_adv <= (train_min_t + 1e-12)) |
        (x_adv >= (train_max_t - 1e-12))
    ).float()
    return float(bmask.mean().item())
