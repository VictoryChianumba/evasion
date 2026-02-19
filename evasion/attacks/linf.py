import torch
import torchattacks as ta
from typing import Callable, List, Optional

from evasion.attacks.transforms import per_channel_clamp, smooth_delta_gauss


def run_fgsm(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    eps_z: float,
    train_min_t: torch.Tensor,
    train_max_t: torch.Tensor,
    batch: int = 128,
    lp_sigma_t: Optional[float] = None,
) -> torch.Tensor:
    """
    Run FGSM attack and return adversarial examples.

    Args:
        model:       target model
        X:           input tensor (N, C, T)
        y:           labels (N,)
        eps_z:       epsilon in normalised (z-score) space
        train_min_t: per-channel lower clamp bound (C, 1) or broadcastable
        train_max_t: per-channel upper clamp bound (C, 1) or broadcastable
        batch:       batch size for attack loop
        lp_sigma_t:  if set, apply Gaussian smoothing to delta (LP variant)

    Returns:
        adversarial tensor (N, C, T)
    """
    atk = ta.FGSM(model, eps=float(eps_z))
    return _linf_loop(atk, model, X, y, eps_z, train_min_t, train_max_t, batch, lp_sigma_t)


def run_pgd(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    eps_z: float,
    train_min_t: torch.Tensor,
    train_max_t: torch.Tensor,
    steps: int = 40,
    alpha_rule: Callable[[float], float] = lambda e: e / 8,
    batch: int = 128,
    lp_sigma_t: Optional[float] = None,
) -> torch.Tensor:
    """
    Run PGD attack and return adversarial examples.

    Args:
        model:       target model
        X:           input tensor (N, C, T)
        y:           labels (N,)
        eps_z:       epsilon in normalised space
        train_min_t: per-channel lower clamp bound
        train_max_t: per-channel upper clamp bound
        steps:       PGD iterations
        alpha_rule:  step size as a function of epsilon
        batch:       batch size for attack loop
        lp_sigma_t:  if set, apply Gaussian smoothing to delta (LP variant)

    Returns:
        adversarial tensor (N, C, T)
    """
    alpha = float(alpha_rule(eps_z))
    atk = ta.PGD(model, eps=float(eps_z), alpha=alpha, steps=steps, random_start=True)
    return _linf_loop(atk, model, X, y, eps_z, train_min_t, train_max_t, batch, lp_sigma_t)


def _linf_loop(
    atk,
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    eps_z: float,
    train_min_t: torch.Tensor,
    train_max_t: torch.Tensor,
    batch: int,
    lp_sigma_t: Optional[float],
) -> torch.Tensor:
    """Shared batched attack loop for Linf attacks."""
    adv_batches = []
    N = X.size(0)

    for i in range(0, N, batch):
        Xi = X[i:i + batch].detach().clone().requires_grad_(True)
        yi = y[i:i + batch]
        xa = atk(Xi, yi).detach()

        if lp_sigma_t is not None:
            delta = xa - Xi
            delta = smooth_delta_gauss(delta, sigma_t=lp_sigma_t)
            delta = delta.clamp(-eps_z, eps_z)
            xa = Xi + delta

        xa = per_channel_clamp(xa, train_min_t, train_max_t)
        adv_batches.append(xa)

    return torch.cat(adv_batches, dim=0)
