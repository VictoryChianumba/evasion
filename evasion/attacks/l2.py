import torch
import torchattacks as ta

from evasion.attacks.transforms import per_channel_clamp


def run_deepfool(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    train_min_t: torch.Tensor,
    train_max_t: torch.Tensor,
    steps: int = 50,
    batch: int = 128,
) -> torch.Tensor:
    """
    Run DeepFool (L2) attack and return adversarial examples.

    Args:
        model:       target model
        X:           input tensor (N, C, T)
        y:           labels (N,)
        train_min_t: per-channel lower clamp bound
        train_max_t: per-channel upper clamp bound
        steps:       DeepFool iterations
        batch:       batch size for attack loop

    Returns:
        adversarial tensor (N, C, T)
    """
    atk = ta.DeepFool(model, steps=steps)
    adv_batches = []
    N = X.size(0)

    for i in range(0, N, batch):
        Xi = X[i:i + batch].detach().clone().requires_grad_(True)
        yi = y[i:i + batch]
        xa = atk(Xi, yi).detach()
        xa = per_channel_clamp(xa, train_min_t, train_max_t)
        adv_batches.append(xa)

    return torch.cat(adv_batches, dim=0)
