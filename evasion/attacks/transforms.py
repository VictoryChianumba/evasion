import torch
import torch.nn.functional as F
import numpy as np


def per_channel_clamp(x: torch.Tensor, vmin: torch.Tensor, vmax: torch.Tensor) -> torch.Tensor:
    """Clamp x to per-channel [vmin, vmax] bounds."""
    device = x.device
    return torch.max(torch.min(x, vmax.to(device)), vmin.to(device))


def smooth_delta_gauss(delta_t: torch.Tensor, sigma_t: float) -> torch.Tensor:
    """
    Apply Gaussian smoothing to a perturbation tensor.
    For 3D inputs (N, C, T), smooth along time.
    For 4D inputs (N, C, H, W), smooth spatially.
    Produces low-pass (LP) attack variants.

    Args:
        delta_t: perturbation tensor of shape (N, C, T) or (N, C, H, W)
        sigma_t: Gaussian std in time-steps. If <= 0, returns delta_t unchanged.
    """
    if sigma_t is None or sigma_t <= 0:
        return delta_t
    radius = int(4 * sigma_t + 0.5)
    device = delta_t.device
    x = torch.arange(-radius, radius + 1, dtype=delta_t.dtype, device=device)
    kernel = torch.exp(-0.5 * (x / sigma_t).pow(2))
    kernel /= kernel.sum()
    C = delta_t.size(1)
    if delta_t.dim() == 3:
        kernel_1d = kernel.expand(C, 1, -1)   # (C, 1, K)
        return F.conv1d(delta_t, kernel_1d, padding=radius, groups=C)
    if delta_t.dim() == 4:
        kernel_2d = (kernel[:, None] * kernel[None, :]).to(device)
        kernel_2d = kernel_2d.expand(C, 1, -1, -1)  # (C, 1, K, K)
        return F.conv2d(delta_t, kernel_2d, padding=radius, groups=C)
    raise ValueError(f"smooth_delta_gauss expects 3D or 4D tensor, got {delta_t.dim()}D")


def snr_db(x: torch.Tensor, x_adv: torch.Tensor) -> np.ndarray:
    """
    Signal-to-noise ratio in dB between clean and adversarial inputs.

    Args:
        x:     clean input (N, C, T)
        x_adv: adversarial input (N, C, T)

    Returns:
        numpy array of per-sample SNR values
    """
    d = x_adv - x
    num = x.pow(2).sum((1, 2)).sqrt()
    den = d.pow(2).sum((1, 2)).sqrt().clamp_min(1e-12)
    return (20.0 * torch.log10(num / den)).detach().cpu().numpy()
