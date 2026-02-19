import numpy as np
import torch
from scipy.stats import spearmanr
from typing import List, Optional

from evasion.metrics.attack_metrics import channel_scores


def spearman_ch(Ec: torch.Tensor, Ea: torch.Tensor) -> float:
    """
    Mean per-sample Spearman correlation between clean and adversarial
    channel attribution scores.

    Args:
        Ec: clean attributions (N, C, T)
        Ea: adversarial attributions (N, C, T)

    Returns:
        mean Spearman correlation across samples
    """
    sc, sa = channel_scores(Ec), channel_scores(Ea)
    vals = []
    for i in range(sc.shape[0]):
        r = spearmanr(sc[i], sa[i]).statistic
        if not np.isnan(r):
            vals.append(r)
    return float(np.mean(vals)) if vals else float("nan")


def roi_share(E: torch.Tensor, roi_idx: Optional[List[int]]) -> float:
    """
    Fraction of total attribution mass concentrated in ROI channels.

    Args:
        E:       attributions (N, C, T)
        roi_idx: list of ROI channel indices (from ChannelConfig.roi_idx)

    Returns:
        float in [0, 1], or nan if roi_idx is None
    """
    if roi_idx is None:
        return float("nan")
    sc = channel_scores(E).mean(axis=0)   # (C,)
    total = sc.sum() + 1e-12
    return float(sc[roi_idx].sum() / total)


def roi_spearman(
    Ec: torch.Tensor,
    Ea: torch.Tensor,
    roi_idx: Optional[List[int]],
) -> float:
    """
    Mean per-sample Spearman correlation restricted to ROI channels.

    Args:
        Ec:      clean attributions (N, C, T)
        Ea:      adversarial attributions (N, C, T)
        roi_idx: list of ROI channel indices (from ChannelConfig.roi_idx)

    Returns:
        mean Spearman correlation across samples, or nan if roi_idx is None
    """
    if roi_idx is None:
        return float("nan")
    sc, sa = channel_scores(Ec), channel_scores(Ea)
    vals = []
    for i in range(sc.shape[0]):
        r = spearmanr(sc[i, roi_idx], sa[i, roi_idx]).statistic
        if not np.isnan(r):
            vals.append(r)
    return float(np.mean(vals)) if vals else float("nan")


def laterality_index(
    E: torch.Tensor,
    left_idx: Optional[int],
    right_idx: Optional[int],
) -> float:
    """
    Laterality index: (left - right) / (left + right) attribution.

    Args:
        E:         attributions (N, C, T)
        left_idx:  index of left hemisphere channel (e.g. C3)
        right_idx: index of right hemisphere channel (e.g. C4)

    Returns:
        laterality index float, or nan if indices are None
    """
    if left_idx is None or right_idx is None:
        return float("nan")
    sc = channel_scores(E).mean(axis=0)   # (C,)
    num = sc[left_idx] - sc[right_idx]
    den = sc[left_idx] + sc[right_idx] + 1e-8
    return float(num / den)


def topk_channels(
    E: torch.Tensor,
    k: int = 5,
    ch_names: Optional[List[str]] = None,
) -> list:
    """
    Top-k channels by mean attribution magnitude.

    Args:
        E:        attributions (N, C, T)
        k:        number of top channels
        ch_names: channel name list (from ChannelConfig.ch_names); returns indices if None

    Returns:
        list of channel names or indices
    """
    sc_mean = channel_scores(E).mean(axis=0)   # (C,)
    idx = np.argsort(sc_mean)[::-1][:k]
    if ch_names is not None:
        return [ch_names[i] for i in idx]
    return idx.tolist()


def topk_roi_share(
    E: torch.Tensor,
    k: int = 5,
    roi_idx: Optional[List[int]] = None,
) -> float:
    """
    Fraction of top-k channels that fall within the ROI.

    Args:
        E:       attributions (N, C, T)
        k:       number of top channels to consider
        roi_idx: list of ROI channel indices (from ChannelConfig.roi_idx)

    Returns:
        float in [0, 1], or nan if roi_idx is None
    """
    if roi_idx is None:
        return float("nan")
    sc_mean = channel_scores(E).mean(axis=0)
    top_idx = set(np.argsort(sc_mean)[::-1][:k].tolist())
    return float(len(top_idx.intersection(set(roi_idx))) / max(1, k))
