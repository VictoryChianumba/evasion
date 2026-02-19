from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class ChannelConfig:
    """
    Holds channel-level metadata for a given dataset/domain.
    ROI indices are computed lazily from ch_names and roi_names.

    Example (EEG motor imagery):
        cfg = ChannelConfig(
            ch_names=["Fp1", "Fp2", "C3", "C4", "Cz", ...],
            roi_names=["C3", "C4", "Cz"],
        )

    Example (domain-agnostic, no ROI):
        cfg = ChannelConfig()
    """
    ch_names: Optional[List[str]] = None
    roi_names: Optional[List[str]] = None

    @property
    def roi_idx(self) -> Optional[List[int]]:
        if self.ch_names is None or self.roi_names is None:
            return None
        return [self.ch_names.index(n) for n in self.roi_names if n in self.ch_names]

    def channel_index(self, name: str) -> Optional[int]:
        if self.ch_names is None:
            return None
        return self.ch_names.index(name) if name in self.ch_names else None


@dataclass
class RunConfig:
    """
    Hyperparameters for the attack/explainer sweep.

    budget_grid:    perturbation budgets in physical units (e.g. microvolts for EEG).
                    For domain-agnostic use, treat as arbitrary perturbation units.
    median_std_pre: median pre-normalisation std, used to convert physical budget -> z-space epsilon.
    prenorm_std_np: per-channel pre-normalisation stds (numpy array, shape [C]).
    attr_max_n:     max samples used for attribution computation.
    n_steps_ig:     integrated gradients interpolation steps.
    seeds:          random seeds for multi-run averaging.
    """
    budget_grid: List[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0])
    median_std_pre: float = 1.0
    prenorm_std_np: Optional[np.ndarray] = None
    attr_max_n: int = 128
    n_steps_ig: int = 16
    seeds: List[int] = field(default_factory=lambda: [42, 123, 2024, 31415, 999])

    def eps_from_units(self, budget: float) -> float:
        """Convert physical-unit budget to z-space epsilon."""
        return float(budget / self.median_std_pre)

    def eps_per_channel(self, eps_z: float) -> Optional[List[float]]:
        """Per-channel physical-unit perturbation implied by z-space epsilon."""
        if self.prenorm_std_np is None:
            return None
        return (eps_z * self.prenorm_std_np).tolist()
