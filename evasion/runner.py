import numpy as np
import torch
from typing import Dict, List, Optional, Callable

from evasion.config import ChannelConfig, RunConfig
from evasion.attacks.linf import run_fgsm, run_pgd
from evasion.attacks.l2 import run_deepfool
from evasion.attacks.transforms import snr_db
from evasion.metrics.attack_metrics import per_class_acc, l2_norms, linf_norms, boundary_fraction
from evasion.metrics import explanation_metrics as em


class EvasionRunner:
    """
    Orchestrates adversarial attack sweeps with paired explanation analysis.

    Args:
        model:          PyTorch model to attack
        X:              input tensor (N, C, T)
        y:              labels (N,)
        train_min_t:    per-channel lower clamp bound
        train_max_t:    per-channel upper clamp bound
        run_cfg:        RunConfig instance
        channel_cfg:    ChannelConfig instance
        explainers:     dict of {name: callable(X, y) -> attribution tensor}
    """

    def __init__(
        self,
        model: torch.nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        train_min_t: torch.Tensor,
        train_max_t: torch.Tensor,
        run_cfg: RunConfig,
        channel_cfg: ChannelConfig,
        explainers: Dict[str, Callable],
    ):
        self.model = model
        self.X = X
        self.y = y
        self.train_min_t = train_min_t
        self.train_max_t = train_max_t
        self.run_cfg = run_cfg
        self.channel_cfg = channel_cfg
        self.explainers = explainers

    def _clean_cache(self, cap_idx: slice) -> Dict[str, torch.Tensor]:
        """Compute and cache clean attributions for the capped subset."""
        X_cap = self.X[cap_idx].detach().clone()
        y_cap = self.y[cap_idx].detach().clone()
        return {name: fn(X_cap, y_cap) for name, fn in self.explainers.items()}

    def _clean_preds(self):
        with torch.no_grad():
            preds = self.model(self.X).argmax(1).cpu()
            n_classes = int(self.model(self.X[:1]).shape[-1])
        acc = float((preds == self.y.cpu()).float().mean().item())
        acc_pc = per_class_acc(self.y.cpu(), preds, n_classes)
        return preds, acc, acc_pc, n_classes

    def _adv_preds(self, X_adv: torch.Tensor):
        with torch.no_grad():
            preds = self.model(X_adv).argmax(1).cpu()
        acc = float((preds == self.y.cpu()).float().mean().item())
        return preds, acc

    def _explanation_row(
        self,
        method_name: str,
        Ec: torch.Tensor,
        Ea: torch.Tensor,
    ) -> dict:
        """Build explanation metric dict for one explainer method."""
        cfg = self.channel_cfg
        left_idx = cfg.channel_index("C3")
        right_idx = cfg.channel_index("C4")

        return {
            f"spearman_{method_name}":             em.spearman_ch(Ec, Ea),
            f"roi_share_clean_{method_name}":      em.roi_share(Ec, cfg.roi_idx),
            f"roi_share_adv_{method_name}":        em.roi_share(Ea, cfg.roi_idx),
            f"roi_delta_share_{method_name}":      em.roi_share(Ea, cfg.roi_idx) - em.roi_share(Ec, cfg.roi_idx),
            f"roi_spearman_{method_name}":         em.roi_spearman(Ec, Ea, cfg.roi_idx),
            f"laterality_clean_{method_name}":     em.laterality_index(Ec, left_idx, right_idx),
            f"laterality_adv_{method_name}":       em.laterality_index(Ea, left_idx, right_idx),
            f"laterality_delta_{method_name}":     (
                em.laterality_index(Ea, left_idx, right_idx) -
                em.laterality_index(Ec, left_idx, right_idx)
            ),
            f"top5_clean_{method_name}":           em.topk_channels(Ec, k=5, ch_names=cfg.ch_names),
            f"top5_adv_{method_name}":             em.topk_channels(Ea, k=5, ch_names=cfg.ch_names),
        }

    def _build_adv_subset(self, X_adv: torch.Tensor, cap_idx: slice) -> torch.Tensor:
        return X_adv[cap_idx].detach()

    def run_linf_sweep(
        self,
        attack_name: str,
        seed: int = 42,
        steps: int = 40,
        alpha_rule: Callable[[float], float] = lambda e: e / 8,
        batch: int = 128,
        lp_sigma_t: Optional[float] = None,
    ) -> List[dict]:
        """
        Sweep over budget_grid for FGSM or PGD, collecting attack and explanation metrics.

        Args:
            attack_name: "FGSM" or "PGD"
            seed:        random seed for this run
            steps:       PGD steps (ignored for FGSM)
            alpha_rule:  PGD step size rule (ignored for FGSM)
            batch:       batch size
            lp_sigma_t:  Gaussian smoothing sigma for LP variant; None for standard

        Returns:
            list of result dicts, one per (budget, explainer)
        """
        assert attack_name in ("FGSM", "PGD"), f"Unknown attack: {attack_name}"

        cap_n = min(self.run_cfg.attr_max_n, self.X.size(0))
        cap_idx = slice(0, cap_n)
        E_CLEAN = self._clean_cache(cap_idx)
        _, clean_acc, clean_acc_pc, n_classes = self._clean_preds()

        rows = []
        for budget in self.run_cfg.budget_grid:
            eps_z = self.run_cfg.eps_from_units(budget)

            if attack_name == "FGSM":
                X_adv = run_fgsm(
                    self.model, self.X, self.y, eps_z,
                    self.train_min_t, self.train_max_t,
                    batch=batch, lp_sigma_t=lp_sigma_t,
                )
            else:
                X_adv = run_pgd(
                    self.model, self.X, self.y, eps_z,
                    self.train_min_t, self.train_max_t,
                    steps=steps, alpha_rule=alpha_rule,
                    batch=batch, lp_sigma_t=lp_sigma_t,
                )

            preds_adv, adv_acc = self._adv_preds(X_adv)
            adv_acc_pc = per_class_acc(self.y.cpu(), preds_adv, n_classes)

            l2 = l2_norms(self.X, X_adv)
            linf = linf_norms(self.X, X_adv)
            snrs = snr_db(self.X, X_adv)
            success_mask = (preds_adv != self.y.cpu()).numpy()
            bf = boundary_fraction(X_adv, self.train_min_t, self.train_max_t)

            X_adv_cap = self._build_adv_subset(X_adv, cap_idx)
            y_cap = self.y[cap_idx]
            E_ADV = {name: fn(X_adv_cap, y_cap) for name, fn in self.explainers.items()}

            row_core = {
                "attack":              f"{attack_name}_LP" if lp_sigma_t else attack_name,
                "muV_budget":          float(budget),
                "eps_uV_per_channel":  self.run_cfg.eps_per_channel(eps_z),
                "seed":                seed,
                "clean_acc":           clean_acc,
                "adv_acc":             adv_acc,
                "ASR":                 1.0 - adv_acc,
                "clean_acc_per_class": clean_acc_pc,
                "adv_acc_per_class":   adv_acc_pc,
                "median_L2_success":   float(np.median(l2[success_mask])) if success_mask.any() else float("nan"),
                "mean_L2_all":         float(np.mean(l2)),
                "mean_Linf_delta":     float(np.mean(linf)),
                "snr_db_mean":         float(np.mean(snrs)),
                "snr_db_std":          float(np.std(snrs)),
                "frac_at_boundary":    bf,
            }

            for name in self.explainers:
                exp_row = self._explanation_row(name, E_CLEAN[name], E_ADV[name])
                rows.append({**row_core, **exp_row})

        return rows

    def run_deepfool_sweep(self, seed: int = 42, steps: int = 50, batch: int = 128) -> List[dict]:
        """
        Run DeepFool (L2) attack and collect attack and explanation metrics.

        Args:
            seed:  random seed tag (DeepFool is deterministic but recorded for bookkeeping)
            steps: DeepFool iterations
            batch: batch size

        Returns:
            list of result dicts, one per explainer
        """
        cap_n = min(self.run_cfg.attr_max_n, self.X.size(0))
        cap_idx = slice(0, cap_n)
        E_CLEAN = self._clean_cache(cap_idx)
        _, clean_acc, clean_acc_pc, n_classes = self._clean_preds()

        X_adv = run_deepfool(
            self.model, self.X, self.y,
            self.train_min_t, self.train_max_t,
            steps=steps, batch=batch,
        )

        preds_adv, adv_acc = self._adv_preds(X_adv)
        adv_acc_pc = per_class_acc(self.y.cpu(), preds_adv, n_classes)

        l2 = l2_norms(self.X, X_adv)
        linf = linf_norms(self.X, X_adv)
        snrs = snr_db(self.X, X_adv)
        success_mask = (preds_adv != self.y.cpu()).numpy()
        bf = boundary_fraction(X_adv, self.train_min_t, self.train_max_t)

        X_adv_cap = self._build_adv_subset(X_adv, cap_idx)
        y_cap = self.y[cap_idx]
        E_ADV = {name: fn(X_adv_cap, y_cap) for name, fn in self.explainers.items()}

        row_core = {
            "attack":              "DeepFool_L2",
            "muV_budget":          None,
            "eps_uV_per_channel":  None,
            "seed":                seed,
            "clean_acc":           clean_acc,
            "adv_acc":             adv_acc,
            "ASR":                 1.0 - adv_acc,
            "clean_acc_per_class": clean_acc_pc,
            "adv_acc_per_class":   adv_acc_pc,
            "median_L2_success":   float(np.median(l2[success_mask])) if success_mask.any() else float("nan"),
            "mean_L2_all":         float(np.mean(l2)),
            "mean_Linf_delta":     float(np.mean(linf)),
            "snr_db_mean":         float(np.mean(snrs)),
            "snr_db_std":          float(np.std(snrs)),
            "frac_at_boundary":    bf,
        }

        rows = []
        for name in self.explainers:
            exp_row = self._explanation_row(name, E_CLEAN[name], E_ADV[name])
            rows.append({**row_core, **exp_row})

        return rows
