"""
Example: EEG adversarial sweep using the evasion library.
Mirrors the original adv_run.py but uses EvasionRunner + ChannelConfig.
"""

import torch
import numpy as np
import pandas as pd
import os

from braindecode.models import EEGNetv4, Deep4Net, CTNet
from evasion import ChannelConfig, RunConfig, EvasionRunner
from evasion.explainers import IGExplainer

# ---- user config ----
SUBJECT_ID = 9   # subjects 1, 3, 8, 9 used in thesis
SEEDS = [42, 123, 2024, 31415, 999]

# ---- load data (project-specific, not part of library) ----
from utils.load_adv_data import load_adv_test_data
from repr import repr_helpers as rp
from models.eeg_mamba_fft import EEGMamba

data = load_adv_test_data(SUBJECT_ID)
X, y, train_min_t, train_max_t, train_std_np, CH_NAMES, train_mean_np, prenorm_std_np = data[:8]

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
X = X.to(device); y = y.to(device)
train_min_t = train_min_t.to(device); train_max_t = train_max_t.to(device)

n_channels = int(X.shape[1])
n_times    = int(X.shape[2])
n_classes  = int(y.max().item() + 1)

# ---- library config ----
channel_cfg = ChannelConfig(
    ch_names=CH_NAMES,
    roi_names=["C3", "C4", "Cz"],
)

run_cfg = RunConfig(
    budget_grid=[0.25, 0.5, 1.0, 2.0],
    median_std_pre=float(np.median(prenorm_std_np)),
    prenorm_std_np=prenorm_std_np,
    attr_max_n=128,
    n_steps_ig=16,
    seeds=SEEDS,
)

MODEL_BUILDERS = {
    "EEGNet":      lambda: EEGNetv4(n_chans=n_channels, n_outputs=n_classes, n_times=n_times),
    "DeepConvNet": lambda: Deep4Net(n_chans=n_channels, n_outputs=n_classes, n_times=n_times),
    "CTNet":       lambda: CTNet(n_chans=n_channels, n_outputs=n_classes, n_times=n_times),
    "Mamba":       lambda: EEGMamba(n_chans=n_channels, n_outputs=n_classes, n_times=n_times),
}

CKPTS = {
    "EEGNet":      {s: f"results/EEGNet/EEGNet_S{SUBJECT_ID}_seed{s}/checkpoint.pth" for s in SEEDS},
    "DeepConvNet": {s: f"results/DeepConvNet/DeepConvNet_S{SUBJECT_ID}_seed{s}/checkpoint.pth" for s in SEEDS},
    "CTNet":       {s: f"results/CTNet/CTNet_S{SUBJECT_ID}_seed{s}/checkpoint.pth" for s in SEEDS},
    "Mamba":       {s: f"results/EEGMamba/EEGMamba_S{SUBJECT_ID}_seed{s}/checkpoint.pth" for s in SEEDS},
}


def run_for_model_seed(model_name: str, seed: int):
    rp.set_all_seeds(seed)

    m = MODEL_BUILDERS[model_name]().to(device)
    ckpt = torch.load(CKPTS[model_name][seed], map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    m.load_state_dict(state_dict)
    m.eval()

    ig_explainer = IGExplainer(m, n_steps=run_cfg.n_steps_ig, max_n=run_cfg.attr_max_n)

    runner = EvasionRunner(
        model=m,
        X=X, y=y,
        train_min_t=train_min_t,
        train_max_t=train_max_t,
        run_cfg=run_cfg,
        channel_cfg=channel_cfg,
        explainers={"IG": ig_explainer.attribute},
    )

    rows = []
    rows += runner.run_linf_sweep("FGSM", seed=seed, lp_sigma_t=None)
    rows += runner.run_linf_sweep("PGD",  seed=seed, lp_sigma_t=None)
    rows += runner.run_linf_sweep("FGSM", seed=seed, lp_sigma_t=3.0)
    rows += runner.run_linf_sweep("PGD",  seed=seed, lp_sigma_t=3.0)
    rows += runner.run_deepfool_sweep(seed=seed)

    for r in rows:
        r["subject_id"] = SUBJECT_ID
        r["model_name"] = model_name
        r["seed"] = seed

    return rows


all_rows = []
for model_name in MODEL_BUILDERS:
    print(f"Running: {model_name}")
    for seed in SEEDS:
        all_rows.extend(run_for_model_seed(model_name, seed))

os.makedirs("results", exist_ok=True)
csv_path = f"results/adversarial_results_{SUBJECT_ID}.csv"
pd.DataFrame(all_rows).to_csv(csv_path, index=False)
print(f"Wrote {csv_path} with {len(all_rows)} rows.")

master_path = "results/adversarial_results_MASTER.csv"
if os.path.isfile(master_path):
    pd.concat([pd.read_csv(master_path), pd.DataFrame(all_rows)], ignore_index=True).to_csv(master_path, index=False)
else:
    pd.DataFrame(all_rows).to_csv(master_path, index=False)
print(f"Updated {master_path}.")
