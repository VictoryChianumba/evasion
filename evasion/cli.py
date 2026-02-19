"""
evasion CLI

Usage:
    evasion run --config experiment.yaml
    evasion run --config experiment.yaml --attack FGSM --output results/out.csv
"""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evasion",
        description="Adversarial attack and explanation stability sweeps.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run subcommand ---
    run_parser = subparsers.add_parser("run", help="Run an adversarial sweep.")
    run_parser.add_argument(
        "--config", required=True, metavar="PATH",
        help="Path to YAML config file.",
    )
    run_parser.add_argument(
        "--attack", metavar="NAME", default=None,
        choices=["FGSM", "PGD", "FGSM_LP", "PGD_LP", "DeepFool", "all"],
        help="Override which attack(s) to run. Overrides config value if set.",
    )
    run_parser.add_argument(
        "--output", metavar="PATH", default=None,
        help="Override output CSV path. Overrides config value if set.",
    )
    run_parser.add_argument(
        "--subject", metavar="ID", type=int, default=None,
        help="Override subject ID. Overrides config value if set.",
    )

    return parser


def cmd_run(args):
    import yaml
    import torch
    import numpy as np
    import pandas as pd
    import os

    from evasion import ChannelConfig, RunConfig, EvasionRunner
    from evasion.explainers import IGExplainer

    # load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.attack is not None:
        cfg["attacks"] = [args.attack] if args.attack != "all" else ["FGSM", "PGD", "FGSM_LP", "PGD_LP", "DeepFool"]
    if args.output is not None:
        cfg["output_csv"] = args.output
    if args.subject is not None:
        cfg["subject_id"] = args.subject

    # data loading — user provides a loader module path in config
    # e.g. data_loader: "utils.load_adv_data.load_adv_test_data"
    loader_path = cfg["data_loader"]
    module_path, fn_name = loader_path.rsplit(".", 1)
    import importlib
    loader_mod = importlib.import_module(module_path)
    load_fn = getattr(loader_mod, fn_name)
    data = load_fn(cfg["subject_id"])
    X, y, train_min_t, train_max_t, _, CH_NAMES, _, prenorm_std_np = data[:8]

    device = cfg.get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    X = X.to(device); y = y.to(device)
    train_min_t = train_min_t.to(device); train_max_t = train_max_t.to(device)

    n_channels = int(X.shape[1])
    n_times    = int(X.shape[2])
    n_classes  = int(y.max().item() + 1)

    # evasion config
    channel_cfg = ChannelConfig(
        ch_names=CH_NAMES if cfg.get("ch_names") == "from_data" else cfg.get("ch_names"),
        roi_names=cfg.get("roi_names"),
    )

    run_cfg = RunConfig(
        budget_grid=cfg.get("budget_grid", [0.25, 0.5, 1.0, 2.0]),
        median_std_pre=float(np.median(prenorm_std_np)),
        prenorm_std_np=prenorm_std_np,
        attr_max_n=cfg.get("attr_max_n", 128),
        n_steps_ig=cfg.get("n_steps_ig", 16),
        seeds=cfg.get("seeds", [42]),
    )

    # model loading
    model_module_path, model_fn_name = cfg["model_builder"].rsplit(".", 1)
    model_mod = importlib.import_module(model_module_path)
    model_builder = getattr(model_mod, model_fn_name)

    attacks_to_run = cfg.get("attacks", ["FGSM", "PGD", "FGSM_LP", "PGD_LP", "DeepFool"])
    lp_sigma = cfg.get("lp_sigma_t", 3.0)
    all_rows = []

    for seed in run_cfg.seeds:
        print(f"Seed {seed}")
        model = model_builder(n_channels=n_channels, n_times=n_times, n_classes=n_classes).to(device)

        ckpt_path = cfg["checkpoint"].format(subject_id=cfg["subject_id"], seed=seed)
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()

        ig_explainer = IGExplainer(model, n_steps=run_cfg.n_steps_ig, max_n=run_cfg.attr_max_n)

        runner = EvasionRunner(
            model=model,
            X=X, y=y,
            train_min_t=train_min_t,
            train_max_t=train_max_t,
            run_cfg=run_cfg,
            channel_cfg=channel_cfg,
            explainers={"IG": ig_explainer.attribute},
        )

        rows = []
        if "FGSM" in attacks_to_run:
            rows += runner.run_linf_sweep("FGSM", seed=seed, lp_sigma_t=None)
        if "PGD" in attacks_to_run:
            rows += runner.run_linf_sweep("PGD", seed=seed, lp_sigma_t=None)
        if "FGSM_LP" in attacks_to_run:
            rows += runner.run_linf_sweep("FGSM", seed=seed, lp_sigma_t=lp_sigma)
        if "PGD_LP" in attacks_to_run:
            rows += runner.run_linf_sweep("PGD", seed=seed, lp_sigma_t=lp_sigma)
        if "DeepFool" in attacks_to_run:
            rows += runner.run_deepfool_sweep(seed=seed)

        for r in rows:
            r["subject_id"] = cfg["subject_id"]
            r["seed"] = seed

        all_rows.extend(rows)

    # save output
    out_path = cfg.get("output_csv", f"results/adversarial_results_{cfg['subject_id']}.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(all_rows)} rows.")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
