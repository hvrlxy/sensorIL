"""
ablation_sensor.py

Runs sensor increment ablation over all possible combinations
of base sensors and new sensors, grouped by number of base sensors.

Sensors: LeftWrist, RightWrist, RightThigh, RightWaist, RightAnkle

Generates configs programmatically — nothing hardcoded in config file.
Runs one set at a time (--n-base flag), saves results incrementally.

Usage:
    # Run 1-base configs (20 combos)
    python scripts/ablation_sensor.py --config configs/pipeline_config.json --n-base 1

    # Run 2-base configs (30 combos)
    python scripts/ablation_sensor.py --config configs/pipeline_config.json --n-base 2

    # Run 3-base configs (20 combos)
    python scripts/ablation_sensor.py --config configs/pipeline_config.json --n-base 3

    # Run 4-base configs (5 combos)
    python scripts/ablation_sensor.py --config configs/pipeline_config.json --n-base 4

    # Run all (75 combos)
    python scripts/ablation_sensor.py --config configs/pipeline_config.json --n-base all
"""

import os
import sys
import json
import argparse
import copy
import torch
import numpy as np
import shutil
from itertools import combinations
from datetime import datetime

from train_base            import train_base
from detect_confusion      import detect_confusion
from estimate_benefit      import estimate_benefit
from incremental_ft        import incremental_ft
from calibrate_thresholds  import calibrate_thresholds
from evaluate              import load_classifiers, get_scores, \
                                  compute_multilabel_f1, encode_split, \
                                  print_combined_table
from simclr_encoder        import load_simclr_encoder
from run_pipeline          import train_oracle


# ─────────────────────────────────────────────────────────────────────────────
# Tee logger
# ─────────────────────────────────────────────────────────────────────────────

class Tee:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log_file = open(path, "w", buffering=1)
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    def close(self):
        self.log_file.close()


# ─────────────────────────────────────────────────────────────────────────────
# Sensor config generation
# ─────────────────────────────────────────────────────────────────────────────

ALL_SENSORS = ["LeftWrist", "RightWrist", "RightThigh", "RightWaist", "RightAnkle"]

SHORT = {
    "LeftWrist" : "LW",
    "RightWrist": "RW",
    "RightThigh": "RT",
    "RightWaist": "RWa",
    "RightAnkle": "RA"
}


def generate_configs(n_base):
    """Generate all sensor configs for a given number of base sensors."""
    configs = []
    for base in combinations(ALL_SENSORS, n_base):
        remaining = [s for s in ALL_SENSORS if s not in base]
        for new in remaining:
            name = "_".join(SHORT[s] for s in base) + "__plus_" + SHORT[new]
            configs.append({
                "name"          : name,
                "known_sensors" : list(base),
                "new_sensor"    : [new]
            })
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# Run one sensor config
# ─────────────────────────────────────────────────────────────────────────────

def run_sensor_config(base_config, sensor_config, device, log_dir, budgets=None):
    name   = sensor_config["name"]
    config = copy.deepcopy(base_config)
    config["sensors"]["known_sensors"] = sensor_config["known_sensors"]
    config["sensors"]["new_sensor"]    = sensor_config["new_sensor"]
    config["checkpoint_name"]          = f"base_{name}.pt"

    ckpt_dir = config["output"]["checkpoint_dir"]

    print(f"\n{'─'*70}")
    print(f"  {name}")
    print(f"  Base: {sensor_config['known_sensors']} → New: {sensor_config['new_sensor']}")
    print(f"{'─'*70}")

    # ── Base training ──────────────────────────────────────────────────────
    base_ckpt = os.path.join(ckpt_dir, f"base_{name}.pt")
    if not os.path.exists(base_ckpt):
        base_ckpt, _, _ = train_base(config, device)
    else:
        print(f"  [cached] base checkpoint")
    calibrate_thresholds(config, base_ckpt, device, verbose=False)

    # ── Confusion + benefit ───────────────────────────────────────────────
    confusion_scores, _  = detect_confusion(config, base_ckpt, device)
    benefit_ranked, target_classes = estimate_benefit(
        config, base_ckpt, confusion_scores, device
    )
    print(f"  Target classes ({len(target_classes)}): {target_classes}")

    # ── Oracle (train once) ──────────────────────────────────────────────
    oracle_ckpt = os.path.join(ckpt_dir, f"oracle_{name}.pt")
    if not os.path.exists(oracle_ckpt):
        config_oracle = copy.deepcopy(config)
        config_oracle["checkpoint_name"] = f"oracle_{name}_tmp.pt"
        tmp = train_oracle(config_oracle, device)
        shutil.copy(tmp, oracle_ckpt)
    else:
        print(f"  [cached] oracle checkpoint")
    calibrate_thresholds(config, oracle_ckpt, device, verbose=False)

    # ── Encode test data once ──────────────────────────────────────────────
    encoder = load_simclr_encoder(config["model"]["encoder_path"], device)
    known   = config["sensors"]["known_sensors"]
    all_s   = known + config["sensors"]["new_sensor"]

    z_known, y, class_names = encode_split(encoder, config, known, device)
    z_full,  _, _           = encode_split(encoder, config, all_s, device)

    def eval_ckpt(ckpt_p):
        clf, idim, _, _ = load_classifiers(ckpt_p, device)
        scores          = get_scores(clf, idim, z_known, z_full, class_names, device)
        thr_p           = ckpt_p.replace(".pt", "_thresholds.pt")
        thr_d           = torch.load(thr_p, map_location="cpu")["thresholds"] \
                          if os.path.exists(thr_p) else None
        f1_arr, macro, weighted = compute_multilabel_f1(
            scores, y, class_names, thresholds_dict=thr_d
        )
        return f1_arr, macro, weighted

    f1_b, macro_b, wtd_b = eval_ckpt(base_ckpt)
    f1_o, macro_o, wtd_o = eval_ckpt(oracle_ckpt)

    # ── Parse budgets ──────────────────────────────────────────────────────
    if budgets is None:
        budgets = ["all"]
    budget_results = []

    for budget in budgets:
        if budget == "all":
            k            = len(target_classes)
            k_targets    = target_classes
            budget_label = "all"
        else:
            k         = min(int(budget), len(target_classes))
            k_targets = target_classes[:k]
            budget_label = str(k)

        print(f"\n  Budget={budget_label} ({k} classes): {k_targets}")

        # Incremental fine-tuning for this budget
        inc_ckpt_k = os.path.join(ckpt_dir, f"incremental_{name}_k{budget_label}.pt")
        inc_tmp, _, _, _ = incremental_ft(
            config, base_ckpt, k_targets, device, verbose=False
        )
        shutil.copy(inc_tmp, inc_ckpt_k)
        calibrate_thresholds(config, inc_ckpt_k, device, verbose=False)

        f1_i, macro_i, wtd_i = eval_ckpt(inc_ckpt_k)

        # Print per-class table
        print_combined_table(
            class_names, f1_b, f1_i, f1_o,
            set(k_targets),
            macro_b, macro_i, macro_o,
            wtd_b,   wtd_i,   wtd_o,
            k
        )

        # Targeted metrics
        tgt_idx       = [class_names.index(c) for c in k_targets if c in class_names]
        tgt_deltas    = f1_i[tgt_idx] - f1_b[tgt_idx]
        oracle_deltas = f1_o[tgt_idx] - f1_b[tgt_idx]
        n_imp = int((tgt_deltas >  0.01).sum())
        n_deg = int((tgt_deltas < -0.01).sum())

        with np.errstate(divide='ignore', invalid='ignore'):
            pct_oracle = float(np.clip(
                np.where(np.abs(oracle_deltas) > 0.01,
                         tgt_deltas / oracle_deltas, 1.0),
                -1, 2
            ).mean()) * 100

        print(f"  → Budget={budget_label} | Baseline={macro_b:.4f} | "
              f"Proposed={macro_i:.4f} ({macro_i-macro_b:+.4f}) | "
              f"Oracle={macro_o:.4f} ({macro_o-macro_b:+.4f}) | "
              f"%Oracle={pct_oracle:.1f}%")

        budget_results.append({
            "budget"        : budget_label,
            "n_target"      : k,
            "target_classes": k_targets,
            "proposed"      : {"macro_f1": float(macro_i), "weighted_f1": float(wtd_i),
                               "delta": float(macro_i - macro_b),
                               "n_improved": n_imp, "n_degraded": n_deg,
                               "mean_delta": float(tgt_deltas.mean()),
                               "median_delta": float(np.median(tgt_deltas))},
            "pct_oracle"    : pct_oracle,
            "per_class"     : {
                class_names[i]: {
                    "baseline": float(f1_b[i]), "proposed": float(f1_i[i]),
                    "oracle"  : float(f1_o[i]), "targeted": class_names[i] in k_targets
                } for i in range(len(class_names))
            }
        })

    result = {
        "name"           : name,
        "known_sensors"  : known,
        "new_sensor"     : config["sensors"]["new_sensor"],
        "n_base"         : len(known),
        "baseline"       : {"macro_f1": float(macro_b), "weighted_f1": float(wtd_b)},
        "oracle"         : {"macro_f1": float(macro_o), "weighted_f1": float(wtd_o),
                            "delta"   : float(macro_o - macro_b)},
        "budget_results" : budget_results,
        "benefit_ranked" : [(n, float(s)) for n, s in benefit_ranked],
        "confusion_scores": [
            {k2: v for k2, v in cs.items() if k2 != "class_id"}
            for cs in confusion_scores
        ]
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results, n_base):
    # Collect all budgets used
    all_budgets = []
    for r in results:
        for br in r.get("budget_results", []):
            if br["budget"] not in all_budgets:
                all_budgets.append(br["budget"])

    for budget in all_budgets:
        print(f"\n{'='*105}")
        print(f"  n_base={n_base} | Budget={budget}  ({len(results)} configs)")
        print(f"{'='*105}")
        print(f"  {'Config':<30} {'Base':<25} {'New':>8} {'#Tgt':>5} "
              f"{'Base':>8} {'Prop':>8} {'Orac':>8} "
              f"{'ΔProp':>7} {'ΔOrac':>7} {'%Orac':>7}")
        print(f"  {'─'*103}")

        deltas, pct_oracs = [], []
        for r in results:
            br = next((b for b in r.get("budget_results", [])
                       if b["budget"] == budget), None)
            if br is None:
                continue
            base_str = "+".join(SHORT[s] for s in r["known_sensors"])
            new_str  = SHORT[r["new_sensor"][0]]
            delta    = br["proposed"]["delta"]
            pct      = br["pct_oracle"]
            deltas.append(delta)
            pct_oracs.append(pct)
            print(f"  {r['name']:<30} {base_str:<25} {new_str:>8} "
                  f"{br['n_target']:>5} "
                  f"{r['baseline']['macro_f1']:>8.4f} "
                  f"{br['proposed']['macro_f1']:>8.4f} "
                  f"{r['oracle']['macro_f1']:>8.4f} "
                  f"{delta:>+7.4f} "
                  f"{r['oracle']['delta']:>+7.4f} "
                  f"{pct:>6.1f}%")

        if deltas:
            print(f"\n  Aggregate: ΔProp mean={np.mean(deltas):+.4f} "
                  f"median={np.median(deltas):+.4f} "
                  f"improved={sum(1 for d in deltas if d>0.005)}/{len(deltas)} "
                  f"%Oracle mean={np.mean(pct_oracs):.1f}%")
        print(f"{'='*105}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n-base", type=str, required=True,
                        help="Number of base sensors: 1, 2, 3, 4, or 'all'")
    parser.add_argument("--budgets", type=str, default="5,10,15,all",
                        help="Comma-separated budgets e.g. '5,10,15,all'")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        base_config = json.load(f)

    device   = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log_dir  = base_config["output"]["checkpoint_dir"]
    os.makedirs(log_dir, exist_ok=True)

    # Parse budgets
    raw_budgets = [b.strip() for b in args.budgets.split(",")]

    # Determine which n_base values to run
    if args.n_base == "all":
        n_base_list = [1, 2, 3, 4]
    else:
        n_base_list = [int(args.n_base)]

    # Start logging
    log_dir_logs = base_config["output"]["log_dir"]
    os.makedirs(log_dir_logs, exist_ok=True)
    budget_tag = args.budgets.replace(",", "_")
    log_path = os.path.join(
        log_dir_logs,
        f"ablation_sensor_n{args.n_base}_b{budget_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    tee = Tee(log_path)
    sys.stdout = tee
    print(f"Logging to: {log_path}")
    print(f"Started: {datetime.now().isoformat()}")

    all_results = []

    for n_base in n_base_list:
        configs = generate_configs(n_base)
        print(f"\n{'='*70}")
        print(f"  Running n_base={n_base}: {len(configs)} configs")
        print(f"{'='*70}")

        set_results = []
        for i, sc in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] ", end="")
            r = run_sensor_config(base_config, sc, device, log_dir,
                                  budgets=raw_budgets)
            set_results.append(r)
            all_results.append(r)

            # Save incrementally after each config
            out_path = os.path.join(log_dir, f"ablation_sensor_n{n_base}_results.json")
            with open(out_path, "w") as f:
                json.dump(set_results, f, indent=2, default=str)

        print_summary(set_results, n_base)

    # Save combined results if running all
    if args.n_base == "all":
        out_all = os.path.join(log_dir, "ablation_sensor_all_results.json")
        with open(out_all, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nAll results → {out_all}")

    print(f"\nLog → {log_path}")
    print(f"Finished: {datetime.now().isoformat()}")
    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()