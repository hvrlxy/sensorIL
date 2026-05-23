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

def run_sensor_config(base_config, sensor_config, device, log_dir):
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

    # ── Incremental fine-tuning ───────────────────────────────────────────
    inc_ckpt, _, _, _ = incremental_ft(
        config, base_ckpt, target_classes, device, verbose=False
    )
    inc_ckpt_named = os.path.join(ckpt_dir, f"incremental_{name}.pt")
    shutil.copy(inc_ckpt, inc_ckpt_named)
    calibrate_thresholds(config, inc_ckpt_named, device, verbose=False)

    # ── Oracle ────────────────────────────────────────────────────────────
    oracle_ckpt = os.path.join(ckpt_dir, f"oracle_{name}.pt")
    if not os.path.exists(oracle_ckpt):
        config_oracle = copy.deepcopy(config)
        config_oracle["checkpoint_name"] = f"oracle_{name}_tmp.pt"
        tmp = train_oracle(config_oracle, device)
        shutil.copy(tmp, oracle_ckpt)
    else:
        print(f"  [cached] oracle checkpoint")
    calibrate_thresholds(config, oracle_ckpt, device, verbose=False)

    # ── Evaluate ──────────────────────────────────────────────────────────
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
    f1_i, macro_i, wtd_i = eval_ckpt(inc_ckpt_named)
    f1_o, macro_o, wtd_o = eval_ckpt(oracle_ckpt)

    # Print per-class table
    print_combined_table(
        class_names, f1_b, f1_i, f1_o,
        set(target_classes),
        macro_b, macro_i, macro_o,
        wtd_b,   wtd_i,   wtd_o,
        len(target_classes)
    )

    # Targeted metrics
    tgt_idx       = [class_names.index(c) for c in target_classes if c in class_names]
    tgt_deltas    = f1_i[tgt_idx] - f1_b[tgt_idx]
    oracle_deltas = f1_o[tgt_idx] - f1_b[tgt_idx]
    n_imp         = int((tgt_deltas >  0.01).sum())
    n_deg         = int((tgt_deltas < -0.01).sum())

    with np.errstate(divide='ignore', invalid='ignore'):
        pct_oracle = float(np.clip(
            np.where(np.abs(oracle_deltas) > 0.01,
                     tgt_deltas / oracle_deltas, 1.0),
            -1, 2
        ).mean()) * 100

    result = {
        "name"          : name,
        "known_sensors" : known,
        "new_sensor"    : config["sensors"]["new_sensor"],
        "n_base"        : len(known),
        "n_target"      : len(target_classes),
        "target_classes": target_classes,
        "baseline"      : {"macro_f1": float(macro_b), "weighted_f1": float(wtd_b)},
        "proposed"      : {"macro_f1": float(macro_i), "weighted_f1": float(wtd_i),
                           "delta"       : float(macro_i - macro_b),
                           "n_improved"  : n_imp,
                           "n_degraded"  : int((tgt_deltas < -0.01).sum()),
                           "mean_delta"  : float(tgt_deltas.mean()),
                           "median_delta": float(np.median(tgt_deltas))},
        "oracle"        : {"macro_f1": float(macro_o), "weighted_f1": float(wtd_o),
                           "delta"   : float(macro_o - macro_b)},
        "pct_oracle"    : pct_oracle,
        "per_class"     : {
            class_names[i]: {
                "baseline": float(f1_b[i]), "proposed": float(f1_i[i]),
                "oracle"  : float(f1_o[i]), "targeted": class_names[i] in target_classes
            } for i in range(len(class_names))
        },
        "benefit_ranked"  : [(n, float(s)) for n, s in benefit_ranked],
        "confusion_scores": [
            {k: v for k, v in cs.items() if k != "class_id"}
            for cs in confusion_scores
        ]
    }

    print(f"\n  → Baseline={macro_b:.4f} | Proposed={macro_i:.4f} "
          f"({macro_i-macro_b:+.4f}) | Oracle={macro_o:.4f} "
          f"({macro_o-macro_b:+.4f}) | %Oracle={pct_oracle:.1f}%")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results, n_base):
    print(f"\n{'='*105}")
    print(f"  n_base={n_base} Ablation Summary  ({len(results)} configs)")
    print(f"{'='*105}")
    print(f"  {'Config':<30} {'Base':<30} {'New':>8} {'#Tgt':>5} "
          f"{'Base':>8} {'Prop':>8} {'Orac':>8} "
          f"{'ΔProp':>7} {'ΔOrac':>7} {'%Orac':>7}")
    print(f"  {'─'*103}")

    for r in results:
        base_str = "+".join(SHORT[s] for s in r["known_sensors"])
        new_str  = SHORT[r["new_sensor"][0]]
        print(f"  {r['name']:<30} {base_str:<30} {new_str:>8} "
              f"{r['n_target']:>5} "
              f"{r['baseline']['macro_f1']:>8.4f} "
              f"{r['proposed']['macro_f1']:>8.4f} "
              f"{r['oracle']['macro_f1']:>8.4f} "
              f"{r['proposed']['delta']:>+7.4f} "
              f"{r['oracle']['delta']:>+7.4f} "
              f"{r['pct_oracle']:>6.1f}%")

    # Aggregate stats
    deltas    = [r["proposed"]["delta"] for r in results]
    pct_oracs = [r["pct_oracle"] for r in results]
    print(f"\n  Aggregate (n={len(results)}):")
    print(f"    ΔProp — mean={np.mean(deltas):+.4f}  "
          f"median={np.median(deltas):+.4f}  "
          f"min={np.min(deltas):+.4f}  max={np.max(deltas):+.4f}")
    print(f"    %Oracle — mean={np.mean(pct_oracs):.1f}%  "
          f"median={np.median(pct_oracs):.1f}%")
    print(f"    Configs improved:  {sum(1 for d in deltas if d > 0.005)}/{len(results)}")
    print(f"    Configs degraded:  {sum(1 for d in deltas if d < -0.005)}/{len(results)}")
    print(f"{'='*105}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n-base", type=str, required=True,
                        help="Number of base sensors: 1, 2, 3, 4, or 'all'")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        base_config = json.load(f)

    device   = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log_dir  = base_config["output"]["checkpoint_dir"]
    os.makedirs(log_dir, exist_ok=True)

    # Determine which n_base values to run
    if args.n_base == "all":
        n_base_list = [1, 2, 3, 4]
    else:
        n_base_list = [int(args.n_base)]

    # Start logging
    log_dir_logs = base_config["output"]["log_dir"]
    os.makedirs(log_dir_logs, exist_ok=True)
    log_path = os.path.join(
        log_dir_logs,
        f"ablation_sensor_n{args.n_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
            r = run_sensor_config(base_config, sc, device, log_dir)
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