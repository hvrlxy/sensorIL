import sys
from datetime import datetime


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


"""
ablation_budget.py

Runs incremental fine-tuning for different annotation budgets
and tracks detailed metrics for targeted classes.

Metrics tracked:
  Overall:
    - Macro F1, Weighted F1 vs baseline and oracle

  Targeted classes:
    - # classes improved / degraded / unchanged
    - Mean / median F1 change
    - Mean / median absolute F1 change
    - % of oracle gain achieved
    - Best and worst performing targeted classes

Usage:
    python scripts/ablation_budget.py \
        --config   configs/pipeline_config.json \
        --base-checkpoint   checkpoints/base_classifiers.pt \
        --oracle-checkpoint checkpoints/oracle_classifiers.pt \
        --budgets  5 10 15 20 25 32
"""

import os
import json
import argparse
import torch
import numpy as np

from detect_confusion     import detect_confusion
from estimate_benefit     import estimate_benefit
from incremental_ft       import incremental_ft
from calibrate_thresholds import calibrate_thresholds
from evaluate             import load_classifiers, get_scores, \
                                 compute_multilabel_f1, encode_split, \
                                 print_combined_table
from simclr_encoder       import load_simclr_encoder


# ─────────────────────────────────────────────────────────────────────────────
# Per-budget run
# ─────────────────────────────────────────────────────────────────────────────

def run_budget(config, base_ckpt, benefit_ranked, budget, device,
               z_known, z_full, y, class_names):
    target_classes = [name for name, _ in benefit_ranked[:budget]]
    print(f"\n{'─'*60}")
    print(f"Budget={budget} | Targets: {target_classes}")
    print(f"{'─'*60}")

    inc_ckpt, _, _, _ = incremental_ft(config, base_ckpt, target_classes, device, verbose=False)
    thr_path = calibrate_thresholds(config, inc_ckpt, device, verbose=False)[1]

    clf_i, idim_i, _, _ = load_classifiers(inc_ckpt, device)
    scores_i = get_scores(clf_i, idim_i, z_known, z_full, class_names, device)
    thr_dict = torch.load(thr_path, map_location="cpu")["thresholds"]

    f1_arr, macro_f1, weighted_f1 = compute_multilabel_f1(
        scores_i, y, class_names, thresholds_dict=thr_dict
    )
    return macro_f1, weighted_f1, f1_arr, target_classes


# ─────────────────────────────────────────────────────────────────────────────
# Targeted class metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_targeted_metrics(f1_base, f1_proposed, f1_oracle,
                              target_classes, class_names):
    """
    Compute detailed metrics for targeted classes only.

    Returns dict with:
      n_improved, n_degraded, n_unchanged
      mean_delta, median_delta
      mean_abs_delta, median_abs_delta
      pct_oracle_gain
      best_class, best_delta
      worst_class, worst_delta
      per_class: {name: {base, proposed, oracle, delta, oracle_delta}}
    """
    target_idx = [class_names.index(c) for c in target_classes
                  if c in class_names]

    if not target_idx:
        return {}

    base_f1     = f1_base[target_idx]
    prop_f1     = f1_proposed[target_idx]
    oracle_f1   = f1_oracle[target_idx]

    deltas      = prop_f1 - base_f1
    oracle_deltas = oracle_f1 - base_f1

    # % of oracle gain achieved (avoid div by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_oracle = np.where(
            np.abs(oracle_deltas) > 0.01,
            deltas / oracle_deltas,
            np.where(np.abs(deltas) < 0.01, 1.0, 0.0)
        )

    n_improved  = (deltas >  0.01).sum()
    n_degraded  = (deltas < -0.01).sum()
    n_unchanged = len(deltas) - n_improved - n_degraded

    best_i  = deltas.argmax()
    worst_i = deltas.argmin()
    names   = [class_names[i] for i in target_idx]

    per_class = {
        names[i]: {
            "base"        : float(base_f1[i]),
            "proposed"    : float(prop_f1[i]),
            "oracle"      : float(oracle_f1[i]),
            "delta"       : float(deltas[i]),
            "oracle_delta": float(oracle_deltas[i]),
            "pct_oracle"  : float(pct_oracle[i])
        }
        for i in range(len(names))
    }

    return {
        "n_targeted"       : len(target_idx),
        "n_improved"       : int(n_improved),
        "n_degraded"       : int(n_degraded),
        "n_unchanged"      : int(n_unchanged),
        "mean_delta"       : float(deltas.mean()),
        "median_delta"     : float(np.median(deltas)),
        "mean_abs_delta"   : float(np.abs(deltas).mean()),
        "median_abs_delta" : float(np.median(np.abs(deltas))),
        "pct_oracle_gain"  : float(np.clip(pct_oracle, -1, 2).mean()),
        "best_class"       : names[best_i],
        "best_delta"       : float(deltas[best_i]),
        "worst_class"      : names[worst_i],
        "worst_delta"      : float(deltas[worst_i]),
        "per_class"        : per_class
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary printing
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results, macro_f1_b, weighted_f1_b,
                  macro_f1_o, weighted_f1_o, n_classes):
    print(f"\n{'='*90}")
    print(f"  Annotation Cost vs Performance")
    print(f"{'='*90}")
    print(f"  {'Condition':<28} {'#Labels':>7} {'MacroF1':>8} {'ΔBase':>7} "
          f"{'WtdF1':>8} {'#Imp':>5} {'#Deg':>5} "
          f"{'MeanΔ':>7} {'%Oracle':>8}")
    print(f"  {'─'*88}")

    print(f"  {'Baseline (n sensors)':<28} {'0':>7} {macro_f1_b:>8.4f} {'—':>7} "
          f"{weighted_f1_b:>8.4f} {'—':>5} {'—':>5} {'—':>7} {'—':>8}")

    for r in results:
        m   = r["targeted_metrics"]
        delta = r["macro_f1"] - macro_f1_b
        pct   = (r["macro_f1"] - macro_f1_b) / (macro_f1_o - macro_f1_b + 1e-8) * 100
        print(f"  {'Proposed':<28} {r['budget']:>7} {r['macro_f1']:>8.4f} "
              f"{delta:>+7.4f} {r['weighted_f1']:>8.4f} "
              f"{m.get('n_improved',0):>5} {m.get('n_degraded',0):>5} "
              f"{m.get('mean_delta',0):>+7.3f} {pct:>7.1f}%")

    delta_o = macro_f1_o - macro_f1_b
    print(f"  {'Oracle (all classes)':<28} {n_classes:>7} {macro_f1_o:>8.4f} "
          f"{delta_o:>+7.4f} {weighted_f1_o:>8.4f} {'—':>5} {'—':>5} {'—':>7} "
          f"{'100.0%':>8}")
    print(f"{'='*90}")

    # Detailed per-budget breakdown
    print(f"\n{'='*90}")
    print(f"  Targeted Class Breakdown by Budget")
    print(f"{'='*90}")
    for r in results:
        m = r["targeted_metrics"]
        if not m:
            continue
        print(f"\n  Budget={r['budget']} ({m['n_targeted']} classes)")
        print(f"    Improved: {m['n_improved']}  Degraded: {m['n_degraded']}  "
              f"Unchanged: {m['n_unchanged']}")
        print(f"    Mean Δ: {m['mean_delta']:+.3f}  Median Δ: {m['median_delta']:+.3f}  "
              f"Mean |Δ|: {m['mean_abs_delta']:.3f}")
        print(f"    % Oracle gain: {m['pct_oracle_gain']*100:.1f}%")
        print(f"    Best:  {m['best_class']:45s} {m['best_delta']:+.3f}")
        print(f"    Worst: {m['worst_class']:45s} {m['worst_delta']:+.3f}")
    print(f"{'='*90}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",            type=str, required=True)
    parser.add_argument("--base-checkpoint",   type=str, required=True)
    parser.add_argument("--oracle-checkpoint", type=str, required=True)
    parser.add_argument("--budgets",           type=int, nargs="+",
                        default=[5, 10, 15, 20, 25, 32])
    parser.add_argument("--device",            type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    device   = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log_dir  = config["output"]["checkpoint_dir"]
    os.makedirs(log_dir, exist_ok=True)
    log_dir_logs = config["output"]["log_dir"]
    os.makedirs(log_dir_logs, exist_ok=True)
    log_path = os.path.join(log_dir_logs, f"ablation_budget_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    tee      = Tee(log_path)
    sys.stdout = tee
    print(f"Logging to: {log_path}")
    print(f"Started: {datetime.now().isoformat()}\n")

    # ── Benefit ranking (once) ────────────────────────────────────────────
    print("\nComputing benefit ranking...")
    confusion_scores, confused_classes = detect_confusion(
        config, args.base_checkpoint, device
    )
    benefit_ranked, _ = estimate_benefit(
        config, args.base_checkpoint, confusion_scores, device
    )

    print(f"\nBenefit ranking (top 20):")
    for i, (name, score) in enumerate(benefit_ranked[:20]):
        print(f"  {i+1:3d}. {name:45s} {score:.4f}")

    # ── Encode test data once ─────────────────────────────────────────────
    known_sensors = config["sensors"]["known_sensors"]
    new_sensor    = config["sensors"]["new_sensor"]
    all_sensors   = known_sensors + new_sensor

    encoder = load_simclr_encoder(config["model"]["encoder_path"], device)
    z_known, y, class_names = encode_split(encoder, config, known_sensors, device)
    z_full,  _, _           = encode_split(encoder, config, all_sensors,   device)
    n_classes               = len(class_names)

    # ── Baseline ─────────────────────────────────────────────────────────
    clf_b, idim_b, _, _ = load_classifiers(args.base_checkpoint, device)
    scores_b = get_scores(clf_b, idim_b, z_known, z_full, class_names, device)
    thr_b_path = args.base_checkpoint.replace(".pt", "_thresholds.pt")
    thr_b = torch.load(thr_b_path, map_location="cpu")["thresholds"] \
            if os.path.exists(thr_b_path) else None
    f1_b, macro_f1_b, weighted_f1_b = compute_multilabel_f1(
        scores_b, y, class_names, thresholds_dict=thr_b
    )

    # ── Oracle ────────────────────────────────────────────────────────────
    clf_o, idim_o, _, _ = load_classifiers(args.oracle_checkpoint, device)
    scores_o = get_scores(clf_o, idim_o, z_known, z_full, class_names, device)
    thr_o_path = args.oracle_checkpoint.replace(".pt", "_thresholds.pt")
    thr_o = torch.load(thr_o_path, map_location="cpu")["thresholds"] \
            if os.path.exists(thr_o_path) else None
    f1_o, macro_f1_o, weighted_f1_o = compute_multilabel_f1(
        scores_o, y, class_names, thresholds_dict=thr_o
    )

    # ── Run each budget ───────────────────────────────────────────────────
    all_results = []
    for budget in sorted(args.budgets):
        macro_f1, weighted_f1, f1_arr, targets = run_budget(
            config, args.base_checkpoint, benefit_ranked,
            budget, device, z_known, z_full, y, class_names
        )
        metrics = compute_targeted_metrics(
            f1_b, f1_arr, f1_o, targets, class_names
        )
        # Print per-class table for this budget
        print_combined_table(
            class_names,
            f1_b, f1_arr, f1_o,
            set(targets),
            macro_f1_b, macro_f1,   macro_f1_o,
            weighted_f1_b, weighted_f1, weighted_f1_o,
            budget
        )

        all_results.append({
            "budget"           : budget,
            "macro_f1"         : macro_f1,
            "weighted_f1"      : weighted_f1,
            "target_classes"   : targets,
            "targeted_metrics" : metrics
        })

    # ── Print summary ─────────────────────────────────────────────────────
    print_summary(all_results, macro_f1_b, weighted_f1_b,
                  macro_f1_o, weighted_f1_o, n_classes)

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = os.path.join(
        config["output"]["checkpoint_dir"], "ablation_budget.json"
    )
    with open(out_path, "w") as f:
        json.dump({
            "baseline"      : {"macro_f1": macro_f1_b, "weighted_f1": weighted_f1_b},
            "oracle"        : {"macro_f1": macro_f1_o, "weighted_f1": weighted_f1_o},
            "proposed"      : all_results,
            "benefit_ranked": [(n, float(s)) for n, s in benefit_ranked]
        }, f, indent=2, default=str)
    print(f"\nResults saved → {out_path}")
    print(f"Log file → {log_path}")
    print(f"\nFinished: {datetime.now().isoformat()}")
    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()