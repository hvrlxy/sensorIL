"""
analyze_benefit_score.py

Post-hoc analysis of benefit score formulations using ablation data.

For each targeted class across all sensor configs, computes:
  - Actual ΔF1 (proposed - baseline)
  - Three benefit score formulations:
      1. Product:          (1 - F1) × discriminability   (current)
      2. Confusion only:   (1 - F1)
      3. Discrim only:     discriminability
      4. Weighted sum:     0.5×(1-F1) + 0.5×discriminability

Reports Spearman correlation of each formulation with actual ΔF1.
Positive correlation = benefit score predicts improvement correctly.

Usage:
    python scripts/analyze_benefit_score.py \
        --results-dir checkpoints/ \
        --n-base 1 2 3 4
"""

import os
import json
import argparse
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict


def load_results(results_dir, n_base_list):
    """Load all ablation JSON results."""
    all_data = []
    for n in n_base_list:
        path = os.path.join(results_dir, f"ablation_sensor_n{n}_results.json")
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        with open(path) as f:
            configs = json.load(f)
        print(f"  Loaded n_base={n}: {len(configs)} configs")
        all_data.extend(configs)
    return all_data


def extract_pairs(all_data):
    """
    For each targeted class in each config, extract:
      - actual ΔF1 (from "all" budget or last budget)
      - confusion signal (1 - F1_base)
      - discriminability
      - benefit score components
    """
    records = []

    for config in all_data:
        name         = config["name"]
        benefit_list = config.get("benefit_ranked", [])
        confusion    = config.get("confusion_scores", [])

        # Use "all" budget results if available, else last budget
        budget_results = config.get("budget_results", [])
        if not budget_results:
            continue

        # Prefer "all" budget
        br = next((b for b in budget_results if b["budget"] == "all"),
                  budget_results[-1])
        per_class = br.get("per_class", {})
        if not per_class:
            continue

        # Build benefit score lookup
        benefit_lookup = {name_: score for name_, score in benefit_list}

        # Build confusion f1 lookup
        conf_lookup = {c["class"]: c for c in confusion}

        for cls_name, benefit_score in benefit_list:
            if cls_name not in per_class:
                continue

            pc       = per_class[cls_name]
            targeted = pc.get("targeted", False)
            if not targeted:
                continue

            delta_f1       = pc["proposed"] - pc["baseline"]   # proposed gain
            delta_f1_oracle = pc["oracle"]   - pc["baseline"]   # oracle gain (true potential)
            f1_base  = pc["baseline"]
            conf_sig = 1.0 - f1_base

            # Get val F1 from confusion scores (used for benefit estimation)
            val_f1   = conf_lookup.get(cls_name, {}).get("f1", f1_base)
            conf_sig_val = 1.0 - val_f1

            # Reconstruct discriminability from additive formula:
            # benefit = conf_sig_val + discrim → discrim = benefit - conf_sig_val
            discrim = max(0.0, benefit_score - conf_sig_val)

            records.append({
                "config"        : name,
                "n_base"        : config.get("n_base", 0),
                "class"         : cls_name,
                "delta_f1"      : delta_f1,         # proposed ΔF1
                "delta_f1_oracle": delta_f1_oracle,  # oracle ΔF1 (true potential)
                "conf_signal"   : conf_sig_val,
                "discrim"       : discrim,
                "benefit"       : benefit_score,
                "product"       : conf_sig_val * discrim,
                "weighted_sum"  : 0.5 * conf_sig_val + 0.5 * discrim,
                "f1_base"       : f1_base
            })

    return records


def compute_correlations(records):
    """
    Compute Spearman correlation of each formulation with:
      - delta_f1_oracle: true potential gain (validates benefit score selection)
      - delta_f1:        proposed gain (end-to-end performance)
    """
    delta_f1        = np.array([r["delta_f1"]         for r in records])
    delta_f1_oracle = np.array([r["delta_f1_oracle"]  for r in records])
    additive        = np.array([r["benefit"]           for r in records])
    product         = np.array([r["product"]           for r in records])
    conf_only       = np.array([r["conf_signal"]       for r in records])
    discrim_only    = np.array([r["discrim"]           for r in records])
    weighted        = np.array([r["weighted_sum"]      for r in records])

    formulas = [
        ("Additive (1-F1) + discrim",  additive),
        ("Product  (1-F1) × discrim",  product),
        ("Confusion only  (1-F1)",     conf_only),
        ("Discrim only",               discrim_only),
        ("Weighted sum  0.5+0.5",      weighted),
    ]

    results = {}
    for fname, scores in formulas:
        rho_oracle, pval_oracle = spearmanr(scores, delta_f1_oracle)
        rho_prop,   pval_prop   = spearmanr(scores, delta_f1)
        results[fname] = {
            "rho_oracle": rho_oracle, "pval_oracle": pval_oracle,
            "rho_prop"  : rho_prop,   "pval_prop"  : pval_prop
        }

    return results, delta_f1, delta_f1_oracle, product, conf_only, discrim_only, weighted


def print_report(records, correlations, delta_f1, delta_f1_oracle):
    n = len(records)
    print(f"\n{'='*85}")
    print(f"  Benefit Score Validation  (n={n} targeted class instances)")
    print(f"{'='*85}")

    def interp(rho):
        if abs(rho) > 0.5: return "strong"
        elif abs(rho) > 0.3: return "moderate"
        elif abs(rho) > 0.1: return "weak"
        else: return "none"

    def sig(pval):
        return "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""

    print(f"\n  Correlation with Oracle ΔF1 (true potential — validates selection):")
    print(f"  {'Formula':<35} {'ρ':>8} {'p-value':>12} {'Interpretation'}")
    print(f"  {'─'*70}")
    for name, r in correlations.items():
        rho, pval = r["rho_oracle"], r["pval_oracle"]
        print(f"  {name:<35} {rho:>8.3f} {pval:>12.4f}  {interp(rho)} {sig(pval)}")

    print(f"\n  Correlation with Proposed ΔF1 (end-to-end performance):")
    print(f"  {'Formula':<35} {'ρ':>8} {'p-value':>12} {'Interpretation'}")
    print(f"  {'─'*70}")
    for name, r in correlations.items():
        rho, pval = r["rho_prop"], r["pval_prop"]
        print(f"  {name:<35} {rho:>8.3f} {pval:>12.4f}  {interp(rho)} {sig(pval)}")

    print(f"\n  ΔF1 distributions:")
    for label, arr in [("Oracle (true potential)", delta_f1_oracle),
                        ("Proposed (end-to-end)", delta_f1)]:
        print(f"    {label}: mean={arr.mean():+.3f}  median={np.median(arr):+.3f}  "
              f"improved={(arr>0.01).mean()*100:.0f}%  "
              f"degraded={(arr<-0.01).mean()*100:.0f}%")
    print(f"{'='*85}")

    # Per n_base breakdown
    by_nbase = defaultdict(lambda: {"prop": [], "oracle": []})
    for r in records:
        nb = r.get("n_base", len(r["config"].split("__")[0].split("_")))
        by_nbase[nb]["prop"].append(r["delta_f1"])
        by_nbase[nb]["oracle"].append(r["delta_f1_oracle"])

    print(f"\n  By number of base sensors:")
    print(f"  {'n_base':>8} {'Oracle mean':>13} {'Prop mean':>11} {'Imp%':>6} {'n':>6}")
    print(f"  {'─'*50}")
    for nb in sorted(by_nbase.keys()):
        op = np.array(by_nbase[nb]["oracle"])
        pp = np.array(by_nbase[nb]["prop"])
        print(f"  {nb:>8} {op.mean():>+13.3f} {pp.mean():>+11.3f} "
              f"{(pp>0.01).mean()*100:>5.0f}% {len(pp):>6}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--n-base",      type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--output",      type=str, default=None)
    args = parser.parse_args()

    print("Loading ablation results...")
    all_data = load_results(args.results_dir, args.n_base)
    if not all_data:
        print("No data found. Run ablation_sensor.py first.")
        return

    print(f"Total configs loaded: {len(all_data)}")

    records = extract_pairs(all_data)
    print(f"Targeted class instances: {len(records)}")

    correlations, delta_f1, delta_f1_oracle, *_ = compute_correlations(records)
    print_report(records, correlations, delta_f1, delta_f1_oracle)

    # Save
    out = args.output or os.path.join(args.results_dir, "benefit_score_analysis.json")
    with open(out, "w") as f:
        json.dump({
            "n_records"   : len(records),
            "correlations": {k: {"rho_oracle": v["rho_oracle"],
                                  "pval_oracle": v["pval_oracle"],
                                  "rho_prop"  : v["rho_prop"],
                                  "pval_prop" : v["pval_prop"]}
                             for k, v in correlations.items()},
            "delta_f1_stats": {
                "mean"    : float(delta_f1.mean()),
                "median"  : float(np.median(delta_f1)),
                "improved": int((delta_f1 > 0.01).sum()),
                "degraded": int((delta_f1 < -0.01).sum()),
                "n"       : len(records)
            },
            "delta_f1_oracle_stats": {
                "mean"    : float(delta_f1_oracle.mean()),
                "median"  : float(np.median(delta_f1_oracle)),
                "improved": int((delta_f1_oracle > 0.01).sum()),
                "degraded": int((delta_f1_oracle < -0.01).sum()),
                "n"       : len(records)
            },
            "records"     : records
        }, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()