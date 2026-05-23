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
      - actual ΔF1
      - confusion signal (1 - F1_base)
      - discriminability
      - product score
    """
    records = []

    for config in all_data:
        name         = config["name"]
        per_class    = config["per_class"]
        benefit_list = config.get("benefit_ranked", [])
        confusion    = config.get("confusion_scores", [])

        # Build lookup: class → discriminability
        # benefit_ranked = [(class, product_score), ...]
        # confusion_scores = [{class, f1, ...}, ...]
        conf_lookup  = {c["class"]: c for c in confusion}

        # Reconstruct discriminability from product and confusion signal
        # product = (1-F1) * discrim  →  discrim = product / (1-F1)
        for cls_name, product_score in benefit_list:
            if cls_name not in per_class:
                continue

            pc       = per_class[cls_name]
            targeted = pc.get("targeted", False)
            if not targeted:
                continue

            delta_f1 = pc["proposed"] - pc["baseline"]
            f1_base  = pc["baseline"]
            conf_sig = 1.0 - f1_base

            # Reconstruct discriminability
            if conf_sig > 0.01:
                discrim = product_score / conf_sig
            else:
                discrim = 0.0

            records.append({
                "config"      : name,
                "class"       : cls_name,
                "delta_f1"    : delta_f1,
                "conf_signal" : conf_sig,
                "discrim"     : discrim,
                "product"     : product_score,
                "weighted_sum": 0.5 * conf_sig + 0.5 * discrim,
                "f1_base"     : f1_base
            })

    return records


def compute_correlations(records):
    """Compute Spearman correlation of each formulation with actual ΔF1."""
    delta_f1    = np.array([r["delta_f1"]     for r in records])
    product     = np.array([r["product"]      for r in records])
    conf_only   = np.array([r["conf_signal"]  for r in records])
    discrim_only= np.array([r["discrim"]      for r in records])
    weighted    = np.array([r["weighted_sum"] for r in records])

    results = {}
    for name, scores in [
        ("Product  (1-F1) × discrim", product),
        ("Confusion only  (1-F1)",    conf_only),
        ("Discrim only",              discrim_only),
        ("Weighted sum  0.5+0.5",     weighted),
    ]:
        rho, pval = spearmanr(scores, delta_f1)
        results[name] = {"rho": rho, "pval": pval}

    return results, delta_f1, product, conf_only, discrim_only, weighted


def print_report(records, correlations, delta_f1):
    n = len(records)
    print(f"\n{'='*65}")
    print(f"  Benefit Score Validation  (n={n} targeted class instances)")
    print(f"{'='*65}")
    print(f"\n  Spearman correlation with actual ΔF1:")
    print(f"  {'Formula':<35} {'ρ':>8} {'p-value':>12} {'Interpretation'}")
    print(f"  {'─'*63}")

    for name, r in correlations.items():
        rho, pval = r["rho"], r["pval"]
        if abs(rho) > 0.5:
            interp = "strong"
        elif abs(rho) > 0.3:
            interp = "moderate"
        elif abs(rho) > 0.1:
            interp = "weak"
        else:
            interp = "none"
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {name:<35} {rho:>8.3f} {pval:>12.4f}  {interp} {sig}")

    print(f"\n  ΔF1 distribution across targeted classes:")
    print(f"    Mean:   {delta_f1.mean():+.3f}")
    print(f"    Median: {np.median(delta_f1):+.3f}")
    print(f"    Improved (>0.01):  {(delta_f1 > 0.01).sum()}/{n} "
          f"({(delta_f1 > 0.01).mean()*100:.1f}%)")
    print(f"    Degraded (<-0.01): {(delta_f1 < -0.01).sum()}/{n} "
          f"({(delta_f1 < -0.01).mean()*100:.1f}%)")
    print(f"{'='*65}")

    # Per n_base breakdown
    by_nbase = defaultdict(list)
    for r in records:
        n_base = len(r["config"].split("__")[0].split("_"))
        by_nbase[n_base].append(r["delta_f1"])

    print(f"\n  ΔF1 by number of base sensors:")
    for nb in sorted(by_nbase.keys()):
        vals = np.array(by_nbase[nb])
        print(f"    n_base={nb}: mean={vals.mean():+.3f}  "
              f"improved={( vals>0.01).mean()*100:.0f}%  "
              f"n={len(vals)}")


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

    correlations, delta_f1, *_ = compute_correlations(records)
    print_report(records, correlations, delta_f1)

    # Save
    out = args.output or os.path.join(args.results_dir, "benefit_score_analysis.json")
    with open(out, "w") as f:
        json.dump({
            "n_records"   : len(records),
            "correlations": {k: {"rho": v["rho"], "pval": v["pval"]}
                             for k, v in correlations.items()},
            "delta_f1_stats": {
                "mean"    : float(delta_f1.mean()),
                "median"  : float(np.median(delta_f1)),
                "improved": int((delta_f1 > 0.01).sum()),
                "degraded": int((delta_f1 < -0.01).sum()),
                "n"       : len(records)
            },
            "records"     : records
        }, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
