"""
run_pipeline.py

Full active sensor increment pipeline:

  Step 1:  Train base binary classifiers (n sensors)
  Step 2:  Detect confused classes from val set
  Step 3:  Estimate which confused classes benefit from new sensor
  Step 4:  (Simulated) annotation request for target classes
  Step 5:  Incremental fine-tuning with PU learning + diverse FL negatives
  Step 5b: Oracle — retrain all classes with n+1 sensors
  Step 6:  Three-way evaluation (baseline vs proposed vs oracle)

Usage:
    python scripts/run_pipeline.py --config configs/pipeline_config.json
    python scripts/run_pipeline.py --config configs/pipeline_config.json \
                                    --base-checkpoint checkpoints/base_classifiers.pt
"""

import os
import json
import argparse
import copy
import torch

from train_base       import train_base
from calibrate_thresholds import calibrate_thresholds
from detect_confusion import detect_confusion
from estimate_benefit import estimate_benefit
from incremental_ft   import incremental_ft
from evaluate         import evaluate


def train_oracle(config, device):
    """Train oracle: ALL classes retrained with n+1 sensors."""
    print(f"\n{'='*60}")
    print(f"Step 5b: Training Oracle (all classes, n+1 sensors)")
    print(f"{'='*60}")

    known   = config["sensors"]["known_sensors"]
    new     = config["sensors"]["new_sensor"]

    # Build oracle config with n+1 sensors as known, no new sensor
    oracle_config = copy.deepcopy(config)
    oracle_config["sensors"]["known_sensors"] = known + new
    oracle_config["sensors"]["new_sensor"]    = []
    oracle_config["checkpoint_name"]          = "oracle_classifiers_tmp.pt"

    ckpt_path, _, _ = train_base(oracle_config, device)

    # Load, fix metadata, save to oracle path
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt["input_dims"]     = {name: ckpt["input_dim"]
                               for name in ckpt["class_names"]}
    ckpt["target_classes"] = ckpt["class_names"]
    ckpt["known_sensors"]  = known + new
    ckpt["new_sensor"]     = []

    oracle_path = os.path.join(
        config["output"]["checkpoint_dir"], "oracle_classifiers.pt"
    )
    torch.save(ckpt, oracle_path)

    # Restore base_classifiers.pt — oracle must not overwrite it
    # (train_base saves to base_classifiers.pt by default)
    import shutil
    base_path = os.path.join(config["output"]["checkpoint_dir"], "base_classifiers.pt")
    if os.path.exists(base_path) and ckpt_path == base_path:
        print("WARNING: oracle overwrote base_classifiers.pt — "
              "please re-run Step 1 or pass --base-checkpoint explicitly")

    print(f"Oracle checkpoint → {oracle_path}")
    return oracle_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",                  type=str, required=True)
    parser.add_argument("--device",                  type=str, default="cuda")
    parser.add_argument("--base-checkpoint",         type=str, default=None)
    parser.add_argument("--incremental-checkpoint",  type=str, default=None)
    parser.add_argument("--oracle-checkpoint",       type=str, default=None)
    parser.add_argument("--top-k-confused",          type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Step 1: Base training ──────────────────────────────────────────────
    if args.base_checkpoint:
        base_ckpt = args.base_checkpoint
        print(f"\n[Skip Step 1] Using: {base_ckpt}")
    else:
        base_ckpt, _, _ = train_base(config, device)

    # ── Steps 2-5 ──────────────────────────────────────────────────────────
    if args.incremental_checkpoint and args.oracle_checkpoint:
        inc_ckpt    = args.incremental_checkpoint
        oracle_ckpt = args.oracle_checkpoint
        print(f"\n[Skip Steps 2-5] Using provided checkpoints.")
    else:
        # Calibrate base thresholds
        calibrate_thresholds(config, base_ckpt, device, verbose=False)

        # Step 2: Confusion detection
        confusion_scores, confused_classes = detect_confusion(
            config, base_ckpt, device,
            top_k=args.top_k_confused
        )

        # Step 3: Benefit estimation
        benefit_ranked, target_classes, all_candidates = estimate_benefit(
            config, base_ckpt, confusion_scores, device
        )

        # Step 4: Simulated annotation
        print(f"\n{'='*60}")
        print(f"Step 4: Annotation request")
        print(f"{'='*60}")
        print(f"Requesting {len(target_classes)} classes from user:")
        for cls in target_classes:
            print(f"  - {cls}")

        # Step 5: Incremental fine-tuning
        inc_ckpt, _, _, _ = incremental_ft(
            config, base_ckpt, target_classes, device
        )

        # Calibrate incremental thresholds
        calibrate_thresholds(config, inc_ckpt, device, verbose=False)

        # Step 5b: Oracle
        oracle_ckpt = train_oracle(config, device)

    # Calibrate oracle thresholds
    calibrate_thresholds(config, oracle_ckpt, device, verbose=False)

    # ── Step 6: Evaluation ────────────────────────────────────────────────
    results = evaluate(config, base_ckpt, inc_ckpt, oracle_ckpt, device)

    print(f"\n{'='*60}")
    print(f"Pipeline complete")
    b = results['baseline']['macro_f1']
    p = results['proposed']['macro_f1']
    o = results['oracle']['macro_f1']
    n = results['proposed']['n_new_labels']
    print(f"  {'Condition':<30} {'Macro F1':>10} {'Delta':>10}")
    print(f"  {'─'*52}")
    print(f"  {'Baseline (n sensors)':<30} {b:>10.4f} {'—':>10}")
    print(f"  {f'Proposed ({n} new labels)':<30} {p:>10.4f} {p-b:>+10.4f}")
    print(f"  {'Oracle (all classes)':<30} {o:>10.4f} {o-b:>+10.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()