"""
run_experiment.py

Full pipeline:
  Phase 1a: Baseline pretraining (BYOL-style, n sensors, augmentation)
  Phase 1b: BYOL pretraining (BYOL + regression + SupCon, n+1 sensors)
  Phase 1c: Compute embedding shift from FL data (byol only)
  Phase 2a: Baseline fine-tuning (frozen encoder + classifier)
  Phase 2b: BYOL fine-tuning (frozen encoder + shift augmentation + classifier)
  Phase 3:  Evaluation

Usage:
    # Full run from scratch
    python scripts/run_experiment.py --config configs/byol_config.json \
                                      --experiment 2to1

    # Skip pretraining
    python scripts/run_experiment.py --config configs/byol_config.json \
                                      --experiment 2to1 \
                                      --baseline-pretrain-ckpt checkpoints/pretrain_baseline_2to1_best.pt \
                                      --byol-pretrain-ckpt     checkpoints/pretrain_byol_2to1_best.pt

    # Skip everything, just evaluate
    python scripts/run_experiment.py --config configs/byol_config.json \
                                      --experiment 2to1 \
                                      --baseline-finetune-ckpt checkpoints/finetune_baseline_2to1_best.pt \
                                      --byol-finetune-ckpt     checkpoints/finetune_byol_2to1_best.pt
"""

import json
import argparse
import os
import torch

from scripts.misc.pretrain        import pretrain
from scripts.misc.finetune        import finetune
from scripts.misc.evaluate        import evaluate
from scripts.misc.compute_shift   import compute_shift


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["2to1", "1to1"])
    parser.add_argument("--device",     type=str, default="cuda")

    parser.add_argument("--baseline-pretrain-ckpt",  type=str, default=None)
    parser.add_argument("--byol-pretrain-ckpt",      type=str, default=None)
    parser.add_argument("--baseline-finetune-ckpt",  type=str, default=None)
    parser.add_argument("--byol-finetune-ckpt",      type=str, default=None)
    parser.add_argument("--shift-ckpt",              type=str, default=None)

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice     : {device}")
    print(f"Experiment : {args.experiment}")

    if args.baseline_finetune_ckpt and args.byol_finetune_ckpt:
        # Skip everything, just evaluate
        print("\n[Skipping Phase 1+2] Using provided finetune checkpoints.")
        baseline_finetune_ckpt = args.baseline_finetune_ckpt
        byol_finetune_ckpt     = args.byol_finetune_ckpt
    else:
        # ── Phase 1: Pretraining ──────────────────────────────────────────
        if args.baseline_pretrain_ckpt:
            baseline_pretrain_ckpt = args.baseline_pretrain_ckpt
            print(f"\n[Baseline] Skipping Phase 1: {baseline_pretrain_ckpt}")
        else:
            baseline_pretrain_ckpt = pretrain(
                config, args.experiment, mode="baseline", device=device
            )

        if args.byol_pretrain_ckpt:
            byol_pretrain_ckpt = args.byol_pretrain_ckpt
            print(f"\n[BYOL] Skipping Phase 1: {byol_pretrain_ckpt}")
        else:
            byol_pretrain_ckpt = pretrain(
                config, args.experiment, mode="byol", device=device
            )

        # ── Phase 1c: Compute embedding shift ────────────────────────────
        if args.shift_ckpt:
            shift_path = args.shift_ckpt
            print(f"\n[BYOL] Using existing shift: {shift_path}")
        else:
            shift_path = compute_shift(
                config, args.experiment, byol_pretrain_ckpt, device
            )

        # ── Phase 2: Fine-tuning ──────────────────────────────────────────
        baseline_finetune_ckpt = finetune(
            config, args.experiment, mode="baseline",
            checkpoint_path=baseline_pretrain_ckpt,
            device=device,
            shift_path=None
        )

        byol_finetune_ckpt = finetune(
            config, args.experiment, mode="byol",
            checkpoint_path=byol_pretrain_ckpt,
            device=device,
            shift_path=shift_path
        )

    # ── Phase 3: Evaluation ───────────────────────────────────────────────
    results = evaluate(
        config, args.experiment,
        baseline_ckpt_path = baseline_finetune_ckpt,
        byol_ckpt_path     = byol_finetune_ckpt,
        device             = device
    )

    print(f"\n{'='*60}")
    print(f"  Experiment {args.experiment} complete")
    print(f"  Baseline acc : {results['baseline_acc']:.4f}")
    print(f"  BYOL acc     : {results['byol_acc']:.4f}")
    print(f"  Delta        : {results['delta_acc']:+.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
