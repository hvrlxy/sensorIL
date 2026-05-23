"""
evaluate.py

Evaluates and compares:
  Baseline : pretrained on n sensors, tested on n sensors
  BYOL     : pretrained on n+1 sensors, tested on n+1 sensors (real, no injection)

At test time the BYOL model uses the REAL n+1 sensor — no injection needed.
The injection was only for fine-tuning (no labeled n+1 data).
At test time we have real sensor data so we just run the full forward pass.

Usage:
    python scripts/evaluate.py --config configs/byol_config.json \\
                                --experiment 2to1 \\
                                --baseline-checkpoint checkpoints/finetune_baseline_2to1_best.pt \\
                                --byol-checkpoint     checkpoints/finetune_byol_2to1_best.pt
"""

import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, classification_report

from scripts.misc.dataset import get_test_loaders
from scripts.misc.stream_encoder import build_encoder, build_classifier


@torch.no_grad()
def run_eval(encoder, classifier, loader, device, mask_idx=None):
    encoder.eval()
    classifier.eval()
    all_preds, all_labels = [], []

    for x, y in loader:
        x      = x.to(device)
        z      = F.normalize(encoder(x, mask_indices=mask_idx), dim=-1)
        logits = classifier(z)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

    return np.array(all_preds), np.array(all_labels)


def print_results(name, preds, labels, class_names):
    acc      = (preds == labels).mean()
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {macro_f1:.4f}")
    print()
    print(classification_report(
        labels, preds,
        target_names=class_names,
        zero_division=0,
        digits=4
    ))
    return acc, macro_f1


def evaluate(config, experiment, baseline_ckpt_path, byol_ckpt_path, device):
    print(f"\n{'='*60}")
    print(f"Evaluation | experiment={experiment}")
    print(f"{'='*60}")

    # ── Baseline: n sensors, no masking ──
    ckpt_b      = torch.load(baseline_ckpt_path, map_location=device)
    n_classes   = ckpt_b["n_classes"]
    class_names = ckpt_b["class_names"]

    enc_b = build_encoder(config).to(device)
    cls_b = build_classifier(config, n_classes).to(device)
    enc_b.load_state_dict(ckpt_b["encoder"])
    cls_b.load_state_dict(ckpt_b["classifier"])

    baseline_loader, _, _ = get_test_loaders(config, experiment, mode="baseline")
    baseline_preds, labels_b = run_eval(
        enc_b, cls_b, baseline_loader, device, mask_idx=None
    )
    baseline_acc, baseline_f1 = print_results(
        "Baseline (n sensors only)", baseline_preds, labels_b, class_names
    )

    # ── BYOL: n+1 sensors, no masking (real sensor at test time) ──
    ckpt_y        = torch.load(byol_ckpt_path, map_location=device)
    n_classes_y   = ckpt_y["n_classes"]
    class_names_y = ckpt_y["class_names"]

    enc_y = build_encoder(config).to(device)
    cls_y = build_classifier(config, n_classes_y).to(device)
    enc_y.load_state_dict(ckpt_y["encoder"])
    cls_y.load_state_dict(ckpt_y["classifier"])

    byol_loader, _, _ = get_test_loaders(config, experiment, mode="byol")
    byol_preds, labels_y = run_eval(
        enc_y, cls_y, byol_loader, device, mask_idx=None
    )
    byol_acc, byol_f1 = print_results(
        "BYOL (n+1 sensors)", byol_preds, labels_y, class_names_y
    )

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  {'':30s} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'Baseline (n sensors)':30s} {baseline_acc:>10.4f} {baseline_f1:>10.4f}")
    print(f"  {'BYOL (n+1 sensors)':30s} {byol_acc:>10.4f} {byol_f1:>10.4f}")
    print(f"  {'Delta':30s} {byol_acc-baseline_acc:>+10.4f} {byol_f1-baseline_f1:>+10.4f}")
    print(f"{'='*60}")

    return {
        "baseline_acc": baseline_acc, "baseline_f1": baseline_f1,
        "byol_acc":     byol_acc,     "byol_f1":     byol_f1,
        "delta_acc":    byol_acc - baseline_acc,
        "delta_f1":     byol_f1  - baseline_f1
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",              type=str, required=True)
    parser.add_argument("--experiment",          type=str, required=True,
                        choices=["2to1", "1to1"])
    parser.add_argument("--baseline-checkpoint", type=str, required=True)
    parser.add_argument("--byol-checkpoint",     type=str, required=True)
    parser.add_argument("--device",              type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    evaluate(config, args.experiment,
             args.baseline_checkpoint, args.byol_checkpoint, device)


if __name__ == "__main__":
    main()
