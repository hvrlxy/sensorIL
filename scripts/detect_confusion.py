"""
detect_confusion.py

Step 2: Identify confused classes from val set performance.

Confusion is counted only between semantically UNRELATED classes —
parent/child pairs in the co-occurrence hierarchy are excluded
(e.g. Treadmill_3mph → Walking is expected, not a confusion).

Ranks by total confusion score descending, takes top_k.

Usage:
    python scripts/detect_confusion.py --config configs/pipeline_config.json \
                                        --base-checkpoint checkpoints/base_classifiers.pt
"""

import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
from cooccurrence import get_multilabel

from simclr_encoder import load_simclr_encoder, encode_sensors, ENCODER_DIM
from dataset import SensorDataset
from train_base import BinaryClassifier
from cooccurrence import get_all_related


def detect_confusion(config, checkpoint_path, device, top_k=None):
    print(f"\n{'='*60}")
    print(f"Step 2: Detecting confused classes")
    print(f"{'='*60}")

    sensors   = config["sensors"]["known_sensors"]
    input_dim = len(sensors) * ENCODER_DIM

    encoder = load_simclr_encoder(config["model"]["encoder_path"], device)

    ckpt        = torch.load(checkpoint_path, map_location=device)
    class_names = ckpt["class_names"]
    n_classes   = len(class_names)

    classifiers = {}
    for name in class_names:
        idim = ckpt.get("input_dims", {}).get(name, ckpt.get("input_dim", input_dim))
        clf  = BinaryClassifier(idim).to(device)
        clf.load_state_dict(ckpt["classifiers"][name])
        clf.eval()
        classifiers[name] = clf

    val_ds = SensorDataset(
        data_dir              = config["data"]["labeled_dir"],
        sensors               = sensors,
        max_samples_per_class = config["finetune"]["few_shot_samples_per_class"],
        split                 = "val",
        val_split             = config["finetune"]["val_split"]
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    all_z, all_y = [], []
    for x, y in val_loader:
        z = encode_sensors(encoder, x, device)
        all_z.append(z)
        all_y.append(y)
    z_val = torch.cat(all_z).to(device)
    y_val = torch.cat(all_y).numpy()

    # Argmax predictions
    scores = np.zeros((len(y_val), n_classes))
    with torch.no_grad():
        for i, name in enumerate(class_names):
            scores[:, i] = torch.sigmoid(
                classifiers[name](z_val)).cpu().numpy()
    preds = scores.argmax(axis=1)

    # Per-class F1 — multilabel-aware (co-occurrence hierarchy)
    # Ground truth: a Treadmill window is also positive for Walking etc.
    ml_matrix = np.array([get_multilabel(name, class_names)
                          for name in class_names], dtype=np.float32)
    # y_ml: (N, n_classes) multilabel ground truth
    y_ml = ml_matrix[y_val]  # (N, n_classes)

    # Predicted scores: use per-class sigmoid scores for binary F1
    f1_scores = np.zeros(n_classes)
    for i in range(n_classes):
        pred_i = (scores[:, i] > 0.5).astype(int)
        true_i = y_ml[:, i].astype(int)
        f1_scores[i] = f1_score(true_i, pred_i, zero_division=0)

    macro_f1 = float(f1_scores.mean())
    wtd_f1   = float(np.average(f1_scores,
                                weights=y_ml.sum(axis=0) + 1e-10))
    print(f"\nBase model val — Macro F1: {macro_f1:.4f} | Weighted F1: {wtd_f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_val, preds, labels=list(range(n_classes)))

    # Build related-class lookup per class
    related_idx = {}
    for i, name in enumerate(class_names):
        related = get_all_related(name)
        related_idx[i] = {j for j, cn in enumerate(class_names) if cn in related}

    confusion_scores = []
    for i, name in enumerate(class_names):
        row = cm[i].copy()
        row[i] = 0
        col = cm[:, i].copy()
        col[i] = 0

        # Zero out related pairs — only count unrelated confusion
        for j in related_idx[i]:
            row[j] = 0
            col[j] = 0

        total_confusion = int(row.sum() + col.sum())

        # Top confused pairs from row (unrelated only)
        top_idx = row.argsort()[::-1][:3]
        pairs   = [(class_names[j], int(row[j]))
                   for j in top_idx if row[j] > 0]

        confusion_scores.append({
            "class"          : name,
            "class_id"       : i,
            "f1"             : float(f1_scores[i]),
            "confusion_score": total_confusion,
            "confused_as"    : pairs
        })

    # Rank by unrelated confusion score descending
    confusion_scores.sort(key=lambda x: x["confusion_score"], reverse=True)

    print(f"\n{'Class':45s} {'F1':>8} {'Confusion':>10}  Confused with")
    print("─" * 90)
    for item in confusion_scores:
        pairs_str = ", ".join(f"{c}({n})" for c, n in item["confused_as"][:2])
        print(f"{item['class']:45s} {item['f1']:8.3f} "
              f"{item['confusion_score']:10d}   {pairs_str}")

    if top_k is None:
        top_k = config.get("active_learning", {}).get("top_k_confused", 15)

    confused_classes = [item["class"] for item in confusion_scores[:top_k]]
    print(f"\nTop {top_k} confused classes: {confused_classes}")
    return confusion_scores, confused_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          type=str, required=True)
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--top-k",           type=int, default=None)
    parser.add_argument("--device",          type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    detect_confusion(config, args.base_checkpoint, device, args.top_k)

if __name__ == "__main__":
    main()