"""
detect_confusion.py

Step 2: Identify confused classes from val set performance.

For each class, compute:
  - Val F1 score from binary classifier
  - Confusion pairs: which other classes are most confused with it

Returns ranked list of classes by confusion level.

Usage:
    python scripts/detect_confusion.py --config configs/pipeline_config.json \
                                        --checkpoint checkpoints/base_classifiers.pt
"""

import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix

from simclr_encoder import load_simclr_encoder, encode_sensors, ENCODER_DIM
from dataset import SensorDataset
from train_base import BinaryClassifier
from cooccurrence import are_related


def detect_confusion(config, checkpoint_path, device, top_k=None):
    print(f"\n{'='*60}")
    print(f"Step 2: Detecting confused classes")
    print(f"{'='*60}")

    sensors   = config["sensors"]["known_sensors"]
    input_dim = len(sensors) * ENCODER_DIM

    # Load encoder
    encoder = load_simclr_encoder(config["model"]["encoder_path"], device)

    # Load classifiers
    ckpt        = torch.load(checkpoint_path, map_location=device)
    class_names = ckpt["class_names"]
    n_classes   = ckpt["n_classes"]

    classifiers = {}
    for name in class_names:
        idim = ckpt.get("input_dims", {}).get(name, ckpt.get("input_dim", input_dim))
        clf  = BinaryClassifier(idim).to(device)
        clf.load_state_dict(ckpt["classifiers"][name])
        clf.eval()
        classifiers[name] = clf

    # Load val data
    val_ds = SensorDataset(
        data_dir              = config["data"]["labeled_dir"],
        sensors               = sensors,
        max_samples_per_class = config["finetune"]["few_shot_samples_per_class"],
        split                 = "val",
        val_split             = config["finetune"]["val_split"]
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    # Encode val set
    all_z, all_y = [], []
    for x, y in val_loader:
        z = encode_sensors(encoder, x, device)
        all_z.append(z)
        all_y.append(y)
    z_val = torch.cat(all_z).to(device)
    y_val = torch.cat(all_y).numpy()

    # Get predictions from all binary classifiers (batched)
    n_val  = len(y_val)
    scores = np.zeros((n_val, n_classes))
    bs     = 4096
    z_val_dev = z_val.to(device)
    with torch.no_grad():
        for i, name in enumerate(class_names):
            clf_scores = []
            for start in range(0, n_val, bs):
                clf_scores.append(
                    torch.sigmoid(classifiers[name](z_val_dev[start:start+bs])).cpu().numpy()
                )
            scores[:, i] = np.concatenate(clf_scores)

    # Independent per-class binary predictions
    # For confusion detection: convert multi-label to single prediction
    # by picking highest-scoring class that fires, or highest overall
    threshold = 0.5
    n = len(scores)
    preds = np.zeros(n, dtype=int)
    for j in range(n):
        fired = np.where(scores[j] > threshold)[0]
        if len(fired) == 1:
            preds[j] = fired[0]
        elif len(fired) > 1:
            preds[j] = fired[scores[j, fired].argmax()]
        else:
            preds[j] = scores[j].argmax()

    # Per-class F1 + overall macro/weighted
    f1_scores    = f1_score(y_val, preds, average=None,
                            labels=list(range(n_classes)),
                            zero_division=0)
    macro_f1     = f1_score(y_val, preds, average="macro",    zero_division=0)
    weighted_f1  = f1_score(y_val, preds, average="weighted", zero_division=0)
    print(f"\nBase model val — Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")

    # Confusion matrix for confusion pairs
    cm = confusion_matrix(y_val, preds, labels=list(range(n_classes)))

    # Confusion score per class: sum of off-diagonal elements
    # EXCLUDING related classes (ancestors/descendants via co-occurrence)
    # e.g. Treadmill confused with Walking is expected — not a real confusion
    confusion_scores = []
    confusion_pairs  = {}

    for i, name in enumerate(class_names):
        row = cm[i].copy()
        row[i] = 0

        # Zero out related class confusions (expected by hierarchy)
        for j, other in enumerate(class_names):
            if are_related(name, other):
                row[j] = 0

        col = cm[:, i].copy()
        col[i] = 0
        for j, other in enumerate(class_names):
            if are_related(name, other):
                col[j] = 0

        confused_as   = row.sum()
        confused_with = col.sum()
        total_confusion = confused_as + confused_with

        # Top confused pairs (unrelated only)
        top_confused_idx = row.argsort()[::-1][:3]
        pairs = [(class_names[j], int(row[j]))
                 for j in top_confused_idx if row[j] > 0]

        confusion_scores.append({
            "class"          : name,
            "class_id"       : i,
            "f1"             : float(f1_scores[i]),
            "confusion_score": int(total_confusion),
            "confused_as"    : pairs
        })
        confusion_pairs[name] = pairs

    # Sort by confusion score descending
    confusion_scores.sort(key=lambda x: x["confusion_score"], reverse=True)

    print(f"\n{'Class':45s} {'F1':>8} {'Confusion':>10} {'Confused with'}")
    print("─" * 90)
    for item in confusion_scores:
        pairs_str = ", ".join([f"{p[0]}({p[1]})" for p in item["confused_as"][:2]])
        print(f"{item['class']:45s} {item['f1']:8.3f} "
              f"{item['confusion_score']:10d}   {pairs_str}")

    # Select top_k most confused classes
    if top_k is None:
        top_k = config.get("active_learning", {}).get("top_k_confused", 10)

    confused_classes = [item["class"] for item in confusion_scores[:top_k]]
    print(f"\nTop {top_k} confused classes: {confused_classes}")

    return confusion_scores, confused_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--top-k",      type=int, default=None)
    parser.add_argument("--device",     type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    detect_confusion(config, args.checkpoint, device, args.top_k)


if __name__ == "__main__":
    main()