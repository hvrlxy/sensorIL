"""
detect_confusion.py

Step 2: Detect classes that struggle under the n-sensor base model.

Ranks ALL classes by F1 score (ascending) — any class with low F1
is a candidate for re-annotation regardless of how it's misclassified.
Also records what each class gets confused with for the benefit
estimation step.

Usage:
    python scripts/detect_confusion.py --config configs/pipeline_config.json \
                                        --base-checkpoint checkpoints/base_classifiers.pt
"""

import json
import argparse
import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader

from simclr_encoder import load_simclr_encoder, encode_sensors, ENCODER_DIM
from dataset import SensorDataset
from train_base import BinaryClassifier


def detect_confusion(config, base_ckpt_path, device, top_k=None):
    print(f"\n{'='*60}")
    print(f"Step 2: Detecting confused classes")
    print(f"{'='*60}")

    known_sensors = config["sensors"]["known_sensors"]
    input_dim     = len(known_sensors) * ENCODER_DIM

    encoder = load_simclr_encoder(config["model"]["encoder_path"], device)

    # Load val data
    val_ds = SensorDataset(
        data_dir              = config["data"]["labeled_dir"],
        sensors               = known_sensors,
        max_samples_per_class = config["finetune"]["few_shot_samples_per_class"],
        split                 = "val",
        val_split             = config["finetune"]["val_split"]
    )
    class_names = val_ds.class_names
    n_classes   = val_ds.n_classes

    loader = DataLoader(val_ds, batch_size=512, shuffle=False,
                        num_workers=2, pin_memory=True)
    all_z, all_y = [], []
    for x, y in loader:
        all_z.append(encode_sensors(encoder, x, device))
        all_y.append(y)
    z_val = torch.cat(all_z)
    y_val = torch.cat(all_y).numpy()

    # Load classifiers
    ckpt        = torch.load(base_ckpt_path, map_location=device)
    classifiers = {}
    for name in class_names:
        idim = ckpt.get("input_dims", {}).get(name, ckpt.get("input_dim", input_dim))
        clf  = BinaryClassifier(idim).to(device)
        clf.load_state_dict(ckpt["classifiers"][name])
        clf.eval()
        classifiers[name] = clf

    # Get scores — batched
    n_val  = len(y_val)
    scores = np.zeros((n_val, n_classes))
    z_dev  = z_val.to(device)
    bs     = 4096

    with torch.no_grad():
        for i, name in enumerate(class_names):
            clf_scores = []
            for start in range(0, n_val, bs):
                clf_scores.append(
                    torch.sigmoid(classifiers[name](z_dev[start:start+bs])).cpu().numpy()
                )
            scores[:, i] = np.concatenate(clf_scores)

    # Predict: threshold-based only — no argmax fallback
    # If nothing fires, mark as -1 (no prediction)
    threshold = 0.5
    preds = np.full(n_val, -1, dtype=int)
    for j in range(n_val):
        fired = np.where(scores[j] > threshold)[0]
        if len(fired) == 1:
            preds[j] = fired[0]
        elif len(fired) > 1:
            preds[j] = fired[scores[j, fired].argmax()]
        # else: no prediction (-1)

    n_no_pred = (preds == -1).sum()
    if n_no_pred > 0:
        print(f"  {n_no_pred}/{n_val} windows had no classifier fire above threshold")

    # Per-class F1
    f1_scores   = f1_score(y_val, preds, average=None,
                           labels=list(range(n_classes)), zero_division=0)
    macro_f1    = f1_score(y_val, preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_val, preds, average="weighted", zero_division=0)
    print(f"\nBase model val — Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")

    # Build per-class confusion using binary scores directly
    # For each class A:
    #   - A window of class A is "confused" if A's classifier did not fire
    #   - "confused_with" = unrelated classes whose classifiers DID fire on A's windows
    #
    # This correctly handles hierarchy:
    #   Treadmill window → Walking fires = correct, not a confusion
    #   Treadmill window → Standing fires = confusion (unrelated)

    from cooccurrence import get_all_related

    confusion_scores = []
    for i, name in enumerate(class_names):
        # Windows that truly belong to class A
        a_mask    = (y_val == i)
        n_a       = a_mask.sum()
        if n_a == 0:
            confusion_scores.append({
                "class": name, "class_id": i, "f1": float(f1_scores[i]),
                "confusion_score": 0, "confused_as": []
            })
            continue

        a_scores  = scores[a_mask]   # (n_a, n_classes)

        # How many of A's windows did NOT have A's classifier fire?
        a_not_fired = (a_scores[:, i] <= threshold).sum()

        # For windows where A did not fire, which UNRELATED classes fired?
        related_classes = get_all_related(name)  # includes self + ancestors
        related_idx     = {j for j, cn in enumerate(class_names)
                           if cn in related_classes}

        confused_with = {}
        for idx in np.where(a_mask)[0]:
            if scores[idx, i] > threshold:
                continue  # A fired correctly, skip
            # Find unrelated classes that fired
            for j in range(n_classes):
                if j == i or j in related_idx:
                    continue
                if scores[idx, j] > threshold:
                    confused_with[class_names[j]] =                         confused_with.get(class_names[j], 0) + 1

        # Sort by count
        confused_as = sorted(confused_with.items(), key=lambda x: -x[1])[:5]

        confusion_scores.append({
            "class"          : name,
            "class_id"       : i,
            "f1"             : float(f1_scores[i]),
            "confusion_score": int(a_not_fired),
            "confused_as"    : confused_as
        })

    # Sort by F1 ascending — worst performing classes first
    confusion_scores.sort(key=lambda x: x["f1"])

    # Apply top_k if specified
    if top_k is not None:
        confusion_scores = confusion_scores[:top_k]

    confused_classes = [item["class"] for item in confusion_scores]

    # Print table
    print(f"\n{'Class':50s} {'F1':>6}  {'Confusion':>9}  {'Confused with'}")
    print("─" * 90)
    for item in confusion_scores:
        pairs_str = ", ".join(f"{c}({n})" for c, n in item["confused_as"][:3])
        print(f"{item['class']:50s} {item['f1']:6.3f}  {item['confusion_score']:9d}  "
              f"{pairs_str}")

    print(f"\nAll {len(confused_classes)} classes ranked by F1 (ascending)")
    return confusion_scores, confused_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          type=str, required=True)
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--device",          type=str, default="cuda")
    parser.add_argument("--top-k",           type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    detect_confusion(config, args.base_checkpoint, device, args.top_k)


if __name__ == "__main__":
    main()