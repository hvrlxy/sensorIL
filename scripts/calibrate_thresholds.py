"""
calibrate_thresholds.py

Calibrate per-classifier thresholds on the validation set.
For each binary classifier, find the threshold that maximizes F1.

Usage:
    python scripts/calibrate_thresholds.py \
        --config configs/pipeline_config.json \
        --checkpoint checkpoints/incremental_classifiers.pt \
        --output    checkpoints/incremental_thresholds.pt
"""

import os
import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from simclr_encoder import load_simclr_encoder, encode_sensors, ENCODER_DIM
from dataset import SensorDataset
from train_base import BinaryClassifier
from cooccurrence import get_multilabel


def calibrate_thresholds(config, ckpt_path, device, output_path=None, verbose=True):
    if verbose:
        print(f"\n{'='*60}")
        print(f"Calibrating thresholds")
        print(f"{'='*60}")

    known_sensors = config["sensors"]["known_sensors"]
    new_sensor    = config["sensors"]["new_sensor"]
    all_sensors   = known_sensors + new_sensor

    encoder = load_simclr_encoder(config["model"]["encoder_path"], device)

    ckpt        = torch.load(ckpt_path, map_location=device)
    class_names = ckpt["class_names"]
    n_classes   = ckpt["n_classes"]
    input_dims  = ckpt.get("input_dims", {})

    # Load classifiers
    classifiers = {}
    for name in class_names:
        idim = input_dims.get(name, ckpt.get("input_dim"))
        clf  = BinaryClassifier(idim).to(device)
        clf.load_state_dict(ckpt["classifiers"][name])
        clf.eval()
        classifiers[name] = clf

    # Encode val data
    ds_known = SensorDataset(
        data_dir              = config["data"]["labeled_dir"],
        sensors               = known_sensors,
        max_samples_per_class = config["finetune"]["few_shot_samples_per_class"],
        split                 = "val",
        val_split             = config["finetune"]["val_split"]
    )
    ds_full = SensorDataset(
        data_dir              = config["data"]["labeled_dir"],
        sensors               = all_sensors,
        max_samples_per_class = config["finetune"]["few_shot_samples_per_class"],
        split                 = "val",
        val_split             = config["finetune"]["val_split"]
    )

    def encode_ds(ds):
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        all_z, all_y = [], []
        for x, y in loader:
            all_z.append(encode_sensors(encoder, x, device))
            all_y.append(y)
        return torch.cat(all_z), torch.cat(all_y).numpy()

    z_known, y = encode_ds(ds_known)
    z_full,  _ = encode_ds(ds_full)
    known_dim  = z_known.shape[1]

    # Build multi-label ground truth (vectorized)
    ml_matrix = np.array([get_multilabel(name, class_names)
                           for name in class_names], dtype=np.int32)
    y_ml = ml_matrix[y.astype(int)]

    # Get scores for all classifiers (batched)
    n      = len(y)
    scores = np.zeros((n, n_classes))
    z_known_dev = z_known.to(device)
    z_full_dev  = z_full.to(device)
    bs = 4096

    with torch.no_grad():
        for i, name in enumerate(class_names):
            idim = input_dims.get(name, ckpt.get("input_dim"))
            z    = z_full_dev if idim > known_dim else z_known_dev
            clf_scores = []
            for start in range(0, n, bs):
                clf_scores.append(torch.sigmoid(
                    classifiers[name](z[start:start+bs])
                ).cpu().numpy())
            scores[:, i] = np.concatenate(clf_scores)

    # Find optimal threshold per class
    thresholds   = {}
    candidates   = np.arange(0.2, 0.81, 0.05)

    if verbose:
        print(f"\n{'Class':45s} {'Default F1':>10} {'Best F1':>10} {'Threshold':>10}")
        print("─" * 80)

    for i, name in enumerate(class_names):
        y_true = y_ml[:, i]
        s      = scores[:, i]

        best_f1   = 0.0
        best_thr  = 0.5
        default_f1 = 0.0

        for thr in candidates:
            preds = (s > thr).astype(int)
            tp    = ((preds == 1) & (y_true == 1)).sum()
            fp    = ((preds == 1) & (y_true == 0)).sum()
            fn    = ((preds == 0) & (y_true == 1)).sum()
            prec  = tp / (tp + fp + 1e-8)
            rec   = tp / (tp + fn + 1e-8)
            f1    = 2 * prec * rec / (prec + rec + 1e-8)

            if abs(thr - 0.5) < 0.01:
                default_f1 = f1

            if f1 > best_f1:
                best_f1  = f1
                best_thr = float(np.clip(thr, 0.2, 0.8))

        thresholds[name] = float(best_thr)

        if verbose and abs(best_thr - 0.5) > 0.1:
            print(f"  {name:45s} {default_f1:10.3f} {best_f1:10.3f} {best_thr:10.2f}")

    if verbose:
        print(f"\nCalibrated {len(thresholds)} thresholds")
        print(f"Thresholds < 0.4: {sum(1 for v in thresholds.values() if v < 0.4)}")
        print(f"Thresholds > 0.6: {sum(1 for v in thresholds.values() if v > 0.6)}")

    if output_path is None:
        output_path = ckpt_path.replace(".pt", "_thresholds.pt")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        "thresholds" : thresholds,
        "class_names": class_names,
        "ckpt_path"  : ckpt_path
    }, output_path)
    if verbose:
        print(f"Saved thresholds → {output_path}")
    return thresholds, output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output",     type=str, default=None)
    parser.add_argument("--device",     type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    calibrate_thresholds(config, args.checkpoint, device, args.output)


if __name__ == "__main__":
    main()