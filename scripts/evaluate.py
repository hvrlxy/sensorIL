"""
evaluate.py

Three-way independent evaluation using per-class binary F1.

Each binary classifier is evaluated completely independently:
  - No competition between classifiers
  - Multi-label ground truth (Treadmill windows positive for Walking too)
  - F1 computed per class as binary classification problem
  - Changing one classifier cannot affect another class's F1

Usage:
    python scripts/evaluate.py --config configs/pipeline_config.json \
                                --base-checkpoint        checkpoints/base_classifiers.pt \
                                --incremental-checkpoint checkpoints/incremental_classifiers.pt \
                                --oracle-checkpoint      checkpoints/oracle_classifiers.pt
"""

import os
import json
import argparse
import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from simclr_encoder import load_simclr_encoder, encode_sensors, ENCODER_DIM
from dataset import SensorDataset
from train_base import BinaryClassifier
from cooccurrence import get_multilabel


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_classifiers(ckpt_path, device):
    ckpt        = torch.load(ckpt_path, map_location=device)
    class_names = ckpt["class_names"]
    classifiers = {}
    input_dims  = {}
    for name in class_names:
        idim = ckpt.get("input_dims", {}).get(name, ckpt.get("input_dim"))
        clf  = BinaryClassifier(idim).to(device)
        clf.load_state_dict(ckpt["classifiers"][name])
        clf.eval()
        classifiers[name] = clf
        input_dims[name]  = idim
    return classifiers, input_dims, class_names, ckpt


def get_scores(classifiers, input_dims, z_known, z_full, class_names, device,
               batch_size=4096):
    """
    Run all classifiers independently in batches.
    Returns (n_samples, n_classes) score matrix.
    Each classifier uses z_known or z_full based on its input_dim.
    """
    n         = len(z_known)
    scores    = torch.zeros(n, len(class_names))
    known_dim = z_known.shape[1]

    z_known_dev = z_known.to(device)
    z_full_dev  = z_full.to(device)

    with torch.no_grad():
        for i, name in enumerate(class_names):
            clf  = classifiers[name]
            idim = input_dims[name]
            z    = z_full_dev if idim > known_dim else z_known_dev

            # Process in batches to avoid OOM
            clf_scores = []
            for start in range(0, n, batch_size):
                zb = z[start:start+batch_size]
                clf_scores.append(torch.sigmoid(clf(zb)).cpu())
            scores[:, i] = torch.cat(clf_scores)

    return scores.numpy()


def encode_split(encoder, config, sensors, device):
    ds = SensorDataset(
        data_dir              = config["data"]["labeled_dir"],
        sensors               = sensors,
        max_samples_per_class = -1,
        split                 = "all",
        val_split             = 0.0
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
    all_z, all_y = [], []
    for x, y in loader:
        all_z.append(encode_sensors(encoder, x, device))
        all_y.append(y)
    return torch.cat(all_z), torch.cat(all_y).numpy(), ds.class_names


def compute_multilabel_f1(scores, y, class_names, threshold=0.5,
                           thresholds_dict=None):
    """
    Compute per-class binary F1 using multi-label ground truth.
    Each class evaluated independently.

    thresholds_dict: optional {class_name: threshold} for per-class thresholds
    """
    n_classes = len(class_names)

    # Build multi-label ground truth (vectorized)
    # Precompute multilabel matrix for all classes
    ml_matrix = np.array([get_multilabel(name, class_names)
                           for name in class_names], dtype=np.int32)  # (n_classes, n_classes)
    y_ml = ml_matrix[y.astype(int)]  # (n_samples, n_classes)

    # Binary predictions per class (with optional per-class thresholds)
    preds_bin = np.zeros_like(scores, dtype=int)
    for i, name in enumerate(class_names):
        thr = thresholds_dict[name] if thresholds_dict and name in thresholds_dict               else threshold
        preds_bin[:, i] = (scores[:, i] > thr).astype(int)

    # Per-class F1
    f1_arr = np.array([
        f1_score(y_ml[:, i], preds_bin[:, i], zero_division=0)
        for i in range(n_classes)
    ])

    macro_f1    = f1_arr.mean()
    support     = np.array([np.sum(y == i) for i in range(n_classes)])
    weighted_f1 = np.average(f1_arr, weights=support)

    return f1_arr, macro_f1, weighted_f1


# ─────────────────────────────────────────────────────────────────────────────
# Combined table
# ─────────────────────────────────────────────────────────────────────────────

def print_combined_table(class_names,
                          f1_b, f1_i, f1_o,
                          target_classes_set,
                          macro_f1_b, macro_f1_i, macro_f1_o,
                          weighted_f1_b, weighted_f1_i, weighted_f1_o,
                          n_new):
    col_w = 45
    print(f"\n{'='*100}")
    print(f"  Per-class F1 comparison  ({n_new} classes re-labeled for Proposed)")
    print(f"{'='*100}")
    print(f"  {'Activity':{col_w}} {'Baseline':>9} {'Proposed':>9} {'Oracle':>9} "
          f"{'Δ Prop':>8} {'Δ Orac':>8}  {'Target'}")
    print(f"  {'─'*col_w} {'─'*9} {'─'*9} {'─'*9} {'─'*8} {'─'*8}  {'─'*6}")

    for i, name in enumerate(class_names):
        marker  = "  ◀◀◀  " if name in target_classes_set else "       "
        delta_p = f1_i[i] - f1_b[i]
        delta_o = f1_o[i] - f1_b[i]
        print(f"  {name:{col_w}} {f1_b[i]:9.3f} {f1_i[i]:9.3f} {f1_o[i]:9.3f} "
              f"{delta_p:>+8.3f} {delta_o:>+8.3f}{marker}")

    print(f"  {'─'*col_w} {'─'*9} {'─'*9} {'─'*9} {'─'*8} {'─'*8}")
    print(f"  {'MACRO F1':{col_w}} {macro_f1_b:9.4f} {macro_f1_i:9.4f} {macro_f1_o:9.4f} "
          f"{macro_f1_i-macro_f1_b:>+8.4f} {macro_f1_o-macro_f1_b:>+8.4f}")
    print(f"  {'WEIGHTED F1':{col_w}} {weighted_f1_b:9.4f} {weighted_f1_i:9.4f} {weighted_f1_o:9.4f} "
          f"{weighted_f1_i-weighted_f1_b:>+8.4f} {weighted_f1_o-weighted_f1_b:>+8.4f}")
    print(f"{'='*100}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(config, base_ckpt, incremental_ckpt, oracle_ckpt, device):
    print(f"\n{'='*60}")
    print(f"Evaluation")
    print(f"{'='*60}")

    known_sensors = config["sensors"]["known_sensors"]
    new_sensor    = config["sensors"]["new_sensor"]
    all_sensors   = known_sensors + new_sensor

    encoder = load_simclr_encoder(config["model"]["encoder_path"], device)

    print("Encoding test data...")
    z_known, y, class_names = encode_split(encoder, config, known_sensors, device)
    z_full,  _, _           = encode_split(encoder, config, all_sensors,   device)

    results = {}

    # Scores from each condition
    clf_b, idim_b, _, _      = load_classifiers(base_ckpt,        device)
    clf_i, idim_i, _, ckpt_i = load_classifiers(incremental_ckpt, device)
    clf_o, idim_o, _, _      = load_classifiers(oracle_ckpt,       device)

    scores_b = get_scores(clf_b, idim_b, z_known, z_full, class_names, device)
    scores_i = get_scores(clf_i, idim_i, z_known, z_full, class_names, device)
    scores_o = get_scores(clf_o, idim_o, z_known, z_full, class_names, device)

    n_new              = len(ckpt_i.get("target_classes", []))
    target_classes_set = set(ckpt_i.get("target_classes", []))

    # Independent per-class binary F1 with multi-label ground truth
    # Load calibrated thresholds if available
    def load_thresholds(ckpt_p):
        thr_path = ckpt_p.replace(".pt", "_thresholds.pt")
        if os.path.exists(thr_path):
            print(f"  Loading calibrated thresholds: {thr_path}")
            return torch.load(thr_path, map_location="cpu")["thresholds"]
        return None

    thr_b = load_thresholds(base_ckpt)
    thr_i = load_thresholds(incremental_ckpt)
    thr_o = load_thresholds(oracle_ckpt)

    f1_b, macro_f1_b, weighted_f1_b = compute_multilabel_f1(
        scores_b, y, class_names, thresholds_dict=thr_b)
    f1_i, macro_f1_i, weighted_f1_i = compute_multilabel_f1(
        scores_i, y, class_names, thresholds_dict=thr_i)
    f1_o, macro_f1_o, weighted_f1_o = compute_multilabel_f1(
        scores_o, y, class_names, thresholds_dict=thr_o)

    results["baseline"] = {"macro_f1": macro_f1_b, "weighted_f1": weighted_f1_b}
    results["proposed"] = {"macro_f1": macro_f1_i, "weighted_f1": weighted_f1_i,
                            "n_new_labels": n_new}
    results["oracle"]   = {"macro_f1": macro_f1_o, "weighted_f1": weighted_f1_o}

    print_combined_table(
        class_names,
        f1_b, f1_i, f1_o,
        target_classes_set,
        macro_f1_b, macro_f1_i, macro_f1_o,
        weighted_f1_b, weighted_f1_i, weighted_f1_o,
        n_new
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",                  type=str, required=True)
    parser.add_argument("--base-checkpoint",         type=str, required=True)
    parser.add_argument("--incremental-checkpoint",  type=str, required=True)
    parser.add_argument("--oracle-checkpoint",       type=str, required=True)
    parser.add_argument("--device",                  type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    evaluate(config, args.base_checkpoint,
             args.incremental_checkpoint,
             args.oracle_checkpoint, device)


if __name__ == "__main__":
    main()