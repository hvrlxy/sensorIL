"""
estimate_benefit.py

Step 3: Estimate which confused classes benefit from the new sensor.

Benefit score (additive):
  benefit(A) = (1 - F1(A)) + discriminability(A)

Discriminability uses a confidence-weighted blend:
  α = min(1, n_pseudo / min_samples)
  discriminability(A) = α × direct + (1-α) × opposition

Selection: elbow detection on sorted benefit scores.

Usage:
    python scripts/estimate_benefit.py --config configs/pipeline_config.json \
                                        --base-checkpoint checkpoints/base_classifiers.pt
"""

import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from simclr_encoder import load_simclr_encoder, encode_sensors, ENCODER_DIM
from dataset import UnlabeledFLDataset
from train_base import BinaryClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Elbow detection
# ─────────────────────────────────────────────────────────────────────────────

def find_elbow(scores):
    n = len(scores)
    if n <= 2:
        return n - 1
    y = np.array(scores, dtype=float)
    x = np.arange(n, dtype=float)
    x_norm = x / (n - 1)
    y_range = y[0] - y[-1]
    y_norm  = (y - y[-1]) / (y_range + 1e-10)
    p1 = np.array([x_norm[0],  y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line = p2 - p1
    dists = []
    for i in range(n):
        p    = np.array([x_norm[i], y_norm[i]])
        proj = p1 + np.dot(p - p1, line) / np.dot(line, line) * line
        dists.append(np.linalg.norm(p - proj))
    return int(np.argmax(dists))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def estimate_benefit(config, base_ckpt_path, confusion_scores, device,
                     min_pseudo_samples=10):
    print(f"\n{'='*60}")
    print(f"Step 3: Estimating benefit of new sensor")
    print(f"{'='*60}")

    known_sensors = config["sensors"]["known_sensors"]
    new_sensor    = config["sensors"]["new_sensor"]
    n_known       = len(known_sensors)
    input_dim     = n_known * ENCODER_DIM

    confusion_lookup = {item["class"]: item for item in confusion_scores}
    confused_classes = [item["class"] for item in confusion_scores]

    encoder = load_simclr_encoder(config["model"]["encoder_path"], device)

    # Load ALL classifiers for pseudo-labeling
    ckpt        = torch.load(base_ckpt_path, map_location=device)
    class_names = ckpt["class_names"]
    classifiers = {}
    for name in class_names:
        idim = ckpt.get("input_dims", {}).get(name, ckpt.get("input_dim", input_dim))
        clf  = BinaryClassifier(idim).to(device)
        clf.load_state_dict(ckpt["classifiers"][name])
        clf.eval()
        classifiers[name] = clf

    # Encode FL data
    print("Encoding FL data with known sensors (pseudo-labeling)...")
    fl_known = UnlabeledFLDataset(config["data"]["unlabeled_dir"], known_sensors)
    all_z = []
    for x in DataLoader(fl_known, batch_size=4096, shuffle=False,
                        num_workers=4, pin_memory=True):
        all_z.append(encode_sensors(encoder, x, device).cpu())
    z_fl_known = torch.cat(all_z)

    print("Encoding FL data with new sensor only (discriminability)...")
    fl_new  = UnlabeledFLDataset(config["data"]["unlabeled_dir"], new_sensor)
    all_z2  = []
    for x in DataLoader(fl_new, batch_size=4096, shuffle=False,
                        num_workers=4, pin_memory=True):
        all_z2.append(encode_sensors(encoder, x, device).cpu())
    z_fl_new = torch.cat(all_z2)

    # Pseudo-label FL data using all classifiers
    print("Pseudo-labeling FL data...")
    threshold = config.get("active_learning", {}).get("pseudo_label_threshold", 0.7)
    scores    = {}
    z_fl_dev  = z_fl_known.to(device)
    with torch.no_grad():
        for name, clf in classifiers.items():
            s = []
            for i in range(0, len(z_fl_known), 4096):
                s.append(torch.sigmoid(clf(z_fl_dev[i:i+4096])).cpu())
            scores[name] = torch.cat(s).numpy()

    # Per-class mean new-sensor embedding
    class_new_embs = {}
    class_n_pseudo = {}
    for name in class_names:
        if name not in scores:
            class_n_pseudo[name] = 0
            continue
        mask = scores[name] > threshold
        n_ps = int(mask.sum())
        class_n_pseudo[name] = n_ps
        class_new_embs[name] = z_fl_new[mask] if n_ps >= min_pseudo_samples else None
        if n_ps > 0:
            print(f"  {name:45s} pseudo_labeled={n_ps}")

    # Compute benefit scores
    benefit_scores = {}

    for item in confusion_scores:
        class_a   = item["class"]
        f1_a      = item["f1"]
        conf_sig  = 1.0 - f1_a
        neighbors = [pair[0] for pair in item["confused_as"]]

        if not neighbors:
            benefit_scores[class_a] = conf_sig  # confusion signal only
            continue

        # Direct discriminability
        direct_dists = []
        if class_new_embs.get(class_a) is not None:
            mean_a = F.normalize(class_new_embs[class_a].mean(0), dim=-1)
            for class_b in neighbors:
                if class_new_embs.get(class_b) is not None:
                    mean_b = F.normalize(class_new_embs[class_b].mean(0), dim=-1)
                    direct_dists.append(1.0 - (mean_a * mean_b).sum().item())
        direct = float(np.mean(direct_dists)) if direct_dists else 0.0

        # Opposition discriminability
        opp_dists = []
        neighbor_embs = [
            (b, F.normalize(class_new_embs[b].mean(0), dim=-1))
            for b in neighbors if class_new_embs.get(b) is not None
        ]
        for i in range(len(neighbor_embs)):
            for j in range(i + 1, len(neighbor_embs)):
                _, e_i = neighbor_embs[i]
                _, e_j = neighbor_embs[j]
                opp_dists.append(1.0 - (e_i * e_j).sum().item())
        opposition = float(np.mean(opp_dists)) if opp_dists else 0.0

        # Confidence-weighted blend
        alpha = min(1.0, class_n_pseudo.get(class_a, 0) / min_pseudo_samples)
        discriminability = alpha * direct + (1.0 - alpha) * opposition

        # Additive formula
        benefit_scores[class_a] = conf_sig + discriminability

    # Sort by benefit score descending
    benefit_ranked = sorted(
        benefit_scores.items(), key=lambda x: x[1], reverse=True
    )

    # Print table
    print(f"\n{'Class':45s} {'1-F1':>8} {'Discrim':>10} {'Benefit':>10}")
    print("─" * 76)
    for name, score in benefit_ranked:
        item     = confusion_lookup.get(name, {})
        f1       = item.get("f1", 0)
        conf_sig = 1 - f1
        neighbors = [pair[0] for pair in item.get("confused_as", [])]
        discrim = 0.0
        if class_new_embs.get(name) is not None and neighbors:
            mean_a = F.normalize(class_new_embs[name].mean(0), dim=-1)
            dd = []
            for b in neighbors:
                if class_new_embs.get(b) is not None:
                    mean_b = F.normalize(class_new_embs[b].mean(0), dim=-1)
                    dd.append(1.0 - (mean_a * mean_b).sum().item())
            discrim = float(np.mean(dd)) if dd else 0.0
        print(f"{name:45s} {conf_sig:8.3f} {discrim:10.4f} {score:10.4f}")

    # Elbow detection on positive-score classes
    pos_scores = [(n, s) for n, s in benefit_ranked if s > 0]
    pos_values = [s for _, s in pos_scores]

    if len(pos_values) >= 2:
        elbow_idx      = find_elbow(pos_values)
        target_classes = [n for n, _ in pos_scores[:elbow_idx + 1]]
    elif len(pos_values) == 1:
        target_classes = [pos_scores[0][0]]
    else:
        target_classes = []

    elbow_score = pos_values[len(target_classes)-1] if target_classes else 0.0
    print(f"\nElbow at rank {len(target_classes)} (score={elbow_score:.4f})")
    print(f"Classes selected for re-annotation ({len(target_classes)}): "
          f"{target_classes}")

    return benefit_ranked, target_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          type=str, required=True)
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--device",          type=str, default="cuda")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    from detect_confusion import detect_confusion
    confusion_scores, _ = detect_confusion(config, args.base_checkpoint, device)
    estimate_benefit(config, args.base_checkpoint, confusion_scores, device)

if __name__ == "__main__":
    main()