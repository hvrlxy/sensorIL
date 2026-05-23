"""
estimate_benefit.py

Step 3: Estimate which confused classes benefit from the new sensor.

Benefit score:
  benefit(A) = (1 - F1(A)) × discriminability(A)

Discriminability uses a confidence-weighted blend:
  - Direct:     dist(e_new_A, e_new_B) using FL pseudo-labels for A
  - Opposition: dist(e_new_B, e_new_C) for B,C in confused_as(A)
                (used when A has few/no FL pseudo-labels)

  α = min(1, n_pseudo / min_samples)
  discriminability(A) = α × direct + (1-α) × opposition

Selection: elbow detection on sorted benefit scores
  (point of maximum curvature = diminishing returns)

Usage:
    python scripts/estimate_benefit.py --config configs/pipeline_config.json \
                                        --base-checkpoint checkpoints/base_classifiers.pt \
                                        --confusion-scores confusion_scores.json
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
    """
    Find elbow point in a sorted (descending) score array.
    Uses maximum curvature (perpendicular distance from line
    connecting first and last point).

    Returns index of elbow — all classes at or above this index
    are selected.
    """
    n = len(scores)
    if n <= 2:
        return n - 1

    # Normalize to [0,1] for stable geometry
    y = np.array(scores, dtype=float)
    x = np.arange(n, dtype=float)

    x_norm = x / (n - 1)
    y_range = y[0] - y[-1]
    y_norm  = (y - y[-1]) / (y_range + 1e-10)

    # Line from first to last point
    p1 = np.array([x_norm[0],  y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line = p2 - p1

    # Perpendicular distance from each point to the line
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

    # Load base classifiers for pseudo-labeling
    ckpt        = torch.load(base_ckpt_path, map_location=device)
    class_names = ckpt["class_names"]

    # Load ALL classifiers for pseudo-labeling (not just confused ones)
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
    all_z    = []
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

    # Pseudo-label FL data
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

    # Per-class mean new-sensor embedding (from pseudo-labels)
    # Computed for ALL classes — needed for opposition-based discriminability
    class_new_embs   = {}
    class_n_pseudo   = {}

    for name in class_names:
        if name not in scores:
            class_n_pseudo[name] = 0
            continue
        mask = scores[name] > threshold
        n_ps = int(mask.sum())
        class_n_pseudo[name] = n_ps

        if n_ps >= min_pseudo_samples:
            class_new_embs[name] = z_fl_new[mask]
        else:
            class_new_embs[name] = None

        print(f"  {name:45s} pseudo_labeled={n_ps}")

    # Compute benefit scores
    benefit_scores = {}

    for item in confusion_scores:
        class_a   = item["class"]
        f1_a      = item["f1"]
        conf_sig  = 1.0 - f1_a
        neighbors = [pair[0] for pair in item["confused_as"]]

        if not neighbors:
            # No confusion pairs — only flag if literally never predicted correctly
            # AND has actual confusion evidence (confusion_score > 0)
            confusion_count = item.get("confusion_score", 0)
            if f1_a == 0.0 and confusion_count > 0:
                benefit_scores[class_a] = conf_sig
            else:
                benefit_scores[class_a] = 0.0
            continue

        # ── Direct discriminability ────────────────────────────────────
        direct_dists = []
        if class_new_embs.get(class_a) is not None:
            mean_a = F.normalize(
                class_new_embs[class_a].mean(0), dim=-1
            )
            for class_b in neighbors:
                if class_new_embs.get(class_b) is not None:
                    mean_b = F.normalize(
                        class_new_embs[class_b].mean(0), dim=-1
                    )
                    direct_dists.append(1.0 - (mean_a * mean_b).sum().item())

        direct = float(np.mean(direct_dists)) if direct_dists else 0.0

        # ── Opposition discriminability ────────────────────────────────
        # Measure distance between A's confusion targets in new-sensor space
        # If confusion targets are well-separated, new sensor discriminates
        # A's neighborhood even if A itself is absent from FL data
        opp_dists = []
        neighbor_embs = [
            (b, F.normalize(class_new_embs[b].mean(0), dim=-1))
            for b in neighbors
            if class_new_embs.get(b) is not None
        ]
        for i in range(len(neighbor_embs)):
            for j in range(i + 1, len(neighbor_embs)):
                _, e_i = neighbor_embs[i]
                _, e_j = neighbor_embs[j]
                opp_dists.append(1.0 - (e_i * e_j).sum().item())

        opposition = float(np.mean(opp_dists)) if opp_dists else 0.0

        # ── Confidence-weighted blend ──────────────────────────────────
        alpha = min(1.0, class_n_pseudo.get(class_a, 0) / min_pseudo_samples)
        discriminability = alpha * direct + (1.0 - alpha) * opposition

        # Special case: if base model has very low F1 and discriminability
        # cannot be estimated (no FL pseudo-labels, no confusion pairs),
        # use confusion signal alone. This handles classes where the new sensor
        # covers a completely different body region — discriminability can't be
        # estimated from FL data but the gain can still be large.
        # benefit = confusion_signal + discriminability
        # Discriminability ADDS to score rather than scaling it
        # → classes with both signals rank above classes with only one
        # Special case: F1=0 AND confused but no discriminability signal
        # → use confusion signal alone (new sensor covers different body region)
        confusion_count = item.get("confusion_score", 0)
        if f1_a == 0.0 and confusion_count > 0 and discriminability == 0.0:
            benefit_scores[class_a] = conf_sig          # confusion signal only
        else:
            benefit_scores[class_a] = conf_sig + discriminability

    # Sort by benefit score
    benefit_ranked = sorted(
        benefit_scores.items(), key=lambda x: x[1], reverse=True
    )

    # Print table
    print(f"\n{'Class':45s} {'1-F1':>8} {'Direct':>8} {'Oppos':>8} "
          f"{'α':>6} {'Discrim':>8} {'Benefit':>8}  {'Formula'}")
    print("─" * 110)

    for name, score in benefit_ranked:
        item     = confusion_lookup.get(name, {})
        f1       = item.get("f1", 0)
        conf_sig = 1 - f1
        n_ps     = class_n_pseudo.get(name, 0)
        alpha    = min(1.0, n_ps / min_pseudo_samples)

        # Recompute for display
        neighbors = [pair[0] for pair in item.get("confused_as", [])]
        direct_d, opp_d = 0.0, 0.0

        if class_new_embs.get(name) is not None and neighbors:
            mean_a = F.normalize(class_new_embs[name].mean(0), dim=-1)
            dd = []
            for b in neighbors:
                if class_new_embs.get(b) is not None:
                    mean_b = F.normalize(class_new_embs[b].mean(0), dim=-1)
                    dd.append(1.0 - (mean_a * mean_b).sum().item())
            direct_d = float(np.mean(dd)) if dd else 0.0

        neighbor_embs = [
            F.normalize(class_new_embs[b].mean(0), dim=-1)
            for b in neighbors if class_new_embs.get(b) is not None
        ]
        od = []
        for i in range(len(neighbor_embs)):
            for j in range(i+1, len(neighbor_embs)):
                od.append(1.0 - (neighbor_embs[i] * neighbor_embs[j]).sum().item())
        opp_d = float(np.mean(od)) if od else 0.0

        discrim = alpha * direct_d + (1 - alpha) * opp_d
        if score == conf_sig and discrim == 0.0:
            formula = "conf only"
        elif discrim == 0.0:
            formula = "conf only"
        else:
            formula = "conf+discrim"
        print(f"{name:45s} {conf_sig:8.3f} {direct_d:8.4f} {opp_d:8.4f} "
              f"{alpha:6.2f} {discrim:8.4f} {score:8.4f}  {formula}")

    # Select all classes with positive benefit score
    target_classes = [n for n, s in benefit_ranked if s > 0]
    print(f"\nClasses selected for re-annotation ({len(target_classes)}): "
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