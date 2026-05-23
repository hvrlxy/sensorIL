"""
estimate_benefit.py

Step 3: Estimate which confused classes benefit from the new sensor.

Benefit score combines two signals:

  1. Confusion signal:
     benefit_confusion(A) = 1 - F1(A)
     How much the base model struggles with class A.

  2. New sensor discriminability:
     Encode ONLY the new sensor stream through SimCLR encoder.
     For class A and its confused neighbors B, measure:
     dist(mean(e_new_A), mean(e_new_B))
     Uses pseudo-labels from base classifier on FL data.

  Combined:
     benefit(A) = confusion(A) * mean_discriminability(A, confused_neighbors)

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


def estimate_benefit(config, base_ckpt_path, confusion_scores, device):
    print(f"\n{'='*60}")
    print(f"Step 3: Estimating benefit of new sensor")
    print(f"{'='*60}")

    known_sensors = config["sensors"]["known_sensors"]
    new_sensor    = config["sensors"]["new_sensor"]
    n_known       = len(known_sensors)
    input_dim     = n_known * ENCODER_DIM

    # confusion_scores is a list of dicts with 'class', 'f1', 'confused_as'
    # Build lookup
    confusion_lookup = {item["class"]: item for item in confusion_scores}
    confused_classes = [item["class"] for item in confusion_scores]

    encoder = load_simclr_encoder(config["model"]["encoder_path"], device)

    # Load base classifiers for pseudo-labeling FL data
    ckpt        = torch.load(base_ckpt_path, map_location=device)
    class_names = ckpt["class_names"]

    classifiers = {}
    for name in confused_classes:
        if name in class_names:
            idim = ckpt.get("input_dims", {}).get(name, ckpt.get("input_dim", input_dim))
            clf  = BinaryClassifier(idim).to(device)
            clf.load_state_dict(ckpt["classifiers"][name])
            clf.eval()
            classifiers[name] = clf

    # Load FL data with NEW SENSOR ONLY for discriminability
    fl_ds_new = UnlabeledFLDataset(
        config["data"]["unlabeled_dir"], new_sensor
    )
    # Load FL data with KNOWN SENSORS for pseudo-labeling
    fl_ds_known = UnlabeledFLDataset(
        config["data"]["unlabeled_dir"], known_sensors
    )

    loader_new   = DataLoader(fl_ds_new,   batch_size=512, shuffle=False,
                              num_workers=4, pin_memory=True)
    loader_known = DataLoader(fl_ds_known, batch_size=512, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Encode FL data with known sensors (for pseudo-labeling)
    print("Encoding FL data with known sensors (for pseudo-labeling)...")
    z_known_all = []
    for x in loader_known:
        z = encode_sensors(encoder, x, device)
        z_known_all.append(z)
    z_known_all = torch.cat(z_known_all)   # (N, n_known * 96)

    # Encode FL data with new sensor ONLY (for discriminability)
    print("Encoding FL data with new sensor only (for discriminability)...")
    z_new_all = []
    for x in loader_new:
        z = encode_sensors(encoder, x, device)
        z_new_all.append(z)
    z_new_all = torch.cat(z_new_all)       # (N, 96)

    # Pseudo-label FL data using base classifiers
    print("Pseudo-labeling FL data...")
    threshold = config.get("active_learning", {}).get("pseudo_label_threshold", 0.7)
    scores    = {}
    batch_size = 512

    z_known_dev = z_known_all.to(device)
    bs_eval     = 4096
    with torch.no_grad():
        for name, clf in classifiers.items():
            class_scores = []
            for start in range(0, len(z_known_all), bs_eval):
                class_scores.append(
                    torch.sigmoid(clf(z_known_dev[start:start+bs_eval])).cpu()
                )
            scores[name] = torch.cat(class_scores).numpy()

    # Get high-confidence pseudo-labeled new sensor embeddings per class
    class_new_embs = {}
    for name in confused_classes:
        if name not in scores:
            continue
        mask = scores[name] > threshold
        if mask.sum() < 10:
            print(f"  {name}: only {mask.sum()} confident FL windows, skipping")
            continue
        class_new_embs[name] = z_new_all[mask]   # (K, 96)
        print(f"  {name}: {mask.sum()} confident FL windows")

    # Compute benefit score for each confused class
    benefit_scores = {}

    for item in confusion_scores:
        class_a = item["class"]
        f1_a    = item["f1"]

        if class_a not in class_new_embs:
            benefit_scores[class_a] = 0.0
            continue

        # Confusion signal: 1 - F1
        confusion_signal = 1.0 - f1_a

        # New sensor discriminability: distance to confused neighbors
        confused_neighbors = [pair[0] for pair in item["confused_as"]]

        discriminability = []
        for class_b in confused_neighbors:
            if class_b not in class_new_embs:
                continue

            mean_a = F.normalize(
                class_new_embs[class_a].mean(0), dim=-1
            )
            mean_b = F.normalize(
                class_new_embs[class_b].mean(0), dim=-1
            )

            # Cosine distance between new-sensor-only embeddings
            dist = 1.0 - (mean_a * mean_b).sum().item()
            discriminability.append(dist)

        if len(discriminability) == 0:
            benefit_scores[class_a] = 0.0
            continue

        avg_discriminability  = np.mean(discriminability)

        # Combined benefit score
        benefit_scores[class_a] = confusion_signal * avg_discriminability

    # Sort by benefit score
    benefit_ranked = sorted(
        benefit_scores.items(), key=lambda x: x[1], reverse=True
    )

    print(f"\n{'Class':45s} {'1-F1':>8} {'Discrim':>10} {'Benefit':>10}")
    print("─" * 76)
    for name, score in benefit_ranked:
        f1       = confusion_lookup[name]["f1"] if name in confusion_lookup else 0
        conf_sig = 1 - f1
        discrim  = score / conf_sig if conf_sig > 0 else 0
        print(f"{name:45s} {conf_sig:8.3f} {discrim:10.4f} {score:10.4f}")

    # Select classes with positive benefit
    threshold_benefit = config.get("active_learning", {}).get("benefit_threshold", 0.0)
    target_classes    = [name for name, score in benefit_ranked
                         if score > threshold_benefit]

    print(f"\nClasses selected for re-annotation ({len(target_classes)}): "
          f"{target_classes}")

    return benefit_ranked, target_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",           type=str, required=True)
    parser.add_argument("--base-checkpoint",  type=str, required=True)
    parser.add_argument("--device",           type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load confusion scores from base checkpoint
    ckpt             = torch.load(args.base_checkpoint, map_location=device)
    from detect_confusion import detect_confusion
    confusion_scores, confused_classes = detect_confusion(
        config, args.base_checkpoint, device
    )
    estimate_benefit(config, args.base_checkpoint, confusion_scores, device)


if __name__ == "__main__":
    main()