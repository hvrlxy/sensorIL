"""
incremental_ft.py

Step 5: Incremental fine-tuning with multi-label encoding.

For target classes (with new n+1 sensor labels):
  - Retrain binary classifier with n+1 sensor embeddings
  - Multi-label positives: new labeled windows of class A AND ancestors
  - Certain negatives: new labeled windows of classes NOT related to A
  - Uncertain negatives: diverse FL pseudo-negatives (weighted by 1-P(A))
  - Hierarchy-aware: related classes (ancestors/descendants) are NOT negatives

For non-target classes: completely untouched.

Usage:
    python scripts/incremental_ft.py --config configs/pipeline_config.json \
                                      --base-checkpoint checkpoints/base_classifiers.pt \
                                      --target-classes "Walking,Standing_Still,..."
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans

from simclr_encoder import load_simclr_encoder, encode_sensors, ENCODER_DIM
from dataset import SensorDataset, UnlabeledFLDataset
from train_base import BinaryClassifier, FocalLoss
from cooccurrence import get_multilabel, are_related, get_ancestors, get_all_related


# ─────────────────────────────────────────────────────────────────────────────
# FL cluster index
# ─────────────────────────────────────────────────────────────────────────────

def build_fl_cluster_index(encoder, config, device,
                           n_clusters=40, batch_size=512):
    known_sensors = config["sensors"]["known_sensors"]
    new_sensor    = config["sensors"]["new_sensor"]
    all_sensors   = known_sensors + new_sensor

    print("  Encoding FL data with known sensors...")
    fl_known = UnlabeledFLDataset(config["data"]["unlabeled_dir"], known_sensors)
    all_z    = []
    for x in DataLoader(fl_known, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True):
        all_z.append(encode_sensors(encoder, x, device).cpu())
    z_fl_known = torch.cat(all_z)

    print("  Encoding FL data with all sensors...")
    fl_full  = UnlabeledFLDataset(config["data"]["unlabeled_dir"], all_sensors)
    all_z2   = []
    for x in DataLoader(fl_full, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True):
        all_z2.append(encode_sensors(encoder, x, device).cpu())
    z_fl_full = torch.cat(all_z2)

    print(f"  Clustering into {n_clusters} clusters...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                             batch_size=4096, n_init=3, max_iter=100)
    cluster_labels = torch.tensor(
        kmeans.fit_predict(z_fl_known.numpy()), dtype=torch.long
    )
    counts = [(cluster_labels == k).sum().item() for k in range(n_clusters)]
    print(f"  Cluster sizes: min={min(counts)} max={max(counts)} "
          f"mean={np.mean(counts):.0f}")

    return z_fl_known, z_fl_full, cluster_labels


def sample_diverse_negatives(z_fl_full, cluster_labels, p_class_a,
                              total_budget, filter_threshold=0.3,
                              n_clusters=40):
    mask = p_class_a < filter_threshold
    if mask.sum() < 10:
        mask = p_class_a < 0.5

    z_f      = z_fl_full[mask]
    labels_f = cluster_labels[mask]
    weights  = 1.0 - p_class_a[mask]

    N = len(z_f)
    if N == 0:
        return torch.zeros(0, z_fl_full.shape[1]), torch.zeros(0)

    sampled_z, sampled_w = [], []
    for k in range(n_clusters):
        cm = labels_f == k
        if cm.sum() == 0:
            continue
        budget = max(1, round(total_budget * cm.sum().item() / N))
        z_k    = z_f[cm]
        w_k    = weights[cm]
        idx    = torch.randperm(len(z_k))[:budget] if len(z_k) >= budget \
                 else torch.randint(0, len(z_k), (budget,))
        sampled_z.append(z_k[idx])
        sampled_w.append(w_k[idx])

    if not sampled_z:
        return torch.zeros(0, z_fl_full.shape[1]), torch.zeros(0)
    return torch.cat(sampled_z), torch.cat(sampled_w)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def incremental_ft(config, base_ckpt_path, target_classes, device, verbose=True):
    print(f"\n{'='*60}")
    print(f"Step 5: Incremental fine-tuning")
    print(f"{'='*60}")
    print(f"Target classes ({len(target_classes)}): {target_classes}")

    known_sensors = config["sensors"]["known_sensors"]
    new_sensor    = config["sensors"]["new_sensor"]
    all_sensors   = known_sensors + new_sensor
    n_known       = len(known_sensors)
    n_clusters    = config.get("active_learning", {}).get("n_clusters", 40)

    input_dim_known = n_known * ENCODER_DIM
    input_dim_full  = (n_known + 1) * ENCODER_DIM

    encoder = load_simclr_encoder(config["model"]["encoder_path"], device)

    # Load base classifiers — non-target ones stay completely unchanged
    ckpt        = torch.load(base_ckpt_path, map_location=device)
    class_names = ckpt["class_names"]
    n_classes   = ckpt["n_classes"]

    classifiers = {}
    input_dims  = {}
    for name in class_names:
        idim = ckpt.get("input_dims", {}).get(name, ckpt.get("input_dim", input_dim_known))
        clf  = BinaryClassifier(idim).to(device)
        clf.load_state_dict(ckpt["classifiers"][name])
        clf.eval()
        classifiers[name] = clf
        input_dims[name]  = idim

    # Build FL cluster index
    print("\nBuilding FL cluster index...")
    z_fl_known, z_fl_full, cluster_labels = build_fl_cluster_index(
        encoder, config, device, n_clusters=n_clusters
    )

    # Compute P(class A) on FL data for each target class (batched)
    print("\nComputing P(class A) for FL windows...")
    p_class    = {}
    bs         = 4096
    z_fl_dev   = z_fl_known.to(device)
    with torch.no_grad():
        for name in target_classes:
            if name not in classifiers:
                continue
            clf_scores = []
            for start in range(0, len(z_fl_known), bs):
                clf_scores.append(
                    torch.sigmoid(classifiers[name](z_fl_dev[start:start+bs])).cpu()
                )
            p_class[name] = torch.cat(clf_scores)

    # Encode new labeled data (n+1 sensors) for target classes
    print("\nEncoding new labeled data (n+1 sensors)...")
    new_ds = SensorDataset(
        data_dir              = config["data"]["labeled_dir"],
        sensors               = all_sensors,
        max_samples_per_class = config["finetune"]["few_shot_samples_per_class"],
        split                 = "train",
        val_split             = config["finetune"]["val_split"],
        include_classes       = target_classes
    )
    new_val_ds = SensorDataset(
        data_dir              = config["data"]["labeled_dir"],
        sensors               = all_sensors,
        max_samples_per_class = config["finetune"]["few_shot_samples_per_class"],
        split                 = "val",
        val_split             = config["finetune"]["val_split"],
        include_classes       = target_classes
    )

    def encode_ds(ds):
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        all_z, all_y = [], []
        for x, y in loader:
            all_z.append(encode_sensors(encoder, x, device))
            all_y.append(y)
        return torch.cat(all_z), torch.cat(all_y)

    z_new_train, y_new_local     = encode_ds(new_ds)
    z_new_val,   y_new_val_local = encode_ds(new_val_ds)

    new_class_names = new_ds.class_names
    new_to_global   = {i: class_names.index(name)
                       for i, name in enumerate(new_class_names)
                       if name in class_names}

    y_new_train = torch.tensor([new_to_global[y.item()] for y in y_new_local])
    y_new_val   = torch.tensor([new_to_global[y.item()] for y in y_new_val_local])

    # Build multi-label matrices (vectorized)
    ml_matrix      = torch.tensor(
        [[1 if c2 in get_all_related(c1) else 0
          for c2 in class_names] for c1 in class_names],
        dtype=torch.float32
    )  # (n_classes, n_classes)
    y_ml_new_train = ml_matrix[y_new_train]   # (N_train, n_classes)
    y_ml_new_val   = ml_matrix[y_new_val]     # (N_val,   n_classes)

    few_shot   = config["finetune"]["few_shot_samples_per_class"]
    neg_budget = (n_classes - 1) * few_shot
    epochs     = config["finetune"]["epochs"]
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

    # Retrain target classifiers only
    for class_name in target_classes:
        if class_name not in class_names:
            if verbose: print(f"  [skip] {class_name} not in base classes")
            continue

        global_id = class_names.index(class_name)
        if verbose: print(f"\n  Retraining: {class_name} (id={global_id})")

        # Multi-label binary targets for this class
        y_bin_train = y_ml_new_train[:, global_id].to(device)
        y_bin_val   = y_ml_new_val[:,   global_id].to(device)

        # Positives: windows where this class label = 1
        pos_mask = y_bin_train == 1
        z_pos    = z_new_train[pos_mask].to(device)
        if verbose: print(f"    Positives: {len(z_pos)}")

        # Certain negatives: other unrelated target classes (real n+1 sensor data)
        # Vectorized: class is negative if y_ml_new_train[:, global_id] == 0
        neg_certain_mask = (y_ml_new_train[:, global_id] == 0)
        z_neg_certain    = z_new_train[neg_certain_mask].to(device)
        if verbose: print(f"    Certain negatives (unrelated target classes): {len(z_neg_certain)}")

        # Uncertain negatives: diverse FL pseudo-negatives
        fl_neg_budget = max(0, neg_budget - len(z_neg_certain))
        z_neg_fl, weights_fl = sample_diverse_negatives(
            z_fl_full, cluster_labels,
            p_class[class_name],
            total_budget     = fl_neg_budget,
            filter_threshold = 0.3,
            n_clusters       = n_clusters
        )
        if verbose: print(f"    Uncertain negatives (FL diverse): {len(z_neg_fl)}")

        # Val data
        z_val_pos = z_new_val[y_bin_val == 1].to(device)
        z_val_neg = z_new_val[y_bin_val == 0].to(device)

        # Combine
        z_train = torch.cat([z_pos, z_neg_certain, z_neg_fl.to(device)])
        y_train_b = torch.cat([
            torch.ones(len(z_pos),          device=device),
            torch.zeros(len(z_neg_certain), device=device),
            torch.zeros(len(z_neg_fl),      device=device)
        ])
        w_train = torch.cat([
            torch.ones(len(z_pos),          device=device),
            torch.ones(len(z_neg_certain),  device=device),
            weights_fl.to(device)
        ])

        z_val   = torch.cat([z_val_pos, z_val_neg])
        y_val_b = torch.cat([
            torch.ones(len(z_val_pos),  device=device),
            torch.zeros(len(z_val_neg), device=device)
        ])

        # Train new classifier with full input dim
        clf = BinaryClassifier(input_dim_full).to(device)
        opt = optim.AdamW(clf.parameters(),
                          lr=config["finetune"]["lr"],
                          weight_decay=config["finetune"]["weight_decay"])
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best_val_loss = float("inf")
        best_state    = None
        bs_ft         = config["finetune"]["batch_size"]

        for epoch in range(epochs):
            clf.train()
            perm   = torch.randperm(len(z_train))
            z_shuf = z_train[perm]
            y_shuf = y_train_b[perm]
            w_shuf = w_train[perm]

            for i in range(0, len(z_shuf), bs_ft):
                opt.zero_grad()
                loss = focal_loss(clf(z_shuf[i:i+bs_ft]),
                                  y_shuf[i:i+bs_ft],
                                  w_shuf[i:i+bs_ft])
                loss.backward()
                opt.step()
            sched.step()

            clf.eval()
            with torch.no_grad():
                val_loss = focal_loss(clf(z_val), y_val_b).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in clf.state_dict().items()}

        clf.load_state_dict(best_state)

        # Val F1
        clf.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(clf(z_val)) > 0.5).long().cpu().numpy()
        from sklearn.metrics import f1_score as _f1
        f1 = _f1(y_val_b.cpu().numpy().astype(int), preds,
                 average="binary", zero_division=0)
        if verbose: print(f"    val_f1={f1:.3f}")

        # Replace only this classifier — non-target classifiers untouched
        classifiers[class_name] = clf
        input_dims[class_name]  = input_dim_full

    # Save
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(
        config["output"]["checkpoint_dir"], "incremental_classifiers.pt"
    )
    torch.save({
        "classifiers"   : {k: v.state_dict() for k, v in classifiers.items()},
        "input_dims"    : input_dims,
        "class_names"   : class_names,
        "n_classes"     : n_classes,
        "known_sensors" : known_sensors,
        "new_sensor"    : new_sensor,
        "target_classes": list(set(target_classes)),  # deduplicate
        "config"        : config
    }, ckpt_path)
    print(f"\nSaved → {ckpt_path}")
    return ckpt_path, classifiers, input_dims, class_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          type=str, required=True)
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--target-classes",  type=str, required=True)
    parser.add_argument("--device",          type=str, default="cuda")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    device         = torch.device(args.device if torch.cuda.is_available() else "cpu")
    target_classes = args.target_classes.split(",")
    incremental_ft(config, args.base_checkpoint, target_classes, device)

if __name__ == "__main__":
    main()