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
from train_base import ParallelBinaryClassifiers

from simclr_encoder import load_simclr_encoder, encode_sensors, ENCODER_DIM
from dataset import SensorDataset, UnlabeledFLDataset
from train_base import BinaryClassifier, FocalLoss
from cooccurrence import get_multilabel, are_related, get_ancestors, get_all_related


# ─────────────────────────────────────────────────────────────────────────────
# FL pseudo-label index
# ─────────────────────────────────────────────────────────────────────────────

def build_fl_pseudolabel_index(encoder, config, base_ckpt, device,
                                batch_size=512):
    """
    Encode FL data and pseudo-label each window using the base classifiers.
    Returns:
      z_fl_known:      (N, known_dim) embeddings with known sensors
      z_fl_full:       (N, full_dim)  embeddings with all sensors
      pseudo_labels:   (N,) long tensor — argmax class assignment
      all_scores:      (N, n_classes) float — per-class probabilities
    """
    known_sensors = config["sensors"]["known_sensors"]
    new_sensor    = config["sensors"]["new_sensor"]
    all_sensors   = known_sensors + new_sensor
    input_dim     = len(known_sensors) * ENCODER_DIM

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

    # Pseudo-label using base classifiers — vectorized over all classes at once
    print("  Pseudo-labeling FL data with base classifiers (vectorized)...")
    ckpt        = torch.load(base_ckpt, map_location=device)
    class_names = ckpt["class_names"]
    n_classes   = len(class_names)
    input_dim_ckpt = ckpt.get("input_dim", input_dim)

    # Stack all classifier weights into a single batched linear op
    # Shape: (n_classes, hidden) then (hidden, 1) → vectorized over classes
    # Use ParallelBinaryClassifiers if available, else fallback to sequential
    z_dev      = z_fl_known.to(device)
    bs         = 4096
    all_scores_list = []

    with torch.no_grad():
        # Load all classifiers at once
        clfs = []
        for name in class_names:
            idim = ckpt.get("input_dims", {}).get(name, input_dim_ckpt)
            clf  = BinaryClassifier(idim).to(device)
            clf.load_state_dict(ckpt["classifiers"][name])
            clf.eval()
            clfs.append(clf)

        # Batched scoring: process all classifiers per chunk
        for start in range(0, len(z_fl_known), bs):
            chunk  = z_dev[start:start+bs]
            scores_chunk = torch.stack(
                [torch.sigmoid(clf(chunk)) for clf in clfs], dim=1
            )  # (chunk_size, n_classes)
            all_scores_list.append(scores_chunk.cpu())

    all_scores = torch.cat(all_scores_list).numpy()  # (N, n_classes)

    pseudo_labels = torch.tensor(all_scores.argmax(axis=1), dtype=torch.long)

    counts = [(pseudo_labels == k).sum().item() for k in range(n_classes)]
    nonzero = sum(1 for c in counts if c > 0)
    print(f"  Pseudo-labeled: {nonzero}/{n_classes} classes present in FL data")
    print(f"  Top-5 pseudo-labeled classes:")
    top5 = sorted(enumerate(counts), key=lambda x: -x[1])[:5]
    for idx, cnt in top5:
        print(f"    {class_names[idx]:45s} {cnt}")

    return z_fl_known, z_fl_full, pseudo_labels, \
           torch.tensor(all_scores), class_names


def build_shared_negative_pool(z_fl_full, pseudo_labels, p_all_classes,
                                total_budget, filter_threshold=0.3):
    """
    Pre-compute a shared negative pool with fixed proportions per
    pseudo-label group. All target classes sample from this same pool,
    differing only in which groups are excluded for each class.
    This ensures consistent negative distribution across all target classes.

    Returns:
        pool_z:      (M, full_dim)
        pool_labels: (M,)           pseudo-label per window
        pool_weights:(M, n_classes) 1 - P(c | x) per window per class
    """
    # Keep only confidently pseudo-labeled windows
    max_conf = p_all_classes.max(dim=1).values
    mask     = max_conf > filter_threshold
    if mask.sum() < 1000:
        mask = torch.ones(len(pseudo_labels), dtype=torch.bool)

    pool_z       = z_fl_full[mask]
    pool_labels  = pseudo_labels[mask]
    pool_weights = 1.0 - p_all_classes[mask]  # (M, n_classes)

    # Subsample proportionally to group size
    N              = len(pool_z)
    unique, counts = pool_labels.unique(return_counts=True)
    sampled_idx    = []

    for k, cnt in zip(unique.tolist(), counts.tolist()):
        group_idx = torch.where(pool_labels == k)[0]
        budget    = max(1, round(total_budget * cnt / N))
        chosen    = group_idx[torch.randperm(len(group_idx))[:budget]]
        sampled_idx.append(chosen)

    idx = torch.cat(sampled_idx)
    return pool_z[idx], pool_labels[idx], pool_weights[idx]


def sample_diverse_negatives(pool_z, pool_labels, pool_weights,
                              class_idx, related_class_ids=None):
    """
    For a specific target class, filter the shared pool to exclude
    windows pseudo-labeled as that class or any related class.
    PU weight = 1 - P(class_a | x) from pool_weights[:, class_idx].
    """
    exclude = {class_idx}
    if related_class_ids:
        exclude.update(related_class_ids)

    keep = torch.ones(len(pool_labels), dtype=torch.bool)
    for rid in exclude:
        keep &= (pool_labels != rid)

    if keep.sum() == 0:
        return torch.zeros(0, pool_z.shape[1]), torch.zeros(0)

    return pool_z[keep], pool_weights[keep, class_idx]



def build_fl_index(config, base_ckpt_path, encoder, device):
    """Build FL pseudo-label index and shared negative pool once,
    to be reused across multiple budget runs."""
    from train_base import BinaryClassifier
    import torch

    known_sensors = config["sensors"]["known_sensors"]
    n_known       = len(known_sensors)
    input_dim     = n_known * ENCODER_DIM

    ckpt        = torch.load(base_ckpt_path, map_location=device)
    class_names = ckpt["class_names"]
    n_classes   = len(class_names)
    few_shot    = config["finetune"]["few_shot_samples_per_class"]

    z_fl_known, z_fl_full, pseudo_labels, all_fl_scores, fl_class_names = \
        build_fl_pseudolabel_index(encoder, config, base_ckpt_path, device)

    neg_budget_pool = (n_classes - 1) * few_shot
    pool_z, pool_labels, pool_weights = build_shared_negative_pool(
        z_fl_full, pseudo_labels, all_fl_scores,
        total_budget=neg_budget_pool, filter_threshold=0.3,
    )
    print(f"  Shared pool size: {len(pool_z)}")
    return (z_fl_known, z_fl_full, pseudo_labels, all_fl_scores,
            fl_class_names, pool_z, pool_labels, pool_weights)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def incremental_ft(config, base_ckpt_path, target_classes, device,
                   verbose=True, fl_index=None):
    """
    fl_index: optional pre-built tuple of
              (z_fl_known, z_fl_full, pseudo_labels, all_fl_scores, fl_class_names,
               pool_z, pool_labels, pool_weights)
              Pass this to avoid recomputing the FL index for each budget.
    """
    print(f"\n{'='*60}")
    print(f"Step 5: Incremental fine-tuning")
    print(f"{'='*60}")
    print(f"Target classes ({len(target_classes)}): {target_classes}")

    known_sensors = config["sensors"]["known_sensors"]
    new_sensor    = config["sensors"]["new_sensor"]
    all_sensors   = known_sensors + new_sensor
    n_known       = len(known_sensors)

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

    # Build FL pseudo-label index — reuse if pre-built
    if fl_index is not None:
        z_fl_known, z_fl_full, pseudo_labels, all_fl_scores, fl_class_names, \
            pool_z, pool_labels, pool_weights = fl_index
        print("  [cached] FL pseudo-label index")
    else:
        print("\nBuilding FL pseudo-label index...")
        z_fl_known, z_fl_full, pseudo_labels, all_fl_scores, fl_class_names = \
            build_fl_pseudolabel_index(encoder, config, base_ckpt_path, device)

        few_shot_tmp    = config["finetune"]["few_shot_samples_per_class"]
        neg_budget_pool = (len(class_names) - 1) * few_shot_tmp
        print("\nBuilding shared negative pool...")
        pool_z, pool_labels, pool_weights = build_shared_negative_pool(
            z_fl_full, pseudo_labels, all_fl_scores,
            total_budget     = neg_budget_pool,
            filter_threshold = 0.3,
        )
        print(f"  Shared pool size: {len(pool_z)}")

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

        # Uncertain negatives: from shared pool, exclude target + related classes
        # All target classes see same activity distribution in their negatives
        related_ids      = {class_names.index(c) for c in get_all_related(class_name)
                            if c in class_names}
        z_neg_fl, weights_fl = sample_diverse_negatives(
            pool_z, pool_labels, pool_weights,
            class_idx         = global_id,
            related_class_ids = related_ids,
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