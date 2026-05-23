"""
build_index.py

Builds a faiss nearest neighbor index over FL unlabeled data
using n-sensor embeddings. Used for cross-dataset alignment
during pretraining.

For each lab window, we find the k nearest FL windows in
n-sensor embedding space — these are likely the same activity
and serve as cross-dataset positives for SupCon.

Usage:
    Called internally by pretrain.py every N epochs.
"""

import numpy as np
import torch
import torch.nn.functional as F
import faiss
from torch.utils.data import DataLoader

from scripts.misc.dataset import UnlabeledFLDataset


def build_fl_index(encoder, config, experiment, device, batch_size=512):
    """
    Encodes all FL windows with n sensors only (masked view)
    and builds a faiss flat L2 index for nearest neighbor search.

    Also stores the full embeddings (n+1 sensors) for retrieval.

    Returns:
        index        : faiss.IndexFlatIP (inner product = cosine on normalized vecs)
        z_fl_full    : (N, emb_dim) np.array of full embeddings
        z_fl_masked  : (N, emb_dim) np.array of masked embeddings
    """
    exp     = config["sensors"]["experiments"][experiment]
    known   = exp["known_sensors"]
    new     = exp["new_sensor"]
    n_known = len(known)

    dataset = UnlabeledFLDataset(config["data"]["unlabeled_dir"], known, new)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, pin_memory=True, drop_last=False)

    encoder.eval()
    all_masked, all_full = [], []

    with torch.no_grad():
        for view_masked, view_full in loader:
            view_masked = view_masked.to(device)
            view_full   = view_full.to(device)

            z_m = F.normalize(
                encoder(view_masked, mask_indices=[n_known]), dim=-1
            ).cpu().numpy().astype(np.float32)

            z_f = F.normalize(
                encoder(view_full, mask_indices=None), dim=-1
            ).cpu().numpy().astype(np.float32)

            all_masked.append(z_m)
            all_full.append(z_f)

    z_fl_masked = np.concatenate(all_masked, axis=0)  # (N, emb_dim)
    z_fl_full   = np.concatenate(all_full,   axis=0)  # (N, emb_dim)

    # Build faiss index on masked embeddings (inner product on normalized = cosine)
    emb_dim = z_fl_masked.shape[1]
    index   = faiss.IndexFlatIP(emb_dim)
    faiss.normalize_L2(z_fl_masked)
    index.add(z_fl_masked)

    print(f"  [Index] Built faiss index: {index.ntotal:,} FL windows | dim={emb_dim}")

    return index, z_fl_full, z_fl_masked


def retrieve_cross_dataset_positives(z_lab_masked, index, z_fl_full,
                                     k=5, device='cpu'):
    """
    For each lab masked embedding, retrieve k nearest FL full embeddings.

    Args:
        z_lab_masked : (B, emb_dim) torch tensor, L2-normalized
        index        : faiss index over FL masked embeddings
        z_fl_full    : (N, emb_dim) np.array of FL full embeddings
        k            : number of neighbors

    Returns:
        z_positives  : (B*k, emb_dim) torch tensor of FL full embeddings
        labels_pos   : (B*k,) torch tensor mapping each positive to its lab sample
    """
    z_np = z_lab_masked.detach().cpu().numpy().astype(np.float32)
    faiss.normalize_L2(z_np)

    _, indices = index.search(z_np, k)   # (B, k)

    B = z_lab_masked.shape[0]
    z_positives = []
    labels_pos  = []

    for i in range(B):
        for j in range(k):
            idx = indices[i, j]
            z_positives.append(z_fl_full[idx])
            labels_pos.append(i)

    z_positives = torch.tensor(
        np.stack(z_positives), dtype=torch.float32
    ).to(device)
    labels_pos = torch.tensor(labels_pos, dtype=torch.long).to(device)

    return z_positives, labels_pos