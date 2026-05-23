"""
compute_shift.py

Computes the mean embedding shift from masked to full view
using unlabeled FL data. This shift is used during fine-tuning
to augment masked embeddings to cover the full embedding distribution.

shift = mean(z_full - z_masked)  over FL windows

Saves shift vector to checkpoints/embedding_shift_{experiment}.pt

Usage:
    python scripts/compute_shift.py --config configs/byol_config.json \
                                     --experiment 2to1 \
                                     --checkpoint checkpoints/pretrain_byol_2to1_best.pt
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from scripts.misc.stream_encoder import build_encoder
from scripts.misc.dataset import UnlabeledFLDataset


def compute_shift(config, experiment, checkpoint_path, device):
    print(f"\nComputing embedding shift | experiment={experiment}")

    exp     = config["sensors"]["experiments"][experiment]
    known   = exp["known_sensors"]
    new     = exp["new_sensor"]
    n_known = len(known)

    # Load encoder
    encoder = build_encoder(config).to(device)
    ckpt    = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    # Load FL data
    dataset = UnlabeledFLDataset(config["data"]["unlabeled_dir"], known, new)
    loader  = DataLoader(dataset, batch_size=512, shuffle=False,
                         num_workers=4, pin_memory=True)

    shifts = []
    print(f"Computing shift over {len(dataset)} windows...")

    with torch.no_grad():
        for i, (view_masked, view_full) in enumerate(loader):
            view_masked = view_masked.to(device)
            view_full   = view_full.to(device)

            z_masked = F.normalize(encoder(view_masked, mask_indices=[n_known]), dim=-1)
            z_full   = F.normalize(encoder(view_full,   mask_indices=None),      dim=-1)

            shift = z_full - z_masked   # (B, emb_dim)
            shifts.append(shift.cpu())

            if (i + 1) % 50 == 0:
                print(f"  Processed {(i+1) * 512:,} windows...")

    shifts     = torch.cat(shifts, dim=0)   # (N, emb_dim)
    mean_shift = shifts.mean(dim=0)         # (emb_dim,)
    std_shift  = shifts.std(dim=0)          # (emb_dim,)

    cos_sim = F.cosine_similarity(
        F.normalize(mean_shift.unsqueeze(0), dim=-1),
        torch.zeros(1, mean_shift.shape[0])  + 1e-8
    )

    print(f"\nMean shift norm:  {mean_shift.norm():.4f}")
    print(f"Std shift norm:   {std_shift.norm():.4f}")
    print(f"Mean shift range: [{mean_shift.min():.4f}, {mean_shift.max():.4f}]")

    # Save
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
    save_path = os.path.join(
        config["output"]["checkpoint_dir"],
        f"embedding_shift_{experiment}.pt"
    )
    torch.save({
        "mean_shift" : mean_shift,
        "std_shift"  : std_shift,
        "experiment" : experiment,
        "config"     : config
    }, save_path)
    print(f"Saved shift to {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["2to1", "1to1"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device",     type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    compute_shift(config, args.experiment, args.checkpoint, device)


if __name__ == "__main__":
    main()
