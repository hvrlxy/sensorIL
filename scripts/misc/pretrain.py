"""
pretrain.py

Phase 1: Joint pretraining with four losses:

  L_BYOL        : asymmetric prediction, masked→full (direction alignment)
  L_regression  : cross-sensor regression, known→new sensor slot
  L_SupCon      : supervised contrastive on lab labeled data (discriminative)
  L_cross       : cross-dataset SupCon using nearest neighbor FL windows
                  pulls lab masked embeddings toward FL full embeddings
                  of the same activity (inferred via n-sensor similarity)

mode='byol':   all four losses
mode='baseline': L_BYOL (augmentation) + L_SupCon only

The cross-dataset loss is the key new addition — it explicitly bridges
the gap between masked (lab fine-tuning) and full (test time) embeddings
in a class-aware way, using unlabeled FL data.

Usage:
    python scripts/pretrain.py --config configs/byol_config.json \
                                --experiment 2to1 --mode byol
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

from scripts.misc.dataset import get_pretrain_loader, get_supcon_loader
from scripts.misc.stream_encoder import build_encoder
from scripts.misc.byol import build_byol, byol_loss, MLP
from scripts.misc.build_index import build_fl_index, retrieve_cross_dataset_positives


# ─────────────────────────────────────────────────────────────────────────────
# Supervised Contrastive Loss
# ─────────────────────────────────────────────────────────────────────────────

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device    = features.device
        B         = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=device)

        sim       = torch.matmul(features, features.T) / self.temperature
        mask_self = torch.eye(B, dtype=torch.bool, device=device)
        labels_2d = labels.unsqueeze(1)
        mask_pos  = (labels_2d == labels_2d.T) & ~mask_self

        if mask_pos.sum() == 0:
            return torch.tensor(0.0, device=device)

        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim        = sim - sim_max.detach()
        exp_sim    = torch.exp(sim) * ~mask_self
        log_prob   = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        n_pos      = mask_pos.sum(dim=1).float().clamp(min=1)
        loss       = -(log_prob * mask_pos).sum(dim=1) / n_pos
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Cross-dataset contrastive loss
# ─────────────────────────────────────────────────────────────────────────────

def cross_dataset_loss(z_lab, z_fl_pos, lab_labels, temperature=0.07):
    """
    Contrastive loss pulling lab masked embeddings toward FL full embeddings
    of nearest neighbors (proxy for same activity).

    For each lab sample i:
      - Positives: FL full embeddings of its k nearest neighbors
      - Negatives: all other lab samples + their FL neighbors

    Args:
        z_lab      : (B, emb_dim) lab masked embeddings, L2-normalized
        z_fl_pos   : (B*k, emb_dim) FL full embeddings of neighbors
        lab_labels : (B*k,) maps each FL positive to its lab sample index
        temperature: softmax temperature
    """
    device = z_lab.device
    B      = z_lab.shape[0]
    Bk     = z_fl_pos.shape[0]
    k      = Bk // B

    # For each lab sample, its FL positives should be close
    # Use InfoNCE: lab_i vs its k FL positives against all others
    loss = torch.tensor(0.0, device=device)
    n    = 0

    for i in range(B):
        # Positive FL embeddings for lab sample i
        pos_idx  = (lab_labels == i).nonzero(as_tuple=True)[0]
        if len(pos_idx) == 0:
            continue

        z_anchor = z_lab[i].unsqueeze(0)               # (1, emb_dim)
        z_pos    = z_fl_pos[pos_idx]                    # (k, emb_dim)

        # Negatives: all FL embeddings not belonging to i
        neg_idx  = (lab_labels != i).nonzero(as_tuple=True)[0]
        z_neg    = z_fl_pos[neg_idx] if len(neg_idx) > 0 else z_fl_pos

        # Compute similarities
        sim_pos  = (z_anchor * z_pos).sum(dim=-1) / temperature   # (k,)
        sim_neg  = (z_anchor * z_neg).sum(dim=-1) / temperature   # (n_neg,)

        # InfoNCE
        all_sim  = torch.cat([sim_pos, sim_neg])                  # (k+n_neg,)
        log_denom = torch.logsumexp(all_sim, dim=0)
        loss     += -(sim_pos - log_denom).mean()
        n        += 1

    return loss / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline BYOL
# ─────────────────────────────────────────────────────────────────────────────

class BaselineBYOL(nn.Module):
    def __init__(self, encoder, embedding_dim=256,
                 projector_dim=256, predictor_hidden_dim=512,
                 ema_decay=0.996):
        super().__init__()
        self.ema_decay        = ema_decay
        self.online_encoder   = encoder
        self.online_projector = MLP(embedding_dim, projector_dim, projector_dim)
        self.predictor        = MLP(projector_dim, predictor_hidden_dim,
                                    projector_dim, n_layers=3)
        self.target_encoder   = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target_network(self):
        for o, t in zip(self.online_encoder.parameters(),
                        self.target_encoder.parameters()):
            t.data = self.ema_decay * t.data + (1 - self.ema_decay) * o.data
        for o, t in zip(self.online_projector.parameters(),
                        self.target_projector.parameters()):
            t.data = self.ema_decay * t.data + (1 - self.ema_decay) * o.data

    def forward(self, view_a, view_b):
        z_online = self.online_encoder(view_a, mask_indices=None)
        q        = self.predictor(self.online_projector(z_online))
        with torch.no_grad():
            z_target = self.target_projector(
                self.target_encoder(view_b, mask_indices=None)
            ).detach()
        return byol_loss(q, z_target)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def pretrain(config, experiment, mode, device):
    assert mode in ("byol", "baseline")

    print(f"\n{'='*60}")
    print(f"Phase 1: Pretraining | experiment={experiment} | mode={mode}")
    print(f"{'='*60}")

    byol_loader   = get_pretrain_loader(config, experiment, mode=mode)
    supcon_loader = get_supcon_loader(config, experiment, mode=mode)

    encoder = build_encoder(config).to(device)

    if mode == "byol":
        model = build_byol(config, encoder, experiment).to(device)
        online_params = (
            list(model.online_encoder.parameters()) +
            list(model.online_projector.parameters()) +
            list(model.predictor.parameters()) +
            list(model.regression_head.parameters())
        )
    else:
        model = BaselineBYOL(
            encoder,
            embedding_dim        = config["model"]["embedding_dim"],
            projector_dim        = config["model"]["embedding_dim"],
            predictor_hidden_dim = config["model"]["predictor_hidden_dim"],
            ema_decay            = config["pretrain"]["ema_decay"]
        ).to(device)
        online_params = (
            list(model.online_encoder.parameters()) +
            list(model.online_projector.parameters()) +
            list(model.predictor.parameters())
        )

    # SupCon projection head
    emb_dim          = config["model"]["embedding_dim"]
    supcon_projector = nn.Sequential(
        nn.Linear(emb_dim, emb_dim),
        nn.GELU(),
        nn.Linear(emb_dim, 128)
    ).to(device)
    supcon_loss_fn = SupConLoss(temperature=config["pretrain"]["supcon_temperature"])

    lam_reg    = config["pretrain"]["regression_lambda"]
    lam_supcon = config["pretrain"]["supcon_lambda"]
    lam_cross  = config["pretrain"].get("cross_lambda", 1.0)
    index_freq = config["pretrain"].get("index_rebuild_freq", 5)

    online_params += list(supcon_projector.parameters())

    optimizer = optim.AdamW(
        online_params,
        lr           = config["pretrain"]["lr"],
        weight_decay = config["pretrain"]["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["pretrain"]["epochs"]
    )

    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)

    # Build initial FL index (byol mode only)
    fl_index = fl_full = None
    if mode == "byol":
        print("\nBuilding initial FL index...")
        fl_index, fl_full, _ = build_fl_index(
            model.online_encoder, config, experiment, device
        )

    best_loss   = float("inf")
    epochs      = config["pretrain"]["epochs"]
    supcon_iter = iter(supcon_loader)
    n_known     = len(config["sensors"]["experiments"][experiment]["known_sensors"])

    for epoch in range(1, epochs + 1):
        model.train()
        supcon_projector.train()

        # Rebuild FL index every index_freq epochs
        if mode == "byol" and epoch > 1 and (epoch - 1) % index_freq == 0:
            print(f"\n  [Epoch {epoch}] Rebuilding FL index...")
            fl_index, fl_full, _ = build_fl_index(
                model.online_encoder, config, experiment, device
            )

        total_loss = total_byol = total_reg = total_sc = total_cross = 0.0
        n_batches  = 0

        for view_a, view_b in byol_loader:
            view_a = view_a.to(device, non_blocking=True)
            view_b = view_b.to(device, non_blocking=True)

            # BYOL / regression losses
            if mode == "byol":
                loss_byol, loss_reg = model(view_a, view_b)
            else:
                loss_byol = model(view_a, view_b)
                loss_reg  = torch.tensor(0.0, device=device)

            model.update_target_network()

            # SupCon loss on lab data
            try:
                x_lab, y_lab = next(supcon_iter)
            except StopIteration:
                supcon_iter  = iter(supcon_loader)
                x_lab, y_lab = next(supcon_iter)

            x_lab = x_lab.to(device, non_blocking=True)
            y_lab = y_lab.to(device, non_blocking=True)

            if mode == "byol":
                z_lab = model.online_encoder(x_lab, mask_indices=[n_known])
            else:
                z_lab = model.online_encoder(x_lab, mask_indices=None)

            z_proj  = supcon_projector(z_lab)
            z_norm  = F.normalize(z_proj, dim=-1)
            loss_sc = supcon_loss_fn(z_norm, y_lab)

            # Cross-dataset alignment loss (byol mode only)
            loss_cross = torch.tensor(0.0, device=device)
            if mode == "byol" and fl_index is not None:
                z_lab_norm  = F.normalize(z_lab, dim=-1)
                z_fl_pos, lab_labels = retrieve_cross_dataset_positives(
                    z_lab_norm, fl_index, fl_full,
                    k=config["pretrain"].get("cross_k", 5),
                    device=device
                )
                loss_cross = cross_dataset_loss(
                    z_lab_norm, z_fl_pos, lab_labels,
                    temperature=config["pretrain"]["supcon_temperature"]
                )

            loss = loss_byol + lam_reg * loss_reg + \
                   lam_supcon * loss_sc + lam_cross * loss_cross

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_params, max_norm=1.0)
            optimizer.step()

            total_loss  += loss.item()
            total_byol  += loss_byol.item()
            total_reg   += loss_reg.item()
            total_sc    += loss_sc.item()
            total_cross += loss_cross.item()
            n_batches   += 1

        scheduler.step()

        avg_loss  = total_loss  / n_batches
        avg_byol  = total_byol  / n_batches
        avg_reg   = total_reg   / n_batches
        avg_sc    = total_sc    / n_batches
        avg_cross = total_cross / n_batches
        lr        = scheduler.get_last_lr()[0]

        if mode == "byol":
            print(f"Epoch {epoch:4d}/{epochs} | loss={avg_loss:.4f} "
                  f"(byol={avg_byol:.4f} reg={avg_reg:.4f} "
                  f"sc={avg_sc:.4f} cross={avg_cross:.4f}) | lr={lr:.2e}")
        else:
            print(f"Epoch {epoch:4d}/{epochs} | loss={avg_loss:.4f} "
                  f"(byol={avg_byol:.4f} sc={avg_sc:.4f}) | lr={lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(
                config["output"]["checkpoint_dir"],
                f"pretrain_{mode}_{experiment}_best.pt"
            )
            torch.save({
                "epoch"      : epoch,
                "encoder"    : model.online_encoder.state_dict(),
                "model"      : model.state_dict(),
                "supcon_proj": supcon_projector.state_dict(),
                "loss"       : best_loss,
                "experiment" : experiment,
                "mode"       : mode,
                "config"     : config
            }, ckpt_path)
            print(f"  -> Saved best checkpoint (loss={best_loss:.4f})")

    # Save final
    final_path = os.path.join(
        config["output"]["checkpoint_dir"],
        f"pretrain_{mode}_{experiment}_final.pt"
    )
    torch.save({
        "epoch"      : epochs,
        "encoder"    : model.online_encoder.state_dict(),
        "model"      : model.state_dict(),
        "supcon_proj": supcon_projector.state_dict(),
        "loss"       : avg_loss,
        "experiment" : experiment,
        "mode"       : mode,
        "config"     : config
    }, final_path)

    print(f"\nPretraining complete | mode={mode} | best loss: {best_loss:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    return ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["2to1", "1to1"])
    parser.add_argument("--mode",       type=str, required=True,
                        choices=["byol", "baseline"])
    parser.add_argument("--device",     type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    pretrain(config, args.experiment, args.mode, device)


if __name__ == "__main__":
    main()
