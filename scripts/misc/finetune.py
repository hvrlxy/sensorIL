"""
finetune.py

Phase 2: Few-shot fine-tuning — frozen encoder + classifier head.

mode='byol':
  For each labeled sample, augments the masked embedding with
  the precomputed mean shift from FL data to simulate the
  distribution of full (n+1 sensor) embeddings:

    z = encoder(x, mask_indices=[new_slot])
    z = z + alpha * mean_shift + beta * randn  (random augmentation)

  This trains the classifier to be robust to the masked→full shift,
  so at test time with real n+1 sensors it doesn't collapse.

mode='baseline':
  Standard forward, no masking, no augmentation.

Usage:
    python scripts/finetune.py --config configs/byol_config.json \\
                                --experiment 2to1 --mode byol \\
                                --checkpoint checkpoints/pretrain_byol_2to1_best.pt \\
                                --shift     checkpoints/embedding_shift_2to1.pt
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scripts.misc.dataset import get_finetune_loaders
from scripts.misc.stream_encoder import build_encoder, build_classifier


def finetune(config, experiment, mode, checkpoint_path, device,
             shift_path=None):
    assert mode in ("byol", "baseline")

    print(f"\n{'='*60}")
    print(f"Phase 2: Fine-tuning | experiment={experiment} | mode={mode}")
    print(f"{'='*60}")

    train_loader, val_loader, n_classes, class_names = get_finetune_loaders(
        config, experiment, mode=mode
    )
    print(f"Classes: {n_classes}")

    exp          = config["sensors"]["experiments"][experiment]
    n_known      = len(exp["known_sensors"])
    new_slot_idx = n_known

    # Load pretrained encoder — frozen
    encoder = build_encoder(config).to(device)
    ckpt    = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    for param in encoder.parameters():
        param.requires_grad = False
    print(f"Encoder frozen: {sum(p.numel() for p in encoder.parameters()):,} params")

    # Load mean shift for augmentation (byol mode only)
    mean_shift = std_shift = None
    if mode == "byol" and shift_path is not None:
        shift_data = torch.load(shift_path, map_location=device)
        mean_shift = shift_data["mean_shift"].to(device)  # (emb_dim,)
        std_shift  = shift_data["std_shift"].to(device)   # (emb_dim,)
        print(f"Loaded embedding shift | "
              f"mean norm={mean_shift.norm():.4f} "
              f"std norm={std_shift.norm():.4f}")

    aug_alpha = config["finetune"].get("shift_alpha", 1.0)   # scale of mean shift
    aug_beta  = config["finetune"].get("shift_beta",  0.5)   # scale of noise

    # Classifier head
    classifier = build_classifier(config, n_classes).to(device)

    optimizer = optim.AdamW(
        classifier.parameters(),
        lr           = config["finetune"]["lr"],
        weight_decay = config["finetune"]["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["finetune"]["epochs"]
    )
    criterion = nn.CrossEntropyLoss()

    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)

    best_val_acc   = 0.0
    best_ckpt_path = None
    epochs         = config["finetune"]["epochs"]

    for epoch in range(1, epochs + 1):
        encoder.eval()
        classifier.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.no_grad():
                if mode == "byol":
                    z = encoder(x, mask_indices=[new_slot_idx])
                    z = F.normalize(z, dim=-1)

                    if mean_shift is not None:
                        # Randomly sample augmentation strength
                        B     = z.shape[0]
                        alpha = torch.empty(B, 1, device=device).uniform_(0, aug_alpha)
                        beta  = torch.empty(B, 1, device=device).uniform_(0, aug_beta)

                        # Apply shift: move toward full embedding distribution
                        noise = torch.randn_like(z) * std_shift.unsqueeze(0)
                        z     = z + alpha * mean_shift.unsqueeze(0) \
                                  + beta  * noise
                        z     = F.normalize(z, dim=-1)
                else:
                    z = F.normalize(encoder(x, mask_indices=None), dim=-1)

            logits = classifier(z)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total   += len(y)

        scheduler.step()
        train_acc  = train_correct / train_total
        train_loss = train_loss / len(train_loader)

        # Validate — no augmentation, use masked view as-is
        classifier.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if mode == "byol":
                    z = F.normalize(encoder(x, mask_indices=[new_slot_idx]), dim=-1)
                else:
                    z = F.normalize(encoder(x, mask_indices=None), dim=-1)

                logits = classifier(z)
                loss   = criterion(logits, y)

                val_loss    += loss.item()
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total   += len(y)

        val_acc  = val_correct / val_total
        val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch:4d}/{epochs} | "
              f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_ckpt_path = os.path.join(
                config["output"]["checkpoint_dir"],
                f"finetune_{mode}_{experiment}_best.pt"
            )
            torch.save({
                "epoch"      : epoch,
                "encoder"    : encoder.state_dict(),
                "classifier" : classifier.state_dict(),
                "val_acc"    : best_val_acc,
                "n_classes"  : n_classes,
                "class_names": class_names,
                "experiment" : experiment,
                "mode"       : mode,
                "config"     : config
            }, best_ckpt_path)
            print(f"  -> Saved best checkpoint (val_acc={best_val_acc:.3f})")

    print(f"\nFine-tuning complete. Best val acc: {best_val_acc:.3f}")
    print(f"Checkpoint: {best_ckpt_path}")
    return best_ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["2to1", "1to1"])
    parser.add_argument("--mode",       type=str, required=True,
                        choices=["byol", "baseline"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--shift",      type=str, default=None,
                        help="Path to embedding shift file (byol mode only)")
    parser.add_argument("--device",     type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    finetune(config, args.experiment, args.mode,
             args.checkpoint, device, args.shift)


if __name__ == "__main__":
    main()
