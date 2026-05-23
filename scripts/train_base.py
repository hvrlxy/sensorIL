"""
train_base.py

Step 1: Train all binary classifiers in parallel using vectorized
independent classifiers (Option B).

Architecture: ParallelBinaryClassifiers
  - n_classes independent MLPs, each with its own weights
  - Implemented as batched linear layers — no shared trunk
  - One forward/backward pass trains all classifiers simultaneously
  - Gradients are fully independent per classifier (no cross-class mixing)

Equivalent to training n_classes separate BinaryClassifier instances
but 10-40x faster via GPU parallelism.

Multi-label encoding:
  A window of class A is positive for A AND all ancestors.

Usage:
    python scripts/train_base.py --config configs/pipeline_config.json
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from simclr_encoder import load_simclr_encoder, encode_sensors, ENCODER_DIM
from dataset import SensorDataset
from cooccurrence import get_multilabel


# ─────────────────────────────────────────────────────────────────────────────
# Parallel Binary Classifiers (vectorized, independent)
# ─────────────────────────────────────────────────────────────────────────────

class ParallelBinaryClassifiers(nn.Module):
    """
    n_classes independent binary classifiers trained in parallel.

    Each classifier is a 3-layer MLP with its own independent weights.
    Implemented as batched matrix ops — no shared parameters.

    Architecture per classifier:
      Linear(input_dim → hidden) → LayerNorm → GELU → Dropout
      Linear(hidden → hidden//2) → LayerNorm → GELU → Dropout
      Linear(hidden//2 → 1)

    Input:  (B, input_dim)
    Output: (B, n_classes) logits
    """

    def __init__(self, input_dim, n_classes,
                 hidden_dim=256, dropout=0.3):
        super().__init__()
        self.n_classes = n_classes
        self.input_dim = input_dim
        h1 = hidden_dim
        h2 = hidden_dim // 2

        # Layer 1: (n_classes, input_dim, h1) — independent per class
        self.w1   = nn.Parameter(torch.empty(n_classes, input_dim, h1))
        self.b1   = nn.Parameter(torch.zeros(n_classes, h1))
        self.ln1  = nn.ModuleList([nn.LayerNorm(h1) for _ in range(n_classes)])

        # Layer 2: (n_classes, h1, h2)
        self.w2   = nn.Parameter(torch.empty(n_classes, h1, h2))
        self.b2   = nn.Parameter(torch.zeros(n_classes, h2))
        self.ln2  = nn.ModuleList([nn.LayerNorm(h2) for _ in range(n_classes)])

        # Layer 3: (n_classes, h2, 1)
        self.w3   = nn.Parameter(torch.empty(n_classes, h2, 1))
        self.b3   = nn.Parameter(torch.zeros(n_classes, 1))

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.w1, a=0.01)
        nn.init.kaiming_uniform_(self.w2, a=0.01)
        nn.init.kaiming_uniform_(self.w3, a=0.01)

    def forward(self, x):
        """
        x : (B, input_dim)
        returns: (B, n_classes) logits
        """
        B = x.shape[0]
        C = self.n_classes

        # x: (B, input_dim) → expand for bmm: (C, B, input_dim)
        x_exp = x.unsqueeze(0).expand(C, B, -1)   # (C, B, D)

        # Layer 1: (C, B, D) x (C, D, h1) → (C, B, h1)
        h = torch.bmm(x_exp, self.w1) + self.b1.unsqueeze(1)
        h = torch.stack([self.ln1[i](h[i]) for i in range(C)])  # (C, B, h1)
        h = self.dropout(F.gelu(h))

        # Layer 2: (C, B, h1) x (C, h1, h2) → (C, B, h2)
        h = torch.bmm(h, self.w2) + self.b2.unsqueeze(1)
        h = torch.stack([self.ln2[i](h[i]) for i in range(C)])  # (C, B, h2)
        h = self.dropout(F.gelu(h))

        # Layer 3: (C, B, h2) x (C, h2, 1) → (C, B, 1) → (B, C)
        out = (torch.bmm(h, self.w3) + self.b3.unsqueeze(1)).squeeze(-1).T
        return out   # (B, C)

    def get_single_classifier(self, class_idx):
        """
        Extract weights for a single class as a BinaryClassifier.
        Used for incremental fine-tuning of individual classifiers.
        """
        from train_base import BinaryClassifier
        clf = BinaryClassifier(self.input_dim)

        with torch.no_grad():
            clf.net[0].weight.copy_(self.w1[class_idx].T)
            clf.net[0].bias.copy_(self.b1[class_idx])
            clf.net[1].weight.copy_(self.ln1[class_idx].weight)
            clf.net[1].bias.copy_(self.ln1[class_idx].bias)
            clf.net[4].weight.copy_(self.w2[class_idx].T)
            clf.net[4].bias.copy_(self.b2[class_idx])
            clf.net[5].weight.copy_(self.ln2[class_idx].weight)
            clf.net[5].bias.copy_(self.ln2[class_idx].bias)
            clf.net[8].weight.copy_(self.w3[class_idx].T)
            clf.net[8].bias.copy_(self.b3[class_idx].squeeze())

        return clf


# ─────────────────────────────────────────────────────────────────────────────
# Single Binary Classifier (used for incremental fine-tuning)
# ─────────────────────────────────────────────────────────────────────────────

class BinaryClassifier(nn.Module):
    """
    Single binary classifier — same architecture as one head
    of ParallelBinaryClassifiers.

    Used during incremental fine-tuning where classifiers
    must be retrained independently.

    Input:  (B, input_dim)
    Output: (B,) logits
    """

    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        h1 = hidden_dim
        h2 = hidden_dim // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),      # 0
            nn.LayerNorm(h1),              # 1
            nn.GELU(),                     # 2
            nn.Dropout(dropout),           # 3
            nn.Linear(h1, h2),             # 4
            nn.LayerNorm(h2),              # 5
            nn.GELU(),                     # 6
            nn.Dropout(dropout),           # 7
            nn.Linear(h2, 1)               # 8
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, weights=None):
        bce  = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none'
        )
        prob = torch.sigmoid(logits)
        pt   = torch.where(targets == 1, prob, 1 - prob)
        at   = torch.where(targets == 1,
                           torch.tensor(self.alpha, device=logits.device),
                           torch.tensor(1 - self.alpha, device=logits.device))
        loss = at * (1 - pt) ** self.gamma * bce
        if weights is not None:
            loss = loss * weights
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_base(config, device):
    print(f"\n{'='*60}")
    print(f"Step 1: Training parallel binary classifiers")
    print(f"{'='*60}")

    sensors   = config["sensors"]["known_sensors"]
    n_sensors = len(sensors)
    input_dim = n_sensors * ENCODER_DIM

    encoder = load_simclr_encoder(config["model"]["encoder_path"], device)

    # Load labeled data
    train_ds = SensorDataset(
        data_dir              = config["data"]["labeled_dir"],
        sensors               = sensors,
        max_samples_per_class = config["finetune"]["few_shot_samples_per_class"],
        split                 = "train",
        val_split             = config["finetune"]["val_split"]
    )
    val_ds = SensorDataset(
        data_dir              = config["data"]["labeled_dir"],
        sensors               = sensors,
        max_samples_per_class = config["finetune"]["few_shot_samples_per_class"],
        split                 = "val",
        val_split             = config["finetune"]["val_split"]
    )

    class_names = train_ds.class_names
    n_classes   = train_ds.n_classes

    # Pre-encode
    print("Encoding windows...")
    def encode_ds(ds):
        loader = DataLoader(ds, batch_size=512, shuffle=False,
                            num_workers=2, pin_memory=True)
        all_z, all_y = [], []
        for x, y in loader:
            all_z.append(encode_sensors(encoder, x, device))
            all_y.append(y)
        return torch.cat(all_z), torch.cat(all_y)

    z_train, y_train = encode_ds(train_ds)
    z_val,   y_val   = encode_ds(val_ds)

    # Build multi-label matrices (vectorized)
    print("Building multi-label targets (vectorized)...")
    ml_matrix  = torch.tensor(
        [get_multilabel(name, class_names) for name in class_names],
        dtype=torch.float32
    )  # (n_classes, n_classes)
    y_ml_train = ml_matrix[y_train]   # (N_train, n_classes)
    y_ml_val   = ml_matrix[y_val]     # (N_val,   n_classes)

    print(f"Train: {z_train.shape} | Val: {z_val.shape}")
    print(f"Avg labels per window: {y_ml_train.sum(1).mean():.2f}")

    # Move to device
    z_tr = z_train.to(device)
    z_vl = z_val.to(device)
    y_tr = y_ml_train.to(device)
    y_vl = y_ml_val.to(device)

    # Train parallel classifiers
    model     = ParallelBinaryClassifiers(input_dim, n_classes).to(device)
    focal     = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(),
                            lr=config["finetune"]["lr"],
                            weight_decay=config["finetune"]["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["finetune"]["epochs"]
    )

    best_val_loss = float("inf")
    best_state    = None
    epochs        = config["finetune"]["epochs"]
    bs            = config["finetune"]["batch_size"]

    for epoch in range(epochs):
        model.train()
        perm   = torch.randperm(len(z_tr), device=device)
        ep_loss = 0.0
        n_batches = 0

        for start in range(0, len(z_tr), bs):
            idx = perm[start:start+bs]
            zb  = z_tr[idx]
            yb  = y_tr[idx]

            optimizer.zero_grad()
            logits = model(zb)          # (B, n_classes)
            loss   = focal(logits, yb)
            loss.backward()
            optimizer.step()
            ep_loss   += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(z_vl)
            val_loss   = focal(val_logits, y_vl).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:4d}/{epochs} | "
                  f"train={ep_loss/n_batches:.4f} val={val_loss:.4f}")

    model.load_state_dict(best_state)

    # Per-class val F1
    model.eval()
    with torch.no_grad():
        val_preds = (torch.sigmoid(model(z_vl)) > 0.5).cpu().numpy().astype(int)
    val_labels = y_vl.cpu().numpy().astype(int)

    from sklearn.metrics import f1_score as _f1
    f1_scores = [_f1(val_labels[:, i], val_preds[:, i],
                     average="binary", zero_division=0)
                 for i in range(n_classes)]
    print(f"\nMean val F1: {np.mean(f1_scores):.3f}")
    for i, (name, f1) in enumerate(zip(class_names, f1_scores)):
        if (i + 1) % 10 == 0 or i == n_classes - 1:
            print(f"  [{i+1:3d}/{n_classes}] {name:40s} val_f1={f1:.3f}")

    # Extract individual BinaryClassifier state dicts for compatibility
    # with incremental_ft and evaluate which expect individual classifiers
    classifiers = {}
    for i, name in enumerate(class_names):
        clf = BinaryClassifier(input_dim).to(device)
        # Copy weights from parallel model
        with torch.no_grad():
            clf.net[0].weight.copy_(model.w1[i].T)
            clf.net[0].bias.copy_(model.b1[i])
            clf.net[1].weight.copy_(model.ln1[i].weight)
            clf.net[1].bias.copy_(model.ln1[i].bias)
            clf.net[4].weight.copy_(model.w2[i].T)
            clf.net[4].bias.copy_(model.b2[i])
            clf.net[5].weight.copy_(model.ln2[i].weight)
            clf.net[5].bias.copy_(model.ln2[i].bias)
            clf.net[8].weight.copy_(model.w3[i].T)
            clf.net[8].bias.copy_(model.b3[i].squeeze())
        classifiers[name] = clf

    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
    ckpt_name = config.get("checkpoint_name", "base_classifiers.pt")
    ckpt_path = os.path.join(config["output"]["checkpoint_dir"], ckpt_name)
    torch.save({
        "classifiers" : {k: v.state_dict() for k, v in classifiers.items()},
        "input_dims"  : {name: input_dim for name in class_names},
        "input_dim"   : input_dim,
        "class_names" : class_names,
        "n_classes"   : n_classes,
        "sensors"     : sensors,
        "config"      : config
    }, ckpt_path)
    print(f"Saved → {ckpt_path}")
    return ckpt_path, classifiers, class_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_base(config, device)

if __name__ == "__main__":
    main()