"""
byol.py

BYOL + cross-sensor regression for sensor increment.

Three components:

1. BYOL asymmetric prediction (direction alignment)
   Online(masked view) → predictor → q ≈ Target(full view)

2. Cross-sensor regression (explicit new sensor prediction)
   Known sensor embeddings → regression MLP → predicted new sensor embedding
   MSE(predicted, actual new sensor embedding)

3. SupCon (in pretrain.py, uses the same encoder)

The regression loss provides direct gradient signal for cross-sensor
alignment — the encoder must explicitly learn what the new sensor
contributes from the known sensors.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.misc.stream_encoder import SensorEncoder


# ─────────────────────────────────────────────────────────────────────────────
# MLP helpers
# ─────────────────────────────────────────────────────────────────────────────

def MLP(in_dim, hidden_dim, out_dim, n_layers=2, dropout=0.3):
    layers = []
    dims   = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            layers += [nn.LayerNorm(dims[i+1]), nn.GELU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# Per-sensor embedding extractor
# ─────────────────────────────────────────────────────────────────────────────

class SensorSlotEncoder(nn.Module):
    """
    Extracts individual sensor slot embeddings from the CNN,
    before the transformer fusion. Used for the regression loss.

    Input:  (batch, n_sensors, 100, 3)
    Output: (batch, n_sensors, cnn_dim)
    """

    def __init__(self, encoder):
        super().__init__()
        self.stream_cnn = encoder.stream_cnn
        self.cnn_proj   = encoder.cnn_proj

    def forward(self, x):
        B, S, T, C = x.shape
        x = x.reshape(B * S, T, C).permute(0, 2, 1)  # (B*S, 3, 100)
        x = self.stream_cnn(x)                         # (B*S, cnn_dim)
        x = x.reshape(B, S, -1)                        # (B, S, cnn_dim)
        x = self.cnn_proj(x)                           # (B, S, d_model)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# BYOL model with regression
# ─────────────────────────────────────────────────────────────────────────────

class BYOL(nn.Module):
    """
    BYOL + cross-sensor regression.

    Args:
        encoder              : SensorEncoder instance
        embedding_dim        : output dim of encoder
        projector_dim        : BYOL projector output dim
        predictor_hidden_dim : BYOL predictor hidden dim
        regression_hidden_dim: regression MLP hidden dim
        ema_decay            : EMA decay for target network
        n_known_sensors      : number of known sensor slots
        new_sensor_idx       : index of new sensor slot in the full input
    """

    def __init__(self, encoder, embedding_dim=256,
                 projector_dim=256, predictor_hidden_dim=512,
                 regression_hidden_dim=512,
                 ema_decay=0.996,
                 n_known_sensors=2,
                 new_sensor_idx=2):
        super().__init__()

        self.ema_decay       = ema_decay
        self.new_sensor_idx  = new_sensor_idx
        self.n_known_sensors = n_known_sensors
        d_model              = encoder.d_model

        # Online network
        self.online_encoder   = encoder
        self.online_projector = MLP(embedding_dim, projector_dim, projector_dim)
        self.predictor        = MLP(projector_dim, predictor_hidden_dim,
                                    projector_dim, n_layers=3)

        # Target network (EMA, no grad)
        self.target_encoder   = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        self._stop_target_gradients()

        # Cross-sensor regression head
        # Input: known sensor slot embeddings (n_known * d_model)
        # Output: predicted new sensor slot embedding (d_model)
        self.regression_head = MLP(
            in_dim     = n_known_sensors * d_model,
            hidden_dim = regression_hidden_dim,
            out_dim    = d_model,
            n_layers   = 3,
            dropout    = 0.3
        )

        # Per-sensor extractor (shares weights with online encoder CNN)
        self.slot_encoder = SensorSlotEncoder(self.online_encoder)

    def _stop_target_gradients(self):
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target_network(self):
        for online, target in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target.data = self.ema_decay * target.data + \
                          (1 - self.ema_decay) * online.data

        for online, target in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters()
        ):
            target.data = self.ema_decay * target.data + \
                          (1 - self.ema_decay) * online.data

    def forward(self, view_masked, view_full):
        """
        view_masked : (B, n_sensors, 100, 3) — new sensor slot masked
        view_full   : (B, n_sensors, 100, 3) — all sensors present

        Returns: (loss_byol, loss_regression)
        """
        # ── BYOL loss ──────────────────────────────────────────────────────
        # Online: masked view with [MASK] token at new sensor slot
        z_online = self.online_encoder(
            view_masked, mask_indices=[self.new_sensor_idx]
        )
        z_proj   = self.online_projector(z_online)
        q        = self.predictor(z_proj)

        # Target: full view, no masking (stop gradient)
        with torch.no_grad():
            z_target = self.target_encoder(view_full, mask_indices=None)
            z_target = self.target_projector(z_target).detach()

        loss_byol = byol_loss(q, z_target)

        # ── Regression loss ────────────────────────────────────────────────
        # Get per-slot CNN embeddings from full view
        slot_embs  = self.slot_encoder(view_full)   # (B, S, d_model)

        # Known sensor slots
        known_idx  = [i for i in range(slot_embs.shape[1])
                      if i != self.new_sensor_idx]
        known_embs = slot_embs[:, known_idx, :]     # (B, n_known, d_model)
        B, K, D    = known_embs.shape
        known_flat = known_embs.reshape(B, K * D)   # (B, n_known * d_model)

        # Predict new sensor slot embedding
        pred_new   = self.regression_head(known_flat)  # (B, d_model)

        # Target: actual new sensor slot embedding (stop gradient)
        with torch.no_grad():
            target_new = slot_embs[:, self.new_sensor_idx, :].detach()

        loss_regression = F.mse_loss(
            F.normalize(pred_new,    dim=-1),
            F.normalize(target_new,  dim=-1)
        )

        return loss_byol, loss_regression

    def get_encoder(self):
        return self.online_encoder

    def get_target_encoder(self):
        return self.target_encoder


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def byol_loss(q, z):
    q = F.normalize(q, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2 - 2 * (q * z).sum(dim=-1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_byol(config, encoder, experiment):
    m    = config["model"]
    p    = config["pretrain"]
    exp  = config["sensors"]["experiments"][experiment]

    n_known      = len(exp["known_sensors"])
    new_sens_idx = n_known   # new sensor is always appended last

    return BYOL(
        encoder               = encoder,
        embedding_dim         = m["embedding_dim"],
        projector_dim         = m["embedding_dim"],
        predictor_hidden_dim  = m["predictor_hidden_dim"],
        regression_hidden_dim = m["regression_hidden_dim"],
        ema_decay             = p["ema_decay"],
        n_known_sensors       = n_known,
        new_sensor_idx        = new_sens_idx
    )


if __name__ == "__main__":
    from scripts.misc.stream_encoder import SensorEncoder

    encoder = SensorEncoder(
        in_channels=3, kernel_size=5,
        n_sensors_max=8, d_model=512,
        n_heads=8, n_layers=4, dropout=0.3,
        embedding_dim=256
    )

    config = {
        "model": {
            "embedding_dim": 256, "predictor_hidden_dim": 512,
            "regression_hidden_dim": 512
        },
        "pretrain": {"ema_decay": 0.996},
        "sensors": {
            "experiments": {
                "2to1": {"known_sensors": ["LeftWrist", "RightAnkle"],
                         "new_sensor":    ["RightThigh"]}
            }
        }
    }

    model = build_byol(config, encoder, "2to1")

    view_masked = torch.randn(8, 3, 100, 3)
    view_full   = torch.randn(8, 3, 100, 3)

    loss_byol, loss_reg = model(view_masked, view_full)
    print(f"BYOL loss:       {loss_byol.item():.4f}")
    print(f"Regression loss: {loss_reg.item():.4f}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params:    {n_params:,}")
    print("BYOL OK")
