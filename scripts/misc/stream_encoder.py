"""
stream_encoder.py

Per-sensor 1D CNN + Transformer encoder with cross-sensor attention.

Architecture:
  1. Per-sensor 1D CNN (shared weights, 4 blocks) → (batch, n_sensors, cnn_dim)
  2. Sensor position embeddings added
  3. Masked sensor slots replaced with learned [MASK] token
  4. [CLS] token prepended
  5. Transformer encoder (4 layers, 8 heads) → cross-sensor attention
  6. [CLS] token extracted → linear projection → embedding_dim

Input:  (batch, n_sensors, 100, 3)
Output: (batch, embedding_dim)

The [MASK] token allows the transformer to infer missing sensor
contributions from context — much stronger than zeroing out sensors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────────────────
# 1D CNN Stream Encoder (shared across sensors)
# ─────────────────────────────────────────────────────────────────────────────

class StreamCNN(nn.Module):
    """
    Deep 1D CNN that encodes a single sensor stream (100, 3) → (cnn_dim,)

    4 blocks of: Conv1d → BN → GELU → MaxPool
    channels: 3 → 64 → 128 → 256 → 512
    """

    def __init__(self, in_channels=3, kernel_size=5):
        super().__init__()

        channels = [in_channels, 64, 128, 256, 512]
        layers   = []

        for i in range(len(channels) - 1):
            layers += [
                nn.Conv1d(channels[i], channels[i+1],
                          kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(channels[i+1]),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ]

        self.cnn     = nn.Sequential(*layers)
        self.out_dim = channels[-1]   # 512

    def forward(self, x):
        """
        x: (B*S, 3, 100)
        returns: (B*S, 512)
        """
        x = self.cnn(x)       # (B*S, 512, T')
        x = x.mean(dim=-1)    # global average pool → (B*S, 512)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Sensor Transformer
# ─────────────────────────────────────────────────────────────────────────────

class SensorTransformer(nn.Module):
    """
    Transformer encoder over sensor tokens.

    Each sensor is a token of size d_model.
    A [CLS] token is prepended for global representation.
    Sensor position embeddings tell the transformer which slot is which.

    Args:
        n_sensors_max : maximum number of sensor slots (for position embeddings)
        d_model       : transformer hidden dimension
        n_heads       : number of attention heads
        n_layers      : number of transformer layers
        dropout       : dropout rate
    """

    def __init__(self, n_sensors_max=8, d_model=512,
                 n_heads=8, n_layers=4, dropout=0.3):
        super().__init__()

        self.d_model = d_model

        # Input projection from CNN dim to d_model
        # (in case cnn_dim != d_model, keeps them decoupled)
        self.input_proj = nn.Linear(d_model, d_model)

        # [CLS] token and [MASK] token (learned)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, d_model))
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Sensor position embeddings (+1 for CLS)
        self.pos_embed  = nn.Embedding(n_sensors_max + 1, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            activation      = 'gelu',
            batch_first     = True,
            norm_first      = True    # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = n_layers,
            norm       = nn.LayerNorm(d_model)
        )

    def forward(self, x, mask_indices=None):
        """
        x            : (B, S, d_model) — per-sensor CNN features
        mask_indices : list of sensor slot indices to replace with [MASK] token
                       None = no masking (all sensors present)

        Returns: (B, d_model) — [CLS] token embedding
        """
        B, S, D = x.shape

        # Replace masked sensor slots with [MASK] token
        if mask_indices is not None and len(mask_indices) > 0:
            mask = self.mask_token.expand(B, 1, D)
            for idx in mask_indices:
                x[:, idx, :] = mask[:, 0, :]

        # Project input
        x = self.input_proj(x)   # (B, S, d_model)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, 1, D)   # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)        # (B, S+1, d_model)

        # Add position embeddings (0=CLS, 1..S=sensors)
        pos = torch.arange(S + 1, device=x.device)   # (S+1,)
        x   = x + self.pos_embed(pos).unsqueeze(0)    # (B, S+1, d_model)

        # Transformer
        x   = self.transformer(x)    # (B, S+1, d_model)

        # Return [CLS] token
        return x[:, 0, :]            # (B, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# Full Sensor Encoder
# ─────────────────────────────────────────────────────────────────────────────

class SensorEncoder(nn.Module):
    """
    Full encoder: per-sensor CNN + transformer + projection.

    Input:  (batch, n_sensors, 100, 3)
    Output: (batch, embedding_dim)

    mask_indices: sensor slot indices to mask (replace with [MASK] token).
    At training time: pass mask_indices=[n] to mask the new sensor.
    At test time:     pass mask_indices=None (all sensors present).
    """

    def __init__(self, in_channels=3, kernel_size=5,
                 n_sensors_max=8, d_model=512,
                 n_heads=8, n_layers=4, dropout=0.3,
                 embedding_dim=256):
        super().__init__()

        self.stream_cnn  = StreamCNN(in_channels, kernel_size)
        cnn_dim          = self.stream_cnn.out_dim   # 512

        # Project CNN output to d_model if different
        self.cnn_proj    = nn.Linear(cnn_dim, d_model) if cnn_dim != d_model \
                           else nn.Identity()

        self.transformer = SensorTransformer(
            n_sensors_max = n_sensors_max,
            d_model       = d_model,
            n_heads       = n_heads,
            n_layers      = n_layers,
            dropout       = dropout
        )

        # Final projection to embedding_dim
        self.projection  = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        self.embedding_dim   = embedding_dim
        self.d_model         = d_model

    def forward(self, x, mask_indices=None):
        """
        x            : (batch, n_sensors, 100, 3)
        mask_indices : list of sensor slot indices to mask, or None

        Returns: (batch, embedding_dim)
        """
        B, S, T, C = x.shape

        # Per-sensor CNN encoding
        x_cnn = x.reshape(B * S, T, C)           # (B*S, 100, 3)
        x_cnn = x_cnn.permute(0, 2, 1)           # (B*S, 3, 100)
        x_cnn = self.stream_cnn(x_cnn)            # (B*S, cnn_dim)
        x_cnn = x_cnn.reshape(B, S, -1)           # (B, S, cnn_dim)
        x_cnn = self.cnn_proj(x_cnn)              # (B, S, d_model)
        x_cnn = F.normalize(x_cnn, dim=-1)        # normalize slot embeddings

        # Cross-sensor transformer
        z = self.transformer(x_cnn, mask_indices) # (B, d_model)

        # Final projection
        z = self.projection(z)                    # (B, embedding_dim)

        return z

    def forward_with_injected_slot(self, x, slot_idx, injected_emb):
        """
        Forward pass where one sensor slot is replaced with a
        pre-computed embedding (e.g. from regression head) instead
        of going through the CNN.

        Used during fine-tuning to produce pseudo-full embeddings:
          - Known sensor slots go through CNN normally
          - New sensor slot is replaced with regression prediction

        Args:
            x            : (batch, n_sensors, 100, 3)  known sensors
                           (new sensor slot can be zeros — it will be replaced)
            slot_idx     : int, which slot to inject into
            injected_emb : (batch, d_model) embedding to inject

        Returns: (batch, embedding_dim)
        """
        B, S, T, C = x.shape

        # Per-sensor CNN encoding for all slots
        x_cnn = x.reshape(B * S, T, C)
        x_cnn = x_cnn.permute(0, 2, 1)
        x_cnn = self.stream_cnn(x_cnn)
        x_cnn = x_cnn.reshape(B, S, -1)
        x_cnn = self.cnn_proj(x_cnn)              # (B, S, d_model)
        x_cnn = F.normalize(x_cnn, dim=-1)        # normalize slot embeddings

        # Replace target slot with injected embedding (also normalized)
        x_cnn = x_cnn.clone()
        x_cnn[:, slot_idx, :] = F.normalize(injected_emb, dim=-1)

        # Transformer with injected slot (no masking)
        z = self.transformer(x_cnn, mask_indices=None)

        # Final projection
        z = self.projection(z)

        return z

    def get_slot_embeddings(self, x):
        """
        Returns per-slot CNN embeddings before transformer fusion.
        Used by regression head to predict the new sensor slot.

        Args:
            x : (batch, n_sensors, 100, 3)

        Returns: (batch, n_sensors, d_model)
        """
        B, S, T, C = x.shape
        x_cnn = x.reshape(B * S, T, C)
        x_cnn = x_cnn.permute(0, 2, 1)
        x_cnn = self.stream_cnn(x_cnn)
        x_cnn = x_cnn.reshape(B, S, -1)
        x_cnn = self.cnn_proj(x_cnn)
        x_cnn = F.normalize(x_cnn, dim=-1)        # normalize slot embeddings
        return x_cnn                               # (B, S, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# Classifier Head
# ─────────────────────────────────────────────────────────────────────────────

class ClassifierHead(nn.Module):
    """
    3-layer MLP classifier on top of encoder embeddings.
    (batch, embedding_dim) → (batch, n_classes)
    """

    def __init__(self, embedding_dim, n_classes, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, n_classes)
        )

    def forward(self, z):
        return self.head(z)


# ─────────────────────────────────────────────────────────────────────────────
# Factory functions
# ─────────────────────────────────────────────────────────────────────────────

def build_encoder(config):
    m = config["model"]
    return SensorEncoder(
        in_channels   = m["input_channels"],
        kernel_size   = m["kernel_size"],
        n_sensors_max = m["n_sensors_max"],
        d_model       = m["d_model"],
        n_heads       = m["n_heads"],
        n_layers      = m["n_layers"],
        dropout       = m["dropout"],
        embedding_dim = m["embedding_dim"]
    )


def build_classifier(config, n_classes):
    return ClassifierHead(
        embedding_dim = config["model"]["embedding_dim"],
        n_classes     = n_classes,
        dropout       = config["model"]["dropout"]
    )


if __name__ == "__main__":
    encoder = SensorEncoder(
        in_channels=3, kernel_size=5,
        n_sensors_max=8, d_model=512,
        n_heads=8, n_layers=4, dropout=0.3,
        embedding_dim=256
    )

    # Test with 3 sensors (2 known + 1 new)
    x = torch.randn(8, 3, 100, 3)

    # Full view (all sensors)
    z_full   = encoder(x, mask_indices=None)
    print(f"Full view:   {z_full.shape}")    # (8, 256)

    # Masked view (new sensor masked)
    z_masked = encoder(x, mask_indices=[2])
    print(f"Masked view: {z_masked.shape}")  # (8, 256)

    cos = F.cosine_similarity(z_full, z_masked).mean()
    print(f"Cosine sim at init: {cos:.4f}")  # random at init, should improve

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total params: {n_params:,}")

    print("SensorEncoder OK")