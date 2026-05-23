"""
SimCLR models for triaxial accelerometer data — PyTorch version.

Architecture mirrors the original TensorFlow TPN-style base model:
    Conv1d(32, k=24) -> Dropout -> Conv1d(64, k=16) -> Dropout
    -> Conv1d(96, k=8) -> Dropout -> GlobalMaxPool -> [projection head]

All models are nn.Module subclasses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Base encoder (TPN-style)
# ---------------------------------------------------------------------------

class BaseEncoder(nn.Module):
    """
    1-D CNN encoder for a single sensor stream.

    Input:  (B, T, C)  — e.g. (B, 100, 3)
    Output: (B, 96)    — global-max-pooled feature vector

    PyTorch Conv1d expects (B, C, T), so we permute inside forward().
    """

    def __init__(self, in_channels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            # block 1
            nn.Conv1d(in_channels, 32, kernel_size=24, padding=0),
            nn.ReLU(),
            nn.Dropout(dropout),
            # block 2
            nn.Conv1d(32, 64, kernel_size=16, padding=0),
            nn.ReLU(),
            nn.Dropout(dropout),
            # block 3
            nn.Conv1d(64, 96, kernel_size=8, padding=0),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.encoder(x)           # (B, 96, T')
        x = x.max(dim=-1).values      # global max pool -> (B, 96)
        return x


# ---------------------------------------------------------------------------
# Projection head (SimCLR head)
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    3-layer MLP projection head attached on top of the base encoder
    for SimCLR contrastive pre-training.

    Architecture: Linear(h1) -> ReLU -> Linear(h2) -> ReLU -> Linear(h3)
    """

    def __init__(self, in_dim: int = 96, hidden_1: int = 256,
                 hidden_2: int = 128, out_dim: int = 50):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# Full SimCLR model (encoder + projection head)
# ---------------------------------------------------------------------------

class SimCLRModel(nn.Module):
    """
    Encoder + projection head used during contrastive pre-training.
    After pre-training, use encoder alone for downstream tasks.
    """

    def __init__(self, in_channels: int = 3, dropout: float = 0.1,
                 hidden_1: int = 256, hidden_2: int = 128, out_dim: int = 50):
        super().__init__()
        self.encoder = BaseEncoder(in_channels=in_channels, dropout=dropout)
        self.head = ProjectionHead(
            in_dim=96,
            hidden_1=hidden_1,
            hidden_2=hidden_2,
            out_dim=out_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


# ---------------------------------------------------------------------------
# Downstream: linear evaluation head
# ---------------------------------------------------------------------------

class LinearClassifier(nn.Module):
    """
    Frozen encoder + single linear layer for linear evaluation.

    Parameters
    ----------
    encoder : BaseEncoder
        Pre-trained encoder (will be frozen).
    num_classes : int
        Number of activity classes.
    """

    def __init__(self, encoder: BaseEncoder, num_classes: int):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.encoder(x)
        return self.fc(z)  # raw logits


# ---------------------------------------------------------------------------
# Downstream: full fine-tuning classification head
# ---------------------------------------------------------------------------

class FullClassifier(nn.Module):
    """
    Partial fine-tuning: encoder layers up to `freeze_until` are frozen,
    the rest + a 2-layer head are trainable.

    Architecture: encoder -> Dense(1024, ReLU) -> Dense(num_classes) -> Softmax
    """

    def __init__(self, encoder: BaseEncoder, num_classes: int,
                 freeze_until: int = 5):
        """
        Parameters
        ----------
        encoder : BaseEncoder
        num_classes : int
        freeze_until : int
            Freeze the first `freeze_until` children of encoder.encoder
            (0-indexed). Mirrors `last_freeze_layer` in the TF version.
        """
        super().__init__()
        self.encoder = encoder

        # Freeze selected layers
        children = list(self.encoder.encoder.children())
        for i, child in enumerate(children):
            for p in child.parameters():
                p.requires_grad = i > freeze_until

        self.head = nn.Sequential(
            nn.Linear(96, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z)


# ---------------------------------------------------------------------------
# Multi-limb wrappers (early fusion & per-limb)
# ---------------------------------------------------------------------------

class MultiLimbEarlyFusionLinear(nn.Module):
    """
    Early fusion linear model for multi-limb data.

    Input:  (B, T, L, C)  — e.g. (B, 100, 8, 3)
    Steps:
        1. Reshape to (B*L, T, C)
        2. Encode with frozen shared encoder -> (B*L, D)
        3. Reshape to (B, L*D)
        4. Single linear head -> (B, num_classes)
    """

    def __init__(self, encoder: BaseEncoder, num_limbs: int,
                 num_classes: int, embed_dim: int = 96):
        super().__init__()
        self.encoder = encoder
        self.num_limbs = num_limbs
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(num_limbs * embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, L, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * L, T, C)  # (B*L, T, C)
        with torch.no_grad():
            z = self.encoder(x)                           # (B*L, D)
        z = z.reshape(B, self.num_limbs * z.shape[-1])   # (B, L*D)
        return self.fc(z)


class MultiLimbEarlyFusionFull(nn.Module):
    """
    Early fusion full fine-tuning model for multi-limb data.

    Input:  (B, T, L, C)
    Output: (B, num_classes) — softmax probabilities
    """

    def __init__(self, encoder: BaseEncoder, num_limbs: int,
                 num_classes: int, embed_dim: int = 96,
                 freeze_until: int = 5):
        super().__init__()
        self.encoder = encoder
        self.num_limbs = num_limbs

        children = list(self.encoder.encoder.children())
        for i, child in enumerate(children):
            for p in child.parameters():
                p.requires_grad = i > freeze_until

        self.head = nn.Sequential(
            nn.Linear(num_limbs * embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, L, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * L, T, C)
        z = self.encoder(x)                              # (B*L, D)
        z = z.reshape(B, self.num_limbs * z.shape[-1])  # (B, L*D)
        return self.head(z)  # raw logits — CrossEntropyLoss applies softmax internally