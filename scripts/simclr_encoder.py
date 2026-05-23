"""
simclr_encoder.py

Frozen SimCLR encoder wrapper + multi-sensor fusion.

Single sensor:  (N, 100, 3) → encoder → (N, 96)
Multi-sensor:   (N, n_sensors, 100, 3) → encoder (shared) → (N, n_sensors * 96)

The encoder is always frozen — only classifiers train on top.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


ENCODER_DIM = 96  # output dim of SimCLR encoder


class SimCLREncoder(nn.Module):
    """
    Frozen SimCLR 1D CNN encoder.
    Input:  (N, 100, 3)
    Output: (N, 96)
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=24),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=16),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=8),
            nn.ReLU(),
        )

    def forward(self, x):
        """x: (N, 100, 3) → (N, 96)"""
        x = x.permute(0, 2, 1)     # (N, 3, 100)
        x = self.encoder(x)         # (N, 96, T')
        x = x.mean(dim=-1)          # (N, 96)
        return x


def load_simclr_encoder(ckpt_path, device):
    """Load pretrained SimCLR encoder, frozen."""
    ckpt    = torch.load(ckpt_path, map_location=device)
    enc     = SimCLREncoder().to(device)

    key_map = {
        'encoder.encoder.0.weight': 'encoder.0.weight',
        'encoder.encoder.0.bias':   'encoder.0.bias',
        'encoder.encoder.3.weight': 'encoder.2.weight',
        'encoder.encoder.3.bias':   'encoder.2.bias',
        'encoder.encoder.6.weight': 'encoder.4.weight',
        'encoder.encoder.6.bias':   'encoder.4.bias',
    }
    enc_state = {key_map[k]: v for k, v in ckpt.items() if k in key_map}
    enc.load_state_dict(enc_state)

    for param in enc.parameters():
        param.requires_grad = False

    enc.eval()
    print(f"Loaded SimCLR encoder from {ckpt_path} | frozen | dim={ENCODER_DIM}")
    return enc


def encode_sensors(encoder, x, device):
    """
    Encode multi-sensor input.

    Args:
        encoder : SimCLREncoder (frozen)
        x       : (N, n_sensors, 100, 3)

    Returns: (N, n_sensors * 96)
    """
    N, S, T, C = x.shape
    x_flat = x.reshape(N * S, T, C)                  # (N*S, 100, 3)
    with torch.no_grad():
        z = encoder(x_flat.to(device))                # (N*S, 96)
    z = z.reshape(N, S * ENCODER_DIM)                 # (N, S*96)
    return z
