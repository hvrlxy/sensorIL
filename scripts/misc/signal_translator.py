"""
signal_translator.py
====================
Conditional 1D U-Net GAN for cross-sensor signal translation.
Translates (wrist, ankle) → thigh in normalized signal space.

Architecture
------------
Generator  : 1D U-Net with per-axis encoding, cross-axis bottleneck,
             transformer for long-range temporal context, skip connections
Discriminator: 1D PatchGAN on (wrist, ankle, thigh) triplets

Normalization
-------------
Per-sample per-axis z-score: (x - mean(T)) / (std(T) + eps)
Applied independently to each window/stream/axis before the model.
The model learns shape only — scaling is separate.

Integration
-----------
Drop-in replacement for impute_missing_streams():
    from signal_translator import SignalTranslator, impute_with_translator
    X_full = impute_with_translator(translator, X_known, known_indices, n_streams_total)

Training inside run_encoder_fraction_sweep:
    translator = train_translator(X_frac, known_stream_indices=[0,1], target_idx=2)
    state["translator"] = translator
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_sample(x: torch.Tensor, eps: float = 1e-6):
    """
    Per-sample per-axis z-score normalization.
    x: (B, T, C) → x_norm: (B, T, C), mean: (B, 1, C), std: (B, 1, C)
    """
    mean = x.mean(dim=1, keepdim=True)
    std  = x.std(dim=1, keepdim=True).clamp(min=eps)
    return (x - mean) / std, mean, std


def denormalize_sample(x_norm, mean, std):
    """Inverse of normalize_sample."""
    return x_norm * std + mean


# ─────────────────────────────────────────────────────────────────────────────
# GENERATOR: 1D U-Net
# ─────────────────────────────────────────────────────────────────────────────

class AxisEncoder(nn.Module):
    """Encodes one axis of one stream: (B, T) → list of feature maps."""
    def __init__(self, T=100, base_ch=32, n_levels=4):
        super().__init__()
        self.levels = nn.ModuleList()
        ch = base_ch
        in_ch = 1
        for i in range(n_levels):
            self.levels.append(nn.Sequential(
                nn.Conv1d(in_ch, ch, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Identity(),
            ))
            in_ch = ch
            ch = min(ch * 2, 256)
        self.out_ch = in_ch

    def forward(self, x):
        """x: (B, T) → features list, each (B, ch, T//2^i)"""
        feats = []
        h = x.unsqueeze(1)   # (B, 1, T)
        for level in self.levels:
            h = level(h)
            feats.append(h)
        return feats


class CrossAxisAttention(nn.Module):
    """
    Combine x/y/z features at bottleneck.
    Each axis contributes — cross-axis attention learns
    e.g. "Y-axis gravity informs X-axis prediction".
    """
    def __init__(self, ch, n_axes=3, n_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, n_heads, batch_first=True,
                                           dropout=0.1)

    def forward(self, axis_feats):
        """
        axis_feats: list of C tensors, each (B, ch, L)
        Returns: list of C tensors, each (B, ch, L)
        """
        B, ch, L = axis_feats[0].shape
        # Stack axes: (B, C*L, ch) — attend across axis×position
        tokens = torch.cat([f.permute(0, 2, 1) for f in axis_feats], dim=1)
        tokens = self.norm(tokens)
        out, _ = self.attn(tokens, tokens, tokens)
        # Split back
        chunks = out.split(L, dim=1)
        return [c.permute(0, 2, 1) for c in chunks]  # list of (B, ch, L)


class TemporalTransformer(nn.Module):
    """Transformer over time at bottleneck for long-range context."""
    def __init__(self, ch, n_heads=4, n_layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=ch, nhead=n_heads, dim_feedforward=ch*4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers,
                                                  norm=nn.LayerNorm(ch))

    def forward(self, x):
        """x: (B, ch, L) → (B, ch, L)"""
        x = x.permute(0, 2, 1)     # (B, L, ch)
        x = self.transformer(x)
        return x.permute(0, 2, 1)  # (B, ch, L)


class AxisDecoder(nn.Module):
    """Decodes one axis: bottleneck + skip connections → (B, T)"""
    def __init__(self, T=100, base_ch=32, n_levels=4, n_skip_streams=2):
        super().__init__()
        # Channel sizes mirror encoder
        ch_list = [min(base_ch * (2**i), 256) for i in range(n_levels)]
        ch_list = ch_list[::-1]   # [256, 128, 64, 32] descending

        self.ups = nn.ModuleList()
        for i, ch in enumerate(ch_list):
            in_ch  = ch_list[i-1] if i > 0 else ch_list[0]
            # Skip connections from all known streams + target encoding
            skip_ch = ch * n_skip_streams
            self.ups.append(nn.Sequential(
                nn.ConvTranspose1d(in_ch + skip_ch, ch,
                                   kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Identity(),
            ))

        self.final = nn.Sequential(
            nn.Conv1d(ch_list[-1], 1, kernel_size=7, padding=3),
            nn.Tanh()
        )
        self.T = T

    def forward(self, bottleneck, skip_list):
        """
        bottleneck: (B, ch, L_bot)
        skip_list : list of tensors per level, each (B, ch*n_streams, L_i)
        """
        h = bottleneck
        for i, (up, skip) in enumerate(zip(self.ups, skip_list)):
            # Resize skip to match h's spatial size before concat
            if skip.shape[2] != h.shape[2]:
                skip = F.adaptive_avg_pool1d(skip, h.shape[2])
            h = torch.cat([h, skip], dim=1)
            h = up(h)
        h = self.final(h)
        h = F.interpolate(h, size=self.T, mode='linear', align_corners=False)
        return h.squeeze(1)


class UNetGenerator(nn.Module):
    """
    1D U-Net: (wrist, ankle) → thigh, all normalized.
    Simple, explicit channel tracking to avoid size bugs.
    Per-axis processing with cross-axis attention at bottleneck.
    """
    def __init__(self, T=100, C=3, n_known=2, base_ch=32, n_levels=4):
        super().__init__()
        self.T = T; self.C = C; self.n_known = n_known
        self.n_levels = n_levels

        # ch[i] = channels at encoder level i
        chs = [min(base_ch * (2**i), 256) for i in range(n_levels)]
        self.chs = chs
        bot_ch = chs[-1] * n_known   # bottleneck after stream concat

        # Encoder: one ModuleList per (stream, axis, level)
        self.enc = nn.ModuleList()
        for s in range(n_known):
            for c in range(C):
                mods = nn.ModuleList()
                ich = 1
                for lvl in range(n_levels):
                    och = chs[lvl]
                    mods.append(nn.Sequential(
                        nn.Conv1d(ich, och, 4, stride=2, padding=1),
                        nn.LeakyReLU(0.2, inplace=False),
                        nn.Dropout(0.1),
                        nn.Identity(),
                    ))
                    ich = och
                self.enc.append(mods)   # index: s*C + c

        # Bottleneck: lightweight cross-axis mixing + temporal conv
        # Cross-axis: 1x1 conv over concatenated axes (B, bot_ch*C, L) -> same
        self.cross_axis_mix = nn.Sequential(
            nn.Conv1d(bot_ch * C, bot_ch * C, kernel_size=1, groups=C),
            nn.GELU(),
            nn.Conv1d(bot_ch * C, bot_ch * C, kernel_size=1),
        )
        # Temporal: dilated conv per axis for long-range context (cheap)
        self.temporal = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(bot_ch, bot_ch, kernel_size=3, padding=2, dilation=2),
                nn.GELU(),
                nn.Conv1d(bot_ch, bot_ch, kernel_size=3, padding=4, dilation=4),
                nn.GELU(),
            )
            for _ in range(C)
        ])

        # Decoder: one ModuleList per axis
        # Level 0: bot_ch → chs[-2]  (no skip)
        # Level k>0: prev_out + chs[n_levels-1-k]*n_known → chs[n_levels-2-k or base_ch]
        self.dec = nn.ModuleList()
        for c in range(C):
            mods = nn.ModuleList()
            ich = bot_ch
            for lvl in range(n_levels):
                och = chs[n_levels-2-lvl] if lvl < n_levels-1 else base_ch
                skip_ch = chs[n_levels-1-lvl] * n_known if lvl > 0 else 0
                mods.append(nn.Sequential(
                    nn.ConvTranspose1d(ich + skip_ch, och, 4, stride=2, padding=1),
                    nn.ReLU(inplace=False),
                    nn.Dropout(0.3),
                    nn.Identity(),
                ))
                ich = och
            self.dec.append(mods)

        self.out_conv = nn.ModuleList([
            nn.Sequential(nn.Conv1d(base_ch, 1, 7, padding=3), nn.Tanh())
            for _ in range(C)
        ])

        # DC and scale heads — take both bottleneck features AND actual
        # input signal statistics so predictions vary per sample
        # Input: concat(bottleneck_pooled, known_dc_flat, known_std_flat)
        # known_dc and known_std are (n_known * C) each → total extra = 2*n_known*C
        extra_dim = 2 * n_known * C
        self.dc_head = nn.Sequential(
            nn.Linear(bot_ch * C + extra_dim, bot_ch), nn.GELU(),
            nn.Linear(bot_ch, bot_ch // 2), nn.GELU(),
            nn.Linear(bot_ch // 2, C),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(bot_ch * C + extra_dim, bot_ch), nn.GELU(),
            nn.Linear(bot_ch, bot_ch // 2), nn.GELU(),
            nn.Linear(bot_ch // 2, C),
            nn.Softplus(),
        )

    def _enc_idx(self, s, c):
        return s * self.C + c

    def encode(self, X):
        """Extract bottleneck embedding from known streams.
        X: (B, T, n_known, C) normalized
        Returns: (B, bot_ch*C) bottleneck pooled embedding
        """
        B, T, n_known, C = X.shape
        feats = {}
        for s in range(n_known):
            for c in range(C):
                h = X[:, :, s, c:c+1].permute(0, 2, 1)
                for mod in self.enc[self._enc_idx(s, c)]:
                    h = mod(h)
                feats[(s, c)] = h

        # Match forward() exactly: concat streams per axis
        bot = [torch.cat([feats[(s, c)] for s in range(n_known)], dim=1)
               for c in range(C)]
        stacked = torch.cat(bot, dim=1)
        stacked = self.cross_axis_mix(stacked)
        bot     = list(stacked.chunk(C, dim=1))
        bot     = [bot[c] + self.temporal[c](bot[c]) for c in range(C)]
        return torch.cat([b.mean(dim=-1) for b in bot], dim=1)  # (B, bot_ch*C)

    def forward(self, X, return_stats=False):
        """
        X: (B, T, n_known, C) → (B, T, C) reconstructed signal (denormalized)
        If return_stats=True, also returns (dc, scale) each (B, C)
        """
        B = X.shape[0]

        # Encode all streams × axes, store feature maps
        feats = {}
        for s in range(self.n_known):
            for c in range(self.C):
                h = X[:, :, s, c].unsqueeze(1)
                for lvl, mod in enumerate(self.enc[self._enc_idx(s, c)]):
                    h = mod(h)
                    feats[(s, c, lvl)] = h

        # Bottleneck per axis: concat streams
        bot = [torch.cat([feats[(s, c, self.n_levels-1)]
                          for s in range(self.n_known)], dim=1)
               for c in range(self.C)]

        # Cross-axis mixing
        stacked = torch.cat(bot, dim=1)
        stacked = self.cross_axis_mix(stacked)
        bot = list(stacked.chunk(self.C, dim=1))

        # Temporal context
        bot = [bot[c] + self.temporal[c](bot[c]) for c in range(self.C)]

        # ── DC and scale prediction ───────────────────────────────────────
        bot_pooled = torch.cat(
            [b.mean(dim=-1) for b in bot], dim=1)        # (B, bot_ch*C)

        # Default: use zero dc, unit std as placeholders
        # Real values injected via _known_dc/_known_std set before forward
        B = X.shape[0]
        known_dc_flat  = getattr(self, '_known_dc',
            torch.zeros(B, self.n_known * self.C, device=X.device))
        known_std_flat = getattr(self, '_known_std',
            torch.ones(B,  self.n_known * self.C, device=X.device))
        # Ensure correct batch size (in case stale value from prev batch)
        if known_dc_flat.shape[0] != B:
            known_dc_flat  = torch.zeros(B, self.n_known * self.C, device=X.device)
            known_std_flat = torch.ones(B,  self.n_known * self.C, device=X.device)

        head_input = torch.cat([bot_pooled, known_dc_flat, known_std_flat], dim=1)
        pred_dc    = self.dc_head(head_input)             # (B, C)
        pred_scale = self.scale_head(head_input)          # (B, C)

        # ── Shape decoder ────────────────────────────────────────────────────
        out = []
        for c in range(self.C):
            h = bot[c]
            for lvl in range(self.n_levels):
                if lvl > 0:
                    enc_lvl = self.n_levels - 1 - lvl
                    skip = torch.cat([feats[(s, c, enc_lvl)]
                                      for s in range(self.n_known)], dim=1)
                    if skip.shape[2] != h.shape[2]:
                        skip = F.adaptive_avg_pool1d(skip, h.shape[2])
                    h = torch.cat([h, skip], dim=1)
                h = self.dec[c][lvl](h)
            h = self.out_conv[c](h)
            h = F.interpolate(h, size=self.T, mode='linear', align_corners=False)
            out.append(h.squeeze(1))

        shape_norm = torch.stack(out, dim=-1)   # (B, T, C)

        # Force unit-variance shape
        shape_dc  = shape_norm.mean(dim=1, keepdim=True)
        shape_dyn = shape_norm - shape_dc
        shape_std = shape_dyn.std(dim=1, keepdim=True).clamp(min=1e-3)
        shape_norm = shape_dyn / shape_std

        # Clamp pred_scale
        known_std_mean = known_std_flat.view(B, self.n_known, self.C).mean(dim=1)
        pred_scale = pred_scale.clamp(min=known_std_mean * 0.1)

        reconstructed = shape_norm * pred_scale.unsqueeze(1) + pred_dc.unsqueeze(1)

        if return_stats:
            return reconstructed, pred_dc, pred_scale
        return reconstructed


# ─────────────────────────────────────────────────────────────────────────────
# DISCRIMINATOR: 1D PatchGAN
# ─────────────────────────────────────────────────────────────────────────────

class PatchDiscriminator(nn.Module):
    """
    1D PatchGAN discriminator with feature extraction for feature matching loss.
    """
    def __init__(self, T=100, C=3, n_streams=3, base_ch=64):
        super().__init__()
        in_ch = n_streams * C
        # Store layers separately so we can extract intermediate features
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, base_ch, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=False),
            ),
            nn.Sequential(
                nn.utils.spectral_norm(
                    nn.Conv1d(base_ch, base_ch*2, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=False),
            ),
            nn.Sequential(
                nn.Conv1d(base_ch*2, 1, kernel_size=4, stride=1, padding=1),
            ),
        ])

    def forward(self, X_known_norm, X_target_norm):
        """Returns (score, feature_list) where feature_list is intermediate activations."""
        B, T, n_known, C = X_known_norm.shape
        known_flat  = X_known_norm.reshape(B, T, n_known * C)
        x = torch.cat([known_flat, X_target_norm], dim=-1).permute(0, 2, 1)
        feats = []
        for layer in self.layers[:-1]:
            x = layer(x)
            feats.append(x)
        score = self.layers[-1](x).squeeze(1)
        return score, feats


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL TRANSLATOR (wraps G + D)
# ─────────────────────────────────────────────────────────────────────────────

class SignalTranslator(nn.Module):
    def __init__(self, T=100, C=3, n_known=2, base_ch=32, n_levels=4):
        super().__init__()
        self.T = T
        self.C = C
        self.n_known = n_known
        self.generator     = UNetGenerator(T, C, n_known, base_ch, n_levels)
        self.discriminator = PatchDiscriminator(T, C, n_known+1, base_ch*2)

    def generate(self, X_known_norm):
        """X_known_norm: (B, T, n_known, C) → (B, T, C) normalized thigh"""
        return self.generator(X_known_norm)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_translator(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    known_stream_indices: list,
    target_stream_idx: int,
    save_path: str,
    T: int = 100, C: int = 3,
    epochs: int = 50,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    batch_size: int = 512,
    lambda_l1: float = 10.0,
    early_stopping_patience: int = 10,
    viz_dir: str = None,
    viz_every: int = 10,
    viz_samples: np.ndarray = None,
    encoders: dict = None,           # SimCLR encoders for DC prediction
    stream_names: list = None,       # names of streams in X_train
    stream_to_encoder: dict = None,  # stream name → encoder key
) -> 'SignalTranslator':
    """
    Train conditional GAN translator.

    X_train: (N, T, S, C) with known + target streams
    known_stream_indices: e.g. [0, 1] for wrist + ankle
    target_stream_idx: e.g. 2 for thigh
    """
    n_known = len(known_stream_indices)
    N = len(X_train)

    print(f"  [Translator] N={N}  known={known_stream_indices}"
          f"  target={target_stream_idx}  epochs={epochs}", flush=True)

    # Use all training windows
    X_train_bal = X_train
    print(f"  [Translator] training on {len(X_train_bal)} windows (all)", flush=True)

    # ── Automatic proxy stream selection ─────────────────────────────────
    # Find which known streams best correlate with target in amplitude and shape
    # Use multiple proxy streams weighted by their per-axis correlation
    tgt_dyn = X_train_bal[:, :, target_stream_idx, :] - \
              X_train_bal[:, :, target_stream_idx, :].mean(axis=1, keepdims=True)  # (N, T, C)

    proxy_weights = np.zeros((n_known, C), dtype=np.float32)  # (n_known, C) per-axis weights
    proxy_std_ratios = np.zeros((n_known, C), dtype=np.float32)

    for k, ki in enumerate(known_stream_indices):
        kn_dyn = X_train_bal[:, :, ki, :] - \
                 X_train_bal[:, :, ki, :].mean(axis=1, keepdims=True)   # (N, T, C)
        for c in range(C):
            # Pearson correlation per window, take median
            tgt_c = tgt_dyn[:, :, c]   # (N, T)
            kn_c  = kn_dyn[:, :, c]
            # Vectorized correlation: (N,)
            tgt_std = tgt_c.std(axis=1).clip(1e-6)
            kn_std  = kn_c.std(axis=1).clip(1e-6)
            corr    = ((tgt_c * kn_c).mean(axis=1)) / (tgt_std * kn_std)
            proxy_weights[k, c]    = float(np.median(np.abs(corr)))
            proxy_std_ratios[k, c] = float(np.median(
                tgt_c.std(axis=1) / kn_c.std(axis=1).clip(1e-6)))

    # Normalize weights per axis so they sum to 1
    w_sum = proxy_weights.sum(axis=0, keepdims=True).clip(1e-6)
    proxy_weights_norm = proxy_weights / w_sum   # (n_known, C)

    print(f"  [Translator] proxy weights per axis (n_known x C):", flush=True)
    for k, ki in enumerate(known_stream_indices):
        name = stream_names[k] if (stream_names and k < len(stream_names)) else f"stream_{ki}"
        print(f"    {name}: corr={proxy_weights[k].round(3)}  "
              f"std_ratio={proxy_std_ratios[k].round(3)}", flush=True)

    # ── Fit DC predictor: SimCLR embeddings → thigh DC ───────────────────
    print(f"  [DC MLP] encoders={encoders is not None}  stream_names={stream_names}  "
          f"stream_to_encoder={stream_to_encoder is not None}", flush=True)
    dc_mlp = None
    dc_mlp_state = None
    if encoders is not None and stream_names is not None and stream_to_encoder is not None:
        print(f"  [Translator] fitting SimCLR DC predictor...", flush=True)
        try:
            from scripts.misc.encoder import extract_all_features
            known_names = [stream_names[i] for i in range(len(known_stream_indices))]
            Z_known = extract_all_features(
                X_train_bal[:, :, known_stream_indices, :], encoders,
                stream_to_encoder, known_names, batch_size=512)  # (N, n_k, embed_dim)
            Z_flat  = Z_known.reshape(len(Z_known), -1).astype(np.float32)
            thigh_dc_flat = X_train_bal[:, :, target_stream_idx, :].mean(axis=1).astype(np.float32)

            embed_dim_dc = Z_flat.shape[1]
            dc_mlp = nn.Sequential(
                nn.Linear(embed_dim_dc, 128), nn.GELU(),
                nn.Linear(128, 64),           nn.GELU(),
                nn.Linear(64, C),
            ).to(DEVICE)

            opt_dc = torch.optim.Adam(dc_mlp.parameters(), lr=1e-3)
            Z_t  = torch.from_numpy(Z_flat).to(DEVICE)
            dc_t = torch.from_numpy(thigh_dc_flat).to(DEVICE)
            for ep in range(200):
                perm = torch.randperm(len(Z_t))
                loss_sum, nb = 0.0, 0
                for i in range(0, len(Z_t), 512):
                    idx = perm[i:i+512]
                    pred = dc_mlp(Z_t[idx])
                    loss = F.mse_loss(pred, dc_t[idx])
                    opt_dc.zero_grad(); loss.backward(); opt_dc.step()
                    loss_sum += loss.item(); nb += 1
                if (ep+1) % 50 == 0:
                    print(f"  [DC MLP] ep={ep+1}  loss={loss_sum/nb:.4f}", flush=True)

            dc_mlp.eval()
            with torch.no_grad():
                pred_all = dc_mlp(Z_t).cpu().numpy()
            r2 = 1 - np.mean((pred_all - thigh_dc_flat)**2) / (thigh_dc_flat.var() + 1e-8)
            print(f"  [DC MLP] R²={r2:.4f}  embed_dim={embed_dim_dc}", flush=True)
            dc_mlp_state = {"state_dict": dc_mlp.state_dict(),
                            "embed_dim": embed_dim_dc, "C": C}
        except Exception as e:
            import traceback
            print(f"  [DC MLP] failed: {e}", flush=True)
            traceback.print_exc()
            dc_mlp = None

    if dc_mlp is None:
        # Fallback: Ridge on raw DC values
        from sklearn.linear_model import Ridge
        known_dc_flat = X_train_bal[:, :, known_stream_indices, :].mean(axis=1).reshape(len(X_train_bal), -1)
        thigh_dc_flat = X_train_bal[:, :, target_stream_idx, :].mean(axis=1)
        dc_ridge = Ridge(alpha=1.0).fit(known_dc_flat, thigh_dc_flat)
        print(f"  [Translator] DC Ridge R²={dc_ridge.score(known_dc_flat, thigh_dc_flat):.4f}", flush=True)
    else:
        dc_ridge = None

    # ── Fit amplitude predictor: SimCLR embeddings → target std ──────────
    amp_mlp = None
    amp_mlp_state = None
    if encoders is not None and stream_names is not None and stream_to_encoder is not None:
        print(f"  [Translator] fitting SimCLR amplitude predictor...", flush=True)
        try:
            from scripts.misc.encoder import extract_all_features
            known_names = [stream_names[i] for i in range(len(known_stream_indices))]
            Z_known_amp = extract_all_features(
                X_train_bal[:, :, known_stream_indices, :], encoders,
                stream_to_encoder, known_names, batch_size=512)
            Z_flat_amp = Z_known_amp.reshape(len(Z_known_amp), -1).astype(np.float32)

            # Target: per-window std of target stream (B, C)
            tgt_dyn_amp = X_train_bal[:, :, target_stream_idx, :] - \
                          X_train_bal[:, :, target_stream_idx, :].mean(axis=1, keepdims=True)
            tgt_std_flat = tgt_dyn_amp.std(axis=1).astype(np.float32)  # (N, C)

            embed_dim_amp = Z_flat_amp.shape[1]
            amp_mlp = nn.Sequential(
                nn.Linear(embed_dim_amp, 128), nn.GELU(),
                nn.Linear(128, 64),            nn.GELU(),
                nn.Linear(64, C),
                nn.Softplus(),  # amplitude must be positive
            ).to(DEVICE)

            opt_amp = torch.optim.Adam(amp_mlp.parameters(), lr=1e-3)
            Z_amp_t   = torch.from_numpy(Z_flat_amp).to(DEVICE)
            std_amp_t = torch.from_numpy(tgt_std_flat).to(DEVICE)
            for ep in range(200):
                perm = torch.randperm(len(Z_amp_t))
                loss_sum, nb = 0.0, 0
                for i in range(0, len(Z_amp_t), 512):
                    idx = perm[i:i+512]
                    pred = amp_mlp(Z_amp_t[idx])
                    loss = F.mse_loss(torch.log(pred + 1e-6),
                                      torch.log(std_amp_t[idx] + 1e-6))
                    opt_amp.zero_grad(); loss.backward(); opt_amp.step()
                    loss_sum += loss.item(); nb += 1
                if (ep+1) % 50 == 0:
                    print(f"  [Amp MLP] ep={ep+1}  loss={loss_sum/nb:.4f}", flush=True)

            amp_mlp.eval()
            with torch.no_grad():
                pred_amp_all = amp_mlp(Z_amp_t).cpu().numpy()
            r2_amp = 1 - np.mean((np.log(pred_amp_all + 1e-6) -
                                   np.log(tgt_std_flat + 1e-6))**2) / \
                         (np.log(tgt_std_flat + 1e-6).var() + 1e-8)
            print(f"  [Amp MLP] R²={r2_amp:.4f}  embed_dim={embed_dim_amp}", flush=True)
            amp_mlp_state = {"state_dict": amp_mlp.state_dict(),
                             "embed_dim": embed_dim_amp, "C": C}
        except Exception as e:
            import traceback
            print(f"  [Amp MLP] failed: {e}", flush=True)
            traceback.print_exc()
            amp_mlp = None

    T = X_train.shape[1]
    C = X_train.shape[3]
    translator = SignalTranslator(T=T, C=C, n_known=n_known, base_ch=32, n_levels=2).to(DEVICE)
    translator.dc_ridge_coef    = dc_ridge.coef_.astype(np.float32) if dc_ridge else None
    translator.dc_ridge_bias    = dc_ridge.intercept_.astype(np.float32) if dc_ridge else None
    translator.dc_mlp           = dc_mlp
    translator.dc_mlp_state     = dc_mlp_state
    translator.amp_mlp          = amp_mlp
    translator.amp_mlp_state    = amp_mlp_state
    translator.proxy_weights    = proxy_weights_norm.astype(np.float32)
    translator.proxy_std_ratios = proxy_std_ratios.astype(np.float32)
    G = translator.generator
    D = translator.discriminator

    # Learnable proxy alpha: (n_known, C) — learned importance of each stream per axis
    # Clamp weights before log to avoid -inf
    proxy_alpha = nn.Parameter(
        torch.from_numpy(np.log(proxy_weights_norm.clip(1e-3) + 1e-6).astype(np.float32)).to(DEVICE)
    )

    # Fixed std ratio from training data — NOT learned (prevents collapse to zero)
    proxy_log_sr = torch.from_numpy(
        np.log(proxy_std_ratios.clip(1e-3)).astype(np.float32)).to(DEVICE)

    for p in D.parameters():
        p.requires_grad = False
    D.eval()

    opt_G = torch.optim.Adam(
        list(G.parameters()),
        lr=lr_g, betas=(0.5, 0.999), weight_decay=5e-3)
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_G, T_max=max(1, epochs // 2), eta_min=lr_g*0.05)

    # Frozen SimCLR encoder for target stream — used for embedding loss
    target_encoder = None
    if encoders is not None and stream_names is not None and stream_to_encoder is not None:
        # stream_names = known_names + [target_name], target is last
        target_name_str = stream_names[-1] if stream_names else None
        enc_key = stream_to_encoder.get(target_name_str) if target_name_str else None
        if enc_key and enc_key in encoders:
            target_encoder = encoders[enc_key]._encoder.to(DEVICE)
            for p in target_encoder.parameters():
                p.requires_grad = False
            target_encoder.eval()
            print(f"  [Translator] embedding loss enabled  "
                  f"target={target_name_str}  encoder={enc_key}", flush=True)
        else:
            print(f"  [Translator] embedding loss disabled  "
                  f"target={target_name_str}  key={enc_key}  "
                  f"available={list(encoders.keys())}", flush=True)

    X_tr_t = torch.from_numpy(X_train_bal.astype(np.float32))
    N      = len(X_tr_t)
    X_vl_t = torch.from_numpy(X_val.astype(np.float32))

    best_val, best_ckpt, patience = float("inf"), None, 0
    history = {"epoch": [], "g_loss": [], "l1_loss": [],
                "dc_loss": [], "feat_loss": [], "val_l1": []}

    # Visualization samples — use provided or fall back to first val windows
    if viz_samples is None:
        viz_samples = X_val[:min(5, len(X_val))]

    if viz_dir:
        Path(viz_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        perm = torch.randperm(N)
        g_loss_sum = 0.0
        l1_sum = dc_sum = feat_sum = embed_sum = 0.0
        nb = 0

        for i in range(0, N, batch_size):
            batch = X_tr_t[perm[i:i+batch_size]].to(DEVICE)
            B_cur = batch.shape[0]

            X_known  = batch[:, :, known_stream_indices, :]
            X_target = batch[:, :, target_stream_idx, :]       # (B, T, C) raw

            # Per-sample per-axis normalization of known streams
            kn_list = []
            for k in range(n_known):
                xk, _, _ = normalize_sample(X_known[:, :, k, :])
                kn_list.append(xk)
            X_known_norm = torch.stack(kn_list, dim=2)

            # Target DC and scale
            target_dc    = X_target.mean(dim=1)                # (B, C)
            target_dyn   = X_target - target_dc.unsqueeze(1)
            target_scale = target_dyn.std(dim=1).clamp(min=1e-6)  # (B, C)

            # ── Train Generator ───────────────────────────────────────────
            G._known_dc  = X_known.mean(dim=1).reshape(B_cur, -1).detach()
            G._known_std = (X_known - X_known.mean(dim=1, keepdim=True)
                            ).std(dim=1).clamp(min=1e-6).reshape(B_cur, -1).detach()
            reconstructed, pred_dc, pred_scale = G(X_known_norm, return_stats=True)

            # Variance weighting
            with torch.no_grad():
                win_var = X_target.var(dim=1).mean(dim=-1)
                upper_clamp = n_known * 20.0
                w = (win_var / (win_var.mean() + 1e-8)).clamp(0.1, upper_clamp)
                w = w / w.mean()

            # ── 1D SSIM loss — best shape fidelity loss ───────────────────
            gen_dc   = reconstructed.mean(dim=1, keepdim=True)
            gen_dyn  = reconstructed - gen_dc
            tgt_dc_  = X_target.mean(dim=1, keepdim=True)
            tgt_dyn_ = X_target - tgt_dc_
            gen_std  = gen_dyn.std(dim=1, keepdim=True).clamp(min=1e-6)
            tgt_std_ = tgt_dyn_.std(dim=1, keepdim=True).clamp(min=1e-6)
            gen_norm = gen_dyn / gen_std
            tgt_norm = tgt_dyn_ / tgt_std_
            # Compares local mean, local variance, and local covariance
            # Flat output → local_std≈0 → SSIM≈0 → loss≈1 (maximum penalty)
            # Scale/DC invariant via local normalization
            def ssim1d(x, y, win=11, C1=0.01**2, C2=0.03**2):
                """1D SSIM per window per axis. x,y: (B, T, C)"""
                x = x.permute(0, 2, 1)  # (B, C, T)
                y = y.permute(0, 2, 1)
                pad = win // 2
                kernel = torch.ones(1, 1, win, device=x.device) / win

                def lmean(z):
                    B_, C_, T_ = z.shape
                    return F.conv1d(z.reshape(B_*C_, 1, T_), kernel,
                                   padding=pad).reshape(B_, C_, T_)

                mu_x  = lmean(x);  mu_y  = lmean(y)
                mu_x2 = lmean(x*x); mu_y2 = lmean(y*y); mu_xy = lmean(x*y)
                # Clamp before sqrt to prevent NaN from floating point negatives
                sig_x  = (mu_x2 - mu_x**2).clamp(min=0).sqrt()
                sig_y  = (mu_y2 - mu_y**2).clamp(min=0).sqrt()
                sig_xy = mu_xy - mu_x * mu_y
                num = (2*mu_x*mu_y + C1) * (2*sig_xy + C2)
                den = (mu_x**2 + mu_y**2 + C1) * (sig_x**2 + sig_y**2 + C2)
                # Clamp den to prevent division by near-zero
                ssim = (num / den.clamp(min=1e-4)).clamp(-1, 1)
                return ssim.permute(0, 2, 1)  # (B, T, C)

            ssim_map = ssim1d(gen_norm, tgt_norm)          # (B, T, C)
            has_var  = (tgt_std_.squeeze(1) > 0.02).float()  # (B, C)
            loss_ssim = ((1.0 - ssim_map.mean(dim=1)) * has_var * w.unsqueeze(-1)
                         ).sum() / (has_var * w.unsqueeze(-1)).sum().clamp(1e-6)

            # Feature matching on normalized shape signals
            with torch.no_grad():
                X_target_norm, _, _ = normalize_sample(X_target)
                _, real_feats = D(X_known_norm, X_target_norm)
            _, fake_feats = D(X_known_norm, gen_norm)
            loss_feat = sum(F.l1_loss(ff, rf.detach())
                            for ff, rf in zip(fake_feats, real_feats))

            loss_dc    = ((pred_dc - target_dc).pow(2).mean(dim=-1) * w).mean()

            # ── Embedding loss ────────────────────────────────────────────────
            loss_embed = torch.tensor(0.0, device=DEVICE)
            if target_encoder is not None:
                with torch.no_grad():
                    z_real = target_encoder(X_target)
                z_gen  = target_encoder(reconstructed)
                loss_embed = F.l1_loss(z_gen, z_real.detach())

            g_loss = (5.0  * loss_ssim
                      + 5.0  * loss_dc
                      + 2.0  * loss_feat
                      + 10.0 * loss_embed)

            if torch.isnan(g_loss) or torch.isinf(g_loss):
                opt_G.zero_grad()
                print(f"  [Translator] Warning: NaN/Inf loss at batch {i}, skipping", flush=True)
                continue

            opt_G.zero_grad(); g_loss.backward()
            # Replace NaN gradients with zero before clipping
            for p in G.parameters():
                if p.grad is not None:
                    p.grad.nan_to_num_(0.0)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_G.step()

            g_loss_sum += g_loss.item()
            l1_sum     += loss_ssim.item()
            dc_sum     += loss_dc.item()
            feat_sum   += loss_feat.item()
            embed_sum  += loss_embed.item()
            nb += 1

        sched_G.step()

        # Validation — correlation on active windows
        G.eval()
        val_l1, nv = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(X_vl_t), batch_size):
                b       = X_vl_t[i:i+batch_size].to(DEVICE)
                X_k     = b[:, :, known_stream_indices, :]
                X_tgt   = b[:, :, target_stream_idx, :]
                kn_list = []
                for k in range(n_known):
                    xk, _, _ = normalize_sample(X_k[:, :, k, :])
                    kn_list.append(xk)
                kn_norm = torch.stack(kn_list, dim=2)
                X_k_raw = b[:, :, known_stream_indices, :]
                G._known_dc  = X_k_raw.mean(dim=1).reshape(b.shape[0], -1)
                G._known_std = (X_k_raw - X_k_raw.mean(dim=1, keepdim=True)
                                ).std(dim=1).clamp(min=1e-6).reshape(b.shape[0], -1)
                recon     = G(kn_norm)
                gen_dyn_v = recon - recon.mean(dim=1, keepdim=True)
                tgt_dyn_v = X_tgt - X_tgt.mean(dim=1, keepdim=True)
                gen_std_v = gen_dyn_v.std(dim=1, keepdim=True).clamp(1e-6)
                tgt_std_v = tgt_dyn_v.std(dim=1, keepdim=True).clamp(1e-6)
                gn = gen_dyn_v / gen_std_v
                tn = tgt_dyn_v / tgt_std_v
                has_var_v = (tgt_std_v.squeeze(1) > 0.02).float()
                if has_var_v.sum() > 0:
                    # Simple correlation as val proxy (fast)
                    corr_v = (gn * tn).mean(dim=1)
                    val_l1 += (1.0 - corr_v[has_var_v.bool().any(dim=-1)]).mean().item()
                nv += 1
        val_l1 /= max(1, nv)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [Translator] epoch={epoch:3d}  "
                  f"G={g_loss_sum/max(1,nb):.4f}  "
                  f"ssim={l1_sum/max(1,nb):.4f}  "
                  f"dc={dc_sum/max(1,nb):.4f}  "
                  f"feat={feat_sum/max(1,nb):.4f}  "
                  f"embed={embed_sum/max(1,nb):.4f}  "
                  f"val_ssim={val_l1:.4f}", flush=True)

        history["epoch"].append(epoch)
        history["g_loss"].append(g_loss_sum / max(1, nb))
        history["l1_loss"].append(l1_sum / max(1, nb))
        history["dc_loss"].append(dc_sum / max(1, nb))
        history["feat_loss"].append(feat_sum / max(1, nb))
        history["val_l1"].append(val_l1)

        # Save loss curve every epoch
        if viz_dir:
            _save_loss_curve(history, os.path.join(viz_dir, "loss_curve.png"))

        # Visualization
        if viz_dir and (epoch % viz_every == 0 or epoch == 1):
            G._ridge_coef   = translator.dc_ridge_coef
            G._ridge_bias   = translator.dc_ridge_bias
            G._proxy_weights    = F.softmax(proxy_alpha, dim=0).detach().cpu().numpy().astype(np.float32)
            G._proxy_std_ratios = torch.exp(proxy_log_sr).detach().cpu().numpy().astype(np.float32)
            # viz_samples has shape (N, T, n_known+1, C) — streams are [0..n_known-1, n_known]
            # Pass sequential indices [0..n_known-1], not FL stream indices
            _visualize_translator(G, viz_samples, list(range(n_known)),
                                  n_known, n_known, T, C,
                                  os.path.join(viz_dir, f"epoch_{epoch:03d}.png"),
                                  title=f"epoch {epoch}")

        # Save best checkpoint based on val L1 but don't early stop —
        # GAN val L1 often increases as generator trades reconstruction
        # for realism. Run all epochs and use best checkpoint.
        if val_l1 < best_val:
            best_val  = val_l1
            best_ckpt = {k: v.cpu().clone() for k,v in translator.state_dict().items()}


    if best_ckpt:
        translator.load_state_dict(best_ckpt)
    print(f"  [Translator] Done  best_val={best_val:.4f}")

    # Store learned proxy weights (after training)
    learned_pw = F.softmax(proxy_alpha, dim=0).detach().cpu().numpy()
    learned_sr = torch.exp(proxy_log_sr).cpu().numpy()
    translator.proxy_weights    = learned_pw.astype(np.float32)
    translator.proxy_std_ratios = learned_sr.astype(np.float32)
    print(f"  [Translator] Learned proxy weights (sr fixed from data):", flush=True)
    for k, ki in enumerate(known_stream_indices):
        name = stream_names[k] if (stream_names and k < len(stream_names)) else f"stream_{ki}"
        print(f"    {name}: weight={learned_pw[k].round(3)}  "
              f"sr={learned_sr[k].round(3)}", flush=True)

    torch.save({
        "state_dict":           translator.state_dict(),
        "T": T, "C": C, "n_known": n_known,
        "base_ch": 32, "n_levels": 2,
        "known_stream_indices": known_stream_indices,
        "target_stream_idx":    target_stream_idx,
        "proxy_weights":        proxy_weights_norm.astype(np.float32),
        "proxy_std_ratios":     proxy_std_ratios.astype(np.float32),
        "dc_ridge_coef":        translator.dc_ridge_coef,
        "dc_ridge_bias":        translator.dc_ridge_bias,
        "dc_mlp_state":         dc_mlp_state,
        "amp_mlp_state":        amp_mlp_state,
    }, save_path)
    return translator


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _save_loss_curve(history, out_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = history["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(ep, history["g_loss"],    label="Total G",   color="#2196F3", lw=2)
    ax1.plot(ep, history["l1_loss"],   label="SSIM",      color="#4CAF50", lw=1.2, ls="--")
    ax1.plot(ep, history["dc_loss"],   label="DC",        color="#E53935", lw=1.2, ls="--")
    ax1.plot(ep, history["feat_loss"], label="Feat match",color="#9C27B0", lw=1.2, ls="--")
    ax1.set_xlabel("Epoch"); ax1.set_title("Generator Loss Components")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.plot(ep, history["val_l1"], label="Val L1 (raw)", color="#4CAF50")
    ax2.set_xlabel("Epoch"); ax2.set_title("Validation L1 (raw signal)")
    ax2.legend(); ax2.grid(alpha=0.3)
    if history["val_l1"]:
        best_ep = ep[history["val_l1"].index(min(history["val_l1"]))]
        ax2.axvline(best_ep, color="gray", ls="--", alpha=0.7, label=f"Best ep {best_ep}")
        ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def _visualize_translator(G, X_samples, known_indices, target_idx,
                           n_known, T, C, out_path, title=""):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_samples = min(5, len(X_samples))
    G.eval()
    with torch.no_grad():
        batch = torch.from_numpy(
            X_samples[:n_samples].astype(np.float32)).to(DEVICE)
        X_k   = batch[:, :, known_indices, :]
        X_tgt = batch[:, :, target_idx, :]
        kn_list = []
        for k in range(n_known):
            xk, _, _ = normalize_sample(X_k[:, :, k, :])
            kn_list.append(xk)
        kn_norm = torch.stack(kn_list, dim=2)
        B_viz = batch.shape[0]
        G._known_dc  = X_k.mean(dim=1).reshape(B_viz, -1)
        G._known_std = (X_k - X_k.mean(dim=1, keepdim=True)
                        ).std(dim=1).clamp(min=1e-6).reshape(B_viz, -1)
        fake_norm = G(kn_norm)   # (B, T, C)

        # Apply full inference path: normalize shape, ankle scale, Ridge DC
        fake_dc  = fake_norm.mean(dim=1, keepdim=True)
        fake_dyn = fake_norm - fake_dc
        fake_std = fake_dyn.std(dim=1, keepdim=True).clamp(min=1e-6)
        shape    = fake_dyn / fake_std   # unit variance shape

        # Weighted proxy for dynamic component
        # Shape: correlation-weighted mean of normalized streams (pw sums to 1)
        # Amplitude: weighted mean of (stream_std * std_ratio) — target amplitude estimate
        pw = getattr(G, '_proxy_weights', None)
        sr = getattr(G, '_proxy_std_ratios', None)
        if pw is not None:
            pw_t = torch.from_numpy(pw).to(DEVICE)
            sr_t = torch.from_numpy(sr).to(DEVICE)
            proxy_shape = torch.zeros(B_viz, T, C, device=DEVICE)
            proxy_amp   = torch.zeros(B_viz, C, device=DEVICE)
            for k in range(n_known):
                kn_raw  = X_k[:, :, k, :]
                kn_dyn  = kn_raw - kn_raw.mean(dim=1, keepdim=True)
                kn_std  = kn_dyn.std(dim=1).clamp(min=1e-6)            # (B, C)
                kn_norm = kn_dyn / kn_std.unsqueeze(1)                 # unit variance
                proxy_shape = proxy_shape + pw_t[k].unsqueeze(0).unsqueeze(0) * kn_norm
                proxy_amp   = proxy_amp   + pw_t[k].unsqueeze(0) * kn_std * sr_t[k].unsqueeze(0)
            proxy_dyn = proxy_shape * proxy_amp.unsqueeze(1)
            proxy_std = proxy_amp.unsqueeze(1)
        else:
            proxy_dyn = X_k[:, :, 0, :] - X_k[:, :, 0, :].mean(dim=1, keepdim=True)
            proxy_std = proxy_dyn.std(dim=1, keepdim=True).clamp(min=1e-6)

        # DC prediction
        if hasattr(G, '_ridge_coef') and G._ridge_coef is not None:
            known_dc_flat = X_k.mean(dim=1).reshape(B_viz, -1)
            pred_dc = (known_dc_flat @ torch.from_numpy(G._ridge_coef).T.to(DEVICE)
                       + torch.from_numpy(G._ridge_bias).to(DEVICE))
        else:
            pred_dc = X_k.mean(dim=2).mean(dim=1)

        fake = shape * proxy_std.squeeze(1).unsqueeze(1) + pred_dc.unsqueeze(1)

    real_np = X_tgt.cpu().numpy()
    fake_np = fake.cpu().numpy()
    axis_names  = ['x', 'y', 'z']
    axis_colors = ['#2196F3', '#4CAF50', '#E53935']
    t = np.arange(T)

    def _make_fig(dual=False):
        fig, axes = plt.subplots(n_samples, C, figsize=(5*C, 2*n_samples))
        if n_samples == 1: axes = axes[np.newaxis]
        for i in range(n_samples):
            for c in range(C):
                ax = axes[i, c]
                r  = real_np[i, :, c]
                f  = fake_np[i, :, c]
                if dual:
                    lo_r, hi_r = r.min(), r.max()
                    pad_r = max((hi_r - lo_r) * 0.1, 0.05)
                    ax.set_ylim(lo_r - pad_r, hi_r + pad_r)
                    ax.plot(t, r, color=axis_colors[c], lw=1.4, alpha=0.9)
                    ax.tick_params(axis='y', labelcolor=axis_colors[c], labelsize=6)
                    ax2 = ax.twinx()
                    lo_f, hi_f = float(np.nanmin(f)), float(np.nanmax(f))
                    if not np.isfinite(lo_f) or not np.isfinite(hi_f):
                        lo_f, hi_f = -1.0, 1.0
                    pad_f = max((hi_f - lo_f) * 0.1, 0.05)
                    ax2.set_ylim(lo_f - pad_f, hi_f + pad_f)
                    ax2.plot(t, f, color='black', lw=1.2, alpha=0.7, linestyle='--')
                    ax2.tick_params(axis='y', labelcolor='black', labelsize=5)
                    if i == 0 and c == 0:
                        lines = [plt.Line2D([0],[0], color=axis_colors[c], lw=1.4, label='Real (←)'),
                                 plt.Line2D([0],[0], color='black', lw=1.2, ls='--', label='Gen (→)')]
                        ax.legend(handles=lines, fontsize=7)
                else:
                    lo = min(r.min(), f.min())
                    hi = max(r.max(), f.max())
                    pad = max((hi - lo) * 0.1, 0.05)
                    ax.set_ylim(lo - pad, hi + pad)
                    ax.plot(t, r, color=axis_colors[c], lw=1.4, alpha=0.9,
                            label='Real' if i == 0 and c == 0 else '_')
                    ax.plot(t, f, color='black', lw=1.2, alpha=0.7, linestyle='--',
                            label='Gen' if i == 0 and c == 0 else '_')
                    ax.tick_params(labelsize=6)
                    if i == 0 and c == 0: ax.legend(fontsize=7)
                if i == 0: ax.set_title(f"Axis {axis_names[c]}", fontsize=9)
                ax.set_xlim(0, T-1)
        suffix = "dual scale (shape)" if dual else "shared scale"
        fig.suptitle(f"Real vs Generated  {title} — {suffix}", fontsize=10)
        plt.tight_layout()
        return fig

    shared_dir = os.path.join(os.path.dirname(out_path), "shared")
    dual_dir   = os.path.join(os.path.dirname(out_path), "dual")
    os.makedirs(shared_dir, exist_ok=True)
    os.makedirs(dual_dir,   exist_ok=True)
    basename   = os.path.basename(out_path)

    _make_fig(dual=False).savefig(os.path.join(shared_dir, basename),
                                   dpi=80, bbox_inches='tight')
    plt.close()
    _make_fig(dual=True).savefig(os.path.join(dual_dir, basename),
                                  dpi=80, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# IMPUTATION (drop-in for impute_missing_streams)
# ─────────────────────────────────────────────────────────────────────────────

def extract_bottleneck(translator, X_known_raw, batch_size=256):
    """Extract translator bottleneck embeddings from raw known streams.
    X_known_raw: (N, T, n_known, C) numpy array
    Returns: (N, bot_ch*C) numpy array
    """
    G = translator.generator
    G.eval()
    N = len(X_known_raw)
    embs = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = torch.from_numpy(
                X_known_raw[i:i+batch_size].astype(np.float32)).to(DEVICE)
            kn_list = []
            for k in range(batch.shape[2]):
                xk, _, _ = normalize_sample(batch[:, :, k, :])
                kn_list.append(xk)
            kn_norm = torch.stack(kn_list, dim=2)
            G._known_dc  = batch.mean(dim=1).reshape(len(batch), -1)
            G._known_std = (batch - batch.mean(dim=1, keepdim=True)
                            ).std(dim=1).clamp(min=1e-6).reshape(len(batch), -1)
            emb = G.encode(kn_norm)  # (B, bot_ch*C)
            embs.append(emb.cpu().numpy())
    return np.concatenate(embs, axis=0)  # (N, bot_ch*C)


def impute_with_translator(
    translator: SignalTranslator,
    X_known: np.ndarray,
    known_indices: list,
    n_streams_total: int,
    T: int = 100, C: int = 3,
    batch_size: int = 512,
    encoders: dict = None,
    stream_names: list = None,
    stream_to_encoder: dict = None,
) -> np.ndarray:
    translator.generator.eval()
    n_known = len(known_indices)
    missing = [j for j in range(n_streams_total) if j not in known_indices]
    results = []

    # Pre-compute SimCLR embeddings for DC MLP if available
    simclr_z = None
    if (getattr(translator, 'dc_mlp', None) is not None
            and encoders is not None and stream_names is not None):
        try:
            from scripts.misc.encoder import extract_all_features
            known_names = [stream_names[k] for k in range(n_known)]
            Z = extract_all_features(X_known, encoders, stream_to_encoder,
                                     known_names, batch_size=batch_size)
            simclr_z = Z.reshape(len(Z), -1).astype(np.float32)  # (N, n_k*embed_dim)
            print(f"  [Translator] SimCLR DC: Z={simclr_z.shape}", flush=True)
        except Exception as e:
            print(f"  [Translator] SimCLR DC failed: {e}", flush=True)

    with torch.no_grad():
        for i in range(0, len(X_known), batch_size):
            b_start, b_end = i, min(i + batch_size, len(X_known))
            bk = torch.from_numpy(
                X_known[b_start:b_end].astype(np.float32)).to(DEVICE)

            # Normalize known streams
            kn_list = []
            for k in range(n_known):
                xk_norm, _, _ = normalize_sample(bk[:, :, k, :])
                kn_list.append(xk_norm)
            kn_norm = torch.stack(kn_list, dim=2)

            _G = translator.generator
            _G._known_dc  = bk.mean(dim=1).reshape(bk.shape[0], -1)
            _G._known_std = (bk - bk.mean(dim=1, keepdim=True)
                            ).std(dim=1).clamp(min=1e-6).reshape(bk.shape[0], -1)
            gen_out = _G(kn_norm)  # (B, T, C)

            # ── DC prediction ─────────────────────────────────────────────
            if simclr_z is not None and getattr(translator, 'dc_mlp', None) is not None:
                z_batch = torch.from_numpy(simclr_z[b_start:b_end]).to(DEVICE)
                pred_dc = translator.dc_mlp(z_batch)
            elif getattr(translator, 'dc_ridge_coef', None) is not None:
                coef_t  = torch.from_numpy(translator.dc_ridge_coef).to(DEVICE)
                bias_t  = torch.from_numpy(translator.dc_ridge_bias).to(DEVICE)
                kdc     = bk.mean(dim=1).reshape(bk.shape[0], -1)
                pred_dc = kdc @ coef_t.T + bias_t
            else:
                pred_dc = bk.mean(dim=2).mean(dim=1)

            # ── Extract unit-variance shape ───────────────────────────────
            gen_dc    = gen_out.mean(dim=1, keepdim=True)
            gen_dyn   = gen_out - gen_dc
            gen_std   = gen_dyn.std(dim=1, keepdim=True).clamp(min=1e-6)
            gen_shape = gen_dyn / gen_std

            # ── Amplitude prediction ──────────────────────────────────────
            if simclr_z is not None and getattr(translator, 'amp_mlp', None) is not None:
                z_batch  = torch.from_numpy(simclr_z[b_start:b_end]).to(DEVICE)
                pred_amp = translator.amp_mlp(z_batch)   # (B, C) predicted std
            else:
                # Fallback: proxy amplitude from known stream stds
                pw = getattr(translator, 'proxy_weights', None)
                sr = getattr(translator, 'proxy_std_ratios', None)
                B_cur = bk.shape[0]
                if pw is not None:
                    pw_t = torch.from_numpy(pw).to(DEVICE)
                    sr_t = torch.from_numpy(sr).to(DEVICE)
                    pred_amp = torch.zeros(B_cur, C, device=DEVICE)
                    for k in range(n_known):
                        kn_raw = bk[:, :, k, :]
                        kn_dyn = kn_raw - kn_raw.mean(dim=1, keepdim=True)
                        kn_std = kn_dyn.std(dim=1).clamp(min=1e-6)
                        pred_amp = pred_amp + pw_t[k].unsqueeze(0) * kn_std * sr_t[k].unsqueeze(0)
                else:
                    pred_amp = (bk - bk.mean(dim=1, keepdim=True)
                                ).std(dim=1).clamp(min=1e-6).mean(dim=1)

            # ── Reconstruct: shape × predicted amplitude + DC ────────────
            fake_denorm = gen_shape * pred_amp.unsqueeze(1) + pred_dc.unsqueeze(1)

            # Assemble full tensor
            B_cur = bk.shape[0]
            X_full = torch.zeros(B_cur, T, n_streams_total, C, device=DEVICE)
            for out_pos, src_pos in enumerate(known_indices):
                X_full[:, :, src_pos, :] = bk[:, :, out_pos, :]
            for m_idx in missing:
                X_full[:, :, m_idx, :] = fake_denorm

            results.append(X_full.cpu().numpy())

    X_out = np.concatenate(results, axis=0)
    imp_var   = float(np.var(X_out[:, :, missing, :]))
    known_var = float(np.var(X_known))
    ratio     = imp_var / (known_var + 1e-8)
    status    = "OK" if 0.3 < ratio < 3.0 else "WARN"
    print(f"  [Translator] imp_var={imp_var:.4f}  known_var={known_var:.4f}"
          f"  ratio={ratio:.2f}  [{status}]")
    return X_out


# ─────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────

def save_translator(translator: SignalTranslator, path: str,
                    known_indices: list, target_idx: int):
    torch.save({
        "state_dict": translator.state_dict(),
        "T": translator.T, "C": translator.C,
        "n_known": translator.n_known,
        "known_stream_indices": known_indices,
        "target_stream_idx": target_idx,
    }, path)


def load_translator(path: str) -> SignalTranslator:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    t = SignalTranslator(T=ckpt["T"], C=ckpt["C"], n_known=ckpt["n_known"],
                         base_ch=ckpt.get("base_ch", 32),
                         n_levels=ckpt.get("n_levels", 2))
    t.load_state_dict(ckpt["state_dict"], strict=False)
    t.std_ratio          = ckpt.get("std_ratio", 1.0)
    t.dc_offset          = ckpt.get("dc_offset", 0.0)
    t.dc_ridge_coef      = ckpt.get("dc_ridge_coef", None)
    t.dc_ridge_bias      = ckpt.get("dc_ridge_bias", None)
    t.proxy_weights      = ckpt.get("proxy_weights", None)    # (n_known, C)
    t.proxy_std_ratios   = ckpt.get("proxy_std_ratios", None) # (n_known, C)
    t.dc_mlp        = None
    t.dc_mlp_state  = ckpt.get("dc_mlp_state", None)
    if t.dc_mlp_state is not None:
        s = t.dc_mlp_state
        mlp = nn.Sequential(
            nn.Linear(s["embed_dim"], 128), nn.GELU(),
            nn.Linear(128, 64),             nn.GELU(),
            nn.Linear(64, s["C"]),
        )
        mlp.load_state_dict(s["state_dict"])
        mlp.eval()
        t.dc_mlp = mlp.to(DEVICE)
    t.amp_mlp       = None
    t.amp_mlp_state = ckpt.get("amp_mlp_state", None)
    if t.amp_mlp_state is not None:
        s = t.amp_mlp_state
        mlp = nn.Sequential(
            nn.Linear(s["embed_dim"], 128), nn.GELU(),
            nn.Linear(128, 64),             nn.GELU(),
            nn.Linear(64, s["C"]),
            nn.Softplus(),
        )
        mlp.load_state_dict(s["state_dict"])
        mlp.eval()
        t.amp_mlp = mlp.to(DEVICE)
    return t.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",       required=True)
    parser.add_argument("--participant",    default="DS_10")
    parser.add_argument("--suffix",         default="_LeftWrist_RightAnkle_RightThigh")
    parser.add_argument("--out-dir",        default="output/translator")
    parser.add_argument("--known-streams",  type=int, nargs="+", default=[0, 1])
    parser.add_argument("--target-stream",  type=int, default=2)
    parser.add_argument("--epochs",         type=int, default=50)
    parser.add_argument("--batch-size",     type=int, default=512)
    parser.add_argument("--lr",             type=float, default=2e-4)
    parser.add_argument("--lambda-l1",      type=float, default=10.0)
    parser.add_argument("--patience",       type=int, default=10)
    parser.add_argument("--max-train",      type=int, default=30000)
    parser.add_argument("--lab-data-dir",   default=None,
                        help="Lab data dir for visualization (e.g. paaws_tuned/DS_11)")
    parser.add_argument("--lab-participant", default="DS_11")
    parser.add_argument("--viz-activity",   default="Cycling_Active_Pedaling_Regular_Bicycle")
    parser.add_argument("--lab-sensor-order", nargs="+",
                        default=["LeftAnkle","LeftThigh","LeftWaist","LeftWrist",
                                 "RightAnkle","RightThigh","RightWaist","RightWrist"])
    parser.add_argument("--initial-sensors", nargs="+",
                        default=["LeftWrist","RightAnkle"])
    parser.add_argument("--target-sensor",   default="RightThigh")
    parser.add_argument("--viz-only",       action="store_true",
                        help="Skip training, just run viz on existing checkpoint")
    parser.add_argument("--checkpoint",     default=None,
                        help="Checkpoint to load for viz-only mode")
    parser.add_argument("--viz-every",      type=int, default=10)
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Load FL data — same format as motion_tokenizer
    data_dir = Path(args.data_dir)
    candidate_dirs = [data_dir / args.participant, data_dir]
    train_path = None
    for d in candidate_dirs:
        t = d / f"encoder_train_raw{args.suffix}.npy"
        if t.exists():
            train_path = t; data_dir = d; break
    if train_path is None:
        for d in candidate_dirs:
            cands = sorted(d.glob("encoder_train_raw*.npy"))
            if cands:
                train_path = cands[0]; data_dir = d
                print(f"Auto-selected: {train_path.name}"); break
    if train_path is None:
        raise FileNotFoundError(f"No encoder_train_raw*.npy found")

    val_path = data_dir / train_path.name.replace("train", "val")
    print(f"Loading: {train_path}")
    X_tr  = np.load(str(train_path)).astype(np.float32)
    if val_path.exists():
        X_val = np.load(str(val_path)).astype(np.float32)
        if len(X_val) > 10_000: X_val = X_val[:10_000]
    else:
        # No separate val file — carve out last 10% of train (not first, to avoid
        # temporal leakage if windows are ordered by time)
        n_val = min(10_000, int(len(X_tr) * 0.1))
        X_val = X_tr[-n_val:].copy()
        X_tr  = X_tr[:-n_val]
        print(f"  No val file found — carved out last {n_val} windows as val")
    print(f"  Train: {X_tr.shape}  Val: {X_val.shape}  (N, T, S, C)")

    if len(X_tr) > args.max_train:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_tr), args.max_train, replace=False)
        X_tr = X_tr[idx]
        print(f"  Subsampled to {len(X_tr)} training windows")

    # Load visualization samples from lab data if provided
    viz_samples = None
    if args.lab_data_dir:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        try:
            from scripts.misc.helpers import create_dataset_file_split
            from scripts.misc.config_loader import cfg
            _, np_val, _, label_dict = create_dataset_file_split(
                args.lab_data_dir, [args.lab_participant], cfg.SEED)
            X_lab, y_lab = np_val[0], np_val[1]   # (N, T, 8, C)

            # Map sensor names to indices in lab data
            known_lab_idx  = [args.lab_sensor_order.index(s)
                              for s in args.initial_sensors]
            target_lab_idx = args.lab_sensor_order.index(args.target_sensor)
            all_lab_idx    = known_lab_idx + [target_lab_idx]

            # Find activity samples for viz
            act_int = label_dict.get(args.viz_activity)
            if act_int is not None:
                mask = (y_lab == act_int)
                if mask.sum() > 0:
                    idx = np.where(mask)[0]
                    np.random.seed(42)
                    idx = np.random.choice(idx, min(5, len(idx)), replace=False)
                    viz_samples = X_lab[idx][:, :, all_lab_idx, :]
                    print(f"  Viz: {args.viz_activity}  n={len(viz_samples)}"
                          f"  streams={all_lab_idx}")
        except Exception as e:
            print(f"  Warning: could not load lab viz data: {e}")

    # Fall back to high-variance FL val windows
    if viz_samples is None:
        var = X_val.var(axis=(1, 2, 3))
        top_idx = np.argsort(var)[::-1][:5]
        viz_samples = X_val[top_idx]
        print(f"  Viz: using top-5 high-variance FL val windows")

    T, S, C = X_tr.shape[1], X_tr.shape[2], X_tr.shape[3]

    if args.viz_only:
        ckpt = args.checkpoint or os.path.join(args.out_dir, "translator.pt")
        print(f"Viz-only mode — loading: {ckpt}")
        translator = load_translator(ckpt)
        translator.generator.eval()
        viz_out = os.path.join(args.out_dir, "viz", "viz_only.png")
        Path(os.path.join(args.out_dir, "viz")).mkdir(parents=True, exist_ok=True)
        from scripts.misc.signal_translator import _visualize_translator
        _visualize_translator(
            translator.generator, viz_samples,
            args.known_streams, args.target_stream,
            len(args.known_streams), T, C,
            viz_out, title="viz-only")
        print(f"Saved: {viz_out}")
        sys.exit(0)

    save_path = os.path.join(args.out_dir, "translator.pt")

    # Load SimCLR encoders for embedding loss
    cli_encoders = None
    cli_stream_to_encoder = None
    # stream_names: sequential list [known_0, known_1, ..., target]
    # Index k in stream_names corresponds to the k-th known stream (0-indexed)
    cli_stream_names = list(args.initial_sensors) + [args.target_sensor]
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from scripts.misc.encoder import load_encoders_from_cfg
        from scripts.misc.config_loader import cfg
        cli_encoders = load_encoders_from_cfg(cfg)
        cli_stream_to_encoder = cfg.STREAM_TO_ENCODER
        print(f"  [CLI] SimCLR encoders loaded for embedding loss", flush=True)
    except Exception as e:
        print(f"  [CLI] Could not load encoders: {e}", flush=True)

    translator = train_translator(
        X_tr, X_val,
        known_stream_indices=args.known_streams,
        target_stream_idx=args.target_stream,
        save_path=save_path,
        T=T, C=C,
        epochs=args.epochs,
        lr_g=args.lr, lr_d=args.lr,
        batch_size=args.batch_size,
        lambda_l1=args.lambda_l1,
        early_stopping_patience=args.patience,
        viz_dir=os.path.join(args.out_dir, "viz"),
        viz_every=args.viz_every,
        viz_samples=viz_samples,
        encoders=cli_encoders,
        stream_names=cli_stream_names,
        stream_to_encoder=cli_stream_to_encoder,
    )
    print(f"\nTranslator saved: {save_path}")
    print(f"Visualizations:   {args.out_dir}/viz/")