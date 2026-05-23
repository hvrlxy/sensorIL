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
                nn.InstanceNorm1d(ch) if i > 0 else nn.Identity(),
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
                nn.InstanceNorm1d(ch) if i < n_levels-1 else nn.Identity(),
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
                        nn.InstanceNorm1d(och) if lvl > 0 else nn.Identity(),
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
                    nn.InstanceNorm1d(och) if lvl < n_levels-1 else nn.Identity(),
                ))
                ich = och
            self.dec.append(mods)

        self.out_conv = nn.ModuleList([
            nn.Sequential(nn.Conv1d(base_ch, 1, 7, padding=3), nn.Tanh())
            for _ in range(C)
        ])

    def _enc_idx(self, s, c):
        return s * self.C + c

    def forward(self, X):
        """X: (B, T, n_known, C) → (B, T, C)"""
        B = X.shape[0]

        # Encode all streams × axes, store feature maps
        feats = {}   # (s, c, lvl) → tensor
        for s in range(self.n_known):
            for c in range(self.C):
                h = X[:, :, s, c].unsqueeze(1)   # (B, 1, T)
                for lvl, mod in enumerate(self.enc[self._enc_idx(s, c)]):
                    h = mod(h)
                    feats[(s, c, lvl)] = h

        # Bottleneck per axis: concat streams
        bot = [torch.cat([feats[(s, c, self.n_levels-1)]
                          for s in range(self.n_known)], dim=1)
               for c in range(self.C)]   # list of (B, bot_ch, L)

        # Cross-axis mixing: stack all axes, mix, split back
        stacked = torch.cat(bot, dim=1)          # (B, bot_ch*C, L)
        stacked = self.cross_axis_mix(stacked)
        bot = stacked.chunk(self.C, dim=1)       # C x (B, bot_ch, L)
        bot = list(bot)

        # Temporal context via dilated conv per axis
        bot = [bot[c] + self.temporal[c](bot[c]) for c in range(self.C)]

        # Decode per axis
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
            out.append(h.squeeze(1))   # (B, T)

        return torch.stack(out, dim=-1)   # (B, T, C)


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
    X_train: np.ndarray,     # (N, T, S, C) — all streams
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
    viz_samples: np.ndarray = None,   # (N, T, S, C) pre-selected samples for viz
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

    translator = SignalTranslator(T=T, C=C, n_known=n_known, base_ch=16).to(DEVICE)
    G = translator.generator
    D = translator.discriminator

    opt_G = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))

    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_G, T_max=epochs, eta_min=lr_g*0.1)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_D, T_max=epochs, eta_min=lr_d*0.1)

    X_tr_t = torch.from_numpy(X_train.astype(np.float32))
    X_vl_t = torch.from_numpy(X_val.astype(np.float32))

    best_val, best_ckpt, patience = float("inf"), None, 0
    history = {"epoch": [], "g_loss": [], "d_loss": [], "val_l1": []}

    # Visualization samples — use provided or fall back to first val windows
    if viz_samples is None:
        viz_samples = X_val[:min(5, len(X_val))]

    if viz_dir:
        Path(viz_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        perm = torch.randperm(N)
        g_loss_sum, d_loss_sum, nb = 0.0, 0.0, 0

        for i in range(0, N, batch_size):
            batch = X_tr_t[perm[i:i+batch_size]].to(DEVICE)  # (B, T, S, C)
            B_cur = batch.shape[0]

            # Extract and normalize known + target streams per sample per axis
            X_known  = batch[:, :, known_stream_indices, :]    # (B, T, n_k, C)
            X_target = batch[:, :, target_stream_idx, :]       # (B, T, C)

            # Per-sample per-axis normalization
            X_known_norm_list = []
            for k in range(n_known):
                xk, _, _ = normalize_sample(X_known[:, :, k, :])
                X_known_norm_list.append(xk)
            X_known_norm = torch.stack(X_known_norm_list, dim=2)  # (B,T,n_k,C)
            X_target_norm, _, _ = normalize_sample(X_target)      # (B, T, C)

            # ── Train Discriminator ───────────────────────────────────────
            with torch.no_grad():
                fake = G(X_known_norm)

            # Add small noise to D inputs to prevent immediate saturation
            noise_std = max(0.05 * (1 - epoch / 50), 0.01)
            real_noisy = X_target_norm + torch.randn_like(X_target_norm) * noise_std
            fake_noisy = fake.detach() + torch.randn_like(fake) * noise_std

            real_score, _ = D(X_known_norm, real_noisy)
            fake_score, _ = D(X_known_norm, fake_noisy)

            d_real = F.mse_loss(real_score, torch.ones_like(real_score) * 0.9)  # label smoothing
            d_fake = F.mse_loss(fake_score, torch.zeros_like(fake_score))
            d_loss = (d_real + d_fake) * 0.5

            opt_D.zero_grad(); d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            opt_D.step()

            # ── Train Generator (2 steps per D step for stability) ────────
            for _ in range(2):
                fake = G(X_known_norm)
                fake_score, fake_feats = D(X_known_norm, fake)
                with torch.no_grad():
                    _, real_feats = D(X_known_norm, X_target_norm)

                g_adv  = F.mse_loss(fake_score, torch.ones_like(fake_score))
                g_feat = sum(F.l1_loss(ff, rf.detach())
                             for ff, rf in zip(fake_feats, real_feats))
                g_l1   = F.l1_loss(fake, X_target_norm)

                # Balanced weights: adversarial drives sharpness,
                # L1 drives accuracy, feature matching stabilizes
                g_loss = g_adv + lambda_l1 * g_l1 + 2.0 * g_feat

                opt_G.zero_grad(); g_loss.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                opt_G.step()

            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()
            nb += 1

        sched_G.step(); sched_D.step()

        # Validation — L1 on normalized signals
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
                kn_norm  = torch.stack(kn_list, dim=2)
                tgt_norm, _, _ = normalize_sample(X_tgt)
                fake     = G(kn_norm)
                val_l1  += F.l1_loss(fake, tgt_norm).item()
                nv      += 1
        val_l1 /= max(1, nv)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [Translator] epoch={epoch:3d}  "
                  f"G={g_loss_sum/max(1,nb):.4f}  "
                  f"D={d_loss_sum/max(1,nb):.4f}  "
                  f"val_l1={val_l1:.4f}", flush=True)

        history["epoch"].append(epoch)
        history["g_loss"].append(g_loss_sum / max(1, nb))
        history["d_loss"].append(d_loss_sum / max(1, nb))
        history["val_l1"].append(val_l1)

        # Save loss curve every epoch
        if viz_dir:
            _save_loss_curve(history, os.path.join(viz_dir, "loss_curve.png"))

        # Visualization
        if viz_dir and (epoch % viz_every == 0 or epoch == 1):
            _visualize_translator(G, viz_samples, known_stream_indices,
                                  target_stream_idx, n_known, T, C,
                                  os.path.join(viz_dir, f"epoch_{epoch:03d}.png"),
                                  title=f"epoch {epoch}")

        # Save best checkpoint based on val L1 but don't early stop —
        # GAN val L1 often increases as generator trades reconstruction
        # for realism. Run all epochs and use best checkpoint.
        if val_l1 < best_val:
            best_val  = val_l1
            best_ckpt = {k: v.cpu().clone() for k,v in translator.state_dict().items()}

        # Only early stop if D loss collapses to ~0 (discriminator dominance)
        d_avg = d_loss_sum / max(1, nb)
        if d_avg < 0.01 and epoch > 5:
            print(f"  [Translator] D collapsed at epoch {epoch} — stopping")
            break

    if best_ckpt:
        translator.load_state_dict(best_ckpt)
    print(f"  [Translator] Done  best_val={best_val:.4f}")

    torch.save({
        "state_dict": translator.state_dict(),
        "T": T, "C": C, "n_known": n_known,
        "known_stream_indices": known_stream_indices,
        "target_stream_idx": target_stream_idx,
    }, save_path)
    return translator


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _save_loss_curve(history, out_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = history["epoch"]
    ax1.plot(ep, history["g_loss"], label="G loss", color="#2196F3")
    ax1.plot(ep, history["d_loss"], label="D loss", color="#E53935")
    ax1.set_xlabel("Epoch"); ax1.set_title("GAN Losses")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(ep, history["val_l1"], label="Val L1", color="#4CAF50")
    ax2.set_xlabel("Epoch"); ax2.set_title("Validation L1 (normalized)")
    ax2.legend(); ax2.grid(alpha=0.3)
    # Mark best epoch
    best_ep = ep[history["val_l1"].index(min(history["val_l1"]))]
    ax2.axvline(best_ep, color="gray", linestyle="--", alpha=0.7,
                label=f"Best epoch {best_ep}")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def _visualize_translator(G, X_samples, known_indices, target_idx,
                           n_known, T, C, out_path, title=""):
    """
    X_samples: (N, T, S, C) — pre-selected samples (e.g. cycling from lab data)
    """
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
        kn_norm  = torch.stack(kn_list, dim=2)
        tgt_norm, _, _ = normalize_sample(X_tgt)
        fake     = G(kn_norm)

    real_np = tgt_norm.cpu().numpy()
    fake_np = fake.cpu().numpy()
    axis_names  = ['x', 'y', 'z']
    axis_colors = ['#2196F3', '#4CAF50', '#E53935']
    t = np.arange(T)

    fig, axes = plt.subplots(n_samples, C, figsize=(5*C, 2*n_samples))
    if n_samples == 1: axes = axes[np.newaxis]
    for i in range(n_samples):
        for c in range(C):
            ax = axes[i, c]
            ax.plot(t, real_np[i, :, c], color=axis_colors[c],
                    lw=1.4, alpha=0.9, label='Real')
            ax.plot(t, fake_np[i, :, c], color='black',
                    lw=1.2, alpha=0.6, linestyle='--', label='Generated')
            if i == 0:
                ax.set_title(f"Axis {axis_names[c]}", fontsize=9)
                ax.legend(fontsize=7)
            ax.set_xlim(0, T-1)
            ax.tick_params(labelsize=6)

    fig.suptitle(f"Normalized: Real vs Generated  {title}", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# IMPUTATION (drop-in for impute_missing_streams)
# ─────────────────────────────────────────────────────────────────────────────

def impute_with_translator(
    translator: SignalTranslator,
    X_known: np.ndarray,         # (N, T, n_known, C)
    known_indices: list,
    n_streams_total: int,
    T: int = 100, C: int = 3,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Drop-in replacement for impute_missing_streams.
    Returns (N, T, n_streams_total, C) with target stream filled in.
    Note: output is in NORMALIZED space — same scale as input per sample.
    """
    translator.generator.eval()
    n_known = len(known_indices)
    missing = [j for j in range(n_streams_total) if j not in known_indices]
    results = []

    with torch.no_grad():
        for i in range(0, len(X_known), batch_size):
            bk   = torch.from_numpy(
                X_known[i:i+batch_size].astype(np.float32)).to(DEVICE)

            # Normalize known streams per sample per axis
            kn_list = []
            kn_stats = []   # save (mean, std) for potential denorm
            for k in range(n_known):
                xk_norm, mn, st = normalize_sample(bk[:, :, k, :])
                kn_list.append(xk_norm)
                kn_stats.append((mn, st))
            kn_norm = torch.stack(kn_list, dim=2)  # (B, T, n_known, C)

            # Generate normalized target
            fake_norm = translator.generator(kn_norm)   # (B, T, C)

            # Denormalize using mean known stream stats as reference
            # (best proxy we have for target scale)
            ref_mean = torch.stack([s[0] for s in kn_stats], dim=0).mean(0)
            ref_std  = torch.stack([s[1] for s in kn_stats], dim=0).mean(0)
            fake_denorm = fake_norm * ref_std + ref_mean

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
    ckpt = torch.load(path, map_location="cpu")
    t    = SignalTranslator(T=ckpt["T"], C=ckpt["C"], n_known=ckpt["n_known"])
    t.load_state_dict(ckpt["state_dict"])
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
    X_val = np.load(str(val_path)).astype(np.float32) \
            if val_path.exists() else X_tr[:10_000]
    if len(X_val) > 10_000: X_val = X_val[:10_000]
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

            # Find activity samples
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
    save_path = os.path.join(args.out_dir, "translator.pt")

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
    )
    print(f"\nTranslator saved: {save_path}")
    print(f"Visualizations:   {args.out_dir}/viz/")