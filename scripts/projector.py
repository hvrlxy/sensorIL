"""
projector.py
============
Patch-based cross-masked transformer MAE for cross-sensor imputation
in sensor-incremental HAR.

Architecture
------------
StreamPatchTokenizer : (B, T, C) -> (B, P, d_model)
StreamPatchDecoder   : (B, P, d_model) -> (B, T, C)
CrossMaskedTransformer: context encoder + cross-attention decoder

Training
--------
  - DC removal, random whole-stream masking, variance-weighted L1+spectral+
    derivative+distribution loss.

Amplitude calibration
---------------------
  stream_std_ratios computed on top-50% highest-variance windows.

Public API (unchanged)
----------------------
  build_projector, train_encoder_on_unlabeled, train_projector_bootstrap,
  measure_encoder_val_loss, save_projector, load_projector,
  impute_missing_streams, generate_synthetic_embeddings,
  retrain_old_heads_on_synthetic, evaluate_with_missing_sensors,
  diagnose_imputation_quality
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# PATCH TOKENIZER  (B, T, C) -> (B, P, d_model)
# ─────────────────────────────────────────────────────────────────────────────

class StreamPatchTokenizer(nn.Module):
    def __init__(self, T: int = 100, C: int = 3,
                 patch_size: int = 10, d_model: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        assert T % patch_size == 0
        self.T          = T
        self.C          = C
        self.patch_size = patch_size
        self.P          = T // patch_size
        self.proj = nn.Linear(patch_size * C, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.reshape(B, self.P, self.patch_size, self.C)
        x = x.reshape(B, self.P, self.patch_size * self.C)
        return self.drop(self.norm(self.proj(x)))


# ─────────────────────────────────────────────────────────────────────────────
# PATCH DECODER  (B, P, d_model) -> (B, T, C)
# ─────────────────────────────────────────────────────────────────────────────

class StreamPatchDecoder(nn.Module):
    def __init__(self, T: int = 100, C: int = 3,
                 patch_size: int = 10, d_model: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.T          = T
        self.C          = C
        self.patch_size = patch_size
        self.P          = T // patch_size
        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.GELU(),
        )
        self.axis_heads = nn.ModuleList([
            nn.Linear(d_model, patch_size) for _ in range(C)
        ])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h    = self.trunk(tokens)
        axes = [head(h) for head in self.axis_heads]
        out  = torch.stack(axes, dim=-1)
        return out.reshape(tokens.shape[0], self.T, self.C)


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-MASKED TRANSFORMER
# ─────────────────────────────────────────────────────────────────────────────

class CrossMaskedTransformer(nn.Module):
    def __init__(
        self,
        n_streams_total: int,
        T:          int   = 100,
        C:          int   = 3,
        patch_size: int   = 10,
        d_model:    int   = 64,
        n_heads:    int   = 4,
        n_layers:   int   = 3,
        dropout:    float = 0.1,
    ):
        super().__init__()
        assert T % patch_size == 0
        self.n_streams_total = n_streams_total
        self.T          = T
        self.C          = C
        self.patch_size = patch_size
        self.P          = T // patch_size
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.n_layers   = n_layers

        self.tokenizers = nn.ModuleList([
            StreamPatchTokenizer(T, C, patch_size, d_model, dropout)
            for _ in range(n_streams_total)
        ])
        self.stream_emb = nn.Parameter(torch.zeros(1, n_streams_total, 1, d_model))
        nn.init.trunc_normal_(self.stream_emb, std=0.02)
        self.patch_emb = nn.Parameter(torch.zeros(1, 1, T // patch_size, d_model))
        nn.init.trunc_normal_(self.patch_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_ffn  = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.query_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.decoders = nn.ModuleList([
            StreamPatchDecoder(T, C, patch_size, d_model, dropout)
            for _ in range(n_streams_total)
        ])

    def _build_context(self, X: torch.Tensor, known_indices: list) -> torch.Tensor:
        B  = X.shape[0]
        Sk = len(known_indices)
        patch_tokens = [self.tokenizers[s](X[:, :, s, :]) for s in known_indices]
        tokens = torch.stack(patch_tokens, dim=1)
        tokens = tokens + self.stream_emb[:, known_indices, :, :] + self.patch_emb
        tokens = tokens.permute(0, 2, 1, 3)
        tokens = tokens.reshape(B, self.P * Sk, self.d_model)
        return self.context_encoder(tokens)

    def _build_queries(self, ctx: torch.Tensor,
                       masked_stream: int, n_known: int) -> torch.Tensor:
        B            = ctx.shape[0]
        ctx_by_patch = ctx.reshape(B, self.P, n_known, self.d_model)
        ctx_avg      = ctx_by_patch.mean(dim=2)
        queries      = self.query_proj(ctx_avg)
        target_emb   = self.stream_emb[:, masked_stream, :, :]
        queries      = queries + target_emb + self.patch_emb[0]
        return queries

    def forward(self, X: torch.Tensor, masked_stream: int) -> torch.Tensor:
        S             = self.n_streams_total
        known_indices = [s for s in range(S) if s != masked_stream]
        ctx     = self._build_context(X, known_indices)
        queries = self._build_queries(ctx, masked_stream, len(known_indices))
        attn_out, _ = self.cross_attn(queries, ctx, ctx)
        queries = self.cross_norm(queries + attn_out)
        queries = queries + self.cross_ffn(queries)
        return self.decoders[masked_stream](queries)

    def impute(self, X_known_c: torch.Tensor,
               known_indices: list, missing_indices: list) -> torch.Tensor:
        B, T, _, C = X_known_c.shape
        S = self.n_streams_total
        X_full_c = torch.zeros(B, T, S, C, device=X_known_c.device)
        for out_pos, src_pos in enumerate(known_indices):
            X_full_c[:, :, src_pos, :] = X_known_c[:, :, out_pos, :]
        for m_idx in missing_indices:
            X_full_c[:, :, m_idx, :] = self.forward(X_full_c, masked_stream=m_idx)
        return X_full_c

    def encode_features(
        self,
        X: torch.Tensor,
        stream_indices: list | None = None,
    ) -> torch.Tensor:
        """
        Extract per-stream MAE embeddings without any masking.

        Runs the context encoder on all available streams, averages the P
        patch tokens per stream, and returns one vector per stream.

        For streams not in stream_indices (missing at training time), the
        output is zero-padded so the shape is always (B, S_total, d_model).

        Parameters
        ----------
        X              : (B, T, S_in, C)  raw DC-removed signals
        stream_indices : list[int] | None
            Which columns of the full S_total-dim tensor these streams
            correspond to.  If None, assumes X covers all S_total streams
            in order.

        Returns
        -------
        Z : (B, S_total, d_model)
        """
        B    = X.shape[0]
        S_in = X.shape[2]
        S    = self.n_streams_total

        if stream_indices is None:
            stream_indices = list(range(S_in))

        # Tokenize each input stream: (B, P, d_model) each
        patch_tokens = []
        for local_idx in range(S_in):
            toks = self.tokenizers[stream_indices[local_idx]](X[:, :, local_idx, :])
            patch_tokens.append(toks)                       # (B, P, d_model)

        # Stack → (B, S_in, P, d_model), add positional embeddings
        tokens = torch.stack(patch_tokens, dim=1)           # (B, S_in, P, d_model)
        tokens = (tokens
                  + self.stream_emb[:, stream_indices, :, :]
                  + self.patch_emb)

        # Interleave by time → (B, P*S_in, d_model), run context encoder
        tokens = tokens.permute(0, 2, 1, 3).reshape(B, self.P * S_in, self.d_model)
        ctx    = self.context_encoder(tokens)                # (B, P*S_in, d_model)

        # Reshape to (B, P, S_in, d_model), average patches → (B, S_in, d_model)
        ctx    = ctx.reshape(B, self.P, S_in, self.d_model)
        Z_in   = ctx.mean(dim=1)                             # (B, S_in, d_model)

        # Zero-pad to full (B, S_total, d_model)
        Z_out  = torch.zeros(B, S, self.d_model, device=X.device)
        for local_idx, global_idx in enumerate(stream_indices):
            Z_out[:, global_idx, :] = Z_in[:, local_idx, :]

        return Z_out                                         # (B, S_total, d_model)


RawSignalEncoder = CrossMaskedTransformer


# ─────────────────────────────────────────────────────────────────────────────
# STREAM PROJECTION HEAD  (B, S, d_model) -> (B, S, proj_dim)
# ─────────────────────────────────────────────────────────────────────────────

class StreamProjectionHead(nn.Module):
    """
    Per-stream MLP that projects MAE patch embeddings to a fixed-size
    downstream feature space.

    Trained separately after MAE reconstruction training on a
    self-supervised objective (reconstruction of held-out patches).
    Frozen MAE backbone, only projection head weights are updated.

    Input  : (B, S, d_model)   — per-stream pooled MAE embeddings
    Output : (B, S, proj_dim)  — projected features for binary heads
    """
    def __init__(self, d_model: int = 64, proj_dim: int = 96,
                 n_streams: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_streams = n_streams
        # Shared projection MLP across streams
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: (B, S, d_model) → (B, S, proj_dim)
        return self.proj(Z)


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────

def _spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Normalized spectral distribution loss — compares relative distribution
    of energy across frequencies rather than absolute magnitudes.
    Prevents the model from defaulting to the average training cadence
    (e.g. walking ~1.8 Hz) regardless of input frequency.
    Normalization is per-axis independently (dim=1 = frequency bins).
    """
    P = torch.fft.rfft(pred,   dim=1).abs()           # (B, F, C)
    T = torch.fft.rfft(target, dim=1).abs()
    P = P / (P.sum(dim=1, keepdim=True) + 1e-8)       # normalize per axis
    T = T / (T.sum(dim=1, keepdim=True) + 1e-8)
    return F.l1_loss(P, T)

def _derivative_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred[:, 1:, :] - pred[:, :-1, :],
                     target[:, 1:, :] - target[:, :-1, :])

def _distribution_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred.std(dim=1), target.std(dim=1))

def reconstruction_loss(
    pred: torch.Tensor, target: torch.Tensor,
    spectral_weight:     float = 1.0,
    deriv_weight:        float = 0.5,
    dist_weight:         float = 1.0,
    orientation_weight:  float = 1.0,
    kurtosis_weight:     float = 0.0,
    envelope_weight:     float = 0.0,
) -> tuple:
    l1          = F.l1_loss(pred, target)
    spectral    = _spectral_loss(pred, target)
    deriv       = _derivative_loss(pred, target)
    dist        = _distribution_loss(pred, target)
    orientation = F.l1_loss(pred.mean(dim=1), target.mean(dim=1))
    total       = (l1
                   + spectral_weight    * spectral
                   + deriv_weight       * deriv
                   + dist_weight        * dist
                   + orientation_weight * orientation)
    return total, l1, spectral, deriv, dist, orientation


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_raw_encoder(
    encoder: CrossMaskedTransformer,
    X_train: np.ndarray, X_val: np.ndarray,
    save_path: str,
    epochs: int = 50, lr: float = 1e-3, batch_size: int = 256,
    early_stopping_patience: int = 10,
    spectral_weight: float = 1.0,
    deriv_weight:    float = 0.5,
    dist_weight:     float = 1.0,
    **kwargs,
) -> CrossMaskedTransformer:
    print(f"  [MAE] X_train={X_train.shape}  X_val={X_val.shape}", flush=True)
    S       = X_train.shape[2]
    encoder = encoder.to(DEVICE)
    opt     = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.05)
    X_tr    = torch.from_numpy(X_train.astype(np.float32))
    X_vl    = torch.from_numpy(X_val.astype(np.float32))
    N       = len(X_tr)
    best_val, best_ckpt, patience_count = float("inf"), None, 0
    rng = np.random.default_rng(42)

    for epoch in range(1, epochs + 1):
        encoder.train()
        idx = torch.randperm(N)
        if epoch == 1:
            P = getattr(encoder, 'P', '?')
            print(f"  [MAE] Epoch 1  batch={batch_size}  lr={lr}  "
                  f"spectral={spectral_weight}  deriv={deriv_weight}  "
                  f"dist={dist_weight}  S={S}  P={P}  "
                  f"seq_len={S*(P if isinstance(P,int) else 0)}", flush=True)

        epoch_loss, n_batches = 0.0, 0
        for i in range(0, N, batch_size):
            batch  = X_tr[idx[i:i+batch_size]].to(DEVICE)
            m      = int(rng.integers(0, S))
            target = batch[:, :, m, :]
            pred   = encoder(batch, masked_stream=m)

            with torch.no_grad():
                win_var  = target.var(dim=1).mean(dim=-1)          # (B,)
                var_med  = win_var.median()
                high     = win_var >= var_med                       # top 50%
                low      = ~high                                    # bottom 50%

            def _batch_loss(mask):
                if mask.sum() == 0:
                    return pred.sum() * 0.0
                p, t = pred[mask], target[mask]
                l1_m = F.l1_loss(p, t)
                _, _, spec, deriv, dist, orient = reconstruction_loss(
                    p, t,
                    spectral_weight=spectral_weight,
                    deriv_weight=deriv_weight,
                    dist_weight=dist_weight,
                )
                return (l1_m
                        + spectral_weight * spec
                        + deriv_weight    * deriv
                        + dist_weight     * dist
                        + orient)

            # Fixed 50/50 split — high-variance windows always contribute
            # equally regardless of their proportion in the batch.
            # Prevents sedentary majority from diluting cycling signal.
            loss = 0.5 * _batch_loss(high) + 0.5 * _batch_loss(low)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches  += 1

        sched.step()

        encoder.eval()
        val_loss, n_val = 0.0, 0
        pm = S - 1
        with torch.no_grad():
            for i in range(0, len(X_vl), batch_size):
                b      = X_vl[i:i+batch_size].to(DEVICE)
                tgt    = b[:, :, pm, :]
                pred   = encoder(b, masked_stream=pm)
                v, *_  = reconstruction_loss(pred, tgt,
                    spectral_weight=spectral_weight,
                    deriv_weight=deriv_weight,
                    dist_weight=dist_weight)
                val_loss += v.item()
                n_val    += 1
                n_val    += 1
        val_loss /= max(1, n_val)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [MAE] epoch={epoch:3d}  "
                  f"train={epoch_loss/max(1,n_batches):.5f}  "
                  f"val={val_loss:.5f}", flush=True)

        if val_loss < best_val:
            best_val       = val_loss
            best_ckpt      = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= early_stopping_patience:
                print(f"  [MAE] Early stop epoch {epoch}  best_val={best_val:.6f}")
                break

    if epoch == epochs:
        print(f"  [MAE] Done epoch {epoch}  best_val={best_val:.6f}")
    if best_ckpt is not None:
        encoder.load_state_dict(best_ckpt)

    torch.save(_checkpoint(encoder), save_path)
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# IMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def impute_missing_streams(
    encoder: CrossMaskedTransformer,
    X_known: np.ndarray,
    known_indices: list,
    n_streams_total: int,
    T: int = 100, C: int = 3,
    batch_size: int = 256, n_samples: int = 1,
) -> np.ndarray:
    encoder.eval()
    missing_indices = [j for j in range(n_streams_total) if j not in known_indices]
    results = []

    with torch.no_grad():
        for i in range(0, len(X_known), batch_size):
            bk = torch.from_numpy(
                X_known[i:i+batch_size].astype(np.float32)
            ).to(DEVICE)

            # Feed raw absolute signals — no DC removal.
            # MAE trained on absolute signals can infer target DC from known streams.
            X_full_c = encoder.impute(bk, known_indices, missing_indices)

            for out_pos, src_pos in enumerate(known_indices):
                X_full_c[:, :, src_pos, :] = bk[:, :, out_pos, :]
            results.append(X_full_c.cpu().numpy())

    X_imputed = np.concatenate(results, axis=0)
    if missing_indices:
        imp_var   = float(np.var(X_imputed[:, :, missing_indices, :]))
        known_var = float(np.var(X_known))
        ratio     = imp_var / (known_var + 1e-8)
        status    = "OK" if 0.3 < ratio < 3.0 else "WARN"
        print(f"  [MAEImpute] imp_var={imp_var:.4f}  "
              f"known_var={known_var:.4f}  ratio={ratio:.2f}  [{status}]")
    return X_imputed


def diagnose_imputation_quality(
    encoder, X_raw_full, known_indices, n_streams_total,
    label_names, y_int, T=100, C=3, batch_size=256, n_per_activity=50,
):
    missing_indices = [i for i in range(n_streams_total) if i not in known_indices]
    if not missing_indices:
        return
    print(f"\n  [Imputation quality per activity]")
    print(f"  {'Activity':<45} {'L1':>8} {'var_real':>10} {'var_imp':>10} {'ratio':>6}")
    X_known_only = X_raw_full[:, :, known_indices, :]
    X_imputed    = impute_missing_streams(encoder, X_known_only, known_indices,
                                          n_streams_total, T, C, batch_size)
    for label_int, act_name in sorted(label_names.items(), key=lambda x: x[1]):
        mask = y_int == label_int
        if mask.sum() < 5:
            continue
        idx          = np.where(mask)[0][:min(n_per_activity, mask.sum())]
        real_missing = X_raw_full[idx][:, :, missing_indices, :]
        imp_missing  = X_imputed[idx][:, :, missing_indices, :]
        l1       = float(np.mean(np.abs(real_missing - imp_missing)))
        var_real = float(np.var(real_missing))
        var_imp  = float(np.var(imp_missing))
        print(f"  {act_name:<45} {l1:>8.4f} {var_real:>10.4f} "
              f"{var_imp:>10.4f} {var_imp/(var_real+1e-8):>6.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD / BUILD
# ─────────────────────────────────────────────────────────────────────────────

def _checkpoint(encoder: CrossMaskedTransformer) -> dict:
    # Save backbone and proj_head separately so load_state_dict is clean
    full_sd = encoder.state_dict()
    backbone_sd = {k: v for k, v in full_sd.items()
                   if not k.startswith("proj_head.")}
    return {
        "state_dict":        backbone_sd,
        "n_streams_total":   encoder.n_streams_total,
        "T":                 encoder.T,
        "C":                 encoder.C,
        "patch_size":        encoder.patch_size,
        "d_model":           encoder.d_model,
        "n_heads":           encoder.n_heads,
        "n_layers":          encoder.n_layers,
        "proj_dim":          getattr(encoder, "proj_dim", None),
        "proj_head_state":   (encoder.proj_head.state_dict()
                              if hasattr(encoder, "proj_head") else None),
        "stream_std_ratios": getattr(encoder, "stream_std_ratios", {}),
        "model_type":        "cross_masked_transformer_patch",
    }

def save_projector(encoder: CrossMaskedTransformer, path: str):
    torch.save(_checkpoint(encoder), path)

def load_projector(path: str, **kwargs) -> CrossMaskedTransformer:
    ckpt = torch.load(path, map_location="cpu")
    proj_dim        = ckpt.get("proj_dim")
    proj_head_state = ckpt.get("proj_head_state")

    enc = CrossMaskedTransformer(
        n_streams_total=ckpt["n_streams_total"],
        T=ckpt["T"], C=ckpt["C"],
        patch_size=ckpt.get("patch_size", 10),
        d_model=ckpt.get("d_model", 64),
        n_heads=ckpt.get("n_heads", 4),
        n_layers=ckpt.get("n_layers", 3),
    )

    # Build proj_head before loading state_dict so all keys are present
    if proj_dim is not None:
        enc.proj_dim  = proj_dim
        enc.proj_head = StreamProjectionHead(
            d_model=enc.d_model, proj_dim=proj_dim,
            n_streams=enc.n_streams_total,
        )

    enc.load_state_dict(ckpt["state_dict"], strict=False)
    enc.stream_std_ratios = ckpt.get("stream_std_ratios", {})

    # Load proj_head weights if saved separately (older checkpoints)
    if proj_head_state is not None and hasattr(enc, "proj_head"):
        enc.proj_head.load_state_dict(proj_head_state)

    return enc.to(DEVICE)

def build_projector(
    n_streams_in: int, n_streams_out: int, embed_dim: int,
    hidden_dim: int = 64, T: int = 100, C: int = 3,
    patch_size: int = 10, dropout: float = 0.1,
    proj_dim: int = 96,   # output dim of projection head (matches SimCLR embed_dim)
    **kwargs,
) -> CrossMaskedTransformer:
    assert n_streams_out > n_streams_in
    assert T % patch_size == 0
    d_model = hidden_dim
    n_heads = 4
    while n_heads > 1 and d_model % n_heads != 0:
        n_heads -= 1
    enc = CrossMaskedTransformer(
        n_streams_total=n_streams_out, T=T, C=C,
        patch_size=patch_size, d_model=d_model,
        n_heads=n_heads, n_layers=3, dropout=dropout,
    ).to(DEVICE)
    enc.proj_dim  = proj_dim
    enc.proj_head = StreamProjectionHead(
        d_model=d_model, proj_dim=proj_dim,
        n_streams=n_streams_out, dropout=dropout,
    ).to(DEVICE)
    return enc


# ─────────────────────────────────────────────────────────────────────────────
# PROJECTION HEAD TRAINING  (cross-stream prediction, separate from MAE)
# ─────────────────────────────────────────────────────────────────────────────

def train_projection_head(
    encoder: CrossMaskedTransformer,
    X_train: np.ndarray,
    X_val:   np.ndarray,
    save_path: str,
    stream_indices: list | None = None,
    epochs: int   = 30,
    lr:     float = 1e-3,
    batch_size: int = 256,
    early_stopping_patience: int = 10,
    **kwargs,
) -> CrossMaskedTransformer:
    """
    Train the projection head on top of a frozen MAE backbone.

    Objective: given known streams, predict the masked stream's MAE embedding.
    Self-supervised — no labels needed.  Backbone frozen, only proj_head updated.

    X_train, X_val : (N, T, S, C) raw signals — all streams present.
    """
    if not hasattr(encoder, "proj_head"):
        raise RuntimeError("encoder has no proj_head — call build_projector with proj_dim")

    print(f"  [ProjHead] Training projection head  "
          f"X_train={X_train.shape}  X_val={X_val.shape}", flush=True)

    # Freeze MAE backbone
    for p in encoder.parameters():
        p.requires_grad = False
    for p in encoder.proj_head.parameters():
        p.requires_grad = True

    encoder  = encoder.to(DEVICE)
    opt      = torch.optim.Adam(encoder.proj_head.parameters(), lr=lr)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                           eta_min=lr * 0.05)
    X_tr     = torch.from_numpy(X_train.astype(np.float32))
    X_vl     = torch.from_numpy(X_val.astype(np.float32))
    N        = len(X_tr)
    S        = X_train.shape[2]
    if stream_indices is None:
        stream_indices = list(range(S))

    best_val, best_ckpt, patience_count = float("inf"), None, 0
    rng = np.random.default_rng(42)

    for epoch in range(1, epochs + 1):
        encoder.proj_head.train()
        idx        = torch.randperm(N)
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, N, batch_size):
            batch    = X_tr[idx[i:i+batch_size]].to(DEVICE)

            m_local  = int(rng.integers(0, S))
            m_global = stream_indices[m_local]
            known_local  = [s for s in range(S) if s != m_local]
            known_global = [stream_indices[s] for s in known_local]

            with torch.no_grad():
                Z_all    = encoder.encode_features(batch, stream_indices)
                Z_target = encoder.proj_head(Z_all)[:, m_global, :]

                X_known  = batch[:, :, known_local, :]
                Z_known  = encoder.encode_features(X_known, known_global)

            Z_pred = encoder.proj_head(Z_known)[:, m_global, :]

            loss = F.mse_loss(Z_pred, Z_target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches  += 1

        sched.step()

        # Validation
        encoder.proj_head.eval()
        val_loss, n_val = 0.0, 0
        pm_local  = S - 1
        pm_global = stream_indices[pm_local]
        kl = list(range(S - 1))
        kg = [stream_indices[s] for s in kl]
        with torch.no_grad():
            for i in range(0, len(X_vl), batch_size):
                b     = X_vl[i:i+batch_size].to(DEVICE)
                Z_all = encoder.encode_features(b, stream_indices)
                Z_tgt = encoder.proj_head(Z_all)[:, pm_global, :]
                X_kn  = b[:, :, kl, :]
                Z_kn  = encoder.encode_features(X_kn, kg)
                Z_pr  = encoder.proj_head(Z_kn)[:, pm_global, :]
                val_loss += F.mse_loss(Z_pr, Z_tgt).item()
                n_val    += 1
        val_loss /= max(1, n_val)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [ProjHead] epoch={epoch:3d}  "
                  f"train={epoch_loss/max(1,n_batches):.5f}  val={val_loss:.5f}",
                  flush=True)

        if val_loss < best_val:
            best_val       = val_loss
            best_ckpt      = {k: v.cpu().clone()
                              for k, v in encoder.proj_head.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= early_stopping_patience:
                print(f"  [ProjHead] Early stop epoch {epoch}  best_val={best_val:.6f}")
                break

    if best_ckpt is not None:
        encoder.proj_head.load_state_dict(best_ckpt)

    for p in encoder.parameters():
        p.requires_grad = True

    torch.save(_checkpoint(encoder), save_path)
    print(f"  [ProjHead] Done  best_val={best_val:.6f}")
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# MAE FEATURE EXTRACTION  (replaces SimCLR for downstream tasks)
# ─────────────────────────────────────────────────────────────────────────────

def extract_mae_features(
    encoder: CrossMaskedTransformer,
    X_raw: np.ndarray,
    stream_indices: list | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Extract MAE projection head features from raw signals.

    Parameters
    ----------
    encoder        : CrossMaskedTransformer with proj_head trained
    X_raw          : (N, T, S_in, C)  raw signals (any subset of streams)
    stream_indices : positions of X_raw streams in the full S_total tensor.
                     If None, assumes streams 0..S_in-1.

    Returns
    -------
    Z : (N, S_total, proj_dim)  — same shape contract as SimCLR embeddings
        Missing streams are zero-padded.
    """
    if not hasattr(encoder, "proj_head"):
        raise RuntimeError("encoder has no proj_head — train projection head first")

    encoder.eval()
    encoder.proj_head.eval()
    S_in = X_raw.shape[2]
    if stream_indices is None:
        stream_indices = list(range(S_in))

    results = []
    with torch.no_grad():
        for i in range(0, len(X_raw), batch_size):
            batch  = torch.from_numpy(
                X_raw[i:i+batch_size].astype(np.float32)
            ).to(DEVICE)
            Z_mae  = encoder.encode_features(batch, stream_indices)
            Z_proj = encoder.proj_head(Z_mae)
            results.append(Z_proj.cpu().numpy())

    return np.concatenate(results, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def train_encoder_on_unlabeled(
    projector, X_unlabeled, n_streams_out, embed_dim, save_path,
    epochs=50, lr=1e-3, batch_size=256, early_stopping_patience=10,
    val_fraction=0.1, Z_val_external=None, **kwargs,
):
    if Z_val_external is not None:
        X_tr, X_vl = X_unlabeled, Z_val_external
    else:
        n_val = max(1, int(len(X_unlabeled) * val_fraction))
        X_tr  = X_unlabeled[n_val:]
        X_vl  = X_unlabeled[:n_val]
    return train_raw_encoder(projector, X_tr, X_vl, save_path,
                             epochs=epochs, lr=lr, batch_size=batch_size,
                             early_stopping_patience=early_stopping_patience)


def measure_encoder_val_loss(encoder, X_val, batch_size=256):
    """Reconstruction loss on the primary missing scenario (last stream)."""
    encoder.eval()
    S   = X_val.shape[2]
    pm  = S - 1
    X_t = torch.from_numpy(X_val.astype(np.float32))
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            b      = X_t[i:i+batch_size].to(DEVICE)
            tgt    = b[:, :, pm, :]
            pred   = encoder(b, masked_stream=pm)
            v, *_  = reconstruction_loss(pred, tgt)
            total += v.item(); n += 1
    return total / max(1, n)


def train_projector_bootstrap(
    projector, X_new_old, X_new_full, n_streams_out, embed_dim, save_path,
    epochs=20, lr=1e-3, batch_size=200, early_stopping_patience=10, **kwargs,
):
    """Bootstrap MAE on labeled val-set windows at sensor-increment time."""
    n_val = max(1, int(len(X_new_full) * 0.1))
    X_tr  = X_new_full[n_val:]
    X_vl  = X_new_full[:n_val]
    return train_raw_encoder(projector, X_tr, X_vl, save_path,
                             epochs=epochs, lr=lr, batch_size=batch_size,
                             early_stopping_patience=early_stopping_patience)


def evaluate_with_missing_sensors(
    heads, thresholds, projector, X_test_raw, Z_test_full,
    y_int, label_dict, fusion, missing_sensors, all_stream_names,
    n_streams_out, embed_dim=None, simclr_encoders=None, stream_to_encoder=None,
    cooccurrence_graph=None, T=100, C=3, batch_size=256,
):
    """
    Evaluate heads when some sensors are missing at test time.
    Imputes missing streams, then extracts MAE features — no SimCLR.
    """
    from helpers_hitl import evaluate_all_heads_fast
    mi = [all_stream_names.index(s) for s in missing_sensors
          if s in all_stream_names]
    ki = [i for i in range(n_streams_out) if i not in mi]

    # Impute missing streams → full raw signal
    X_k    = X_test_raw[:, :, ki, :]
    X_full = impute_missing_streams(projector, X_k, ki, n_streams_out,
                                    T, C, batch_size)
    # Extract MAE features from imputed full signal
    Z = extract_mae_features(projector, X_full,
                             stream_indices=list(range(n_streams_out)),
                             batch_size=batch_size)
    return evaluate_all_heads_fast(heads, Z, y_int, label_dict, thresholds, fusion,
                                   cooccurrence_graph=cooccurrence_graph)


def retrain_old_heads_on_synthetic(*args, **kwargs):
    raise NotImplementedError(
        "retrain_old_heads_on_synthetic() is no longer used. "
        "Old-activity heads are retrained directly on MAE features "
        "in experiment_runner.py."
    )


def generate_synthetic_embeddings(*args, **kwargs):
    raise NotImplementedError(
        "generate_synthetic_embeddings() is no longer used. "
        "Use extract_mae_features() instead."
    )


def generate_synthetic_full(*args, **kwargs):
    raise NotImplementedError("generate_synthetic_full() is no longer supported.")