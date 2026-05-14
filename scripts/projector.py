"""
projector.py
============
Patch-based cross-masked transformer MAE for cross-sensor imputation
in sensor-incremental HAR.

Architecture
------------
MultiScaleStreamTokenizer: (B, T, C) -> (B, P_total, d_model)  [per stream]
StreamPatchDecoder       : (B, P, d_model) -> (B, T, C)
CrossMaskedTransformer   : multi-scale context encoder + cross-attention decoder
  - Context encoder sees ALL scales (17 tokens/stream)
  - Decoder queries use ONLY fine-scale tokens (P=10) — no aggregator needed
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-SCALE PATCH TOKENIZER  (B, T, C) -> (B, P_total, d_model)
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleStreamTokenizer(nn.Module):
    """
    Parallel patch tokenizers at physiologically meaningful scales (20Hz, T=100).
      scale=10 (0.5s): sub-stride events, heel-strike impulses
      scale=20 (1.0s): full stride cycle, push-up half-cycle
      scale=50 (2.5s): slow postural changes, full push-up cycle
    Token counts: [10, 5, 2] = 17 total per stream.
    Each stream has independent projection weights and scale embeddings.
    """
    SCALES = [10, 20, 50]

    def __init__(self, T: int = 100, C: int = 3,
                 d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        assert all(T % s == 0 for s in self.SCALES)
        self.T           = T
        self.C           = C
        self.d_model     = d_model
        self.scales      = self.SCALES
        self.P_per_scale = [T // s for s in self.SCALES]   # [10, 5, 2]
        self.P_total     = sum(self.P_per_scale)            # 17
        self.projs = nn.ModuleList([
            nn.Linear(s * C, d_model) for s in self.SCALES
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in self.SCALES])
        self.scale_emb = nn.Parameter(torch.zeros(len(self.SCALES), d_model))
        nn.init.trunc_normal_(self.scale_emb, std=0.02)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, P_total, d_model)"""
        B = x.shape[0]
        scale_tokens = []
        for i, (s, P_s) in enumerate(zip(self.SCALES, self.P_per_scale)):
            patches = x.reshape(B, P_s, s, self.C).reshape(B, P_s, s * self.C)
            toks    = self.drop(self.norms[i](self.projs[i](patches)))
            toks    = toks + self.scale_emb[i]
            scale_tokens.append(toks)
        return torch.cat(scale_tokens, dim=1)   # (B, P_total, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-SCALE TOKENIZER (kept for backward compat with old checkpoints)
# ─────────────────────────────────────────────────────────────────────────────

class StreamPatchTokenizer(nn.Module):
    def __init__(self, T=100, C=3, patch_size=10, d_model=64, dropout=0.1):
        super().__init__()
        assert T % patch_size == 0
        self.T = T; self.C = C; self.patch_size = patch_size
        self.P = T // patch_size
        self.proj = nn.Linear(patch_size * C, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B, self.P, self.patch_size, self.C)
        x = x.reshape(B, self.P, self.patch_size * self.C)
        return self.drop(self.norm(self.proj(x)))


# ─────────────────────────────────────────────────────────────────────────────
# PATCH DECODER  (B, P, d_model) -> (B, T, C)
# ─────────────────────────────────────────────────────────────────────────────

class StreamPatchDecoder(nn.Module):
    def __init__(self, T=100, C=3, patch_size=10, d_model=64, dropout=0.1):
        super().__init__()
        self.T = T; self.C = C; self.patch_size = patch_size
        self.P = T // patch_size
        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.GELU(),
        )
        self.axis_heads = nn.ModuleList([
            nn.Linear(d_model, patch_size) for _ in range(C)
        ])

    def forward(self, tokens):
        h   = self.trunk(tokens)
        out = torch.stack([head(h) for head in self.axis_heads], dim=-1)
        return out.reshape(tokens.shape[0], self.T, self.C)


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-MASKED TRANSFORMER  (multi-scale)
# ─────────────────────────────────────────────────────────────────────────────

class CrossMaskedTransformer(nn.Module):
    """
    Context encoder sees ALL scales (17 tokens/stream) for rich temporal context.
    Decoder queries use ONLY fine-scale tokens (P=10) — no aggregator, no mismatch.
    """
    def __init__(self, n_streams_total, T=100, C=3, patch_size=10,
                 d_model=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        assert T % patch_size == 0
        self.n_streams_total = n_streams_total
        self.T = T; self.C = C; self.patch_size = patch_size
        self.P = T // patch_size   # fine-scale count = 10
        self.d_model = d_model; self.n_heads = n_heads; self.n_layers = n_layers

        self.tokenizers = nn.ModuleList([
            MultiScaleStreamTokenizer(T, C, d_model, dropout)
            for _ in range(n_streams_total)
        ])
        self.P_total = self.tokenizers[0].P_total  # 17

        self.stream_emb = nn.Parameter(torch.zeros(1, n_streams_total, 1, d_model))
        nn.init.trunc_normal_(self.stream_emb, std=0.02)
        if n_streams_total == 3:
            with torch.no_grad():
                self.stream_emb[0, 0, 0, :d_model//4]          += 0.1
                self.stream_emb[0, 1, 0, d_model//4:d_model//2] += 0.05
                self.stream_emb[0, 2, 0, d_model//4:d_model//2] += 0.05

        self.patch_emb = nn.Parameter(torch.zeros(1, 1, self.P_total, d_model))
        nn.init.trunc_normal_(self.patch_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.context_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model))

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_ffn  = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model*4, d_model))
        self.query_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model))
        self.decoders = nn.ModuleList([
            StreamPatchDecoder(T, C, patch_size, d_model, dropout)
            for _ in range(n_streams_total)])

        # Learned scale weights for decoder queries
        self.scale_query_weights = nn.Parameter(torch.ones(3))

        # DC predictor — handles orientation separately from shape
        n_known = n_streams_total - 1
        self.dc_predictor = DCPredictor(n_known=n_known, C=C)

    def _build_context(self, X, known_indices):
        B, Sk = X.shape[0], len(known_indices)
        tokens = torch.stack([self.tokenizers[s](X[:, :, s, :])
                               for s in known_indices], dim=1)  # (B,Sk,P_total,d)
        tokens = tokens + self.stream_emb[:, known_indices, :, :] + self.patch_emb
        tokens = tokens.permute(0, 2, 1, 3).reshape(B, self.P_total * Sk, self.d_model)
        return self.context_encoder(tokens)   # (B, P_total*Sk, d)

    def _build_queries(self, ctx, masked_stream, n_known):
        B = ctx.shape[0]
        # Use all scales, averaged over streams, then pool to P tokens.
        # Each scale contributes: fine (10 tokens), medium (5), coarse (2).
        # A learned scale-weighting vector balances fine vs coarse contribution.
        ctx_3d  = ctx.reshape(B, self.P_total, n_known, self.d_model)
        ctx_avg = ctx_3d.mean(dim=2)          # (B, P_total, d) — avg over streams

        # Learned scale weights: one scalar per scale, applied to token groups
        # Build weight tensor matching P_total positions — no inplace ops
        sw = torch.softmax(self.scale_query_weights, dim=0) * 3  # (3,)
        P0, P1, P2 = self.tokenizers[0].P_per_scale              # 10, 5, 2
        scale_mask = torch.cat([
            sw[0].expand(P0),
            sw[1].expand(P1),
            sw[2].expand(P2),
        ], dim=0)                                                  # (P_total,)
        ctx_weighted = ctx_avg * scale_mask.unsqueeze(0).unsqueeze(-1)  # (B, P_total, d)

        # Pool P_total → P via adaptive average pooling
        ctx_pooled = F.adaptive_avg_pool1d(
            ctx_weighted.permute(0, 2, 1),   # (B, d, P_total)
            self.P
        ).permute(0, 2, 1)              # (B, P, d)

        queries = self.query_proj(ctx_pooled)
        queries = (queries
                   + self.stream_emb[:, masked_stream, :, :]
                   + self.patch_emb[:, :, :self.P, :][0])
        return queries

    def forward(self, X, masked_stream):
        known = [s for s in range(self.n_streams_total) if s != masked_stream]
        ctx     = self._build_context(X, known)
        queries = self._build_queries(ctx, masked_stream, len(known))
        attn_out, _ = self.cross_attn(queries, ctx, ctx)
        queries = self.cross_norm(queries + attn_out)
        queries = queries + self.cross_ffn(queries)
        return self.decoders[masked_stream](queries)

    def forward_with_attn(self, X, masked_stream, known_indices):
        ctx     = self._build_context(X, known_indices)
        queries = self._build_queries(ctx, masked_stream, len(known_indices))
        attn_out, attn_w = self.cross_attn(
            queries, ctx, ctx, need_weights=True, average_attn_weights=False)
        queries = self.cross_norm(queries + attn_out)
        queries = queries + self.cross_ffn(queries)
        pred    = self.decoders[masked_stream](queries)
        tok_labels = []
        for ki in known_indices:
            tok = self.tokenizers[ki]
            for scale_i, (s, P_s) in enumerate(zip(tok.scales, tok.P_per_scale)):
                for t_i in range(P_s):
                    tok_labels.append(f"S{ki}_sc{s}_t{t_i}")
        return pred, attn_w, tok_labels

    def impute(self, X_known, known_indices, missing_indices):
        B, T, _, C = X_known.shape
        X_full = torch.zeros(B, T, self.n_streams_total, C, device=X_known.device)
        for out_pos, src_pos in enumerate(known_indices):
            X_full[:, :, src_pos, :] = X_known[:, :, out_pos, :]
        for m_idx in missing_indices:
            X_full[:, :, m_idx, :] = self.forward(X_full, masked_stream=m_idx)
        return X_full

    def encode_features(self, X, stream_indices=None):
        B, S_in = X.shape[0], X.shape[2]
        S = self.n_streams_total
        if stream_indices is None:
            stream_indices = list(range(S_in))
        tokens = torch.stack([
            self.tokenizers[stream_indices[i]](X[:, :, i, :])
            for i in range(S_in)], dim=1)                   # (B,S_in,P_total,d)
        tokens = (tokens
                  + self.stream_emb[:, stream_indices, :, :]
                  + self.patch_emb)
        tokens = tokens.permute(0,2,1,3).reshape(B, self.P_total*S_in, self.d_model)
        ctx    = self.context_encoder(tokens)
        ctx    = ctx.reshape(B, self.P_total, S_in, self.d_model)
        Z_in   = ctx.mean(dim=1)                             # (B, S_in, d)
        Z_out  = torch.zeros(B, S, self.d_model, device=X.device)
        for li, gi in enumerate(stream_indices):
            Z_out[:, gi, :] = Z_in[:, li, :]
        return Z_out


RawSignalEncoder = CrossMaskedTransformer


# ─────────────────────────────────────────────────────────────────────────────
# DC PREDICTOR  (wrist_dc, ankle_dc) -> thigh_dc
# ─────────────────────────────────────────────────────────────────────────────

class DCPredictor(nn.Module):
    """
    Predicts the target stream's DC offset (orientation/gravity component)
    from the known streams' DC offsets.

    Input : (B, S_known * C)  — concatenated per-axis means of known streams
    Output: (B, C)            — predicted DC offset of target stream

    Trained with MSE loss on window means. Simple MLP sufficient since
    the DC-to-DC mapping is approximately linear for gravity-dominated signals.
    """
    def __init__(self, n_known: int = 2, C: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_known * C, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),      nn.GELU(),
            nn.Linear(hidden, C),
        )

    def forward(self, dc_known: torch.Tensor) -> torch.Tensor:
        """dc_known: (B, S_known, C) → (B, C)"""
        B = dc_known.shape[0]
        return self.net(dc_known.reshape(B, -1))


# ─────────────────────────────────────────────────────────────────────────────
# AMPLITUDE PREDICTOR  (wrist_dyn_std, ankle_dyn_std) -> thigh_dyn_std
# ─────────────────────────────────────────────────────────────────────────────

class AmplitudePredictor(nn.Module):
    """
    Predicts the target stream's dynamic amplitude (per-axis std of DC-removed
    signal) from the known streams' dynamic amplitudes.

    Input : (B, S_known * C)  — concatenated per-axis stds of known streams
    Output: (B, C)            — predicted per-axis std of target stream

    Trained with MSE loss on log-stds (more stable than raw stds).
    """
    def __init__(self, n_known: int = 2, C: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_known * C, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),      nn.GELU(),
            nn.Linear(hidden, C),
        )

    def forward(self, std_known: torch.Tensor) -> torch.Tensor:
        """std_known: (B, S_known, C) → (B, C) predicted std (positive)"""
        B = std_known.shape[0]
        # Predict in log space, exponentiate for positivity
        return torch.exp(self.net(std_known.reshape(B, -1))).clamp(1e-4, 10.0)


# ─────────────────────────────────────────────────────────────────────────────
# STREAM PROJECTION HEAD
# ─────────────────────────────────────────────────────────────────────────────

class StreamProjectionHead(nn.Module):
    def __init__(self, d_model=64, proj_dim=96, n_streams=2, dropout=0.1):
        super().__init__()
        self.n_streams = n_streams
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model*2, proj_dim),
            nn.LayerNorm(proj_dim))

    def forward(self, Z):
        return self.proj(Z)


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────

def _spectral_loss(pred, target):
    P = torch.fft.rfft(pred, dim=1).abs()
    T = torch.fft.rfft(target, dim=1).abs()
    return F.l1_loss(P, T)

def _derivative_loss(pred, target):
    return F.l1_loss(pred[:,1:,:]-pred[:,:-1,:], target[:,1:,:]-target[:,:-1,:])

def _distribution_loss(pred, target):
    return F.l1_loss(pred.std(dim=1), target.std(dim=1))

def reconstruction_loss(pred, target, spectral_weight=0.0,
                        deriv_weight=0.5, dist_weight=1.0, **kwargs):
    l1       = F.l1_loss(pred, target)
    spectral = _spectral_loss(pred, target)
    deriv    = _derivative_loss(pred, target)
    dist     = _distribution_loss(pred, target)
    total    = l1 + spectral_weight*spectral + deriv_weight*deriv + dist_weight*dist
    return total, l1, spectral, deriv, dist


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_raw_encoder(encoder, X_train, X_val, save_path,
                      epochs=50, lr=1e-3, batch_size=256,
                      early_stopping_patience=10,
                      spectral_weight=0.0, deriv_weight=0.5, dist_weight=1.0,
                      **kwargs):
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
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    for epoch in range(1, epochs+1):
        encoder.train()
        idx = torch.randperm(N)
        if epoch == 1:
            P = getattr(encoder, 'P_total', getattr(encoder, 'P', '?'))
            print(f"  [MAE] Epoch 1  batch={batch_size}  lr={lr}  "
                  f"spectral={spectral_weight}  deriv={deriv_weight}  "
                  f"dist={dist_weight}  S={S}  P_total={P}  "
                  f"seq_len={S*(P if isinstance(P,int) else 0)}", flush=True)

        epoch_loss, n_batches = 0.0, 0
        for i in range(0, N, batch_size):
            batch  = X_tr[idx[i:i+batch_size]].to(DEVICE)   # (B, T, S, C) absolute
            m      = int(rng.integers(0, S))
            known  = [s for s in range(S) if s != m]

            # ── Decompose signal ──────────────────────────────────────────
            # DC component (window mean per axis)
            target_dc  = batch[:, :, m, :].mean(dim=1)          # (B, C)
            known_dc   = batch[:, :, known, :].mean(dim=1)       # (B, S_k, C)

            # Dynamic component (DC-removed) — MAE predicts this directly
            batch_dc   = batch.mean(dim=1, keepdim=True)
            batch_dyn  = batch - batch_dc                        # (B, T, S, C)
            target_dyn = batch_dyn[:, :, m, :]                   # (B, T, C)

            # ── MAE loss on raw DC-removed signal ─────────────────────────
            # Build input: known streams DC-removed, missing zeroed.
            # Must use torch.zeros (not zeros_like) to avoid sharing
            # gradient graph with batch_dyn.
            B_cur = batch.shape[0]
            batch_in = torch.zeros(
                B_cur, batch.shape[1], S, batch.shape[3],
                device=DEVICE, dtype=torch.float32)
            for ks in known:
                batch_in[:, :, ks, :] = batch_dyn.detach()[:, :, ks, :]

            pred_dyn = encoder(batch_in, masked_stream=m)

            with torch.no_grad():
                win_var = target_dyn.var(dim=1).mean(dim=-1)
                w = (win_var / (win_var.mean()+1e-8)).clamp(0.2, 5.0)
                w = w / w.mean()

            per_sample_l1 = (pred_dyn - target_dyn).abs().mean(dim=(1,2))
            l1_weighted   = (per_sample_l1 * w).mean()
            _, _, spec, deriv, dist = reconstruction_loss(
                pred_dyn, target_dyn,
                spectral_weight=spectral_weight,
                deriv_weight=deriv_weight,
                dist_weight=dist_weight)
            loss_shape = (l1_weighted
                          + spectral_weight * spec
                          + deriv_weight    * deriv
                          + dist_weight     * dist)

            # ── DC loss (only when n_known matches dc_predictor) ─────────
            expected_known = encoder.dc_predictor.net[0].in_features // batch.shape[3]
            if len(known) == expected_known:
                pred_dc  = encoder.dc_predictor(known_dc)
                loss_dc  = F.mse_loss(pred_dc, target_dc)
                loss = loss_shape + loss_dc
            else:
                loss = loss_shape
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item(); n_batches += 1

        sched.step()
        encoder.eval()
        val_loss, n_val = 0.0, 0
        pm = S - 1
        with torch.no_grad():
            for i in range(0, len(X_vl), batch_size):
                b       = X_vl[i:i+batch_size].to(DEVICE)
                b_dyn   = b - b.mean(dim=1, keepdim=True)
                tgt_dyn = b_dyn[:, :, pm, :]
                b_in    = b_dyn.clone(); b_in[:, :, pm, :] = 0.0
                pred    = encoder(b_in, masked_stream=pm)
                v,*_    = reconstruction_loss(pred, tgt_dyn,
                            spectral_weight=spectral_weight,
                            deriv_weight=deriv_weight,
                            dist_weight=dist_weight)
                val_loss += v.item(); n_val += 1
        val_loss /= max(1, n_val)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [MAE] epoch={epoch:3d}  "
                  f"train={epoch_loss/max(1,n_batches):.5f}  val={val_loss:.5f}", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            best_ckpt = {k: v.cpu().clone() for k,v in encoder.state_dict().items()}
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

    # Amplitude calibration
    encoder.eval()
    arr    = X_train[:min(10_000, len(X_train))]
    arr_dc = arr - arr.mean(axis=1, keepdims=True)
    win_var = arr_dc.var(axis=(1,2,3))
    arr_hi  = arr_dc[win_var > float(np.percentile(win_var, 50))]
    stream_std_ratios = {}
    for s in range(S):
        ks  = [i for i in range(S) if i != s]
        tgt = arr_hi[:, :, s,  :].std(axis=(1,2))
        kno = arr_hi[:, :, ks, :].std(axis=(1,2,3))
        stream_std_ratios[s] = float(np.median(tgt / (kno + 1e-6)))
    encoder.stream_std_ratios = stream_std_ratios
    print(f"  [MAE] stream_std_ratios: {[f's{k}:{v:.3f}' for k,v in stream_std_ratios.items()]}")

    torch.save(_checkpoint(encoder), save_path)
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# IMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def impute_missing_streams(encoder, X_known, known_indices, n_streams_total,
                           T=100, C=3, batch_size=256, n_samples=1):
    """
    Decomposed imputation: thigh = shape * amplitude + dc

    Step 1: DC predictor    → thigh window mean (orientation)
    Step 2: Amplitude pred  → thigh dynamic std per axis
    Step 3: MAE (normalized)→ thigh shape at unit variance
    Final : shape * amplitude + dc
    """
    encoder.eval()
    missing_indices = [j for j in range(n_streams_total) if j not in known_indices]
    results = []

    with torch.no_grad():
        for i in range(0, len(X_known), batch_size):
            bk = torch.from_numpy(
                X_known[i:i+batch_size].astype(np.float32)).to(DEVICE)
            X_full = torch.zeros(
                bk.shape[0], T, n_streams_total, C, device=bk.device)
            for out_pos, src_pos in enumerate(known_indices):
                X_full[:, :, src_pos, :] = bk[:, :, out_pos, :]

            for m_idx in missing_indices:
                bk_dyn = bk - bk.mean(dim=1, keepdim=True)      # (B, T, S_k, C)

                # Step 1: predict DC (orientation)
                known_dc = bk.mean(dim=1)                        # (B, S_k, C)
                pred_dc  = encoder.dc_predictor(known_dc)        # (B, C)

                # Step 2: predict dynamic component (DC-removed shape+amplitude)
                X_norm = torch.zeros(
                    bk.shape[0], T, n_streams_total, C, device=bk.device)
                for ki, ks in enumerate(known_indices):
                    X_norm[:, :, ks, :] = bk_dyn[:, :, ki, :]
                pred_dyn = encoder.forward(X_norm, masked_stream=m_idx)  # (B,T,C)

                # Step 3: reassemble
                X_full[:, :, m_idx, :] = pred_dyn + pred_dc.unsqueeze(1)

            for out_pos, src_pos in enumerate(known_indices):
                X_full[:, :, src_pos, :] = bk[:, :, out_pos, :]
            results.append(X_full.cpu().numpy())

    X_imputed = np.concatenate(results, axis=0)
    if missing_indices:
        imp_var   = float(np.var(X_imputed[:, :, missing_indices, :]))
        known_var = float(np.var(X_known))
        ratio     = imp_var / (known_var + 1e-8)
        status    = "OK" if 0.3 < ratio < 3.0 else "WARN"
        print(f"  [MAEImpute] imp_var={imp_var:.4f}  known_var={known_var:.4f}"
              f"  ratio={ratio:.2f}  [{status}]")
    return X_imputed


def visualize_token_attention(encoder, X_sample, known_indices, missing_idx,
                               stream_names, out_path, sample_idx=0):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    encoder.eval()
    with torch.no_grad():
        batch = torch.from_numpy(
            X_sample[:sample_idx+1].astype(np.float32)).to(DEVICE)
        pred, attn_w, tok_labels = encoder.forward_with_attn(
            batch, missing_idx, known_indices)
    attn    = attn_w[0].mean(dim=0).cpu().numpy()
    P, ctx_len = attn.shape
    T_len   = X_sample.shape[1]
    pred_np = pred[0].cpu().numpy()
    stream_colors = ['#2196F3','#4CAF50','#E53935','#FF9800','#9C27B0']
    tok_colors = [stream_colors[int(l.split('_')[0][1:]) % len(stream_colors)]
                  for l in tok_labels]
    fig = plt.figure(figsize=(max(14, ctx_len*0.5+4), 10))
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.5, height_ratios=[2,1])
    ax_attn = fig.add_subplot(gs[0])
    im = ax_attn.imshow(attn, aspect='auto', cmap='Blues', vmin=0, vmax=attn.max())
    ax_attn.set_xlabel("Context token (stream_scale_timepos)", fontsize=9)
    ax_attn.set_ylabel(f"Decoder query → {stream_names[missing_idx]}", fontsize=9)
    ax_attn.set_xticks(range(ctx_len))
    ax_attn.set_xticklabels(tok_labels, rotation=90, fontsize=6)
    ax_attn.set_yticks(range(P))
    ax_attn.set_yticklabels([f"q{i}" for i in range(P)], fontsize=7)
    ax_attn.set_title(
        f"Cross-attention: {stream_names[missing_idx]} ← "
        f"{[stream_names[k] for k in known_indices]}", fontsize=9)
    plt.colorbar(im, ax=ax_attn, shrink=0.6)
    for tick, color in zip(ax_attn.get_xticklabels(), tok_colors):
        tick.set_color(color)
    P_total = encoder.P_total
    for sep in range(1, len(known_indices)):
        ax_attn.axvline(sep*P_total - 0.5, color='red', lw=1.5, alpha=0.7)
    if hasattr(encoder.tokenizers[0], 'P_per_scale'):
        for ki_pos in range(len(known_indices)):
            cum = 0
            for p_s in encoder.tokenizers[0].P_per_scale[:-1]:
                cum += p_s
                ax_attn.axvline(ki_pos*P_total + cum - 0.5,
                                color='gray', lw=0.8, alpha=0.5, ls='--')
    ax_sig = fig.add_subplot(gs[1])
    t = np.arange(T_len)
    real = X_sample[sample_idx, :, missing_idx, :]
    for c, (clr, name) in enumerate(zip(['#2196F3','#4CAF50','#E53935'], ['x','y','z'])):
        ax_sig.plot(t, real[:,c],    color=clr, lw=1.4, alpha=0.9, label=f"Real {name}")
        ax_sig.plot(t, pred_np[:,c], color=clr, lw=1.2, alpha=0.55,
                    linestyle='--', label=f"Pred {name}")
    ax_sig.legend(fontsize=7, ncol=3, loc='upper right')
    ax_sig.set_title("Reconstructed vs real target signal", fontsize=9)
    q_step = T_len // P
    for qi in range(1, P):
        ax_sig.axvline(qi*q_step, color='gray', lw=0.6, alpha=0.4)
    fig.suptitle(
        f"Token attention  |  Known: {[stream_names[k] for k in known_indices]}"
        f"  →  Target: {stream_names[missing_idx]}", fontsize=10, fontweight='bold')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Attention plot saved: {out_path}")
    return attn, tok_labels


def diagnose_imputation_quality(encoder, X_raw_full, known_indices, n_streams_total,
                                 label_names, y_int, T=100, C=3,
                                 batch_size=256, n_per_activity=50):
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
        if mask.sum() < 5: continue
        idx          = np.where(mask)[0][:min(n_per_activity, mask.sum())]
        real_missing = X_raw_full[idx][:, :, missing_indices, :]
        imp_missing  = X_imputed[idx][:, :, missing_indices, :]
        l1           = float(np.mean(np.abs(real_missing - imp_missing)))
        var_real     = float(np.var(real_missing))
        var_imp      = float(np.var(imp_missing))
        print(f"  {act_name:<45} {l1:>8.4f} {var_real:>10.4f} "
              f"{var_imp:>10.4f} {var_imp/(var_real+1e-8):>6.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD / BUILD
# ─────────────────────────────────────────────────────────────────────────────

def _checkpoint(encoder):
    full_sd     = encoder.state_dict()
    backbone_sd = {k: v for k,v in full_sd.items() if not k.startswith("proj_head.")}
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
        "model_type":        "cross_masked_transformer_multiscale_decomposed",
    }

def save_projector(encoder, path):
    torch.save(_checkpoint(encoder), path)

def load_projector(path, **kwargs):
    ckpt = torch.load(path, map_location="cpu")
    proj_dim        = ckpt.get("proj_dim")
    proj_head_state = ckpt.get("proj_head_state")

    # Detect architecture from checkpoint — single-scale has patch_emb (1,1,P,d)
    # where P = T // patch_size = 10. Multiscale has P_total = 17.
    patch_emb_shape = ckpt["state_dict"].get("patch_emb", None)
    if patch_emb_shape is not None:
        saved_P = patch_emb_shape.shape[2]
        T       = ckpt["T"]
        patch_size = ckpt.get("patch_size", 10)
        P_single = T // patch_size   # 10
        is_multiscale = (saved_P != P_single)
    else:
        is_multiscale = False

    if is_multiscale:
        # Saved with multiscale — use CrossMaskedTransformer as-is
        enc = CrossMaskedTransformer(
            n_streams_total=ckpt["n_streams_total"],
            T=ckpt["T"], C=ckpt["C"],
            patch_size=ckpt.get("patch_size", 10),
            d_model=ckpt.get("d_model", 64),
            n_heads=ckpt.get("n_heads", 4),
            n_layers=ckpt.get("n_layers", 3),
        )
    else:
        # Saved with single-scale — build a compatible single-scale model
        # using StreamPatchTokenizer instead of MultiScaleStreamTokenizer
        print("  [load_projector] Legacy single-scale checkpoint detected — "
              "using StreamPatchTokenizer for compatibility")

        class _LegacyCMT(CrossMaskedTransformer):
            """Single-scale version for loading old checkpoints."""
            def __init__(self, n_streams_total, T, C, patch_size,
                         d_model, n_heads, n_layers, dropout=0.1):
                # Call nn.Module.__init__ directly, bypass CrossMaskedTransformer
                nn.Module.__init__(self)
                assert T % patch_size == 0
                self.n_streams_total = n_streams_total
                self.T = T; self.C = C; self.patch_size = patch_size
                self.P = T // patch_size
                self.P_total = self.P   # same as P for single-scale
                self.d_model = d_model; self.n_heads = n_heads; self.n_layers = n_layers
                self.tokenizers = nn.ModuleList([
                    StreamPatchTokenizer(T, C, patch_size, d_model, dropout)
                    for _ in range(n_streams_total)])
                self.stream_emb = nn.Parameter(
                    torch.zeros(1, n_streams_total, 1, d_model))
                self.patch_emb = nn.Parameter(
                    torch.zeros(1, 1, self.P, d_model))
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
                    dropout=dropout, activation="gelu",
                    batch_first=True, norm_first=True)
                self.context_encoder = nn.TransformerEncoder(
                    enc_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model))
                self.cross_attn = nn.MultiheadAttention(
                    d_model, n_heads, dropout=dropout, batch_first=True)
                self.cross_norm = nn.LayerNorm(d_model)
                self.cross_ffn  = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(d_model*4, d_model))
                self.query_proj = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model), nn.GELU(),
                    nn.Linear(d_model, d_model))
                self.decoders = nn.ModuleList([
                    StreamPatchDecoder(T, C, patch_size, d_model, dropout)
                    for _ in range(n_streams_total)])

            def _build_context(self, X, known_indices):
                B, Sk = X.shape[0], len(known_indices)
                tokens = torch.stack([self.tokenizers[s](X[:, :, s, :])
                                       for s in known_indices], dim=1)
                tokens = tokens + self.stream_emb[:, known_indices, :, :] + self.patch_emb
                tokens = tokens.permute(0,2,1,3).reshape(B, self.P*Sk, self.d_model)
                return self.context_encoder(tokens)

            def _build_queries(self, ctx, masked_stream, n_known):
                B = ctx.shape[0]
                ctx_avg = ctx.reshape(B, self.P, n_known, self.d_model).mean(dim=2)
                queries = self.query_proj(ctx_avg)
                queries = queries + self.stream_emb[:, masked_stream, :, :] + self.patch_emb[0]
                return queries

        enc = _LegacyCMT(
            n_streams_total=ckpt["n_streams_total"],
            T=ckpt["T"], C=ckpt["C"],
            patch_size=ckpt.get("patch_size", 10),
            d_model=ckpt.get("d_model", 64),
            n_heads=ckpt.get("n_heads", 4),
            n_layers=ckpt.get("n_layers", 3),
        )

    if proj_dim is not None:
        enc.proj_dim  = proj_dim
        enc.proj_head = StreamProjectionHead(
            d_model=enc.d_model, proj_dim=proj_dim, n_streams=enc.n_streams_total)
    enc.load_state_dict(ckpt["state_dict"], strict=False)
    enc.stream_std_ratios = ckpt.get("stream_std_ratios", {})
    if proj_head_state is not None and hasattr(enc, "proj_head"):
        enc.proj_head.load_state_dict(proj_head_state)
    return enc.to(DEVICE)

def build_projector(n_streams_in, n_streams_out, embed_dim,
                    hidden_dim=64, T=100, C=3, patch_size=10,
                    dropout=0.1, proj_dim=96, **kwargs):
    assert n_streams_out > n_streams_in
    assert T % patch_size == 0
    d_model = hidden_dim
    n_heads = 4
    while n_heads > 1 and d_model % n_heads != 0:
        n_heads -= 1
    enc = CrossMaskedTransformer(
        n_streams_total=n_streams_out, T=T, C=C,
        patch_size=patch_size, d_model=d_model,
        n_heads=n_heads, n_layers=3, dropout=dropout).to(DEVICE)
    enc.proj_dim  = proj_dim
    enc.proj_head = StreamProjectionHead(
        d_model=d_model, proj_dim=proj_dim,
        n_streams=n_streams_out, dropout=dropout).to(DEVICE)
    return enc


# ─────────────────────────────────────────────────────────────────────────────
# PROJECTION HEAD TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_projection_head(encoder, X_train, X_val, save_path,
                          stream_indices=None, epochs=30, lr=1e-3,
                          batch_size=256, early_stopping_patience=10, **kwargs):
    if not hasattr(encoder, "proj_head"):
        raise RuntimeError("encoder has no proj_head")
    print(f"  [ProjHead] X_train={X_train.shape}  X_val={X_val.shape}", flush=True)
    for p in encoder.parameters(): p.requires_grad = False
    for p in encoder.proj_head.parameters(): p.requires_grad = True
    encoder = encoder.to(DEVICE)
    opt   = torch.optim.Adam(encoder.proj_head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.05)
    X_tr  = torch.from_numpy(X_train.astype(np.float32))
    X_vl  = torch.from_numpy(X_val.astype(np.float32))
    N     = len(X_tr)
    S     = X_train.shape[2]
    if stream_indices is None:
        stream_indices = list(range(S))
    best_val, best_ckpt, patience_count = float("inf"), None, 0
    rng = np.random.default_rng(42)

    for epoch in range(1, epochs+1):
        encoder.proj_head.train()
        idx = torch.randperm(N)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, N, batch_size):
            batch        = X_tr[idx[i:i+batch_size]].to(DEVICE)
            m_local      = int(rng.integers(0, S))
            m_global     = stream_indices[m_local]
            known_local  = [s for s in range(S) if s != m_local]
            known_global = [stream_indices[s] for s in known_local]
            with torch.no_grad():
                Z_all    = encoder.encode_features(batch, stream_indices)
                Z_target = encoder.proj_head(Z_all)[:, m_global, :]
                X_known  = batch[:, :, known_local, :]
                Z_known  = encoder.encode_features(X_known, known_global)
            Z_pred = encoder.proj_head(Z_known)[:, m_global, :]
            loss   = F.mse_loss(Z_pred, Z_target)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); n_batches += 1
        sched.step()

        encoder.proj_head.eval()
        val_loss, n_val = 0.0, 0
        pm_local  = S - 1
        pm_global = stream_indices[pm_local]
        kl = list(range(S-1))
        kg = [stream_indices[s] for s in kl]
        with torch.no_grad():
            for i in range(0, len(X_vl), batch_size):
                b     = X_vl[i:i+batch_size].to(DEVICE)
                Z_all = encoder.encode_features(b, stream_indices)
                Z_tgt = encoder.proj_head(Z_all)[:, pm_global, :]
                X_kn  = b[:, :, kl, :]
                Z_kn  = encoder.encode_features(X_kn, kg)
                Z_pr  = encoder.proj_head(Z_kn)[:, pm_global, :]
                val_loss += F.mse_loss(Z_pr, Z_tgt).item(); n_val += 1
        val_loss /= max(1, n_val)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [ProjHead] epoch={epoch:3d}  "
                  f"train={epoch_loss/max(1,n_batches):.5f}  val={val_loss:.5f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_ckpt = {k: v.cpu().clone() for k,v in encoder.proj_head.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= early_stopping_patience:
                print(f"  [ProjHead] Early stop epoch {epoch}  best_val={best_val:.6f}")
                break

    if best_ckpt is not None:
        encoder.proj_head.load_state_dict(best_ckpt)
    for p in encoder.parameters(): p.requires_grad = True
    torch.save(_checkpoint(encoder), save_path)
    print(f"  [ProjHead] Done  best_val={best_val:.6f}")
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION + WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def extract_mae_features(encoder, X_raw, stream_indices=None, batch_size=256):
    if not hasattr(encoder, "proj_head"):
        raise RuntimeError("encoder has no proj_head")
    encoder.eval(); encoder.proj_head.eval()
    S_in = X_raw.shape[2]
    if stream_indices is None:
        stream_indices = list(range(S_in))
    results = []
    with torch.no_grad():
        for i in range(0, len(X_raw), batch_size):
            batch  = torch.from_numpy(X_raw[i:i+batch_size].astype(np.float32)).to(DEVICE)
            Z_mae  = encoder.encode_features(batch, stream_indices)
            Z_proj = encoder.proj_head(Z_mae)
            results.append(Z_proj.cpu().numpy())
    return np.concatenate(results, axis=0).astype(np.float32)


def train_encoder_on_unlabeled(projector, X_unlabeled, n_streams_out, embed_dim,
                                save_path, epochs=50, lr=1e-3, batch_size=256,
                                early_stopping_patience=10, val_fraction=0.1,
                                Z_val_external=None, **kwargs):
    if Z_val_external is not None:
        X_tr, X_vl = X_unlabeled, Z_val_external
    else:
        n_val = max(1, int(len(X_unlabeled) * val_fraction))
        X_tr  = X_unlabeled[n_val:]
        X_vl  = X_unlabeled[:n_val]
    return train_raw_encoder(projector, X_tr, X_vl, save_path, epochs=epochs,
                             lr=lr, batch_size=batch_size,
                             early_stopping_patience=early_stopping_patience)


def measure_encoder_val_loss(encoder, X_val, batch_size=256):
    encoder.eval()
    S, pm = X_val.shape[2], X_val.shape[2]-1
    X_t   = torch.from_numpy(X_val.astype(np.float32))
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            b    = X_t[i:i+batch_size].to(DEVICE)
            tgt  = b[:, :, pm, :]
            pred = encoder(b, masked_stream=pm)
            v,*_ = reconstruction_loss(pred, tgt)
            total += v.item(); n += 1
    return total / max(1, n)


def train_projector_bootstrap(projector, X_new_old, X_new_full, n_streams_out,
                               embed_dim, save_path, epochs=20, lr=1e-3,
                               batch_size=200, early_stopping_patience=10, **kwargs):
    n_val = max(1, int(len(X_new_full) * 0.1))
    return train_raw_encoder(projector, X_new_full[n_val:], X_new_full[:n_val],
                             save_path, epochs=epochs, lr=lr, batch_size=batch_size,
                             early_stopping_patience=early_stopping_patience)


def evaluate_with_missing_sensors(heads, thresholds, projector, X_test_raw,
                                   Z_test_full, y_int, label_dict, fusion,
                                   missing_sensors, all_stream_names, n_streams_out,
                                   embed_dim=None, simclr_encoders=None,
                                   stream_to_encoder=None, cooccurrence_graph=None,
                                   T=100, C=3, batch_size=256, translator=None):
    from helpers_hitl import evaluate_all_heads_fast
    mi  = [all_stream_names.index(s) for s in missing_sensors if s in all_stream_names]
    ki  = [i for i in range(n_streams_out) if i not in mi]
    X_k = X_test_raw[:, :, ki, :]

    if translator is not None and simclr_encoders is not None:
        # Reconstruct missing stream with translator, then encode all streams
        # with SimCLR (per-stream, independent — no embedding space mismatch)
        from signal_translator import impute_with_translator
        from encoder import extract_all_features
        X_full = impute_with_translator(translator, X_k, ki, n_streams_out,
                                         T, C, batch_size)
        Z = extract_all_features(
            X_full, simclr_encoders, stream_to_encoder,
            stream_names=all_stream_names,
            batch_size=batch_size,
        )
    elif translator is not None:
        # Translator but no SimCLR — fall back to MAE on reconstructed signal
        from signal_translator import impute_with_translator
        X_full = impute_with_translator(translator, X_k, ki, n_streams_out,
                                         T, C, batch_size)
        Z = extract_mae_features(projector, X_full,
                                 stream_indices=list(range(n_streams_out)),
                                 batch_size=batch_size)
    else:
        # No translator — MAE imputation fallback
        X_full = impute_missing_streams(projector, X_k, ki, n_streams_out,
                                         T, C, batch_size)
        Z = extract_mae_features(projector, X_full,
                                 stream_indices=list(range(n_streams_out)),
                                 batch_size=batch_size)

    return evaluate_all_heads_fast(heads, Z, y_int, label_dict, thresholds, fusion,
                                   cooccurrence_graph=cooccurrence_graph)


def retrain_old_heads_on_synthetic(*args, **kwargs):
    raise NotImplementedError("Use extract_mae_features() instead.")

def generate_synthetic_embeddings(*args, **kwargs):
    raise NotImplementedError("Use extract_mae_features() instead.")

def generate_synthetic_full(*args, **kwargs):
    raise NotImplementedError("generate_synthetic_full() is no longer supported.")

