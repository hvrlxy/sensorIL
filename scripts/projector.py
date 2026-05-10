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


RawSignalEncoder = CrossMaskedTransformer


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────

def _spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    P = torch.fft.rfft(pred,   dim=1).abs()
    T = torch.fft.rfft(target, dim=1).abs()
    return F.l1_loss(P, T)

def _derivative_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred[:, 1:, :] - pred[:, :-1, :],
                     target[:, 1:, :] - target[:, :-1, :])

def _distribution_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred.std(dim=1), target.std(dim=1))

def reconstruction_loss(
    pred: torch.Tensor, target: torch.Tensor,
    spectral_weight: float = 1.0, deriv_weight: float = 0.5,
    dist_weight: float = 1.0,
) -> tuple:
    l1       = F.l1_loss(pred, target)
    spectral = _spectral_loss(pred, target)
    deriv    = _derivative_loss(pred, target)
    dist     = _distribution_loss(pred, target)
    total    = l1 + spectral_weight*spectral + deriv_weight*deriv + dist_weight*dist
    return total, l1, spectral, deriv, dist


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_raw_encoder(
    encoder: CrossMaskedTransformer,
    X_train: np.ndarray, X_val: np.ndarray,
    save_path: str,
    epochs: int = 50, lr: float = 1e-3, batch_size: int = 256,
    early_stopping_patience: int = 10,
    spectral_weight: float = 1.0, deriv_weight: float = 0.5,
    dist_weight: float = 1.0, **kwargs,
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
            batch    = X_tr[idx[i:i+batch_size]].to(DEVICE)
            m        = int(rng.integers(0, S))
            batch_c  = batch - batch.mean(dim=1, keepdim=True)
            target_c = batch_c[:, :, m, :]
            pred_c   = encoder(batch_c, masked_stream=m)

            with torch.no_grad():
                win_var = target_c.var(dim=1).mean(dim=-1)
                w       = win_var / (win_var.mean() + 1e-8)
                w       = w.clamp(0.2, 5.0)
                w       = w / w.mean()

            per_sample_l1 = (pred_c - target_c).abs().mean(dim=(1, 2))
            l1_weighted   = (per_sample_l1 * w).mean()
            _, _, spec, deriv, dist = reconstruction_loss(
                pred_c, target_c,
                spectral_weight=spectral_weight,
                deriv_weight=deriv_weight,
                dist_weight=dist_weight,
            )
            loss = l1_weighted + spectral_weight*spec + deriv_weight*deriv + dist_weight*dist
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
                b_c    = b - b.mean(dim=1, keepdim=True)
                tgt_c  = b_c[:, :, pm, :]
                pred_c = encoder(b_c, masked_stream=pm)
                v, *_  = reconstruction_loss(pred_c, tgt_c,
                    spectral_weight=spectral_weight,
                    deriv_weight=deriv_weight,
                    dist_weight=dist_weight)
                val_loss += v.item()
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

    # ── Amplitude calibration ─────────────────────────────────────────────────
    encoder.eval()
    arr = X_train[:min(10_000, len(X_train))]
    win_var   = arr.std(axis=(1, 2, 3))
    threshold = float(np.percentile(win_var, 50))
    arr_high  = arr[win_var > threshold]
    n_high    = len(arr_high)

    stream_std_ratios = {}
    for s in range(S):
        ks  = [i for i in range(S) if i != s]
        tgt = arr_high[:, :, s,  :].std(axis=(1, 2))
        kno = arr_high[:, :, ks, :].std(axis=(1, 2, 3))
        stream_std_ratios[s] = float(np.median(tgt / (kno + 1e-6)))

    encoder.stream_std_ratios = stream_std_ratios
    print(f"  [MAE] stream_std_ratios (top-50%, n={n_high}): "
          f"{[f's{k}:{v:.3f}' for k, v in stream_std_ratios.items()]}")

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
    std_ratios      = getattr(encoder, "stream_std_ratios", {})
    missing_indices = [j for j in range(n_streams_total) if j not in known_indices]
    results = []

    with torch.no_grad():
        for i in range(0, len(X_known), batch_size):
            bk = torch.from_numpy(
                X_known[i:i+batch_size].astype(np.float32)
            ).to(DEVICE)
            bk_mean = bk.mean(dim=1, keepdim=True)
            bk_c    = bk - bk_mean
            X_full_c = encoder.impute(bk_c, known_indices, missing_indices)

            # Add DC back using known stream mean as proxy
            dc = bk_mean.mean(dim=2, keepdim=True)   # (B, 1, 1, C)
            for m_idx in missing_indices:
                X_full_c[:, :, m_idx, :] = X_full_c[:, :, m_idx, :] + dc.squeeze(2)

            # Amplitude correction
            known_energy = bk.std(dim=1).mean(dim=(1, 2), keepdim=True)  # (B,1,1)
            for m_idx in missing_indices:
                ratio    = std_ratios.get(m_idx, 1.0)
                pred     = X_full_c[:, :, m_idx, :]
                pred_std = pred.std(dim=1, keepdim=True).clamp(min=1e-6)
                scale    = (known_energy * ratio / pred_std).clamp(0.3, 3.0)
                X_full_c[:, :, m_idx, :] = pred * scale

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
# SYNTHETIC FEATURE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_embeddings(
    encoder, X_old_raw, known_indices, n_streams_total,
    simclr_encoders, stream_names, stream_to_encoder,
    embed_dim=96, batch_size=256, T=100, C=3,
):
    from encoder import extract_all_features
    X_full = impute_missing_streams(encoder, X_old_raw, known_indices,
                                    n_streams_total, T, C, batch_size)
    return extract_all_features(X_full, simclr_encoders, stream_to_encoder,
                                stream_names, batch_size=batch_size,
                                stream_indices=list(range(n_streams_total)))


# ─────────────────────────────────────────────────────────────────────────────
# HEAD RETRAINING
# ─────────────────────────────────────────────────────────────────────────────

def retrain_old_heads_on_synthetic(
    projector, heads, head_streams, full_streams,
    X_train_old, X_val_old, X_test_old,
    Z_val_full,
    y_train_int, y_val_int, y_test_int,
    label_dict, thresholds, fusion,
    n_streams_out, embed_dim, known_indices, stream_names,
    simclr_encoders, stream_to_encoder,
    working_dir, timestamp,
    pre_increment_activities=None,
    epochs=20, lr=1e-3, batch_size=200,
    focal_gamma=2.0, max_class_weight=10.0,
    early_stopping_patience=10,
    cfg=None, T=100, C=3,
):
    """Retrain pre-increment heads on imputed synthetic embeddings.
    Returns (updated_heads, updated_thresholds, Z_test_synth)."""
    from helpers_hitl import (build_gated_head_from_features, train_head_fast,
                               find_optimal_threshold_fast, evaluate_head_fast)

    print(f"\n  [Synthetic retrain] Generating synthetic embeddings (train/val/test)...")
    kw = dict(known_indices=known_indices, n_streams_total=n_streams_out,
              simclr_encoders=simclr_encoders, stream_names=stream_names,
              stream_to_encoder=stream_to_encoder, embed_dim=embed_dim,
              batch_size=batch_size, T=T, C=C)
    Z_train_synth = generate_synthetic_embeddings(projector, X_train_old, **kw)
    Z_val_synth   = generate_synthetic_embeddings(projector, X_val_old,   **kw)
    Z_test_synth  = generate_synthetic_embeddings(projector, X_test_old,  **kw)
    print(f"  [Synthetic retrain] Train:{Z_train_synth.shape} "
          f"Val:{Z_val_synth.shape} Test:{Z_test_synth.shape}")

    _si        = n_streams_out - 1
    Z_rn, Z_sn = Z_val_full[:, _si, :], Z_val_synth[:, _si, :]
    ns  = np.linalg.norm(Z_sn, axis=-1, keepdims=True).clip(min=1e-8)
    nr  = np.linalg.norm(Z_rn, axis=-1, keepdims=True).clip(min=1e-8)
    cos = float(np.mean(np.sum((Z_sn/ns)*(Z_rn/nr), axis=-1)))
    print(f"  [Imputation diag] synth_var={np.var(Z_sn):.3f}  "
          f"real_var={np.var(Z_rn):.3f}  cosine={cos:.3f}  "
          f"({'good' if cos > 0.5 and np.var(Z_sn) > np.var(Z_rn)*0.1 else 'POOR'})")

    updated_heads      = dict(heads)
    updated_thresholds = dict(thresholds)
    pre_increment      = set(pre_increment_activities or [])
    print(f"  [Synthetic retrain] pre_increment={sorted(pre_increment)}")

    for (activity, f), model in heads.items():
        if f != fusion or activity not in label_dict:
            continue
        h_streams = head_streams.get((activity, f), full_streams)
        is_old    = (activity in pre_increment) or (len(h_streams) < len(full_streams))
        print(f"  [Synthetic retrain] '{activity}': is_old={is_old}")
        if not is_old:
            continue

        class_idx = label_dict[activity]
        y_tr_bin  = (y_train_int == class_idx).astype(np.int32)
        y_vl_bin  = (y_val_int   == class_idx).astype(np.int32)
        y_te_bin  = (y_test_int  == class_idx).astype(np.int32)
        n_pos = int(y_tr_bin.sum())
        if n_pos == 0 or y_vl_bin.sum() == 0:
            continue

        max_neg  = min(int((y_tr_bin == 0).sum()), 10 * n_pos)
        pos_idx  = np.where(y_tr_bin == 1)[0]
        neg_idx  = np.where(y_tr_bin == 0)[0]
        if len(neg_idx) > max_neg:
            neg_idx = np.random.default_rng(42).choice(neg_idx, size=max_neg, replace=False)
        keep_idx = np.concatenate([pos_idx, neg_idx])
        Z_tr_bal = Z_train_synth[keep_idx]
        y_tr_bal = y_tr_bin[keep_idx]
        print(f"  [Synthetic retrain] '{activity}' "
              f"pos:{n_pos} neg:{len(neg_idx)} ratio:{len(neg_idx)/max(n_pos,1):.1f}:1")

        Z_pos = Z_train_synth[y_tr_bin == 1]
        Z_neg = Z_train_synth[y_tr_bin == 0]
        if len(Z_pos) > 0 and len(Z_neg) > 0:
            pn  = float(np.linalg.norm(Z_pos.reshape(len(Z_pos),-1), axis=1).mean())
            nn_ = float(np.linalg.norm(Z_neg.reshape(len(Z_neg),-1), axis=1).mean())
            cs  = float(np.mean([
                np.dot(Z_pos[i].flatten(), Z_neg[i%len(Z_neg)].flatten()) /
                (np.linalg.norm(Z_pos[i]) * np.linalg.norm(Z_neg[i%len(Z_neg)]) + 1e-8)
                for i in range(min(100, len(Z_pos)))]))
            print(f"    [Diag] pos_norm={pn:.3f}  neg_norm={nn_:.3f}  cosine={cs:.3f}")

        safe      = activity.replace(" ", "_")
        save_path = os.path.join(working_dir, f"{timestamp}_synth_retrain_{safe}.pt")
        best_model, best_auc = None, -1.0
        for seed in [42, 123, 777]:
            torch.manual_seed(seed)
            cand = build_gated_head_from_features([1.0]*n_streams_out, embed_dim, fusion=fusion)
            cand = train_head_fast(cand, Z_tr_bal, y_tr_bal, Z_val_synth, y_vl_bin,
                                   save_path, epochs=epochs, lr=lr, batch_size=batch_size,
                                   focal_gamma=focal_gamma, max_class_weight=max_class_weight,
                                   early_stopping_patience=early_stopping_patience)
            m = evaluate_head_fast(cand, Z_val_synth, y_vl_bin, threshold=0.5)
            if m["auc"] > best_auc:
                best_auc, best_model = m["auc"], cand

        t_min    = cfg.THRESHOLD_MIN      if cfg else 0.20
        t_max    = cfg.THRESHOLD_MAX      if cfg else 0.80
        fallback = cfg.THRESHOLD_FALLBACK if cfg else 0.50
        thresh   = find_optimal_threshold_fast(best_model, Z_val_synth, y_vl_bin,
                                               t_min=t_min, t_max=t_max, fallback=fallback)
        updated_heads[(activity, f)]      = best_model
        updated_thresholds[(activity, f)] = thresh
        m = evaluate_head_fast(best_model, Z_test_synth, y_te_bin, threshold=thresh)
        print(f"    AUC:{m['auc']:.4f}  F1:{m['f1']:.4f}  thresh:{thresh:.3f}  [synthetic]")

    return updated_heads, updated_thresholds, Z_test_synth


# ─────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD / BUILD
# ─────────────────────────────────────────────────────────────────────────────

def _checkpoint(encoder: CrossMaskedTransformer) -> dict:
    return {
        "state_dict":        encoder.state_dict(),
        "n_streams_total":   encoder.n_streams_total,
        "T":                 encoder.T,
        "C":                 encoder.C,
        "patch_size":        encoder.patch_size,
        "d_model":           encoder.d_model,
        "n_heads":           encoder.n_heads,
        "n_layers":          encoder.n_layers,
        "stream_std_ratios": getattr(encoder, "stream_std_ratios", {}),
        "model_type":        "cross_masked_transformer_patch",
    }

def save_projector(encoder: CrossMaskedTransformer, path: str):
    torch.save(_checkpoint(encoder), path)

def load_projector(path: str, **kwargs) -> CrossMaskedTransformer:
    ckpt = torch.load(path, map_location="cpu")
    enc  = CrossMaskedTransformer(
        n_streams_total=ckpt["n_streams_total"],
        T=ckpt["T"], C=ckpt["C"],
        patch_size=ckpt.get("patch_size", 10),
        d_model=ckpt.get("d_model", 64),
        n_heads=ckpt.get("n_heads", 4),
        n_layers=ckpt.get("n_layers", 3),
    )
    enc.load_state_dict(ckpt["state_dict"])
    enc.stream_std_ratios = ckpt.get("stream_std_ratios", {})
    return enc.to(DEVICE)

def build_projector(
    n_streams_in: int, n_streams_out: int, embed_dim: int,
    hidden_dim: int = 64, T: int = 100, C: int = 3,
    patch_size: int = 10, dropout: float = 0.1, **kwargs,
) -> CrossMaskedTransformer:
    assert n_streams_out > n_streams_in
    assert T % patch_size == 0
    d_model = hidden_dim
    n_heads = 4
    while n_heads > 1 and d_model % n_heads != 0:
        n_heads -= 1
    return CrossMaskedTransformer(
        n_streams_total=n_streams_out, T=T, C=C,
        patch_size=patch_size, d_model=d_model,
        n_heads=n_heads, n_layers=3, dropout=dropout,
    ).to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def train_encoder_on_unlabeled(
    projector, X_unlabeled, n_streams_out, embed_dim, save_path,
    epochs=50, lr=1e-3, batch_size=256, early_stopping_patience=10,
    val_fraction=0.1, Z_val_external=None,
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
    encoder.eval()
    S   = X_val.shape[2]
    pm  = S - 1
    X_t = torch.from_numpy(X_val.astype(np.float32))
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            b      = X_t[i:i+batch_size].to(DEVICE)
            b_c    = b - b.mean(dim=1, keepdim=True)
            tgt_c  = b_c[:, :, pm, :]
            pred_c = encoder(b_c, masked_stream=pm)
            v, *_  = reconstruction_loss(pred_c, tgt_c)
            total += v.item(); n += 1
    return total / max(1, n)

def train_projector_bootstrap(
    projector, X_new_old, X_new_full, n_streams_out, embed_dim, save_path,
    epochs=20, lr=1e-3, batch_size=200, early_stopping_patience=10,
    simclr_encoders=None, stream_names=None, stream_to_encoder=None, **kwargs,
):
    n_val = max(1, int(len(X_new_full) * 0.1))
    X_tr  = X_new_full[n_val:]
    X_vl  = X_new_full[:n_val]
    return train_raw_encoder(projector, X_tr, X_vl, save_path,
                             epochs=epochs, lr=lr, batch_size=batch_size,
                             early_stopping_patience=early_stopping_patience)

def evaluate_with_missing_sensors(
    heads, thresholds, projector, X_test_raw, Z_test_full,
    y_int, label_dict, fusion, missing_sensors, all_stream_names,
    n_streams_out, embed_dim, simclr_encoders, stream_to_encoder,
    cooccurrence_graph=None, T=100, C=3, batch_size=256,
):
    from helpers_hitl import evaluate_all_heads_fast
    mi  = [all_stream_names.index(s) for s in missing_sensors if s in all_stream_names]
    ki  = [i for i in range(n_streams_out) if i not in mi]
    X_k = X_test_raw[:, :, ki, :]
    Z   = generate_synthetic_embeddings(projector, X_k, ki, n_streams_out,
                                        simclr_encoders, all_stream_names,
                                        stream_to_encoder, embed_dim, batch_size, T, C)
    return evaluate_all_heads_fast(heads, Z, y_int, label_dict, thresholds, fusion,
                                   cooccurrence_graph=cooccurrence_graph)

def generate_synthetic_full(projector, Z_old, n_streams_out, embed_dim, **kwargs):
    raise NotImplementedError(
        "generate_synthetic_full() is no longer supported. "
        "Use generate_synthetic_embeddings() with raw signals instead."
    )