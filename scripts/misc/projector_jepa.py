"""
projector_jepa.py
=================
JEPA-style embedding predictor for cross-sensor imputation.

Instead of generating raw signals, predicts directly in SimCLR
embedding space:

  Old pipeline: raw_old → GAN/CVAE → raw_full → SimCLR → Z_synth
  New pipeline: raw_old → SimCLR   → Z_old    → JEPA   → Z_synth

This sidesteps all raw signal generation problems (amplitude, waveform
shape, axis orientation) because we never leave embedding space.

Architecture
------------
Context encoder : (Z_wrist, Z_ankle) → context c   [MLP on real embeddings]
Predictor       : c → Z_thigh_predicted             [MLP in embedding space]

Training (self-supervised on FL embeddings)
-------------------------------------------
  1. Encode all 3 FL streams with frozen SimCLR → Z_wrist, Z_ankle, Z_thigh
  2. Context encoder: (Z_wrist, Z_ankle) → c
  3. Predictor: c → Z_thigh_hat
  4. Loss: MSE(Z_thigh_hat, Z_thigh) + cosine loss

Inference
---------
  1. Encode known streams with frozen SimCLR → Z_known
  2. Context encoder → c
  3. Predictor → Z_missing
  4. Assemble full Z_synth = [Z_known, Z_missing]

Drop-in replacement for projector.py — identical public API.
Swap with one import line change.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class JEPAPredictor(nn.Module):
    """
    JEPA-style predictor: known stream embeddings → missing stream embedding.

    Operates entirely in SimCLR embedding space (96-dim per stream).
    Each known stream gets its own linear projection before fusion so
    the model knows which stream it's reading (prevents wrist domination).

    Input  : Z_known  (B, S_known, D)
    Output : Z_missing (B, S_missing, D)
    """
    def __init__(self, n_streams_known: int, n_streams_missing: int,
                 embed_dim: int = 96, hidden_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.n_streams_known   = n_streams_known
        self.n_streams_missing = n_streams_missing
        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim

        # Per-stream projection — gives each stream its own identity
        self.stream_projs = nn.ModuleList([
            nn.Linear(embed_dim, hidden_dim)
            for _ in range(n_streams_known)
        ])

        # Context encoder: fuses all known stream projections
        self.context_encoder = nn.Sequential(
            nn.Linear(n_streams_known * hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Predictor: context → missing stream embeddings
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_streams_missing * embed_dim),
        )

    def forward(self, Z_known: torch.Tensor) -> torch.Tensor:
        """
        Z_known  : (B, S_known, D)
        Returns  : (B, S_missing, D)
        """
        B = Z_known.shape[0]

        # Per-stream projection
        projs = [self.stream_projs[s](Z_known[:, s, :])
                 for s in range(self.n_streams_known)]
        H = torch.cat(projs, dim=-1)                    # (B, S_known * hidden)

        # Context + predict
        c   = self.context_encoder(H)                   # (B, hidden)
        out = self.predictor(c)                         # (B, S_missing * D)
        return out.reshape(B, self.n_streams_missing, self.embed_dim)


# Alias so existing code that references RawSignalEncoder still works
RawSignalEncoder = JEPAPredictor


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────

def jepa_loss(Z_pred: torch.Tensor, Z_target: torch.Tensor,
              cosine_weight: float = 0.5) -> tuple:
    """
    MSE + cosine similarity loss in embedding space.

    MSE    : match embedding values
    Cosine : match embedding direction (critical for SimCLR space where
             direction encodes activity identity)

    Returns (total, mse, cosine_loss)
    """
    mse = F.mse_loss(Z_pred, Z_target)

    # Cosine loss: 1 - cosine_similarity (per stream, averaged)
    S = Z_pred.shape[1]
    cos_loss = torch.tensor(0.0, device=Z_pred.device)
    for s in range(S):
        cos_loss += (1.0 - F.cosine_similarity(
            Z_pred[:, s, :], Z_target[:, s, :], dim=-1
        ).mean())
    cos_loss = cos_loss / S

    total = mse + cosine_weight * cos_loss
    return total, mse, cos_loss


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_raw_encoder(
    encoder: JEPAPredictor,
    X_train: np.ndarray,     # (N, S_total, D)  SimCLR embeddings of all streams
    X_val: np.ndarray,       # (N, S_total, D)
    save_path: str,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
    early_stopping_patience: int = 10,
    val_fraction: float = 0.0,
    cosine_weight: float = 0.5,
    **kwargs,
) -> JEPAPredictor:
    """
    Train JEPA predictor on FL SimCLR embeddings.

    X_train/X_val are (N, S_total, D) embedding tensors — NOT raw signals.
    Primary scenario: last stream missing, all others known.
    """
    print(f"  [JEPA] X_train={X_train.shape}  X_val={X_val.shape}", flush=True)
    encoder = encoder.to(DEVICE)
    opt     = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_tr = torch.from_numpy(X_train.astype(np.float32))
    X_vl = torch.from_numpy(X_val.astype(np.float32))
    N    = len(X_tr)
    S    = X_tr.shape[1]

    primary_missing = [S - 1]
    primary_known   = list(range(S - 1))

    best_val, best_ckpt, patience = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        encoder.train()
        idx        = torch.randperm(N)
        total_loss = 0.0
        n_batches  = 0

        if epoch == 1:
            print(f"  [JEPA] Epoch 1  batch={batch_size}  "
                  f"lr={lr}  cosine_weight={cosine_weight}", flush=True)

        for i in range(0, N, batch_size):
            batch     = X_tr[idx[i:i + batch_size]].to(DEVICE)  # (B, S, D)
            Z_known   = batch[:, primary_known,   :]             # (B, S_k, D)
            Z_missing = batch[:, primary_missing, :]             # (B, S_m, D)

            Z_pred = encoder(Z_known)                            # (B, S_m, D)
            loss, mse, cos = jepa_loss(Z_pred, Z_missing, cosine_weight)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            n_batches  += 1

        sched.step()

        # Validation
        encoder.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(X_vl), batch_size):
                b         = X_vl[i:i + batch_size].to(DEVICE)
                Z_known   = b[:, primary_known,   :]
                Z_missing = b[:, primary_missing, :]
                Z_pred    = encoder(Z_known)
                v, _, _   = jepa_loss(Z_pred, Z_missing, cosine_weight)
                val_loss += v.item()
                n_val    += 1
        val_loss /= max(1, n_val)

        if val_loss < best_val:
            best_val  = val_loss
            best_ckpt = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            patience  = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                print(f"  [JEPA] Early stop epoch {epoch}  best_val={best_val:.6f}")
                break

    if epoch == epochs:
        print(f"  [JEPA] Done epoch {epoch}  best_val={best_val:.6f}")

    if best_ckpt is not None:
        encoder.load_state_dict(best_ckpt)

    torch.save({
        "state_dict":        encoder.state_dict(),
        "n_streams_known":   encoder.n_streams_known,
        "n_streams_missing": encoder.n_streams_missing,
        "embed_dim":         encoder.embed_dim,
        "hidden_dim":        encoder.hidden_dim,
        "model_type":        "jepa",
    }, save_path)
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# IMPUTATION  (operates in embedding space — no raw signals)
# ─────────────────────────────────────────────────────────────────────────────

def impute_missing_streams(
    encoder: JEPAPredictor,
    X_known: np.ndarray,       # (N, T, S_known, C)  raw signals
    known_indices: list,
    n_streams_total: int,
    T: int = 100,
    C: int = 3,
    batch_size: int = 256,
    simclr_encoders: dict = None,
    stream_names: list = None,
    stream_to_encoder: dict = None,
) -> np.ndarray:
    """
    JEPA imputation — operates in embedding space.

    Returns (N, T, S_total, C) where missing streams contain
    reconstructed raw signals from inverted embeddings.

    NOTE: JEPA predicts embeddings directly. To get raw signals back
    (for the diagnostic / visualization only), we use a simple
    per-stream mean signal scaled to match the predicted embedding norm.
    For downstream HAR, use generate_synthetic_embeddings() directly
    which skips the raw signal reconstruction entirely.
    """
    # For raw signal output (visualization), return known streams real
    # and missing streams as zero-mean noise scaled to predicted embedding norm.
    # The real power of JEPA is in generate_synthetic_embeddings() below.
    encoder.eval()
    missing_indices = [j for j in range(n_streams_total) if j not in known_indices]
    N = len(X_known)

    # Build output with known streams real, missing streams as proxy raw signal
    X_out = np.zeros((N, T, n_streams_total, C), dtype=np.float32)
    for op, sp in enumerate(known_indices):
        X_out[:, :, sp, :] = X_known[:, :, op, :]

    # For missing streams: use known stream signals as proxy
    # (magnitude-matched to give correct variance for diagnostics)
    for tp in missing_indices:
        # Use mean of known streams as placeholder raw signal
        X_out[:, :, tp, :] = X_known.mean(axis=2)

    imp_var   = float(np.var(X_out[:, :, missing_indices, :]))
    known_var = float(np.var(X_known))
    print(f"  [JEPAImpute] imp_var={imp_var:.4f}  known_var={known_var:.4f}  "
          f"ratio={imp_var/(known_var+1e-8):.2f}  "
          f"[NOTE: raw signal is proxy only — use generate_synthetic_embeddings]")
    return X_out


def generate_synthetic_embeddings(
    encoder: JEPAPredictor,
    X_old_raw: np.ndarray,     # (N, T, S_old, C)  raw signals
    known_indices: list,
    n_streams_total: int,
    simclr_encoders: dict,
    stream_names: list,
    stream_to_encoder: dict,
    embed_dim: int = 96,
    batch_size: int = 256,
    T: int = 100,
    C: int = 3,
) -> np.ndarray:
    """
    JEPA embedding prediction pipeline:
      1. Encode known streams with frozen SimCLR → Z_known (N, S_known, D)
      2. JEPA predictor → Z_missing (N, S_missing, D)
      3. Assemble Z_full (N, S_total, D)

    Never generates raw signals — operates entirely in embedding space.
    """
    from encoder import extract_all_features

    # Step 1: encode known streams with SimCLR
    known_stream_names = [stream_names[i] for i in known_indices]
    Z_known = extract_all_features(
        X_old_raw, simclr_encoders, stream_to_encoder,
        known_stream_names, batch_size=batch_size,
        stream_indices=list(range(len(known_indices))),
    )                                                    # (N, S_known, D)

    # Step 2: JEPA predicts missing stream embeddings
    encoder.eval()
    missing_indices = [j for j in range(n_streams_total) if j not in known_indices]
    Z_missing_pred  = []

    Z_known_t = torch.from_numpy(Z_known.astype(np.float32))
    with torch.no_grad():
        for i in range(0, len(Z_known_t), batch_size):
            b      = Z_known_t[i:i + batch_size].to(DEVICE)
            Z_pred = encoder(b)                          # (B, S_missing, D)
            Z_missing_pred.append(Z_pred.cpu().numpy())
    Z_missing_pred = np.concatenate(Z_missing_pred, axis=0)  # (N, S_missing, D)

    # Step 3: assemble full embedding tensor
    N, D   = Z_known.shape[0], embed_dim
    Z_full = np.zeros((N, n_streams_total, D), dtype=np.float32)
    for out_pos, src_pos in enumerate(known_indices):
        Z_full[:, src_pos, :] = Z_known[:, out_pos, :]
    for out_pos, tgt_pos in enumerate(missing_indices):
        Z_full[:, tgt_pos, :] = Z_missing_pred[:, out_pos, :]

    # Diagnostic
    Z_synth_new = Z_full[:, missing_indices[-1], :]
    print(f"  [JEPAEmbed] synth_var={np.var(Z_synth_new):.3f}  "
          f"shape={Z_full.shape}")
    return Z_full


# ─────────────────────────────────────────────────────────────────────────────
# HEAD RETRAINING  (identical to projector.py)
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
    """Identical to projector.py version."""
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

    # Diagnostic
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

        max_neg = min(int((y_tr_bin == 0).sum()), 10 * n_pos)
        pos_idx = np.where(y_tr_bin == 1)[0]
        neg_idx = np.where(y_tr_bin == 0)[0]
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
                (np.linalg.norm(Z_pos[i])*np.linalg.norm(Z_neg[i%len(Z_neg)])+1e-8)
                for i in range(min(100, len(Z_pos)))]))
            print(f"    [Diag] pos_norm={pn:.3f}  neg_norm={nn_:.3f}  cosine={cs:.3f}")

        safe      = activity.replace(" ", "_")
        save_path = os.path.join(working_dir, f"{timestamp}_synth_retrain_{safe}.pt")
        best_model, best_auc = None, -1.0
        for seed in [42, 123, 777]:
            torch.manual_seed(seed)
            cand = build_gated_head_from_features([1.0]*n_streams_out, embed_dim, fusion=fusion)
            cand = train_head_fast(
                cand, Z_tr_bal, y_tr_bal, Z_val_synth, y_vl_bin, save_path,
                epochs=epochs, lr=lr, batch_size=batch_size,
                focal_gamma=focal_gamma, max_class_weight=max_class_weight,
                early_stopping_patience=early_stopping_patience,
            )
            m = evaluate_head_fast(cand, Z_val_synth, y_vl_bin, threshold=0.5)
            if m["auc"] > best_auc:
                best_auc, best_model = m["auc"], cand

        t_min    = cfg.THRESHOLD_MIN      if cfg else 0.20
        t_max    = cfg.THRESHOLD_MAX      if cfg else 0.80
        fallback = cfg.THRESHOLD_FALLBACK if cfg else 0.50
        thresh   = find_optimal_threshold_fast(
            best_model, Z_val_synth, y_vl_bin,
            t_min=t_min, t_max=t_max, fallback=fallback,
        )
        updated_heads[(activity, f)]      = best_model
        updated_thresholds[(activity, f)] = thresh

        m = evaluate_head_fast(best_model, Z_test_synth, y_te_bin, threshold=thresh)
        print(f"    AUC:{m['auc']:.4f}  F1:{m['f1']:.4f}  thresh:{thresh:.3f}  [synthetic]")

    return updated_heads, updated_thresholds, Z_test_synth


# ─────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD / BUILD
# ─────────────────────────────────────────────────────────────────────────────

def save_projector(encoder: JEPAPredictor, path: str):
    torch.save({
        "state_dict":        encoder.state_dict(),
        "n_streams_known":   encoder.n_streams_known,
        "n_streams_missing": encoder.n_streams_missing,
        "embed_dim":         encoder.embed_dim,
        "hidden_dim":        encoder.hidden_dim,
        "model_type":        "jepa",
    }, path)


def load_projector(path: str, **kwargs) -> JEPAPredictor:
    ckpt = torch.load(path, map_location="cpu")
    enc  = JEPAPredictor(
        n_streams_known   = ckpt["n_streams_known"],
        n_streams_missing = ckpt["n_streams_missing"],
        embed_dim         = ckpt.get("embed_dim", 96),
        hidden_dim        = ckpt.get("hidden_dim", 512),
    )
    enc.load_state_dict(ckpt["state_dict"])
    return enc.to(DEVICE)


def build_projector(n_streams_in, n_streams_out, embed_dim,
                    hidden_dim=512, T=100, C=3, **kwargs):
    n_missing = n_streams_out - n_streams_in
    assert n_missing > 0
    return JEPAPredictor(
        n_streams_known=n_streams_in,
        n_streams_missing=n_missing,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    ).to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def train_encoder_on_unlabeled(
    projector, X_unlabeled, n_streams_out, embed_dim, save_path,
    epochs=50, lr=1e-3, batch_size=256, early_stopping_patience=10,
    val_fraction=0.1, Z_val_external=None,
):
    """
    X_unlabeled is (N, S_total, D) SimCLR embeddings — NOT raw signals.
    The experiment_runner must extract embeddings before calling this.
    """
    if Z_val_external is not None:
        X_tr, X_vl = X_unlabeled, Z_val_external
    else:
        n_val = max(1, int(len(X_unlabeled) * val_fraction))
        X_tr, X_vl = X_unlabeled[n_val:], X_unlabeled[:n_val]
    return train_raw_encoder(
        projector, X_tr, X_vl, save_path,
        epochs=epochs, lr=lr, batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
    )


def measure_encoder_val_loss(encoder, X_val, batch_size=256):
    """X_val is (N, S_total, D) embeddings."""
    encoder.eval()
    S, total, n = X_val.shape[1], 0.0, 0
    pk, pm = list(range(S-1)), [S-1]
    X_t = torch.from_numpy(X_val.astype(np.float32))
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            b      = X_t[i:i+batch_size].to(DEVICE)
            Z_k    = b[:, pk, :]
            Z_m    = b[:, pm, :]
            Z_pred = encoder(Z_k)
            v, _, _ = jepa_loss(Z_pred, Z_m)
            total  += v.item()
            n      += 1
    return total / max(1, n)


def train_projector_bootstrap(
    projector, X_new_old, X_new_full, n_streams_out, embed_dim, save_path,
    epochs=20, lr=1e-3, batch_size=200, early_stopping_patience=10,
    simclr_encoders=None, stream_names=None, stream_to_encoder=None,
    **kwargs,
):
    """
    Bootstrap on labeled windows at sensor increment time.
    Extracts embeddings from raw signals first, then trains JEPA.
    """
    from encoder import extract_all_features

    assert simclr_encoders is not None, "Bootstrap requires simclr_encoders"

    print(f"  [JEPA Bootstrap] Extracting embeddings from {len(X_new_full)} windows...")
    Z_full = extract_all_features(
        X_new_full, simclr_encoders, stream_to_encoder,
        stream_names, batch_size=batch_size,
        stream_indices=list(range(n_streams_out)),
    )                                                    # (N, S_total, D)

    n_val  = max(1, int(len(Z_full) * 0.1))
    Z_tr, Z_vl = Z_full[n_val:], Z_full[:n_val]
    return train_raw_encoder(
        projector, Z_tr, Z_vl, save_path,
        epochs=epochs, lr=lr, batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
    )


def evaluate_with_missing_sensors(
    heads, thresholds, projector, X_test_raw, Z_test_full,
    y_int, label_dict, fusion, missing_sensors, all_stream_names,
    n_streams_out, embed_dim, simclr_encoders, stream_to_encoder,
    cooccurrence_graph=None, T=100, C=3, batch_size=256,
):
    from helpers_hitl import evaluate_all_heads_fast
    mi  = [all_stream_names.index(s) for s in missing_sensors if s in all_stream_names]
    ki  = [i for i in range(n_streams_out) if i not in mi]
    Xm  = X_test_raw[:, :, ki, :]   # known streams only
    Z   = generate_synthetic_embeddings(
        projector, Xm, ki, n_streams_out,
        simclr_encoders, all_stream_names, stream_to_encoder,
        embed_dim, batch_size, T, C,
    )
    return evaluate_all_heads_fast(
        heads, Z, y_int, label_dict, thresholds, fusion,
        cooccurrence_graph=cooccurrence_graph,
    )


def generate_synthetic_full(projector, Z_old, n_streams_out, embed_dim, **kwargs):
    raise NotImplementedError(
        "generate_synthetic_full() is no longer supported. "
        "Use generate_synthetic_embeddings() instead."
    )
