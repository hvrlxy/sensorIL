"""
E2_add_activity.py
==================
Add one new activity to an existing HITL-HAR system.

Reads all settings from paths.json / dataset config / hparams.json via config_loader.
HITL interactions (co-occurrence confirmation, ME marking) are handled by
hitl_simulation.py — swap that module to change simulation strategy.

Usage
-----
  python E2_add_activity.py --activity Treadmill_2mph_Lab --state output/DS_11_e1_state_early.pkl
"""
import sys
from pathlib import Path
# Ensure repo root (parent of scripts/) is on sys.path so config_loader and
# helpers resolve correctly regardless of where the script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))              # scripts/

import os
import time
import pickle
import argparse
import random
import numpy as np
import torch
from datetime import datetime

from scripts.misc.config_loader import cfg
from scripts.misc.logger import RunLogger
from scripts.misc.hitl_simulation import (
    simulate_cooccurrence_confirmation,
    should_retrain_for_fn,
)
from scripts.misc.helpers import create_dataset_file_split
from scripts.misc.helpers_hitl import (
    make_binary_labels, build_gated_head_from_features,
    train_head_fast, evaluate_head_fast,
    find_optimal_threshold_fast,
    FeatureCache,
    check_cooccurrence, retrain_head_fast,
    NegativeBuffer, ReplayBuffer, UnlabeledBuffer,
    evaluate_all_heads_fast, make_multilabel_binary,
    load_heads_from_state, save_head_weights,
)



# ─────────────────────────────────────────────────────────────────────────────
# STATE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_state(state_path: str) -> dict:
    print(f"Loading state from {state_path}...")
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    fusion = state["fusion"]
    D      = state.get("feature_dim")
    if D is None:
        raise ValueError("State missing 'feature_dim' — re-run E1.")
    state["heads"] = load_heads_from_state(state, fusion, cfg=cfg)
    print(f"  Loaded {len(state['heads'])} heads  D={D}")
    if state.get("sensor_incremented") and state.get("projector_path"):
        from scripts.misc.projector import load_projector
        hp = cfg.PROJECTOR_HPARAMS
        state["projector"] = load_projector(
            state["projector_path"], hidden_dim=hp.get("hidden_dim", 256)
        )
    else:
        state.setdefault("projector", None)
    state.setdefault("head_streams",             {})
    # full_streams and initial_streams are always set by experiment_runner
    # before calling run_add_activity. The fallback here is only reached
    # when running E2 standalone (outside experiment_runner), in which case
    # cfg.INITIAL_STREAM_NAMES is still the right default.
    state.setdefault("full_streams",    list(cfg.INITIAL_STREAM_NAMES))
    state.setdefault("initial_streams", list(cfg.INITIAL_STREAM_NAMES))
    state.setdefault("sensor_incremented",       False)
    state.setdefault("pre_increment_activities", [])
    # Unlabeled buffer persists across E2 calls so passive data accumulates
    if "unlabeled_buffer" not in state:
        ul_hp = cfg.UNLABELED_ENCODER_HPARAMS
        state["unlabeled_buffer"] = UnlabeledBuffer(
            max_windows=ul_hp.get("max_buffer_windows", 10_000)
        )
    return state


def save_state(state: dict, state_path: str):
    """Save updated weights for any new/retrained heads, then pickle metadata."""
    timestamp = state.get("timestamp")
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    for (activity, f), model in state["heads"].items():
        if state.get("_updated", {}).get((activity, f), False):
            wpath = save_head_weights(
                activity, f, model, cfg.WORKING_DIR, timestamp
            )
            state["weights_paths"][(activity, f)] = wpath

    state_to_save = {k: v for k, v in state.items()
                     if k not in ("heads", "_updated")}
    with open(state_path, "wb") as f:
        pickle.dump(state_to_save, f)
    print(f"  State saved to {state_path}")



# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# PASSIVE DATA INGESTION
# ─────────────────────────────────────────────────────────────────────────────

def _load_passive_into_buffer(
    unlabeled_buffer,
    cfg,
    encoders,          # kept for API compat — no longer used
    feat_cache,        # kept for API compat — no longer used
    n_streams_out: int,
    embed_dim: int,    # kept for API compat — no longer used
    np_val_raw,
    timestamp: str,
) -> None:
    """
    Scan cfg.UNLABELED_DATA_DIR for .npy files not yet ingested and add their
    RAW SIGNALS to the unlabeled buffer.

    The buffer now stores raw (N, T, S, C) windows — the MAE trains directly
    on raw signals, no SimCLR feature extraction needed.

    Gracefully skips if UNLABELED_DATA_DIR is not configured or is empty.
    """
    if unlabeled_buffer is None:
        return
    data_dir = cfg.UNLABELED_DATA_DIR
    if not data_dir:
        return

    from pathlib import Path

    passive_dir = Path(data_dir)
    if not passive_dir.exists():
        print(f"  [Passive] UNLABELED_DATA_DIR not found: {passive_dir} — skipping")
        return

    npy_files = sorted(passive_dir.rglob("*.npy"))
    if not npy_files:
        return

    if not hasattr(unlabeled_buffer, "_ingested_files"):
        unlabeled_buffer._ingested_files = set()

    new_files = [f for f in npy_files if str(f) not in unlabeled_buffer._ingested_files]
    if not new_files:
        return

    full_stream_cols = [cfg.FULL_DATASET_STREAMS.index(s)
                        for s in cfg.FULL_STREAM_NAMES]
    n_added = 0

    for fpath in new_files:
        try:
            X = np.load(str(fpath), allow_pickle=False)
        except Exception as e:
            print(f"  [Passive] Could not load {fpath.name}: {e} — skipping")
            continue

        if X.ndim == 3:
            X = X[None]
        if X.ndim != 4:
            print(f"  [Passive] Unexpected shape {X.shape} in {fpath.name} — skipping")
            continue

        try:
            # Store raw signals — MAE trains on raw (N, T, S, C)
            X_sliced = X[:, :, full_stream_cols, :].astype(np.float32)
            unlabeled_buffer.add(X_sliced)
            n_added += X_sliced.shape[0]
            unlabeled_buffer._ingested_files.add(str(fpath))
        except Exception as e:
            print(f"  [Passive] Failed for {fpath.name}: {e} — skipping")

    if n_added > 0:
        print(f"  [Passive] Ingested {n_added} windows from {len(new_files)} file(s)  "
              f"buffer={len(unlabeled_buffer)}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HITL STEP
# ─────────────────────────────────────────────────────────────────────────────

def run_add_activity(new_activity, state,
                     feat_cache, np_train_raw, np_val_raw, np_test_raw,
                     label_dict, logger: RunLogger, encoders=None,
                     timestamp=None):
    from scripts.misc.helpers_hitl import (
        make_binary_labels, build_gated_head_from_features,
        train_head_fast, evaluate_head_fast,
        find_optimal_threshold_fast, FeatureCache,
        check_cooccurrence, retrain_head_fast,
        NegativeBuffer, ReplayBuffer, UnlabeledBuffer,
        evaluate_all_heads_fast, make_multilabel_binary,
        load_heads_from_state, save_head_weights,
    )
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    TIMESTAMP = timestamp

    fusion             = state["fusion"]
    heads              = state["heads"]
    thresholds         = state["thresholds"]
    replay_buffer      = state["replay_buffer"]
    trained_activities = state["trained_activities"]
    D                  = state["feature_dim"]

    if new_activity not in label_dict:
        logger.warn(f"'{new_activity}' not in label_dict — skipping.")
        return state, None, None

    class_idx = label_dict[new_activity]
    mask_val  = (np_val_raw[1]   == class_idx)
    mask_tr   = (np_train_raw[1] == class_idx)
    Z_new_val = feat_cache.val[mask_val]

    print(f"\n{'='*60}")
    print(f"Adding: '{new_activity}'")
    logger.event("INFO", f"Adding activity: '{new_activity}'")

    if Z_new_val.shape[0] == 0:
        logger.warn(f"No val samples for '{new_activity}' — skipping.")
        return state, None, None

    step_info = {"activity": new_activity}

    # ── Sensor increment state ────────────────────────────────────────────────
    projector          = state.get("projector")
    head_streams       = state.get("head_streams", {})
    full_streams       = state.get("full_streams",    cfg.FULL_STREAM_NAMES)
    initial_streams    = state.get("initial_streams", list(cfg.INITIAL_STREAM_NAMES))
    n_streams_out      = len(full_streams)
    sensor_incremented = state.get("sensor_incremented", False)
    unlabeled_buffer   = state.get("unlabeled_buffer")

    # ── Bootstrap projector on first activity after sensor add ─────────────────
    _streams_expanded = (len(full_streams) > len(initial_streams))
    if _streams_expanded and not sensor_incremented:
        from scripts.misc.projector import (build_projector, train_projector_bootstrap,
                               train_encoder_on_unlabeled, save_projector,
                               train_projection_head, extract_mae_features)

        sensor_incremented                = True
        state["sensor_incremented"]       = True
        state["pre_increment_activities"] = list(trained_activities)
        full_streams                      = list(full_streams)
        n_streams_out                     = len(full_streams)
        n_initial                         = len(initial_streams)
        hp    = cfg.PROJECTOR_HPARAMS
        ul_hp = cfg.UNLABELED_ENCODER_HPARAMS
        full_stream_cols = [cfg.FULL_DATASET_STREAMS.index(s) for s in full_streams]
        init_cols        = [cfg.FULL_DATASET_STREAMS.index(s) for s in initial_streams]

        if state.get("projector_from_sweep") and projector is not None \
                and hasattr(projector, "proj_head"):
            # ── Sweep projector available — skip MAE retraining ───────────────
            # The sweep already trained a good MAE-2 on unlabeled data and
            # retrained the seed heads.  Just mark sensor_incremented and
            # rebuild feat_cache with the sweep projector's features.
            print(f"\n[Sensor increment] Using sweep projector — skipping bootstrap MAE.")
            Z_mae_tr_full = extract_mae_features(
                projector, np_train_raw[0][:, :, full_stream_cols, :],
                stream_indices=list(range(n_streams_out)),
                batch_size=hp.get("batch_size", 256),
            )
            Z_mae_vl_full = extract_mae_features(
                projector, np_val_raw[0][:, :, full_stream_cols, :],
                stream_indices=list(range(n_streams_out)),
                batch_size=hp.get("batch_size", 256),
            )
            Z_mae_te_full = extract_mae_features(
                projector, np_test_raw[0][:, :, full_stream_cols, :],
                stream_indices=list(range(n_streams_out)),
                batch_size=hp.get("batch_size", 256),
            )
            feat_cache = FeatureCache(Z_mae_tr_full, Z_mae_vl_full, Z_mae_te_full)
            D = Z_mae_tr_full.shape[-1]
            state["feature_dim"] = D
            state["head_streams"] = {
                (act, fusion): list(full_streams)
                for act in trained_activities
                if (act, fusion) in heads
            }
            print(f"  feat_cache rebuilt from sweep projector: {Z_mae_tr_full.shape}")

        else:
            print(f"\n[Sensor increment] First activity — bootstrapping projector")

            hp    = cfg.PROJECTOR_HPARAMS
            ul_hp = cfg.UNLABELED_ENCODER_HPARAMS
            projector = build_projector(n_initial, n_streams_out, D,
                                        hp.get("hidden_dim", 64),
                                        proj_dim=hp.get("proj_dim", 96),
                                        T=np_val_raw[0].shape[1],
                                        C=np_val_raw[0].shape[3])
    
            proj_path_b = os.path.join(cfg.WORKING_DIR,
                                       f"{TIMESTAMP}_projector_bootstrap.pt")
    
            # Load passive unlabeled data (raw signals)
            _load_passive_into_buffer(
                unlabeled_buffer, cfg, None, None, n_streams_out, D,
                np_val_raw, TIMESTAMP
            )
            Z_passive = unlabeled_buffer.get() if unlabeled_buffer is not None else None
    
            full_stream_cols = [cfg.FULL_DATASET_STREAMS.index(s) for s in full_streams]
            X_boot_full = np_val_raw[0][:, :, full_stream_cols, :]
    
            if Z_passive is not None and len(Z_passive) >= ul_hp["min_new_windows"]:
                print(f"  [Bootstrap] Training encoder on {len(Z_passive)} passive windows")
                projector = train_encoder_on_unlabeled(
                    projector=projector, X_unlabeled=Z_passive,
                    n_streams_out=n_streams_out, embed_dim=D,
                    save_path=proj_path_b,
                    epochs=ul_hp["epochs"], lr=ul_hp["learning_rate"],
                    batch_size=ul_hp["batch_size"],
                    early_stopping_patience=ul_hp["early_stopping_patience"],
                    val_fraction=ul_hp["val_fraction"],
                )
                if unlabeled_buffer is not None:
                    unlabeled_buffer.mark_trained()
            else:
                n_passive = len(Z_passive) if Z_passive is not None else 0
                print(f"  [Bootstrap] Passive buffer too small ({n_passive} windows, "
                      f"need {ul_hp['min_new_windows']}) — falling back to val-set bootstrap")
                projector = train_projector_bootstrap(
                    projector=projector, X_new_old=None, X_new_full=X_boot_full,
                    n_streams_out=n_streams_out, embed_dim=D, save_path=proj_path_b,
                    epochs=hp.get("bootstrap_epochs", 50),
                    lr=hp.get("learning_rate", 1e-3),
                    batch_size=hp.get("batch_size", 200),
                    early_stopping_patience=hp.get("early_stopping_patience", 10),
                )
    
            # Train projection head on bootstrap data
            proj_head_path = proj_path_b.replace(".pt", "_proj.pt")
            projector = train_projection_head(
                projector, X_boot_full,
                X_boot_full[:max(1, int(len(X_boot_full)*0.1))],
                proj_head_path,
                stream_indices=list(range(n_streams_out)),
                epochs=hp.get("proj_epochs", 30),
                lr=hp.get("proj_lr", 1e-3),
                batch_size=hp.get("batch_size", 200),
            )
    
            save_projector(projector, proj_path_b)
            state["projector"]      = projector
            state["projector_path"] = proj_path_b
    
            # ── Retrain old-activity heads on MAE features ─────────────────────────
            # Old activities have only initial-stream samples.
            # Extract MAE features with missing stream zero-padded.
            init_cols      = [cfg.FULL_DATASET_STREAMS.index(s) for s in initial_streams]
            cooc_graph_b   = cfg._dataset["cooccurrence_graph"]
            Z_mae_tr_old   = extract_mae_features(
                projector, np_train_raw[0][:, :, init_cols, :],
                stream_indices=list(range(n_initial)),
                batch_size=hp.get("batch_size", 256),
            )
            Z_mae_vl_old   = extract_mae_features(
                projector, np_val_raw[0][:, :, init_cols, :],
                stream_indices=list(range(n_initial)),
                batch_size=hp.get("batch_size", 256),
            )
            Z_mae_te_old   = extract_mae_features(
                projector, np_test_raw[0][:, :, init_cols, :],
                stream_indices=list(range(n_initial)),
                batch_size=hp.get("batch_size", 256),
            )
            mae_D = Z_mae_tr_old.shape[-1]
            D     = mae_D
            state["feature_dim"] = D
    
            for activity_b in list(trained_activities):
                if (activity_b, fusion) not in heads or activity_b not in label_dict:
                    continue
                class_idx_b = label_dict[activity_b]
                y_tr_b = (np_train_raw[1] == class_idx_b).astype(np.int32)
                y_vl_b = (np_val_raw[1]   == class_idx_b).astype(np.int32)
                y_te_b = make_multilabel_binary(
                    activity_b, np_test_raw[1], label_dict,
                    cooccurrence_graph=cooc_graph_b)
                n_pos_b = int(y_tr_b.sum())
                if n_pos_b == 0 or y_vl_b.sum() == 0:
                    continue
                max_neg_b = min(int((y_tr_b==0).sum()), 10 * n_pos_b)
                pos_idx_b = np.where(y_tr_b == 1)[0]
                neg_idx_b = np.where(y_tr_b == 0)[0]
                if len(neg_idx_b) > max_neg_b:
                    neg_idx_b = np.random.default_rng(42).choice(
                        neg_idx_b, size=max_neg_b, replace=False)
                keep_b   = np.concatenate([pos_idx_b, neg_idx_b])
                Z_tr_bal = Z_mae_tr_old[keep_b]
                y_tr_bal = y_tr_b[keep_b]
    
                save_path_b = os.path.join(
                    cfg.WORKING_DIR,
                    f"{TIMESTAMP}_mae_head_{activity_b.replace(' ','_')}_{fusion}.pt"
                )
                best_model_b, best_auc_b = None, -1.0
                for seed_b in [42, 123, 777]:
                    torch.manual_seed(seed_b)
                    cand_b = build_gated_head_from_features(
                        [1.0]*n_streams_out, mae_D, fusion=fusion)
                    cand_b = train_head_fast(
                        cand_b, Z_tr_bal, y_tr_bal, Z_mae_vl_old, y_vl_b,
                        save_path_b,
                        epochs=hp.get("retrain_epochs", 20),
                        lr=hp.get("learning_rate", 1e-3),
                        batch_size=hp.get("batch_size", 200),
                        focal_gamma=cfg.FOCAL_GAMMA,
                        max_class_weight=cfg.MAX_CLASS_WEIGHT,
                        early_stopping_patience=hp.get("early_stopping_patience", 10),
                    )
                    m_b = evaluate_head_fast(cand_b, Z_mae_vl_old, y_vl_b, threshold=0.5)
                    if m_b["auc"] > best_auc_b:
                        best_auc_b, best_model_b = m_b["auc"], cand_b
    
                thresh_b = find_optimal_threshold_fast(
                    best_model_b, Z_mae_vl_old, y_vl_b,
                    t_min=cfg.THRESHOLD_MIN, t_max=cfg.THRESHOLD_MAX,
                    fallback=cfg.THRESHOLD_FALLBACK,
                )
                heads[(activity_b, fusion)]      = best_model_b
                thresholds[(activity_b, fusion)] = thresh_b
                head_streams[(activity_b, fusion)] = list(full_streams)
                m_b = evaluate_head_fast(best_model_b, Z_mae_te_old, y_te_b,
                                         threshold=thresh_b)
                print(f"    '{activity_b}'  AUC:{m_b['auc']:.4f}  F1:{m_b['f1']:.4f}")
    
            state["head_streams"] = head_streams
            # Update state heads with retrained 3-stream heads
            state["heads"] = heads
            state["thresholds"] = thresholds
    
            # Rebuild feat_cache with MAE full-stream features for new activity training
            Z_mae_tr_full = extract_mae_features(
                projector, np_train_raw[0][:, :, full_stream_cols, :],
                stream_indices=list(range(n_streams_out)),
                batch_size=hp.get("batch_size", 256),
            )
            Z_mae_vl_full = extract_mae_features(
                projector, np_val_raw[0][:, :, full_stream_cols, :],
                stream_indices=list(range(n_streams_out)),
                batch_size=hp.get("batch_size", 256),
            )
            Z_mae_te_full = extract_mae_features(
                projector, np_test_raw[0][:, :, full_stream_cols, :],
                stream_indices=list(range(n_streams_out)),
                batch_size=hp.get("batch_size", 256),
            )
            from scripts.misc.helpers_hitl import FeatureCache
            feat_cache = FeatureCache(Z_mae_tr_full, Z_mae_vl_full, Z_mae_te_full)
    
            # Save retrained head weights
            for act_b in trained_activities:
                if (act_b, fusion) in heads:
                    wpath_b = save_head_weights(act_b, fusion, heads[(act_b, fusion)],
                                                cfg.WORKING_DIR, TIMESTAMP)
                    state["weights_paths"][(act_b, fusion)] = wpath_b
            print(f"  Bootstrap done — seed heads retrained on MAE features.")

    # ── Step 1: Co-occurrence check ───────────────────────────────────────────
    print(f"\n[Step 1] Co-occurrence check")
    cooc_results = check_cooccurrence(
        new_activity, Z_new_val, heads, thresholds,
        fusion=fusion,
        fire_threshold=cfg.COOC_FIRE_THRESHOLD,
    )
    print(f"  {'Activity':<40} {'Mean P':>7} {'%Pos':>6} {'Fires':>6} {'Thresh':>7}")
    print(f"  {'-'*65}")
    for act, r in sorted(cooc_results.items()):
        print(f"  {act:<40} {r['mean_prob']:>7.3f} "
              f"{r['pct_positive']:>6.2%} "
              f"{'✓' if r['fires'] else '✗':>6} "
              f"{r['threshold']:>7.2f}")

    # ── Step 2: User confirms co-occurrences (simulated) ─────────────────────
    print(f"\n[Step 2] User confirms co-occurrences")
    confirmed, missed, false_pos = simulate_cooccurrence_confirmation(
        new_activity, cooc_results, trained_activities, cfg
    )
    step_info.update({"confirmed": confirmed, "missed": missed, "false_pos": false_pos})

    # ── Pre-step snapshot: capture head state BEFORE projector update ──────────
    # This is used as BEF in the OOD table so changes from synthetic retrain show up
    pre_val_ood_before_proj  = evaluate_all_heads_fast(
        heads, feat_cache.val,  np_val_raw[1],  label_dict, thresholds, fusion,
        cooccurrence_graph=cfg._dataset["cooccurrence_graph"])
    pre_test_ood_before_proj = evaluate_all_heads_fast(
        heads, feat_cache.test, np_test_raw[1], label_dict, thresholds, fusion,
        cooccurrence_graph=cfg._dataset["cooccurrence_graph"])

    # ── Step 2b: Update encoder + retrain old heads on synthetic data ──────────
    if sensor_incremented:
        from scripts.misc.projector import (train_projector_bootstrap, train_encoder_on_unlabeled,
                               save_projector, train_projection_head,
                               extract_mae_features)
        hp    = cfg.PROJECTOR_HPARAMS
        ul_hp = cfg.UNLABELED_ENCODER_HPARAMS
        proj_path = os.path.join(cfg.WORKING_DIR,
            f"{TIMESTAMP}_projector_after_{new_activity.replace(' ','_')}.pt")

        # Column indices for initial streams in the raw lab data
        full_stream_cols    = [cfg.FULL_DATASET_STREAMS.index(s) for s in full_streams]
        initial_stream_cols = [cfg.FULL_DATASET_STREAMS.index(s) for s in initial_streams]
        known_indices       = list(range(len(initial_streams)))

        # ── Step 2b-0: Ingest passive windows into unlabeled buffer ──────────
        _load_passive_into_buffer(
            unlabeled_buffer, cfg, encoders, feat_cache, n_streams_out, D,
            np_val_raw, TIMESTAMP
        )
        logger.event("INFO",
            f"UnlabeledBuffer: {unlabeled_buffer}" if unlabeled_buffer else
            "UnlabeledBuffer: not initialised")

        # ── Step 2b-i: Re-train encoder only if passive buffer has grown enough ──
        # The encoder was trained during the sweep on unlabeled data and is
        # kept frozen between steps. Only update it when significant new passive
        # data has accumulated. Never fine-tune on labeled activity windows.
        encoder_updated = False
        if (unlabeled_buffer is not None and
                unlabeled_buffer.new_since_last_train() >= ul_hp["min_new_windows"]):
            X_passive = unlabeled_buffer.get()
            print(f"\n[Step 2b-i] Re-training encoder on {len(X_passive)} "
                  f"passive raw windows ({unlabeled_buffer.new_since_last_train()} new)")
            projector = train_encoder_on_unlabeled(
                projector=projector,
                X_unlabeled=X_passive,
                n_streams_out=n_streams_out, embed_dim=D,
                save_path=proj_path,
                epochs=ul_hp["epochs"],
                lr=ul_hp["learning_rate"],
                batch_size=ul_hp["batch_size"],
                early_stopping_patience=ul_hp["early_stopping_patience"],
                val_fraction=ul_hp["val_fraction"],
            )
            unlabeled_buffer.mark_trained()
            encoder_updated = True
        # No fallback — keep sweep encoder frozen if buffer insufficient

        if encoder_updated:
            save_projector(projector, proj_path)
            state["projector"]      = projector
            state["projector_path"] = proj_path

            # Re-train projection head with updated encoder
            proj_head_path = proj_path.replace(".pt", "_proj.pt")
            full_stream_cols_2b = [cfg.FULL_DATASET_STREAMS.index(s)
                                   for s in full_streams]
            X_boot_2b = np_val_raw[0][:, :, full_stream_cols_2b, :]
            projector = train_projection_head(
                projector, X_boot_2b,
                X_boot_2b[:max(1, int(len(X_boot_2b)*0.1))],
                proj_head_path,
                stream_indices=list(range(n_streams_out)),
                epochs=hp.get("proj_epochs", 30),
                lr=hp.get("proj_lr", 1e-3),
                batch_size=hp.get("batch_size", 200),
            )

            # ── Step 2b-iii: Retrain old heads on updated MAE features ──────
            print(f"\n[Step 2b-iii] Re-extracting MAE features + retraining heads")
            Z_mae_tr_2b = extract_mae_features(
                projector, np_train_raw[0][:, :, initial_stream_cols, :],
                stream_indices=list(range(len(initial_streams))),
                batch_size=hp.get("batch_size", 256),
            )
            Z_mae_vl_2b = extract_mae_features(
                projector, np_val_raw[0][:, :, initial_stream_cols, :],
                stream_indices=list(range(len(initial_streams))),
                batch_size=hp.get("batch_size", 256),
            )
            mae_D_2b = Z_mae_tr_2b.shape[-1]

            pre_acts_2b = set(state.get("pre_increment_activities", []))
            for act_2b in list(pre_acts_2b):
                if (act_2b, fusion) not in heads or act_2b not in label_dict:
                    continue
                class_idx_2b = label_dict[act_2b]
                y_tr_2b = (np_train_raw[1] == class_idx_2b).astype(np.int32)
                y_vl_2b = (np_val_raw[1]   == class_idx_2b).astype(np.int32)
                n_pos_2b = int(y_tr_2b.sum())
                if n_pos_2b == 0 or y_vl_2b.sum() == 0:
                    continue
                max_neg_2b = min(int((y_tr_2b==0).sum()), 10*n_pos_2b)
                pos_idx_2b = np.where(y_tr_2b==1)[0]
                neg_idx_2b = np.where(y_tr_2b==0)[0]
                if len(neg_idx_2b) > max_neg_2b:
                    neg_idx_2b = np.random.default_rng(42).choice(
                        neg_idx_2b, size=max_neg_2b, replace=False)
                keep_2b  = np.concatenate([pos_idx_2b, neg_idx_2b])
                save_2b  = os.path.join(
                    cfg.WORKING_DIR,
                    f"{TIMESTAMP}_mae_head2b_{act_2b.replace(' ','_')}_{fusion}.pt"
                )
                best_2b, best_auc_2b = None, -1.0
                for seed_2b in [42, 123, 777]:
                    torch.manual_seed(seed_2b)
                    cand_2b = build_gated_head_from_features(
                        [1.0]*n_streams_out, mae_D_2b, fusion=fusion)
                    cand_2b = train_head_fast(
                        cand_2b, Z_mae_tr_2b[keep_2b], y_tr_2b[keep_2b],
                        Z_mae_vl_2b, y_vl_2b, save_2b,
                        epochs=hp.get("retrain_epochs", 20),
                        lr=hp.get("learning_rate", 1e-3),
                        batch_size=hp.get("batch_size", 200),
                        focal_gamma=cfg.FOCAL_GAMMA,
                        max_class_weight=cfg.MAX_CLASS_WEIGHT,
                        early_stopping_patience=hp.get("early_stopping_patience", 10),
                    )
                    m_2b = evaluate_head_fast(cand_2b, Z_mae_vl_2b, y_vl_2b,
                                              threshold=0.5)
                    if m_2b["auc"] > best_auc_2b:
                        best_auc_2b, best_2b = m_2b["auc"], cand_2b
                thresh_2b = find_optimal_threshold_fast(
                    best_2b, Z_mae_vl_2b, y_vl_2b,
                    t_min=cfg.THRESHOLD_MIN, t_max=cfg.THRESHOLD_MAX,
                    fallback=cfg.THRESHOLD_FALLBACK,
                )
                heads[(act_2b, fusion)]      = best_2b
                thresholds[(act_2b, fusion)] = thresh_2b

    # ── Step 3: Pre-training OOD snapshot ─────────────────────────────────────
    print(f"\n[Step 5] Pre-training OOD evaluation")
    # Use snapshot from before step 2b so BEF reflects state before
    # projector update AND synthetic head retraining
    pre_val_ood  = pre_val_ood_before_proj
    pre_test_ood = pre_test_ood_before_proj

    # ── Step 6: Retrain for FNs ───────────────────────────────────────────────
    if missed:
        print(f"\n[Step 6] Retraining for FNs: {missed}")
        for activity in missed:
            if not should_retrain_for_fn(activity, cfg):
                continue
            if (activity, fusion) not in heads:
                continue
            y_val_bin = (np_val_raw[1] == label_dict[activity]).astype(np.int32)
            heads[(activity, fusion)], _, _ = retrain_head_fast(
                activity=activity,
                model=heads[(activity, fusion)],
                replay_buffer=replay_buffer,
                Z_new=Z_new_val, y_new_label=1,
                Z_val=feat_cache.val, y_val=y_val_bin,
                Z_train_all=feat_cache.train, y_train_int=np_train_raw[1],
                label_dict=label_dict,
                trained_activities=trained_activities,
                epochs=cfg.RETRAIN_EPOCHS,
                lr=cfg.RETRAIN_LR,
                timestamp=TIMESTAMP,
                working_dir=cfg.WORKING_DIR,
            )
            thresholds[(activity, fusion)] = find_optimal_threshold_fast(
                heads[(activity, fusion)], feat_cache.val, y_val_bin,
                t_min=cfg.THRESHOLD_MIN, t_max=cfg.THRESHOLD_MAX,
                fallback=cfg.THRESHOLD_FALLBACK,
            )
            state.setdefault("_updated", {})[(activity, fusion)] = True
    else:
        print(f"\n[Step 6] No FNs — no retraining needed.")

    # ── Step 7: Retrain for FPs ───────────────────────────────────────────────
    if false_pos:
        print(f"\n[Step 7] Retraining for FPs: {false_pos}")
        for activity in false_pos:
            if (activity, fusion) not in heads:
                continue
            y_val_bin = (np_val_raw[1] == label_dict[activity]).astype(np.int32)
            heads[(activity, fusion)], _, _ = retrain_head_fast(
                activity=activity,
                model=heads[(activity, fusion)],
                replay_buffer=replay_buffer,
                Z_new=Z_new_val, y_new_label=0,
                Z_val=feat_cache.val, y_val=y_val_bin,
                Z_train_all=feat_cache.train, y_train_int=np_train_raw[1],
                label_dict=label_dict,
                trained_activities=trained_activities,
                epochs=cfg.RETRAIN_EPOCHS,
                lr=cfg.RETRAIN_LR,
                timestamp=TIMESTAMP,
                working_dir=cfg.WORKING_DIR,
            )
            thresholds[(activity, fusion)] = find_optimal_threshold_fast(
                heads[(activity, fusion)], feat_cache.val, y_val_bin,
                t_min=cfg.THRESHOLD_MIN, t_max=cfg.THRESHOLD_MAX,
                fallback=cfg.THRESHOLD_FALLBACK,
            )
            state.setdefault("_updated", {})[(activity, fusion)] = True
    else:
        print(f"\n[Step 7] No FPs — no retraining needed.")

    # ── Step 8: Train new activity head ───────────────────────────────────────
    print(f"\n[Step 8] Training head for '{new_activity}'")
    excl_idxs = {label_dict[a]
                 for a in cfg.get_training_exclusions(new_activity)
                 if a in label_dict}
    tr_mask   = np.array([i == class_idx or i not in excl_idxs
                          for i in np_train_raw[1]])
    vl_mask_t = np.array([i == class_idx or i not in excl_idxs
                          for i in np_val_raw[1]])
    y_tr_new  = (np_train_raw[1][tr_mask]  == class_idx).astype(np.int32)
    y_vl_new  = (np_val_raw[1][vl_mask_t]  == class_idx).astype(np.int32)
    Z_tr_new  = feat_cache.train[tr_mask]
    Z_vl_new  = feat_cache.val[vl_mask_t]

    n_hint = len(full_streams) if sensor_incremented else len(initial_streams)
    hint   = [1.0] * n_hint  # no gating
    model  = build_gated_head_from_features(hint, D, fusion=fusion)
    safe_name = new_activity.replace(" ", "_")
    save_path = os.path.join(cfg.WORKING_DIR,
                             f"{TIMESTAMP}_head_{safe_name}_{fusion}.pt")

    t0    = time.time()
    model = train_head_fast(
        model, Z_tr_new, y_tr_new,
        Z_vl_new, y_vl_new,
        save_path,
        epochs=cfg.INCREMENTAL_HEAD_EPOCHS,
        lr=cfg.LEARNING_RATE,
        focal_gamma=cfg.FOCAL_GAMMA,
        max_class_weight=cfg.MAX_CLASS_WEIGHT,
    )
    print(f"  Trained in {time.time()-t0:.1f}s")

    thresh = find_optimal_threshold_fast(
        model, Z_vl_new, y_vl_new,
        t_min=cfg.THRESHOLD_MIN, t_max=cfg.THRESHOLD_MAX,
        fallback=cfg.THRESHOLD_FALLBACK,
    )
    y_te_new    = (np_test_raw[1] == class_idx).astype(np.int32)
    new_metrics = evaluate_head_fast(model, feat_cache.test, y_te_new,
                                     threshold=thresh)
    print(f"  Threshold: {thresh:.2f}  "
          f"AUC:{new_metrics['auc']:.4f} F1:{new_metrics['f1']:.4f}")
    step_info["new_head_metrics"] = new_metrics
    step_info["threshold"]        = thresh

    heads[(new_activity, fusion)]      = model
    thresholds[(new_activity, fusion)] = thresh
    wpath = save_head_weights(new_activity, fusion, model,
                              cfg.WORKING_DIR, TIMESTAMP)
    state["weights_paths"][(new_activity, fusion)] = wpath
    # Store positives for new head using full streams (new head is 2-stream)
    Z_pos_new = feat_cache.train[np_train_raw[1] == class_idx]
    replay_buffer.store_positives(new_activity, Z_pos_new)

    # Also store projected 1-stream version under a separate key so old-head
    # retrains can use it as negatives without shape mismatch.
    # We do this by ensuring feat_cache.train is consistent for old head retraining.
    trained_activities.append(new_activity)
    state.setdefault("_updated", {})[(new_activity, fusion)] = True
    head_streams[(new_activity, fusion)] = (
        list(full_streams) if sensor_incremented else list(initial_streams)
    )
    state["head_streams"] = head_streams

    # ── Step 9: Recalibrate thresholds for retrained heads ────────────────────
    retrained = set(missed) | set(false_pos)
    if retrained:
        trained_idxs = {label_dict[a] for a in trained_activities if a in label_dict}
        mask_seen    = np.array([i in trained_idxs for i in np_val_raw[1]])
        y_val_seen   = np_val_raw[1][mask_seen]
        Z_val_seen   = feat_cache.val[mask_seen]
        Z_val_seen_old = feat_cache.val[mask_seen]  # projected for old heads
        for activity in retrained:
            if (activity, fusion) not in heads or activity not in label_dict:
                continue
            y_bin = make_multilabel_binary(activity, y_val_seen, label_dict, cooccurrence_graph=cfg._dataset["cooccurrence_graph"])
            if y_bin.sum() == 0:
                continue
            h_streams  = head_streams.get((activity, fusion), full_streams)
            Z_recal    = Z_val_seen
            thresholds[(activity, fusion)] = find_optimal_threshold_fast(
                heads[(activity, fusion)], Z_recal, y_bin,
                t_min=cfg.THRESHOLD_MIN, t_max=cfg.THRESHOLD_MAX,
                fallback=cfg.THRESHOLD_FALLBACK,
            )

    # ── Step 10/11: OOD evaluation ─────────────────────────────────────────────
    print(f"\n[Step 10] Val OOD evaluation")
    val_ood = evaluate_all_heads_fast(
        heads, feat_cache.val, np_val_raw[1], label_dict, thresholds, fusion,
        cooccurrence_graph=cfg._dataset["cooccurrence_graph"])
    print(f"\n[Step 11] Test OOD evaluation")
    test_ood = evaluate_all_heads_fast(
        heads, feat_cache.test, np_test_raw[1], label_dict, thresholds, fusion,
        cooccurrence_graph=cfg._dataset["cooccurrence_graph"])

    step_info["pre_val_ood"]   = pre_val_ood
    step_info["post_val_ood"]  = val_ood
    step_info["pre_test_ood"]  = pre_test_ood
    step_info["post_test_ood"] = test_ood
    step_info["retrained"]     = sorted(retrained)

    # ── Before/After table ────────────────────────────────────────────────────
    all_acts = sorted(set(pre_val_ood) | set(val_ood))
    print(f"\n  {'─'*112}")
    print(f"  {'Activity':<45} {'':^4} {'':^6} "
          f"{'Val AUC':>8} {'Val F1':>7} {'Tst AUC':>8} {'Tst F1':>7}")
    print(f"  {'─'*112}")
    for act in all_acts:
        seed_tag = " *" if act in set(cfg.SEED_ACTIVITIES) else ""
        role     = "[NEW]" if act == new_activity else \
                   ("[RTR]" if act in retrained else "")
        pre_v, pre_t = pre_val_ood.get(act, {}), pre_test_ood.get(act, {})
        pst_v, pst_t = val_ood.get(act, {}),     test_ood.get(act, {})
        def fmt(d, k): return f"{d[k]:>7.4f}" if k in d else f"{'—':>7}"
        print(f"  {act+seed_tag:<45} {'BEF':>4} {role:>6} "
              f"{fmt(pre_v,'auc'):>8} {fmt(pre_v,'f1'):>7} "
              f"{fmt(pre_t,'auc'):>8} {fmt(pre_t,'f1'):>7}")
        print(f"  {'':<45} {'AFT':>4} {role:>6} "
              f"{fmt(pst_v,'auc'):>8} {fmt(pst_v,'f1'):>7} "
              f"{fmt(pst_t,'auc'):>8} {fmt(pst_t,'f1'):>7}")
        # Always print delta
        d_val_auc = pst_v.get('auc',0) - pre_v.get('auc',0) if pre_v else 0
        d_val_f1  = pst_v.get('f1', 0) - pre_v.get('f1', 0) if pre_v else 0
        d_tst_auc = pst_t.get('auc',0) - pre_t.get('auc',0) if pre_t else 0
        d_tst_f1  = pst_t.get('f1', 0) - pre_t.get('f1', 0) if pre_t else 0
        print(f"  {'':<45} {'Δ':>4} {role:>6} "
              f"{d_val_auc:>+8.4f} {d_val_f1:>+7.4f} "
              f"{d_tst_auc:>+8.4f} {d_tst_f1:>+7.4f}")
        print(f"  {'·'*112}")

    # ── Update state ──────────────────────────────────────────────────────────
    state["heads"]              = heads
    state["thresholds"]         = thresholds
    state["replay_buffer"]      = replay_buffer
    state["trained_activities"] = trained_activities
    state.setdefault("ood_history",          {})[new_activity] = val_ood
    state.setdefault("test_ood_history",     {})[new_activity] = test_ood
    state.setdefault("pre_ood_history",      {})[new_activity] = pre_val_ood
    state.setdefault("pre_test_ood_history", {})[new_activity] = pre_test_ood
    state.setdefault("head_birth_step",      {})[new_activity] = \
        len(trained_activities) - 1

    logger.log_activity_step(new_activity, step_info)
    return state, val_ood, test_ood


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if cfg is None:
        raise RuntimeError("configs/paths.json not found. Copy configs/paths.example.json to configs/paths.json and fill in your paths.")
    DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--activity", type=str, required=True,
                        help="Name of the activity to add (must exist in label_dict).")
    parser.add_argument("--state",    type=str,
                        default=None,
                        help="Path to .pkl state file from E1 or a previous E2 run.")
    args = parser.parse_args()

    logger = RunLogger(cfg.WORKING_DIR, run_id=TIMESTAMP, script="E2_add_activity")
    logger.log_run_start(cfg)

    print("Loading dataset...")
    np_train_raw, np_val_raw, np_test_raw, label_dict = create_dataset_file_split(
        cfg.DATA_DIR, participant_lst=cfg.PARTICIPANTS
    )

    # ── Load state ────────────────────────────────────────────────────────────
    state_path = args.state
    if state_path is None:
        import glob
        matches = sorted(glob.glob(
            os.path.join(cfg.WORKING_DIR, f"*_e1_state_{cfg.FUSION}.pkl")
        ))
        if not matches:
            raise FileNotFoundError(
                f"No E1 state file found in {cfg.WORKING_DIR}. "
                "Run E1_train_seeds.py first, or pass --state explicitly."
            )
        state_path = matches[-1]
        print(f"Auto-selected state: {state_path}")

    state = load_state(state_path)
    D     = state.get("feature_dim")
    if D is None:
        raise ValueError("State missing 'feature_dim' — re-run E1.")

    # ── Build feat_cache from MAE features ───────────────────────────────────
    # Use the projector stored in state to extract MAE features.
    # If no projector yet (pre-increment), feat_cache will be rebuilt at the
    # bootstrap step inside run_add_activity.
    encoders   = None   # SimCLR no longer used
    projector  = state.get("projector")
    full_streams_m = state.get("full_streams", list(cfg.FULL_STREAM_NAMES))
    full_cols_m    = [cfg.FULL_DATASET_STREAMS.index(s) for s in full_streams_m]

    if projector is not None and hasattr(projector, "proj_head"):
        from scripts.misc.projector import extract_mae_features
        print("Extracting MAE features for feat_cache...")
        Z_train = extract_mae_features(
            projector, np_train_raw[0][:, :, full_cols_m, :],
            stream_indices=list(range(len(full_streams_m))),
            batch_size=cfg.BATCH_SIZE,
        )
        Z_val   = extract_mae_features(
            projector, np_val_raw[0][:, :, full_cols_m, :],
            stream_indices=list(range(len(full_streams_m))),
            batch_size=cfg.BATCH_SIZE,
        )
        Z_test  = extract_mae_features(
            projector, np_test_raw[0][:, :, full_cols_m, :],
            stream_indices=list(range(len(full_streams_m))),
            batch_size=cfg.BATCH_SIZE,
        )
        D = Z_train.shape[-1]
    else:
        # Pre-increment: no projector yet — use zeros as placeholder.
        # run_add_activity bootstrap will rebuild feat_cache with MAE features.
        print("[WARN] No MAE projector in state — feat_cache will be rebuilt at bootstrap.")
        n_streams_ph = len(full_streams_m)
        proj_dim_ph  = cfg.PROJECTOR_HPARAMS.get("proj_dim", 96)
        N_tr = len(np_train_raw[0])
        N_vl = len(np_val_raw[0])
        N_te = len(np_test_raw[0])
        Z_train = np.zeros((N_tr, n_streams_ph, proj_dim_ph), dtype=np.float32)
        Z_val   = np.zeros((N_vl, n_streams_ph, proj_dim_ph), dtype=np.float32)
        Z_test  = np.zeros((N_te, n_streams_ph, proj_dim_ph), dtype=np.float32)

    feat_cache = FeatureCache(Z_train, Z_val, Z_test)
    state["feature_dim"] = D
    logger.event("INFO",
        f"feat_cache: train={Z_train.shape}  val={Z_val.shape}  test={Z_test.shape}"
    )

    state, val_ood, test_ood = run_add_activity(
        new_activity=args.activity,
        state=state,
        feat_cache=feat_cache,
        np_train_raw=np_train_raw,
        np_val_raw=np_val_raw,
        np_test_raw=np_test_raw,
        label_dict=label_dict,
        logger=logger,
        encoders=None,
        timestamp=TIMESTAMP,
    )

    safe_act  = args.activity.replace(" ", "_")
    out_path  = os.path.join(
        cfg.WORKING_DIR,
        f"{TIMESTAMP}_state_after_{safe_act}.pkl"
    )
    save_state(state, out_path)
    log_path = logger.save_alongside(out_path)

    print(f"\nDone.")
    print(f"  State : {out_path}")
    print(f"  Log   : {log_path}")
    print(f"\nNext run: python E2_add_activity.py --activity <NEXT_ACTIVITY> --state {out_path}")