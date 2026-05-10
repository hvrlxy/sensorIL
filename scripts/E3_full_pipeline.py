"""
E3_full_pipeline.py
===================
Runs the complete sensor-incremental HITL pipeline end to end:

  Phase 1 — Seed training (E1):
    Train binary heads for seed activities on initial n streams.

  Phase 2 — Sensor increment + incremental activity addition (E2):
    On the first activity after sensor_incremented=True:
      - Bootstrap projector on all available windows (labels ignored)
      - Retrain seed heads on synthetic n+1-stream data
    On every subsequent activity:
      - Update projector with new activity windows
      - Retrain all pre-increment heads on improved synthetic data
      - Add new activity head (real n+1-stream data)

  Tracks per-step metrics for all heads so you can plot F1 improvement
  of old heads over time as the projector improves.

Usage
-----
  python scripts/E3_full_pipeline.py

  Reads incremental_activities from dataset config in order.
  Set sensor_incremented=true in paths.json before running.

Output
------
  <ts>_e3_history.pkl     — full per-step metrics history
  <ts>_e3_history_log.json — run log with config snapshot
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os, time, pickle, random, glob
import numpy as np
import torch
from datetime import datetime

from config_loader import cfg
from logger import RunLogger
from helpers import create_dataset_file_split
from helpers_hitl import (
    FeatureCache, load_heads_from_state,
    evaluate_all_heads_fast, make_multilabel_binary,
    find_optimal_threshold_fast, evaluate_head_fast,
)
from E2_add_activity import run_add_activity, load_state, save_state

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# E1 — SEED TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def run_e1(np_train_raw, np_val_raw, np_test_raw, label_dict,
           feat_cache, D, logger, timestamp):
    from helpers_hitl import (
        build_gated_head_from_features, train_head_fast,
        find_optimal_threshold_fast, evaluate_head_fast,
        ReplayBuffer, save_head_weights,
    )

    print(f"\n{'='*60}")
    print("PHASE 1 — SEED TRAINING")
    print(f"{'='*60}")

    heads         = {}
    thresholds    = {}
    weights_paths = {}
    replay_buffer = ReplayBuffer()
    results       = {}

    seed_idxs = {label_dict[a] for a in cfg.SEED_ACTIVITIES if a in label_dict}

    for activity in cfg.SEED_ACTIVITIES:
        if activity not in label_dict:
            logger.warn(f"Seed '{activity}' not in label_dict — skipping.")
            continue

        act_idx     = label_dict[activity]
        other_seeds = [s for s in cfg.SEED_ACTIVITIES if s != activity]
        other_idxs  = {label_dict[s] for s in other_seeds if s in label_dict}
        keep_idxs   = {act_idx} | other_idxs
        excl_idxs   = {label_dict[a]
                       for a in cfg.get_training_exclusions(activity)
                       if a in label_dict}

        tr_mask   = np.array([i in keep_idxs and (i == act_idx or i not in excl_idxs)
                               for i in np_train_raw[1]])
        vl_mask   = np.array([i in keep_idxs and (i == act_idx or i not in excl_idxs)
                               for i in np_val_raw[1]])

        y_tr = (np_train_raw[1][tr_mask] == act_idx).astype(np.int32)
        y_vl = (np_val_raw[1][vl_mask]   == act_idx).astype(np.int32)
        # Full test set evaluation
        y_te = (np_test_raw[1] == act_idx).astype(np.int32)

        Z_tr = feat_cache.train[tr_mask]
        Z_vl = feat_cache.val[vl_mask]
        Z_te = feat_cache.test

        print(f"\nSeed: '{activity}'  "
              f"train pos:{y_tr.sum()} neg:{(y_tr==0).sum()}  "
              f"test pos:{y_te.sum()} neg:{(y_te==0).sum()}")

        hint      = [1.0] * len(cfg.INITIAL_STREAM_NAMES)
        model     = build_gated_head_from_features(hint, D, fusion=cfg.FUSION)
        safe      = activity.replace(" ", "_")
        save_path = os.path.join(cfg.WORKING_DIR,
                                 f"{timestamp}_seed_{safe}_{cfg.FUSION}.pt")

        t0    = time.time()
        model = train_head_fast(
            model, Z_tr, y_tr, Z_vl, y_vl, save_path,
            epochs=cfg.SEED_HEAD_EPOCHS, lr=cfg.LEARNING_RATE,
            focal_gamma=cfg.FOCAL_GAMMA, max_class_weight=cfg.MAX_CLASS_WEIGHT,
        )
        thresh  = find_optimal_threshold_fast(
            model, Z_vl, y_vl,
            t_min=cfg.THRESHOLD_MIN, t_max=cfg.THRESHOLD_MAX,
            fallback=cfg.THRESHOLD_FALLBACK,
        )
        metrics = evaluate_head_fast(model, Z_te, y_te, threshold=thresh)
        print(f"  AUC:{metrics['auc']:.4f}  F1:{metrics['f1']:.4f}  "
              f"thresh:{thresh:.2f}  ({time.time()-t0:.1f}s)")

        heads[(activity, cfg.FUSION)]      = model
        thresholds[(activity, cfg.FUSION)] = thresh
        wpath = save_head_weights(activity, cfg.FUSION, model,
                                  cfg.WORKING_DIR, timestamp)
        weights_paths[(activity, cfg.FUSION)] = wpath
        replay_buffer.store_positives(
            activity, feat_cache.train[np_train_raw[1] == act_idx]
        )
        results[activity] = metrics
        logger.log_seed_result(activity, metrics, thresh)

    print(f"\n{'─'*60}")
    print("E1 SUMMARY")
    print(f"{'Activity':<45} {'AUC':>7} {'F1':>7}")
    for act, m in results.items():
        print(f"  {act:<43} {m['auc']:>7.4f} {m['f1']:>7.4f}")

    state = {
        "weights_paths":             weights_paths,
        "thresholds":                thresholds,
        "trained_activities":        list(cfg.SEED_ACTIVITIES),
        "label_dict":                label_dict,
        "replay_buffer":             replay_buffer,
        "fusion":                    cfg.FUSION,
        "feature_dim":               D,
        "timestamp":                 timestamp,
        "dataset":                   cfg.DATASET_NAME,
        "heads":                     heads,
        "head_streams":              {},
        "projector":                 None,
        "projector_path":            None,
        "sensor_incremented":        False,
        "initial_streams":           cfg.INITIAL_STREAM_NAMES,
        "full_streams":              cfg.INITIAL_STREAM_NAMES,
        "pre_increment_activities":  [],
    }

    state_path = os.path.join(cfg.WORKING_DIR,
                              f"{timestamp}_e1_state_{cfg.FUSION}.pkl")
    state_to_save = {k: v for k, v in state.items() if k != "heads"}
    with open(state_path, "wb") as f:
        pickle.dump(state_to_save, f)
    logger.event("INFO", f"E1 state saved to {state_path}")
    return state, state_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if cfg is None:
        raise RuntimeError(
            "configs/paths.json not found. "
            "Copy configs/paths.example.json to configs/paths.json."
        )

    DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    logger = RunLogger(cfg.WORKING_DIR, run_id=TIMESTAMP, script="E3_full_pipeline")
    logger.log_run_start(cfg)
    logger.event("INFO", f"Device: {DEVICE}")
    logger.event("INFO",
        f"sensor_incremented={cfg.SENSOR_INCREMENTED}  "
        f"initial={cfg.INITIAL_STREAM_NAMES}  full={cfg.FULL_STREAM_NAMES}")
    cfg.summary()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    np_train_raw, np_val_raw, np_test_raw, label_dict = create_dataset_file_split(
        cfg.DATA_DIR, participant_lst=cfg.PARTICIPANTS
    )

    # ── Load encoder + extract features ───────────────────────────────────────
    print("\nLoading encoder(s)...")
    from encoder import load_encoders_from_cfg, extract_all_features
    encoders = load_encoders_from_cfg(cfg)

    # Always extract full streams — E1 will slice to initial streams
    print("\nExtracting features...")
    stream_names  = cfg.FULL_STREAM_NAMES
    stream_indices = cfg.get_stream_indices(stream_names)

    Z_train = extract_all_features(np_train_raw[0], encoders,
                                   cfg.STREAM_TO_ENCODER, stream_names,
                                   batch_size=cfg.BATCH_SIZE,
                                   stream_indices=stream_indices)
    Z_val   = extract_all_features(np_val_raw[0],   encoders,
                                   cfg.STREAM_TO_ENCODER, stream_names,
                                   batch_size=cfg.BATCH_SIZE,
                                   stream_indices=stream_indices)
    Z_test  = extract_all_features(np_test_raw[0],  encoders,
                                   cfg.STREAM_TO_ENCODER, stream_names,
                                   batch_size=cfg.BATCH_SIZE,
                                   stream_indices=stream_indices)
    D = Z_train.shape[-1]
    logger.event("INFO",
        f"Full features: train={Z_train.shape}  val={Z_val.shape}  test={Z_test.shape}")

    # For E1 we need initial-stream features only
    idx_initial   = cfg.get_stream_indices(cfg.INITIAL_STREAM_NAMES)
    Z_train_init  = Z_train[:, [cfg.FULL_STREAM_NAMES.index(s)
                                for s in cfg.INITIAL_STREAM_NAMES], :]
    Z_val_init    = Z_val[:,   [cfg.FULL_STREAM_NAMES.index(s)
                                for s in cfg.INITIAL_STREAM_NAMES], :]
    Z_test_init   = Z_test[:,  [cfg.FULL_STREAM_NAMES.index(s)
                                for s in cfg.INITIAL_STREAM_NAMES], :]

    feat_cache_init = FeatureCache(Z_train_init, Z_val_init, Z_test_init)
    feat_cache_full = FeatureCache(Z_train, Z_val, Z_test)

    # ── Phase 1: Seed training ─────────────────────────────────────────────────
    state, e1_path = run_e1(
        np_train_raw, np_val_raw, np_test_raw, label_dict,
        feat_cache_init, D, logger, TIMESTAMP
    )

    # ── Phase 2: Incremental activity addition ────────────────────────────────
    incremental_activities = cfg.INCREMENTAL_ACTIVITIES
    if not incremental_activities:
        raise ValueError("No incremental_activities defined in dataset config.")

    print(f"\n{'='*60}")
    print(f"PHASE 2 — INCREMENTAL ADDITION ({len(incremental_activities)} activities)")
    print(f"  sensor_incremented: {cfg.SENSOR_INCREMENTED}")
    print(f"{'='*60}")

    # History: track per-step metrics for all heads
    history = {
        "seed_metrics_e1":    {},   # E1 baseline (initial stream)
        "steps":              [],   # per-step info
        "final_metrics":      None,
    }

    # Record E1 baseline on initial-stream features
    for act in cfg.SEED_ACTIVITIES:
        if act not in label_dict: continue
        y_te  = (np_test_raw[1] == label_dict[act]).astype(np.int32)
        if len(np.unique(y_te)) < 2: continue
        thresh = state["thresholds"].get((act, cfg.FUSION), 0.5)
        m = evaluate_head_fast(state["heads"][(act, cfg.FUSION)],
                               Z_test_init, y_te, threshold=thresh)
        history["seed_metrics_e1"][act] = m

    t_phase2 = time.time()

    for step, activity in enumerate(incremental_activities):
        if activity not in label_dict:
            logger.warn(f"'{activity}' not in label_dict — skipping.")
            continue

        print(f"\n{'─'*60}")
        print(f"[{step+1:02d}/{len(incremental_activities)}] Adding: '{activity}'")
        t_step = time.time()

        # Use full-stream feat_cache after sensor increment, initial before
        is_incremented = state.get("sensor_incremented", False) or cfg.SENSOR_INCREMENTED
        feat_cache_e2  = feat_cache_full if is_incremented else feat_cache_init

        state, val_ood, test_ood = run_add_activity(
            new_activity=activity,
            state=state,
            feat_cache=feat_cache_e2,
            np_train_raw=np_train_raw,
            np_val_raw=np_val_raw,
            np_test_raw=np_test_raw,
            label_dict=label_dict,
            logger=logger,
            encoders=encoders,
            timestamp=TIMESTAMP,
        )

        # Record step metrics
        step_info = {
            "step":     step,
            "activity": activity,
            "val_ood":  val_ood,
            "test_ood": test_ood,
            "duration": time.time() - t_step,
        }

        # Track seed head F1 at this step
        seed_f1 = {}
        for act in cfg.SEED_ACTIVITIES:
            if val_ood and act in val_ood:
                seed_f1[act] = {
                    "val_f1":  val_ood[act].get("f1", 0),
                    "test_f1": test_ood[act].get("f1", 0) if test_ood else 0,
                }
        step_info["seed_f1"] = seed_f1
        history["steps"].append(step_info)

        # Save state after each step
        step_state_path = os.path.join(
            cfg.WORKING_DIR,
            f"{TIMESTAMP}_state_after_{activity.replace(' ', '_')}.pkl"
        )
        save_state(state, step_state_path)
        logger.event("INFO",
            f"Step {step+1} done in {time.time()-t_step:.1f}s — "
            f"saved to {step_state_path}")

    history["final_metrics"] = {
        "val":  val_ood  if val_ood  else {},
        "test": test_ood if test_ood else {},
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("E3 FINAL SUMMARY")
    print(f"Phase 2 total time: {time.time()-t_phase2:.1f}s")
    print(f"\nSeed head F1 progression (test):")
    print(f"  {'Activity':<45} {'E1 F1':>7}", end="")
    for s in history["steps"]:
        print(f"  {s['activity'][:8]:>10}", end="")
    print()
    print(f"  {'-'*120}")
    for act in cfg.SEED_ACTIVITIES:
        e1_f1 = history["seed_metrics_e1"].get(act, {}).get("f1", 0)
        print(f"  {act:<45} {e1_f1:>7.4f}", end="")
        for s in history["steps"]:
            f1 = s["seed_f1"].get(act, {}).get("test_f1", 0)
            print(f"  {f1:>10.4f}", end="")
        print()

    print(f"\nFinal test metrics (all heads):")
    if history["final_metrics"]["test"]:
        print(f"  {'Activity':<45} {'AUC':>7} {'F1':>7}")
        for act, m in sorted(history["final_metrics"]["test"].items()):
            seed_tag = " *" if act in set(cfg.SEED_ACTIVITIES) else ""
            print(f"  {act+seed_tag:<45} {m.get('auc',0):>7.4f} {m.get('f1',0):>7.4f}")

    # ── Save history ──────────────────────────────────────────────────────────
    hist_path = os.path.join(cfg.WORKING_DIR,
                             f"{TIMESTAMP}_e3_history.pkl")
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)

    log_path = logger.save_alongside(hist_path)
    print(f"\nDone.")
    print(f"  History : {hist_path}")
    print(f"  Log     : {log_path}")