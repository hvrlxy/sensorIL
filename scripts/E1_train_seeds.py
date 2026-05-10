"""
E1_train_seeds.py
=================
Train binary GatedHead classifiers for the seed activities.

Reads all settings from:
  paths.json         — data/model/output paths
  configs/<dataset>.json  — activities, gating, ME
  configs/hparams.json    — training hyperparameters

Output
------
  <TIMESTAMP>_e1_state_<fusion>.pkl   — state file (loaded by E2/E3)
  <TIMESTAMP>_e1_state_<fusion>_log.json  — full run log (config snapshot + metrics)
  <TIMESTAMP>_seed_<activity>_<fusion>.pt — per-head weights
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
import random
import numpy as np
import torch
from datetime import datetime

from config_loader import cfg
from logger import RunLogger
from helpers import create_dataset_file_split
from helpers_hitl import (
    build_gated_head_from_features,
    train_head_fast, evaluate_head_fast,
    find_optimal_threshold_fast,
    FeatureCache,
    ReplayBuffer, save_head_weights,
)



if __name__ == "__main__":
    # ── Setup ────────────────────────────────────────────────────────────────
    if cfg is None:
        raise RuntimeError("configs/paths.json not found. Copy configs/paths.example.json to configs/paths.json and fill in your paths.")
    DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)


    logger = RunLogger(cfg.WORKING_DIR, run_id=TIMESTAMP, script="E1_train_seeds")
    logger.log_run_start(cfg)
    logger.event("INFO", f"Device: {DEVICE}")
    cfg.summary()

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.event("INFO", "Loading dataset...")
    np_train_raw, np_val_raw, np_test_raw, label_dict = create_dataset_file_split(
        cfg.DATA_DIR, participant_lst=cfg.PARTICIPANTS
    )
    num_classes = len(label_dict)
    logger.event("INFO",
        f"Classes: {num_classes}  "
        f"Train: {np_train_raw[0].shape[0]}  "
        f"Val: {np_val_raw[0].shape[0]}  "
        f"Test: {np_test_raw[0].shape[0]}"
    )

    # ── Load encoder(s) ──────────────────────────────────────────────────────
    logger.event("INFO", "Loading encoder(s)...")
    from encoder import load_encoders_from_cfg, extract_all_features
    encoders = load_encoders_from_cfg(cfg)

    # ── Precompute encoder features ───────────────────────────────────────────
    logger.event("INFO", "Precomputing encoder features...")
    stream_indices = cfg.get_stream_indices(cfg.INITIAL_STREAM_NAMES)
    Z_train = extract_all_features(np_train_raw[0], encoders,
                                   cfg.STREAM_TO_ENCODER, cfg.INITIAL_STREAM_NAMES,
                                   batch_size=cfg.BATCH_SIZE,
                                   stream_indices=stream_indices)
    Z_val   = extract_all_features(np_val_raw[0],   encoders,
                                   cfg.STREAM_TO_ENCODER, cfg.INITIAL_STREAM_NAMES,
                                   batch_size=cfg.BATCH_SIZE,
                                   stream_indices=stream_indices)
    Z_test  = extract_all_features(np_test_raw[0],  encoders,
                                   cfg.STREAM_TO_ENCODER, cfg.INITIAL_STREAM_NAMES,
                                   batch_size=cfg.BATCH_SIZE,
                                   stream_indices=stream_indices)
    feat_cache = FeatureCache(Z_train, Z_val, Z_test)
    D = Z_train.shape[-1]
    logger.event("INFO",
        f"Train: {Z_train.shape}  Val: {Z_val.shape}  Test: {Z_test.shape}"
    )

    # ── Train one head per seed activity ──────────────────────────────────────
    heads         = {}
    thresholds    = {}
    weights_paths = {}
    replay_buffer = ReplayBuffer()
    results       = {}

    for activity in cfg.SEED_ACTIVITIES:
        if activity not in label_dict:
            logger.warn(f"Seed activity '{activity}' not in label_dict — skipping.")
            continue

        print(f"\n{'='*55}")
        print(f"Seed activity: '{activity}'")
        logger.event("INFO", f"Training seed head: '{activity}'")

        other_seeds     = [s for s in cfg.SEED_ACTIVITIES if s != activity]
        other_seed_idxs = {label_dict[s] for s in other_seeds if s in label_dict}
        act_idx         = label_dict[activity]
        keep_idxs       = {act_idx} | other_seed_idxs

        # Exclude known co-occurring activities from negatives
        excl_idxs = {label_dict[a]
                     for a in cfg.get_training_exclusions(activity)
                     if a in label_dict}

        # Train/val: filtered to seed activities only (clean negatives)
        tr_mask   = np.array([i in keep_idxs and (i == act_idx or i not in excl_idxs)
                               for i in np_train_raw[1]])
        vl_mask_s = np.array([i in keep_idxs and (i == act_idx or i not in excl_idxs)
                               for i in np_val_raw[1]])

        y_tr = (np_train_raw[1][tr_mask]  == act_idx).astype(np.int32)
        y_vl = (np_val_raw[1][vl_mask_s]  == act_idx).astype(np.int32)
        Z_tr = feat_cache.train[tr_mask]
        Z_vl = feat_cache.val[vl_mask_s]

        # Test: full 41-activity set (realistic evaluation)
        y_te = (np_test_raw[1] == act_idx).astype(np.int32)
        Z_te = feat_cache.test

        print(f"  Train — pos:{(y_tr==1).sum()} neg:{(y_tr==0).sum()} "
              f"ratio:{(y_tr==0).sum()/max((y_tr==1).sum(),1):.1f}:1")
        print(f"  Val   — pos:{(y_vl==1).sum()} neg:{(y_vl==0).sum()}")
        print(f"  Test  — pos:{(y_te==1).sum()} neg:{(y_te==0).sum()}")

        hint  = [1.0] * len(cfg.INITIAL_STREAM_NAMES)  # no gating
        model = build_gated_head_from_features(hint, D, fusion=cfg.FUSION)

        safe_name = activity.replace(" ", "_")
        save_path = os.path.join(
            cfg.WORKING_DIR,
            f"{TIMESTAMP}_seed_{safe_name}_{cfg.FUSION}.pt"
        )

        t0    = time.time()
        model = train_head_fast(
            model, Z_tr, y_tr, Z_vl, y_vl,
            save_path,
            epochs=cfg.SEED_HEAD_EPOCHS,
            lr=cfg.LEARNING_RATE,
            focal_gamma=cfg.FOCAL_GAMMA,
            max_class_weight=cfg.MAX_CLASS_WEIGHT,
        )
        duration = time.time() - t0
        print(f"  Done in {duration:.1f}s")

        thresh  = find_optimal_threshold_fast(
            model, Z_vl, y_vl,
            t_min=cfg.THRESHOLD_MIN,
            t_max=cfg.THRESHOLD_MAX,
            fallback=cfg.THRESHOLD_FALLBACK,
        )
        metrics = evaluate_head_fast(model, Z_te, y_te, threshold=thresh)
        print(f"  Threshold: {thresh:.2f}")
        print(f"  Test — AUC:{metrics['auc']:.4f} F1:{metrics['f1']:.4f} "
              f"Acc:{metrics['accuracy']:.4f} "
              f"P:{metrics['precision']:.4f} R:{metrics['recall']:.4f}")

        heads[(activity, cfg.FUSION)]      = model
        thresholds[(activity, cfg.FUSION)] = thresh
        wpath = save_head_weights(activity, cfg.FUSION, model,
                                  cfg.WORKING_DIR, TIMESTAMP)
        weights_paths[(activity, cfg.FUSION)] = wpath
        Z_pos = feat_cache.train[np_train_raw[1] == act_idx]
        replay_buffer.store_positives(activity, Z_pos)
        results[activity] = metrics

        logger.log_seed_result(activity, metrics, thresh, duration_s=duration)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("E1 SEED TRAINING SUMMARY")
    print(f"{'Activity':<45} {'AUC':>7} {'F1':>7} {'Acc':>7} {'P':>7} {'R':>7}")
    print("-" * 80)
    for activity, m in results.items():
        print(f"  {activity:<43} {m['auc']:>7.4f} {m['f1']:>7.4f} "
              f"{m['accuracy']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f}")

    # ── Save state ────────────────────────────────────────────────────────────
    # Record which streams each seed head was trained on
    head_streams = {
        (activity, cfg.FUSION): cfg.INITIAL_STREAM_NAMES
        for activity in cfg.SEED_ACTIVITIES
        if activity in label_dict
    }

    state = {
        "weights_paths":      weights_paths,
        "thresholds":         thresholds,
        "trained_activities": list(cfg.SEED_ACTIVITIES),
        "label_dict":         label_dict,
        "replay_buffer":      replay_buffer,
        "fusion":             cfg.FUSION,
        "feature_dim":        D,
        "timestamp":          TIMESTAMP,
        "dataset":            cfg.DATASET_NAME,
        "head_streams":       head_streams,
        "projector":          None,
        "projector_path":     None,
        "sensor_incremented": False,
        "initial_streams":    cfg.INITIAL_STREAM_NAMES,
        "full_streams":       cfg.INITIAL_STREAM_NAMES,
    }
    state_path = os.path.join(
        cfg.WORKING_DIR,
        f"{TIMESTAMP}_e1_state_{cfg.FUSION}.pkl"
    )
    with open(state_path, "wb") as f:
        pickle.dump(state, f)
    logger.event("INFO", f"State saved to {state_path}")

    log_path = logger.save_alongside(state_path)
    print(f"\nDone.")
    print(f"  State : {state_path}")
    print(f"  Log   : {log_path}")
    print(f"\nNext step: python E2_add_activity.py --activity <ACTIVITY> --state {state_path}")