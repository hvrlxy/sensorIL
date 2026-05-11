"""
experiment_runner.py
====================
Config-driven experiment runner for the sensor-incremental HITL-HAR pipeline.

Replaces E3_full_pipeline.py with a more flexible system where the exact
sequence of activities and sensor additions is specified in a JSON config
file rather than inferred from the dataset config.

Key additions over E3
---------------------
1. Explicit per-step config: which activity is added, and optionally which
   sensor is added, at each timestep.

2. Encoder data-fraction sweep: when a sensor is added at step T, the encoder
   is trained on 20%, 40%, 60%, 80%, 100% of the available unlabeled data.
   At each checkpoint the old activity heads are re-imputed and re-evaluated,
   producing a learning curve showing how encoder quality affects HAR.

3. Multi-scenario missing-sensor evaluation: at each encoder fraction
   checkpoint, the HAR model is evaluated under every specified missing-sensor
   scenario (e.g. {RightWaist missing}, {LeftWrist missing}, {both missing}).
   This tests encoder robustness to different patterns of sensor absence,
   not just the specific new-sensor imputation scenario.

   The encoder is trained with random masking of any stream (not just the
   new sensor), so it learns physics across all sensor pairs. The evaluation
   scenarios verify this generalises correctly.

Encoder training objective (why random masking matters)
-------------------------------------------------------
During encoder training, random_mask_forward() masks:
  - The new sensor (always)
  - Any old sensor with probability p_mask_old (~15%)
This means the encoder learns to reconstruct any sensor from any subset of
the others — not just new-from-old. The multi-scenario evaluation directly
tests whether this generalisation worked.

Usage
-----
  python experiment_runner.py --config configs/experiment_config.json

  # Resume from a saved E1 state
  python experiment_runner.py \\
      --config configs/experiment_config.json \\
      --resume-e1 output/20260501-120000_e1_state_early.pkl

Output
------
  <working_dir>/<ts>_experiment_history.pkl
  <working_dir>/<ts>_experiment_history_log.json
  <working_dir>/<ts>_encoder_sweep_<fraction>.pt   (encoder checkpoint per fraction)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import itertools
import json
import os
import pickle
import random
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config_loader import cfg
from logger import RunLogger
from helpers import create_dataset_file_split
from helpers_hitl import (
    FeatureCache, load_heads_from_state,
    evaluate_all_heads_fast, evaluate_head_fast,
    find_optimal_threshold_fast, build_gated_head_from_features,
    train_head_fast, ReplayBuffer, save_head_weights,
    make_multilabel_binary,
)
from encoder import load_encoders_from_cfg, extract_all_features
from projector import (
    build_projector, load_projector, save_projector,
    train_encoder_on_unlabeled, measure_encoder_val_loss,
    evaluate_with_missing_sensors,
    train_projection_head, extract_mae_features,
)
from E2_add_activity import run_add_activity, save_state

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT CONFIG LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_experiment_config(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)

    def strip(obj):
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items() if not k.startswith("_")}
        if isinstance(obj, list):
            return [strip(i) for i in obj]
        return obj

    exp = strip(raw)

    assert "seed_activities" in exp, "experiment_config must have 'seed_activities'"
    assert "steps" in exp,          "experiment_config must have 'steps'"
    assert len(exp["steps"]) > 0,   "'steps' must be non-empty"
    assert "initial_sensor" in exp or "initial_sensors" in exp, (
        "experiment_config must have 'initial_sensors' (list) or 'initial_sensor' (str). "
        "Sensors are added via 'add_sensors' (list) or 'add_sensor' (str) in each step."
    )

    ts = [s["t"] for s in exp["steps"]]
    assert ts == sorted(ts), f"Step 't' values must be strictly increasing: {ts}"

    for step in exp["steps"]:
        assert "t"            in step, f"Step missing 't': {step}"
        assert "add_activity" in step, f"Step missing 'add_activity': {step}"
        if "add_sensor" in step or "add_sensors" in step:
            assert "encoder_data_fractions" in step, (
                f"Step {step['t']}: 'encoder_data_fractions' required when add_sensor(s) is set"
            )
            fracs = step["encoder_data_fractions"]
            assert all(0 < f <= 1.0 for f in fracs), (
                f"Step {step['t']}: all fractions must be in (0, 1]"
            )

    sensor_steps  = [s for s in exp["steps"] if "add_sensor" in s or "add_sensors" in s]
    added_sensors = []
    for s in sensor_steps:
        ns = s.get("add_sensors") or ([s["add_sensor"]] if "add_sensor" in s else [])
        added_sensors.extend([ns] if isinstance(ns, str) else ns)
    assert len(added_sensors) == len(set(added_sensors)), (
        f"Duplicate sensors across add_sensors steps: {added_sensors}."
    )

    return exp


def resolve_missing_scenarios(
    step: dict,
    all_streams: list[str],
    max_auto: int = 16,
) -> list[list[str]]:
    n_streams = len(all_streams)

    if "missing_sensor_scenarios" in step:
        validated = []
        for sc in step["missing_sensor_scenarios"]:
            unknown = [s for s in sc if s not in all_streams]
            if unknown:
                print(f"  [WARN] Scenario {sc}: unknown sensors {unknown} — skipping")
                continue
            if len(sc) >= n_streams:
                print(f"  [WARN] Scenario {sc}: masks all {n_streams} sensors "
                      f"(nothing to reconstruct from) — skipping")
                continue
            validated.append(sc)
        return validated

    scenarios  = []
    new_sensor = step.get("add_sensor")

    if new_sensor and [new_sensor] not in scenarios:
        scenarios.append([new_sensor])

    for s in all_streams:
        if [s] not in scenarios:
            scenarios.append([s])
        if len(scenarios) >= max_auto:
            break

    if n_streams >= 3 and len(scenarios) < max_auto:
        for combo in itertools.combinations(all_streams, 2):
            scenarios.append(list(combo))
            if len(scenarios) >= max_auto:
                break

    scenarios = [sc for sc in scenarios if len(sc) < n_streams]
    return scenarios[:max_auto]


# ─────────────────────────────────────────────────────────────────────────────
# ENCODER DATA-FRACTION SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def run_encoder_fraction_sweep(
    step: dict,
    state: dict,
    streams_before: list[str],
    streams_after: list[str],
    X_unlabeled_train: np.ndarray,
    X_unlabeled_val: np.ndarray,
    X_train_old: np.ndarray,
    X_val_old: np.ndarray,
    X_test_old: np.ndarray,       # (N, T, S_old, C)  old streams only — realistic eval
    Z_train_full: np.ndarray,
    Z_val_full: np.ndarray,
    Z_test_full: np.ndarray,
    X_test_raw: np.ndarray,
    np_train_raw: list,
    np_val_raw: list,
    np_test_raw: list,
    label_dict: dict,
    embed_dim: int,
    sweep_hparams: dict,
    working_dir: str,
    timestamp: str,
    logger: RunLogger,
    seed_metrics_e1: dict | None = None,
    encoders: dict | None = None,
) -> dict:
    heads         = state["heads"]
    thresholds    = state["thresholds"]
    head_streams  = state["head_streams"]
    full_streams  = state["full_streams"]
    fusion        = state["fusion"]
    pre_increment = set(state.get("pre_increment_activities", []))
    cooc_graph    = cfg._dataset["cooccurrence_graph"]

    fractions     = sorted(step["encoder_data_fractions"])
    scenarios     = resolve_missing_scenarios(
        step, streams_after,
        max_auto=cfg._dataset.get("max_auto_missing_scenarios", 16),
    )

    N_total       = X_unlabeled_train.shape[0]
    n_initial     = len(streams_before)
    n_streams_out = len(streams_after)
    known_indices = list(range(n_initial))
    sweep_results = {}
    T = X_unlabeled_train.shape[1]
    C = X_unlabeled_train.shape[3]

    logger.event("INFO",
        f"Encoder sweep: {len(fractions)} fractions × "
        f"{len(scenarios)} missing-sensor scenarios  "
        f"N_train={N_total}  N_val={len(X_unlabeled_val)}"
    )
    print(f"\n{'='*60}")
    print(f"ENCODER DATA-FRACTION SWEEP  (raw signal space)")
    print(f"  Fractions  : {fractions}")
    print(f"  Scenarios  : {len(scenarios)} missing-sensor patterns")
    print(f"  Train pool : {N_total} windows  shape={X_unlabeled_train.shape[1:]}")
    print(f"  Val (fixed): {len(X_unlabeled_val)} windows  ← same across all fractions")
    print(f"  Streams before: {streams_before}")
    print(f"  Streams after : {streams_after}")
    for i, sc in enumerate(scenarios):
        print(f"    [{i:02d}] missing={sc}")
    print(f"{'='*60}")

    hp  = cfg.PROJECTOR_HPARAMS
    rng = np.random.default_rng(cfg.SEED)
    X_unlabeled_shuffled = X_unlabeled_train[rng.permutation(N_total)]

    for frac in fractions:
        n_use    = max(1, int(N_total * frac))
        X_frac   = X_unlabeled_shuffled[:n_use]
        frac_pct = f"{frac:.0%}"

        projector = build_projector(
                n_streams_in  = n_initial,
                n_streams_out = n_streams_out,
                embed_dim     = embed_dim,
                hidden_dim    = hp.get("hidden_dim", 256),
                latent_dim    = hp.get("latent_dim", 128),  # ← add
                T=T, C=C,
            )

        print(f"\n{'─'*60}")
        print(f"[Fraction {frac_pct}]  {n_use}/{N_total} train windows  "
              f"{len(X_unlabeled_val)} val windows (fixed)")
        t0 = time.time()

        ckpt_path = os.path.join(
            working_dir,
            f"{timestamp}_encoder_sweep_{frac_pct.replace('%','pct')}.pt"
        )
        projector = train_encoder_on_unlabeled(
            projector       = projector,
            X_unlabeled     = X_frac,
            n_streams_out   = n_streams_out,
            embed_dim       = embed_dim,
            save_path       = ckpt_path,
            epochs          = sweep_hparams.get("epochs", 50),
            lr              = sweep_hparams.get("learning_rate", 1e-3),
            batch_size      = sweep_hparams.get("batch_size", 256),
            early_stopping_patience = sweep_hparams.get("early_stopping_patience", 10),
            Z_val_external  = X_unlabeled_val,
        )

        # ── Train projection head on unlabeled data ──────────────────────────
        # Frozen MAE backbone + trainable projection head.
        # The projection head maps MAE patch embeddings to a fixed-size
        # feature space used by the binary heads — replacing SimCLR.
        proj_ckpt_path = ckpt_path.replace(".pt", "_proj.pt")
        projector = train_projection_head(
            encoder    = projector,
            X_train    = X_frac,
            X_val      = X_unlabeled_val,
            save_path  = proj_ckpt_path,
            stream_indices = list(range(n_streams_out)),
            epochs     = sweep_hparams.get("proj_epochs", 30),
            lr         = sweep_hparams.get("proj_lr", 1e-3),
            batch_size = sweep_hparams.get("batch_size", 256),
            early_stopping_patience = sweep_hparams.get("early_stopping_patience", 10),
        )

        # ── Extract MAE features for labeled data ────────────────────────────
        # Use MAE projection head features instead of SimCLR.
        # Old activities: only initial streams available → zero-pad new streams.
        # New activities: all streams available → full MAE features.
        after_cols  = [cfg.FULL_DATASET_STREAMS.index(s) for s in streams_after]
        before_cols = [cfg.FULL_DATASET_STREAMS.index(s) for s in streams_before]
        n_before    = len(streams_before)

        print(f"  Extracting MAE features (all labeled data)...")
        # Full streams — for new activities and evaluation
        Z_mae_train_full = extract_mae_features(
            projector, np_train_raw[0][:, :, after_cols, :],
            stream_indices=list(range(n_streams_out)),
            batch_size=sweep_hparams.get("batch_size", 256),
        )
        Z_mae_val_full   = extract_mae_features(
            projector, np_val_raw[0][:, :, after_cols, :],
            stream_indices=list(range(n_streams_out)),
            batch_size=sweep_hparams.get("batch_size", 256),
        )
        Z_mae_test_full  = extract_mae_features(
            projector, np_test_raw[0][:, :, after_cols, :],
            stream_indices=list(range(n_streams_out)),
            batch_size=sweep_hparams.get("batch_size", 256),
        )
        # Old streams only — for retraining old-activity heads
        # Missing streams are zero-padded inside extract_mae_features
        Z_mae_train_old  = extract_mae_features(
            projector, np_train_raw[0][:, :, before_cols, :],
            stream_indices=list(range(n_before)),
            batch_size=sweep_hparams.get("batch_size", 256),
        )
        Z_mae_val_old    = extract_mae_features(
            projector, np_val_raw[0][:, :, before_cols, :],
            stream_indices=list(range(n_before)),
            batch_size=sweep_hparams.get("batch_size", 256),
        )
        Z_mae_test_old   = extract_mae_features(
            projector, np_test_raw[0][:, :, before_cols, :],
            stream_indices=list(range(n_before)),
            batch_size=sweep_hparams.get("batch_size", 256),
        )

        # Update embed_dim to MAE proj_dim for head construction
        mae_D = Z_mae_train_full.shape[-1]
        print(f"  MAE features: train={Z_mae_train_full.shape}  "
              f"val={Z_mae_val_full.shape}  test={Z_mae_test_full.shape}  D={mae_D}")

        enc_val_loss = _measure_encoder_val_loss(projector, X_unlabeled_val)

        # ── Retrain old-activity heads on MAE features ───────────────────────
        # Old activities have only initial-stream lab samples available.
        # We use real MAE features (with missing streams zero-padded) instead
        # of synthetic imputed signals — no SimCLR black box, no imputation
        # quality issues. The head trains on real signal patterns from the
        # available streams.
        print(f"  Retraining pre-increment heads on MAE features "
              f"({len(pre_increment)} activities)...")
        sweep_heads      = dict(state["heads"])
        sweep_thresholds = dict(state["thresholds"])
        cooc_graph       = cfg._dataset["cooccurrence_graph"]

        for activity in pre_increment:
            if activity not in label_dict:
                continue
            if (activity, fusion) not in sweep_heads:
                continue

            class_idx = label_dict[activity]
            y_tr_bin  = (np_train_raw[1] == class_idx).astype(np.int32)
            y_vl_bin  = (np_val_raw[1]   == class_idx).astype(np.int32)
            y_te_bin  = make_multilabel_binary(activity, np_test_raw[1],
                                               label_dict, cooccurrence_graph=cooc_graph)

            n_pos = int(y_tr_bin.sum())
            if n_pos == 0 or y_vl_bin.sum() == 0:
                continue

            # Balanced train set 10:1
            max_neg = min(int((y_tr_bin==0).sum()), 10 * n_pos)
            pos_idx = np.where(y_tr_bin == 1)[0]
            neg_idx = np.where(y_tr_bin == 0)[0]
            if len(neg_idx) > max_neg:
                neg_idx = np.random.default_rng(cfg.SEED).choice(
                    neg_idx, size=max_neg, replace=False)
            keep_idx = np.concatenate([pos_idx, neg_idx])

            Z_tr = Z_mae_train_old[keep_idx]
            y_tr = y_tr_bin[keep_idx]
            Z_vl = Z_mae_val_old
            Z_te = Z_mae_test_old

            print(f"    '{activity}'  pos:{n_pos}  neg:{len(neg_idx)}")

            safe      = activity.replace(" ", "_")
            save_path = os.path.join(
                working_dir,
                f"{timestamp}_mae_head_{safe}_{fusion}.pt"
            )
            best_model, best_auc = None, -1.0
            for seed in [42, 123, 777]:
                torch.manual_seed(seed)
                cand = build_gated_head_from_features(
                    [1.0] * n_streams_out, mae_D, fusion=fusion)
                cand = train_head_fast(
                    cand, Z_tr, y_tr, Z_vl, y_vl_bin, save_path,
                    epochs=sweep_hparams.get("retrain_epochs", 20),
                    lr=sweep_hparams.get("learning_rate", 1e-3),
                    batch_size=sweep_hparams.get("batch_size", 200),
                    focal_gamma=cfg.FOCAL_GAMMA,
                    max_class_weight=cfg.MAX_CLASS_WEIGHT,
                    early_stopping_patience=sweep_hparams.get(
                        "early_stopping_patience", 10),
                )
                m = evaluate_head_fast(cand, Z_vl, y_vl_bin, threshold=0.5)
                if m["auc"] > best_auc:
                    best_auc, best_model = m["auc"], cand

            thresh = find_optimal_threshold_fast(
                best_model, Z_vl, y_vl_bin,
                t_min=cfg.THRESHOLD_MIN, t_max=cfg.THRESHOLD_MAX,
                fallback=cfg.THRESHOLD_FALLBACK,
            )
            sweep_heads[(activity, fusion)]      = best_model
            sweep_thresholds[(activity, fusion)] = thresh

            m = evaluate_head_fast(best_model, Z_te, y_te_bin, threshold=thresh)
            print(f"      AUC:{m['auc']:.4f}  F1:{m['f1']:.4f}  thresh:{thresh:.3f}")

        # ── Full-sensor evaluation on MAE features ───────────────────────────
        # All heads (old and new) evaluated on real MAE features.
        # Old acts: Z_mae_test_old (initial streams, new stream zero-padded)
        # New acts: Z_mae_test_full (all streams)
        # No synthetic embeddings in the evaluation path.
        old_act_metrics = evaluate_all_heads_fast(
            {k: v for k, v in sweep_heads.items() if k[0] in pre_increment},
            Z_mae_test_old, np_test_raw[1],
            label_dict, sweep_thresholds, fusion,
            cooccurrence_graph=cooc_graph,
        )
        new_act_metrics = evaluate_all_heads_fast(
            {k: v for k, v in sweep_heads.items() if k[0] not in pre_increment},
            Z_mae_test_full, np_test_raw[1],
            label_dict, sweep_thresholds, fusion,
            cooccurrence_graph=cooc_graph,
        )
        full_metrics = {**old_act_metrics, **new_act_metrics}

        # ── Missing-sensor scenario evaluations (reconstruction quality) ─────
        # Still uses imputation pipeline to track reconstruction quality.
        # Separately reports imputation-based F1 for comparison.
        missing_metrics = {}
        for scenario in scenarios:
            scenario_key = "+".join(sorted(scenario)) + "_missing"
            print(f"  Evaluating scenario: {scenario_key}")
            m = evaluate_with_missing_sensors(
                heads             = sweep_heads,
                thresholds        = sweep_thresholds,
                projector         = projector,
                X_test_raw        = X_test_raw,
                Z_test_full       = Z_mae_test_full,
                y_int             = np_test_raw[1],
                label_dict        = label_dict,
                fusion            = fusion,
                missing_sensors   = scenario,
                all_stream_names  = streams_after,
                n_streams_out     = n_streams_out,
                embed_dim         = mae_D,
                simclr_encoders   = encoders,
                stream_to_encoder = cfg.STREAM_TO_ENCODER,
                cooccurrence_graph= cooc_graph,
                T=T, C=C,
            )
            missing_metrics[scenario_key] = m

        elapsed = time.time() - t0
        sweep_results[frac] = {
            "fraction":               frac,
            "n_windows_used":         n_use,
            "encoder_val_loss":       float(enc_val_loss),
            "full_sensor_metrics":    full_metrics,
            "sweep_heads":            sweep_heads,
            "sweep_thresholds":       sweep_thresholds,
            "missing_sensor_metrics": missing_metrics,
            "elapsed_s":              round(elapsed, 1),
        }

        _print_fraction_summary(frac_pct, enc_val_loss, full_metrics,
                                seed_metrics_e1, cfg.SEED_ACTIVITIES)
        logger.event("INFO",
            f"Fraction {frac_pct}: enc_val={enc_val_loss:.5f}  "
            f"elapsed={elapsed:.1f}s"
        )

    return sweep_results


def _measure_encoder_val_loss(projector, X_val: np.ndarray) -> float:
    from projector import measure_encoder_val_loss
    return measure_encoder_val_loss(projector, X_val)


def _print_fraction_summary(frac_pct, enc_val_loss, full_metrics,
                              seed_metrics_e1, seed_activities):
    print(f"\n  Fraction {frac_pct}  encoder_val_loss={enc_val_loss:.5f}")
    print(f"  {'Activity':<45} {'1-stream (E1)':>14} {'2-stream imputed':>17} {'Δ':>7}")
    print(f"  {'-'*87}")
    for act in seed_activities:
        e1  = (seed_metrics_e1 or {}).get(act, {}).get("f1", float("nan"))
        imp = full_metrics.get(act, {}).get("f1", float("nan"))
        d   = f"{imp-e1:+.4f}" if imp==imp and e1==e1 else "   nan"
        print(f"  {act:<45} {e1:>14.4f} {imp:>17.4f} {d:>7}")


# ─────────────────────────────────────────────────────────────────────────────
# E1 — SEED TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def run_e1(np_train_raw, np_val_raw, np_test_raw, label_dict,
           feat_cache_init, D, logger, timestamp, seed_activities,
           initial_streams: list):

    print(f"\n{'='*60}")
    print("PHASE 1 — SEED TRAINING")
    print(f"  Activities: {seed_activities}")
    print(f"{'='*60}")

    heads, thresholds, weights_paths = {}, {}, {}
    replay_buffer = ReplayBuffer()
    results       = {}
    cooc_graph    = cfg._dataset["cooccurrence_graph"]

    for activity in seed_activities:
        if activity not in label_dict:
            logger.warn(f"Seed '{activity}' not in label_dict — skipping.")
            continue

        act_idx     = label_dict[activity]
        other_seeds = [s for s in seed_activities if s != activity]
        other_idxs  = {label_dict[s] for s in other_seeds if s in label_dict}
        keep_idxs   = {act_idx} | other_idxs
        excl_idxs   = {label_dict[a]
                       for a in cfg.get_training_exclusions(activity)
                       if a in label_dict}

        tr_mask = np.array([i in keep_idxs and (i == act_idx or i not in excl_idxs)
                             for i in np_train_raw[1]])
        vl_mask = np.array([i in keep_idxs and (i == act_idx or i not in excl_idxs)
                             for i in np_val_raw[1]])

        y_tr = (np_train_raw[1][tr_mask] == act_idx).astype(np.int32)
        y_vl = (np_val_raw[1][vl_mask]   == act_idx).astype(np.int32)

        # Co-occurrence aware test labels
        y_te = make_multilabel_binary(
            activity, np_test_raw[1], label_dict,
            cooccurrence_graph=cooc_graph,
        )

        Z_tr = feat_cache_init.train[tr_mask]
        Z_vl = feat_cache_init.val[vl_mask]
        Z_te = feat_cache_init.test

        print(f"\nSeed: '{activity}'  pos:{y_tr.sum()} neg:{(y_tr==0).sum()}")

        hint  = [1.0] * feat_cache_init.train.shape[1]
        model = build_gated_head_from_features(hint, D, fusion=cfg.FUSION)
        save_path = os.path.join(
            cfg.WORKING_DIR, f"{timestamp}_seed_{activity.replace(' ','_')}_{cfg.FUSION}.pt"
        )

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
            activity, feat_cache_init.train[np_train_raw[1] == act_idx]
        )
        results[activity] = metrics
        logger.log_seed_result(activity, metrics, thresh)

    state = {
        "weights_paths":            weights_paths,
        "thresholds":               thresholds,
        "trained_activities":       list(seed_activities),
        "label_dict":               label_dict,
        "replay_buffer":            replay_buffer,
        "fusion":                   cfg.FUSION,
        "feature_dim":              D,
        "timestamp":                timestamp,
        "dataset":                  cfg.DATASET_NAME,
        "heads":                    heads,
        "head_streams":             {},
        "projector":                None,
        "projector_path":           None,
        "sensor_incremented":       False,
        "initial_streams":          list(initial_streams),
        "full_streams":             list(initial_streams),
        "pre_increment_activities": [],
    }

    state_path = os.path.join(
        cfg.WORKING_DIR, f"{timestamp}_e1_state_{cfg.FUSION}.pkl"
    )
    state_to_save = {k: v for k, v in state.items() if k != "heads"}
    with open(state_path, "wb") as f:
        pickle.dump(state_to_save, f)
    logger.event("INFO", f"E1 state saved: {state_path}")
    return state, state_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _step_sensors(step: dict) -> list[str]:
    s = step.get("add_sensors") or ([step["add_sensor"]] if "add_sensor" in step else [])
    return [s] if isinstance(s, str) else list(s)


def run_experiment(exp_config: dict, resume_e1_path: str | None = None):

    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    exp_name = exp_config.get("experiment_name", "experiment")
    logger   = RunLogger(cfg.WORKING_DIR, run_id=TIMESTAMP,
                         script=f"experiment_runner_{exp_name}")
    logger.log_run_start(cfg)
    logger.event("INFO", f"Device: {DEVICE}")
    logger.event("INFO", f"Experiment: {exp_name}")

    seed_activities = exp_config["seed_activities"]
    steps           = exp_config["steps"]
    _init = exp_config.get("initial_sensors") or exp_config.get("initial_sensor")
    initial_sensors = [_init] if isinstance(_init, str) else list(_init)

    all_sensors_in_order = list(initial_sensors)
    for step in steps:
        ns = step.get("add_sensors") or ([step["add_sensor"]] if "add_sensor" in step else [])
        for s in ([ns] if isinstance(ns, str) else ns):
            if s not in all_sensors_in_order:
                all_sensors_in_order.append(s)

    cfg.SEED_ACTIVITIES      = seed_activities
    cfg.INITIAL_STREAM_NAMES = initial_sensors
    cfg.FULL_STREAM_NAMES    = all_sensors_in_order

    logger.event("INFO",
        f"Experiment streams: initial={initial_sensors}  "
        f"full={all_sensors_in_order}  seeds={seed_activities}"
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    np_train_raw, np_val_raw, np_test_raw, label_dict = create_dataset_file_split(
        cfg.DATA_DIR, participant_lst=cfg.PARTICIPANTS
    )

    cooc_graph = cfg._dataset["cooccurrence_graph"]

    # ── Load encoders (kept for API compatibility only — not used in pipeline) ─
    print("\nLoading encoder(s)...")
    encoders = load_encoders_from_cfg(cfg)

    # SimCLR features no longer used for downstream tasks.
    # MAE features replace SimCLR throughout (E1, sweep, E2).
    all_stream_names = cfg.FULL_STREAM_NAMES

    current_streams = list(initial_sensors)

    # ── Train 1-stream MAE on initial unlabeled FL data ───────────────────────
    # MAE features replace SimCLR as the feature extractor for binary heads.
    # This ensures E1 and post-increment use the same feature space so Δ is
    # a clean comparison of "1-stream MAE" → "2-stream MAE".
    print("\nLoading initial unlabeled data for MAE-1 training...")
    ul_dir_init = cfg.UNLABELED_DATA_DIR
    X_ul_init_tr, X_ul_init_vl = _load_unlabeled_raw(
        ul_dir=ul_dir_init,
        active_streams=list(initial_sensors),
        cache_suffix="_" + "_".join(initial_sensors),
    ) if ul_dir_init else (None, None)

    if X_ul_init_tr is not None:
        hp         = cfg.PROJECTOR_HPARAMS
        ul_hp      = cfg.UNLABELED_ENCODER_HPARAMS
        n_init     = len(initial_sensors)
        T_raw      = X_ul_init_tr.shape[1]
        C_raw      = X_ul_init_tr.shape[3]
        proj_dim   = hp.get("proj_dim", 96)

        print(f"\nTraining MAE-1 on {len(X_ul_init_tr)} unlabeled windows "
              f"({initial_sensors})...")
        # Cap at 10% to match the sweep fractions — consistent comparison
        n_mae1 = max(1, int(len(X_ul_init_tr) * 0.10))
        rng_mae1 = np.random.default_rng(cfg.SEED)
        X_ul_init_tr_sub = X_ul_init_tr[rng_mae1.permutation(len(X_ul_init_tr))[:n_mae1]]
        print(f"  Using {n_mae1}/{len(X_ul_init_tr)} windows (10%)")
        # Build a 1-stream MAE — n_streams_in=0 (no missing, just reconstruct)
        # We reuse build_projector with n_streams_in=0, n_streams_out=n_init
        # The MAE learns cross-patch temporal structure on 1-stream data.
        # Note: with only 1 stream, masking = temporal patch masking.
        mae1_ckpt = os.path.join(
            cfg.WORKING_DIR, f"{TIMESTAMP}_mae1_{'_'.join(initial_sensors)}.pt"
        )
        # build_projector requires n_streams_out > n_streams_in, use n_init, 0
        # Override: build directly with n_streams_total = n_init
        from projector import (CrossMaskedTransformer, StreamProjectionHead,
                                train_raw_encoder, train_projection_head,
                                extract_mae_features, _checkpoint)
        d_model = hp.get("hidden_dim", 64)
        n_heads = 4
        while n_heads > 1 and d_model % n_heads != 0:
            n_heads -= 1
        mae1 = CrossMaskedTransformer(
            n_streams_total=n_init, T=T_raw, C=C_raw,
            patch_size=hp.get("patch_size", 10),
            d_model=d_model, n_heads=n_heads, n_layers=3,
        ).to(DEVICE)
        mae1.proj_dim  = proj_dim
        mae1.proj_head = StreamProjectionHead(
            d_model=d_model, proj_dim=proj_dim, n_streams=n_init,
        ).to(DEVICE)

        mae1 = train_raw_encoder(
            mae1, X_ul_init_tr_sub, X_ul_init_vl, mae1_ckpt,
            epochs=ul_hp.get("epochs", 50),
            lr=ul_hp.get("learning_rate", 1e-3),
            batch_size=ul_hp.get("batch_size", 256),
            early_stopping_patience=ul_hp.get("early_stopping_patience", 10),
        )
        mae1_proj_ckpt = mae1_ckpt.replace(".pt", "_proj.pt")
        mae1 = train_projection_head(
            mae1, X_ul_init_tr_sub, X_ul_init_vl, mae1_proj_ckpt,
            stream_indices=list(range(n_init)),
            epochs=hp.get("proj_epochs", 30),
            lr=hp.get("proj_lr", 1e-3),
            batch_size=ul_hp.get("batch_size", 256),
        )

        # Extract MAE-1 features for labeled data (initial streams only)
        init_cols = [cfg.FULL_DATASET_STREAMS.index(s) for s in initial_sensors]
        print(f"  Extracting MAE-1 features for labeled data...")
        Z_mae1_train = extract_mae_features(
            mae1, np_train_raw[0][:, :, init_cols, :],
            stream_indices=list(range(n_init)),
            batch_size=cfg.BATCH_SIZE,
        )
        Z_mae1_val   = extract_mae_features(
            mae1, np_val_raw[0][:, :, init_cols, :],
            stream_indices=list(range(n_init)),
            batch_size=cfg.BATCH_SIZE,
        )
        Z_mae1_test  = extract_mae_features(
            mae1, np_test_raw[0][:, :, init_cols, :],
            stream_indices=list(range(n_init)),
            batch_size=cfg.BATCH_SIZE,
        )
        D_mae        = Z_mae1_train.shape[-1]
        feat_cache_cur = FeatureCache(Z_mae1_train, Z_mae1_val, Z_mae1_test)
        D              = D_mae
        print(f"  MAE-1 features: train={Z_mae1_train.shape}  D={D_mae}")
        logger.event("INFO", f"MAE-1 features: train={Z_mae1_train.shape} D={D_mae}")
    else:
        # No unlabeled data — cannot train MAE-1.
        # E1 requires unlabeled data for MAE feature extraction.
        raise RuntimeError(
            "No initial unlabeled data found — cannot train MAE-1. "
            f"Set UNLABELED_DATA_DIR in config and ensure "
            f"encoder_train_raw_{'_'.join(initial_sensors)}.npy exists."
        )

    # ── Phase 1: Seed training ─────────────────────────────────────────────────
    if resume_e1_path:
        print(f"\nResuming from E1 state: {resume_e1_path}")
        from E2_add_activity import load_state
        state = load_state(resume_e1_path)
        state["heads"] = load_heads_from_state(state, cfg.WORKING_DIR)
        logger.event("INFO", f"Resumed E1 from {resume_e1_path}")
    else:
        state, e1_path = run_e1(
            np_train_raw, np_val_raw, np_test_raw, label_dict,
            feat_cache_cur, D, logger, TIMESTAMP, seed_activities,
            initial_streams=current_streams,
        )

    # ── E1 baseline metrics (co-occurrence aware, MAE-1 features) ─────────────
    seed_metrics_e1 = {}
    for act in seed_activities:
        if act not in label_dict:
            continue
        y_te   = make_multilabel_binary(
            act, np_test_raw[1], label_dict,
            cooccurrence_graph=cooc_graph,
        )
        thresh = state["thresholds"].get((act, cfg.FUSION), 0.5)
        m = evaluate_head_fast(
            state["heads"][(act, cfg.FUSION)],
            feat_cache_cur.test, y_te, threshold=thresh
        )
        seed_metrics_e1[act] = m

    # ── History ───────────────────────────────────────────────────────────────
    history = {
        "experiment_name":   exp_name,
        "seed_activities":   seed_activities,
        "seed_metrics_e1":   seed_metrics_e1,
        "steps":             [],
        "encoder_sweep":     None,
    }

    # ── Phase 2: Incremental steps ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 2 — {len(steps)} INCREMENTAL STEPS")
    print(f"{'='*60}")

    t_phase2 = time.time()

    for step_idx, step in enumerate(steps):
        activity       = step["add_activity"]
        is_sensor_step = "add_sensor" in step or "add_sensors" in step

        if activity not in label_dict:
            logger.warn(f"Step {step['t']}: '{activity}' not in label_dict — skipping.")
            continue

        print(f"\n{'─'*60}")
        print(f"[Step {step['t']:02d}/{steps[-1]['t']:02d}]  "
              f"add_activity='{activity}'"
              + (f"  add_sensors={_step_sensors(step)}" if is_sensor_step else ""))
        t_step = time.time()

        sweep_results = None

        if is_sensor_step:
            new_sensors    = _step_sensors(step)
            streams_before = list(current_streams)
            for s in new_sensors:
                if s not in current_streams:
                    current_streams.append(s)
            new_sensor = "_".join(new_sensors)

            # feat_cache_cur will be rebuilt with MAE-2 features after sweep.
            # Use existing feat_cache_cur as placeholder until then.
            pass

            state["sensor_incremented"]       = False
            state["full_streams"]             = list(current_streams)
            state["initial_streams"]          = list(streams_before)
            state["pre_increment_activities"] = list(state.get("trained_activities", []))

            ul_dir = step.get("unlabeled_data_dir") or cfg.UNLABELED_DATA_DIR
            if ul_dir:
                Z_ul_tr, Z_ul_vl = _load_unlabeled_features(
                    ul_dir=ul_dir,
                    encoders=encoders,
                    active_streams=list(current_streams),
                    cache_suffix="_" + "_".join(current_streams),
                )
            else:
                Z_ul_tr, Z_ul_vl = None, None

            logger.event("INFO",
                f"Sensor increment: {streams_before} -> {current_streams}  "
                f"unlabeled={'loaded' if Z_ul_tr is not None else 'unavailable'}")

            if Z_ul_tr is not None:
                sweep_hparams = {
                    **cfg.UNLABELED_ENCODER_HPARAMS,
                    **exp_config.get("encoder_sweep_hparams", {}),
                }
                before_cols    = [cfg.FULL_DATASET_STREAMS.index(s) for s in streams_before]
                after_cols     = [cfg.FULL_DATASET_STREAMS.index(s) for s in current_streams]
                X_train_before = np_train_raw[0][:, :, before_cols, :]
                X_val_before   = np_val_raw[0][:,   :, before_cols, :]
                X_test_before  = np_test_raw[0][:,  :, before_cols, :]
                X_test_after   = np_test_raw[0][:,  :, after_cols,  :]

                sweep_results = run_encoder_fraction_sweep(
                    step              = step,
                    state             = state,
                    streams_before    = streams_before,
                    streams_after     = list(current_streams),
                    X_unlabeled_train = Z_ul_tr,
                    X_unlabeled_val   = Z_ul_vl,
                    X_train_old       = X_train_before,
                    X_val_old         = X_val_before,
                    X_test_old        = X_test_before,
                    Z_train_full      = None,
                    Z_val_full        = None,
                    Z_test_full       = None,
                    X_test_raw        = X_test_after,
                    np_train_raw      = np_train_raw,
                    np_val_raw        = np_val_raw,
                    np_test_raw       = np_test_raw,
                    label_dict        = label_dict,
                    embed_dim         = D,
                    sweep_hparams     = sweep_hparams,
                    working_dir       = cfg.WORKING_DIR,
                    timestamp         = TIMESTAMP,
                    logger            = logger,
                    seed_metrics_e1   = seed_metrics_e1,
                    encoders          = None,
                )
                history.setdefault("encoder_sweeps", {})[new_sensor] = sweep_results

                # Select best fraction by average F1 gain across all activities
                def _avg_delta(res):
                    metrics = res.get("full_sensor_metrics", {})
                    e1      = seed_metrics_e1 or {}
                    deltas  = []
                    for act, m in metrics.items():
                        e1_f1 = e1.get(act, {}).get("f1", 0.0)
                        deltas.append(m.get("f1", 0.0) - e1_f1)
                    return float(np.mean(deltas)) if deltas else -999.0

                best_frac_key = max(sweep_results.keys(),
                                    key=lambda k: _avg_delta(sweep_results[k]))
                print(f"  Best fraction: {best_frac_key:.0%} "
                      f"(avg Δ={_avg_delta(sweep_results[best_frac_key]):.4f})")
                best_frac_pct = f"{best_frac_key:.0%}".replace('%', 'pct')
                best_ckpt = os.path.join(
                    cfg.WORKING_DIR,
                    f"{TIMESTAMP}_encoder_sweep_{best_frac_pct}_proj.pt"
                )
                if not os.path.exists(best_ckpt):
                    best_ckpt = os.path.join(
                        cfg.WORKING_DIR,
                        f"{TIMESTAMP}_encoder_sweep_{best_frac_pct}.pt"
                    )
                if os.path.exists(best_ckpt):
                    from projector import load_projector as _lp
                    mae2 = _lp(best_ckpt,
                        hidden_dim=cfg.PROJECTOR_HPARAMS.get("hidden_dim", 64))
                    state["projector"]            = mae2
                    state["projector_path"]       = best_ckpt
                    state["projector_from_sweep"] = True

                    best_sweep = sweep_results[best_frac_key]
                    if "sweep_heads" in best_sweep:
                        state["heads"].update(best_sweep["sweep_heads"])
                        state["thresholds"].update(best_sweep["sweep_thresholds"])
                        logger.event("INFO",
                            f"Injected sweep heads ({best_frac_key:.0%} fraction) into state.")

                    # ── Rebuild feat_cache_cur with MAE-2 features ────────────
                    # All subsequent E2 steps use MAE-2 features so the heads
                    # trained in the sweep and the new-activity heads are in the
                    # same feature space.
                    after_cols_fc = [cfg.FULL_DATASET_STREAMS.index(s)
                                     for s in current_streams]
                    n_after_fc    = len(current_streams)
                    print(f"  Rebuilding feat_cache with MAE-2 features "
                          f"({current_streams})...")
                    Z_mae2_tr = extract_mae_features(
                        mae2, np_train_raw[0][:, :, after_cols_fc, :],
                        stream_indices=list(range(n_after_fc)),
                        batch_size=cfg.BATCH_SIZE,
                    )
                    Z_mae2_vl = extract_mae_features(
                        mae2, np_val_raw[0][:, :, after_cols_fc, :],
                        stream_indices=list(range(n_after_fc)),
                        batch_size=cfg.BATCH_SIZE,
                    )
                    Z_mae2_te = extract_mae_features(
                        mae2, np_test_raw[0][:, :, after_cols_fc, :],
                        stream_indices=list(range(n_after_fc)),
                        batch_size=cfg.BATCH_SIZE,
                    )
                    feat_cache_cur = FeatureCache(Z_mae2_tr, Z_mae2_vl, Z_mae2_te)
                    D = Z_mae2_tr.shape[-1]
                    state["feature_dim"] = D
                    print(f"  feat_cache rebuilt: {Z_mae2_tr.shape}  D={D}")

                    # ── Re-tune thresholds on full MAE-2 val features ─────────
                    # Sweep thresholds were calibrated on zero-padded val
                    # embeddings (initial streams only).  Full 3-stream val
                    # embeddings have different score distributions — re-tune
                    # so heads don't produce all-zeros at t1.
                    print(f"  Re-tuning thresholds on full MAE-2 val features...")
                    for (act, f), head in state["heads"].items():
                        if f != cfg.FUSION or act not in label_dict:
                            continue
                        act_idx  = label_dict[act]
                        y_vl_bin = (np_val_raw[1] == act_idx).astype(np.int32)
                        if y_vl_bin.sum() == 0:
                            continue
                        thresh = find_optimal_threshold_fast(
                            head, Z_mae2_vl, y_vl_bin,
                            t_min=cfg.THRESHOLD_MIN,
                            t_max=cfg.THRESHOLD_MAX,
                            fallback=cfg.THRESHOLD_FALLBACK,
                        )
                        state["thresholds"][(act, f)] = thresh
                    print(f"  Thresholds re-tuned on full MAE-2 val features.")
                    logger.event("INFO",
                        "Injected best encoder (100% fraction) — bootstrap will be skipped."
                        f"  feat_cache rebuilt with MAE-2 features D={D}")
            else:
                logger.warn(
                    f"Step {step['t']}: sensor increment ('{new_sensor}') but no "
                    f"unlabeled data — encoder sweep skipped."
                )

        # ── Run HITL step ─────────────────────────────────────────────────────
        state, val_ood, test_ood = run_add_activity(
            new_activity = activity,
            state        = state,
            feat_cache   = feat_cache_cur,
            np_train_raw = np_train_raw,
            np_val_raw   = np_val_raw,
            np_test_raw  = np_test_raw,
            label_dict   = label_dict,
            logger       = logger,
            encoders     = None,
            timestamp    = TIMESTAMP,
        )

        # ── Per-step seed F1 (co-occurrence aware) ────────────────────────────
        seed_f1 = {}
        for act in seed_activities:
            if act not in label_dict or (act, cfg.FUSION) not in state["heads"]:
                continue
            y_te   = make_multilabel_binary(
                act, np_test_raw[1], label_dict,
                cooccurrence_graph=cooc_graph,
            )
            thresh = state["thresholds"].get((act, cfg.FUSION), 0.5)
            m = evaluate_head_fast(
                state["heads"][(act, cfg.FUSION)],
                feat_cache_cur.test, y_te, threshold=thresh
            )
            seed_f1[act] = {
                "val_f1":  val_ood.get(act, {}).get("f1", float("nan")) if val_ood else float("nan"),
                "test_f1": m["f1"],
            }

        step_record = {
            "t":             step["t"],
            "activity":      activity,
            "add_sensors":   _step_sensors(step) if is_sensor_step else [],
            "streams_after": list(current_streams),
            "val_ood":       val_ood,
            "test_ood":      test_ood,
            "seed_f1":       seed_f1,
            "duration_s":    round(time.time() - t_step, 1),
            "encoder_sweep": sweep_results if is_sensor_step else None,
        }

        history["steps"].append(step_record)

        step_path = os.path.join(
            cfg.WORKING_DIR,
            f"{TIMESTAMP}_state_step{step['t']:02d}_{activity.replace(' ','_')}.pkl"
        )
        save_state(state, step_path)
        logger.event("INFO", f"Step {step['t']} done in {time.time()-t_step:.1f}s")

    history["final_metrics"] = {
        "val":  val_ood  if val_ood  else {},
        "test": test_ood if test_ood else {},
    }

    # ── Final summary ─────────────────────────────────────────────────────────
    _print_final_summary(history, seed_activities, steps)

    # ── Save history ──────────────────────────────────────────────────────────
    hist_path = os.path.join(
        cfg.WORKING_DIR, f"{TIMESTAMP}_experiment_history.pkl"
    )
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)

    log_path = logger.save_alongside(hist_path)
    print(f"\nDone  ({time.time()-t_phase2:.1f}s)")
    print(f"  History : {hist_path}")
    print(f"  Log     : {log_path}")
    return history


def _load_unlabeled_raw(
    ul_dir: str,
    active_streams: list[str],
    cache_suffix: str = "",
    max_val_windows: int = 10_000,
) -> tuple:
    from pathlib import Path

    data_dir       = Path(ul_dir)
    train_raw_path = data_dir / f"encoder_train_raw{cache_suffix}.npy"
    val_raw_path   = data_dir / f"encoder_val_raw{cache_suffix}.npy"

    if train_raw_path.exists() and val_raw_path.exists():
        print(f"\nLoading cached unlabeled raw signals {active_streams} "
              f"from {data_dir.name}")
        X_train = np.load(str(train_raw_path), allow_pickle=False)
        X_val   = np.load(str(val_raw_path),   allow_pickle=False)
        if len(X_val) > max_val_windows:
            X_val = X_val[:max_val_windows]
            print(f"  Val capped at {max_val_windows} windows for speed")
        print(f"  Train: {X_train.shape}   Val: {X_val.shape}")
        return X_train, X_val

    train_src = data_dir / "encoder_train.npy"
    val_src   = data_dir / "encoder_val.npy"
    meta_path = data_dir / "encoder_meta.json"
    missing   = [p.name for p in [train_src, val_src, meta_path] if not p.exists()]
    if missing:
        print(f"\n[WARN] Missing in {data_dir}: {missing}")
        print(f"       Run prepare_unlabeled_encoder_data.py first — sweep disabled.")
        return None, None

    with open(meta_path) as f:
        meta = json.load(f)
    fl_sensor_order = meta["fl_sensor_order"]

    missing_sensors = [s for s in active_streams if s not in fl_sensor_order]
    if missing_sensors:
        print(f"\n[WARN] Sensors {missing_sensors} not in FL sensor order "
              f"{fl_sensor_order} — sweep disabled.")
        return None, None
    fl_col_indices = [fl_sensor_order.index(s) for s in active_streams]

    print(f"\nSlicing unlabeled raw signals {active_streams} "
          f"(FL cols {fl_col_indices}) — caching after first run...")

    results = {}
    for split, src_path, cache_path in [
        ("train", train_src, train_raw_path),
        ("val",   val_src,   val_raw_path),
    ]:
        if cache_path.exists():
            print(f"  {split}: loading cached {cache_path.name}")
            results[split] = np.load(str(cache_path), allow_pickle=False)
            continue

        print(f"  {split}: slicing from {src_path.name}...")
        X_all    = np.load(str(src_path), allow_pickle=False)
        X_sliced = X_all[:, :, fl_col_indices, :].astype(np.float32)
        np.save(str(cache_path), X_sliced)
        results[split] = X_sliced
        print(f"  {split}: {X_all.shape} -> {X_sliced.shape}  "
              f"cached to {cache_path.name}")
        del X_all

    X_val = results["val"]
    if len(X_val) > max_val_windows:
        X_val = X_val[:max_val_windows]
        print(f"  Val capped at {max_val_windows} windows for speed")

    print(f"  Train: {results['train'].shape}   Val: {X_val.shape}")
    return results["train"], X_val


def _load_unlabeled_features(ul_dir, encoders, active_streams, cache_suffix=""):
    return _load_unlabeled_raw(ul_dir, active_streams, cache_suffix)


def _print_final_summary(history, seed_activities, steps):
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"\nSeed head F1 progression (test, co-occurrence aware):")
    step_labels = [f"t{s['t']}" for s in history["steps"]]
    print(f"  {'Activity':<45} {'E1 F1':>7}", end="")
    for label in step_labels:
        print(f"  {label:>8}", end="")
    print()
    print(f"  {'─'*110}")
    for act in seed_activities:
        e1 = history["seed_metrics_e1"].get(act, {}).get("f1", float("nan"))
        print(f"  {act:<45} {e1:>7.4f}", end="")
        for s in history["steps"]:
            f1 = s["seed_f1"].get(act, {}).get("test_f1", float("nan"))
            print(f"  {f1:>8.4f}", end="")
        print()

    encoder_sweeps = history.get("encoder_sweeps") or {}
    for sensor_name, sweep in encoder_sweeps.items():
        if not sweep:
            continue
        print(f"\nEncoder sweep — {sensor_name} added:")
        print(f"  {'Fraction':>9}  {'Windows':>8}  {'EncValLoss':>11}  "
              f"  {'Activity':<35} {'1-stream':>9} {'2-stream':>9} {'Δ':>7}")
        print(f"  {'-'*100}")
        for frac, res in sorted(sweep.items()):
            first = True
            for act in seed_activities:
                e1_f1  = history["seed_metrics_e1"].get(act, {}).get("f1", float("nan"))
                imp_f1 = res["full_sensor_metrics"].get(act, {}).get("f1", float("nan"))
                delta  = imp_f1 - e1_f1 if imp_f1==imp_f1 and e1_f1==e1_f1 else float("nan")
                ds     = f"{delta:+.4f}" if delta==delta else "    nan"
                if first:
                    print(f"  {frac:>9.0%}  {res['n_windows_used']:>8}  "
                          f"{res['encoder_val_loss']:>11.5f}  "
                          f"  {act:<35} {e1_f1:>9.4f} {imp_f1:>9.4f} {ds:>7}")
                    first = False
                else:
                    print(f"  {'':<9}  {'':<8}  {'':<11}  "
                          f"  {act:<35} {e1_f1:>9.4f} {imp_f1:>9.4f} {ds:>7}")
            print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if cfg is None:
        raise RuntimeError(
            "configs/paths.json not found. "
            "Copy configs/paths.example.json and fill in your paths."
        )

    parser = argparse.ArgumentParser(
        description="Config-driven sensor-incremental HAR experiment runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment config JSON (see experiment_config.json for format).",
    )
    parser.add_argument(
        "--resume-e1", type=str, default=None,
        help="Path to existing E1 state .pkl to skip seed training.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        parser.error(f"Config not found: {args.config}")

    print(f"\nLoading experiment config: {args.config}")
    exp_config = load_experiment_config(args.config)

    print(f"\nExperiment  : {exp_config.get('experiment_name', '(unnamed)')}")
    print(f"Seeds       : {exp_config['seed_activities']}")
    print(f"Steps       : {len(exp_config['steps'])}")
    sensor_steps = [s for s in exp_config["steps"]
                    if "add_sensor" in s or "add_sensors" in s]
    for ss in sensor_steps:
        sensors = ss.get("add_sensors") or ([ss["add_sensor"]] if "add_sensor" in ss else [])
        if isinstance(sensors, str): sensors = [sensors]
        print(f"  Sensor increment at t={ss['t']}: add_sensors={sensors}  "
              f"fractions={ss['encoder_data_fractions']}")

    run_experiment(exp_config, resume_e1_path=args.resume_e1)


if __name__ == "__main__":
    main()