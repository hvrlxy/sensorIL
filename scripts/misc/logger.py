"""
logger.py
=========
Structured logger for HITL-HAR experiment runs.

Each experiment step writes a JSON log entry alongside the .pkl state file,
making every saved state self-describing and reproducible.

Usage
-----
  from logger import RunLogger

  logger = RunLogger(working_dir=cfg.WORKING_DIR, run_id=TIMESTAMP)
  logger.log_run_start(cfg)
  logger.log_seed_result("Walking", metrics, threshold)
  logger.log_activity_step("Treadmill_2mph_Lab", step_info)
  logger.save()           # writes <TIMESTAMP>_run_log.json
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _serialisable(obj: Any) -> Any:
    """Recursively convert numpy/torch objects to JSON-safe types."""
    # Lazy imports — logger should not force numpy/torch at module level
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialisable(i) for i in obj]
    if isinstance(obj, set):
        return sorted([_serialisable(i) for i in obj])
    if isinstance(obj, Path):
        return str(obj)
    return obj


class RunLogger:
    """
    Accumulates structured log entries for one experiment run (E1, E2, or E3)
    and writes them to a JSON file alongside the .pkl state.

    JSON structure
    --------------
    {
      "run_id":        "DS_11",
      "script":        "E1_train_seeds",
      "started_at":    "2026-05-04T12:00:00Z",
      "finished_at":   null,
      "duration_s":    null,
      "environment":   { python, torch, numpy, platform },
      "config_snapshot": { ... full config at run start ... },
      "seed_results":  { activity: { metrics, threshold } },
      "incremental_steps": [ { step, activity, cooc, retrain, metrics } ],
      "ablation_results":  { condition: { flags, final_metrics } },
      "events":        [ { time, level, message } ]
    }
    """

    def __init__(self, working_dir: str, run_id: str, script: str = "unknown"):
        self.working_dir  = working_dir
        self.run_id       = run_id
        self.script       = script
        self._started_at  = datetime.now(timezone.utc).isoformat()
        self._start_time  = time.time()

        self._data: dict = {
            "run_id":             run_id,
            "script":             script,
            "started_at":         self._started_at,
            "finished_at":        None,
            "duration_s":         None,
            "environment":        self._get_environment(),
            "config_snapshot":    {},
            "seed_results":       {},
            "incremental_steps":  [],
            "ablation_results":   {},
            "events":             [],
        }

    # ── Environment ───────────────────────────────────────────────────────────

    @staticmethod
    def _get_environment() -> dict:
        env = {
            "python":   sys.version,
            "platform": platform.platform(),
        }
        try:
            import torch
            env["torch"]  = torch.__version__
            env["cuda"]   = torch.cuda.is_available()
            env["device"] = str(torch.cuda.get_device_name(0)) \
                            if torch.cuda.is_available() else "cpu"
        except ImportError:
            pass
        try:
            import numpy as np
            env["numpy"] = np.__version__
        except ImportError:
            pass
        try:
            import sklearn
            env["sklearn"] = sklearn.__version__
        except ImportError:
            pass
        return env

    # ── Config snapshot ───────────────────────────────────────────────────────

    def log_run_start(self, cfg) -> None:
        """
        Snapshot the full config at run start so the log is self-describing.
        Pass the Config object from config_loader.
        """
        snapshot = {
            "dataset_name":       cfg.DATASET_NAME,
            "dim":                cfg.DIM,
            "stream_names":       cfg.STREAM_NAMES,
            "seed_activities":    cfg.SEED_ACTIVITIES,
            "fusion":             cfg.FUSION,
            "participants":       cfg.PARTICIPANTS,
            "encoder_paths":      cfg.ENCODER_PATHS,
            "batch_size":         cfg.BATCH_SIZE,
            "seed_head_epochs":   cfg.SEED_HEAD_EPOCHS,
            "incremental_epochs": cfg.INCREMENTAL_HEAD_EPOCHS,
            "retrain_epochs":     cfg.RETRAIN_EPOCHS,
            "learning_rate":      cfg.LEARNING_RATE,
            "retrain_lr":         cfg.RETRAIN_LR,
            "focal_gamma":        cfg.FOCAL_GAMMA,
            "threshold_min":      cfg.THRESHOLD_MIN,
            "threshold_max":      cfg.THRESHOLD_MAX,
            "cooc_fire_threshold":cfg.COOC_FIRE_THRESHOLD,
            "projector_hparams":  cfg.PROJECTOR_HPARAMS,
            "hitl_simulation":    cfg.HITL_SIMULATION,
        }
        self._data["config_snapshot"] = snapshot
        self.event("INFO", f"Run started — dataset={cfg.DATASET_NAME} "
                           f"fusion={cfg.FUSION} participants={cfg.PARTICIPANTS}")

    # ── Seed results (E1) ─────────────────────────────────────────────────────

    def log_seed_result(self, activity: str, metrics: dict,
                        threshold: float, duration_s: float = 0.0) -> None:
        self._data["seed_results"][activity] = {
            "metrics":    _serialisable(metrics),
            "threshold":  float(threshold),
            "duration_s": float(duration_s),
        }
        self.event("INFO",
                   f"Seed '{activity}' — AUC={metrics.get('auc', 0):.4f} "
                   f"F1={metrics.get('f1', 0):.4f} thresh={threshold:.2f}")

    # ── Incremental step (E2 / E3) ────────────────────────────────────────────

    def log_activity_step(self, activity: str, step_info: dict) -> None:
        """
        step_info keys (all optional):
          step           int       index in incremental sequence
          confirmed      list      confirmed co-occurrences
          missed         list      missed co-occurrences (FNs)
          false_pos      list      false positive co-occurrences (FPs)
          retrained      list      heads retrained this step
          new_head_metrics dict    metrics for the new head
          pre_val_ood    dict      all-head metrics before this step (val)
          post_val_ood   dict      all-head metrics after this step (val)
          pre_test_ood   dict      all-head metrics before this step (test)
          post_test_ood  dict      all-head metrics after this step (test)
          threshold      float
          condition      str       ablation condition name (E3 only)
        """
        entry = {"activity": activity}
        entry.update(_serialisable(step_info))
        self._data["incremental_steps"].append(entry)

        cooc_summary = (f"confirmed={step_info.get('confirmed',[])} "
                        f"missed={step_info.get('missed',[])} "
                        f"fp={step_info.get('false_pos',[])}")
        nm = step_info.get("new_head_metrics", {})
        self.event("INFO",
                   f"Step '{activity}' — {cooc_summary} | "
                   f"new head AUC={nm.get('auc',0):.4f} F1={nm.get('f1',0):.4f}")

    # ── Ablation results (E3) ─────────────────────────────────────────────────

    def log_ablation_condition(self, condition_name: str,
                                flags: dict, final_metrics: dict,
                                retraining_events: int = 0) -> None:
        self._data["ablation_results"][condition_name] = {
            "flags":             flags,
            "final_metrics":     _serialisable(final_metrics),
            "retraining_events": retraining_events,
        }
        self.event("INFO",
                   f"Ablation '{condition_name}' done — "
                   f"retraining_events={retraining_events}")

    # ── Generic event log ─────────────────────────────────────────────────────

    def event(self, level: str, message: str) -> None:
        """Append a timestamped event to the log and print it."""
        ts = datetime.now(timezone.utc).isoformat()
        self._data["events"].append({
            "time":    ts,
            "level":   level,
            "message": message,
        })
        prefix = {"INFO": "ℹ", "WARN": "⚠", "ERROR": "✗"}.get(level, "·")
        print(f"[{level}] {message}")

    def warn(self, message: str) -> None:
        self.event("WARN", message)

    def error(self, message: str) -> None:
        self.event("ERROR", message)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, suffix: str = "") -> str:
        """
        Finalise and write the JSON log to working_dir.
        Returns the path written.
        Call this at the end of each script (E1/E2/E3).
        """
        self._data["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._data["duration_s"]  = round(time.time() - self._start_time, 1)

        fname = f"{self.run_id}_{self.script}{suffix}_log.json"
        path  = os.path.join(self.working_dir, fname)

        with open(path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

        self.event("INFO", f"Log saved to {path}")
        return path

    def save_alongside(self, state_pkl_path: str) -> str:
        """
        Convenience: save JSON log with same stem as the .pkl state file.
        E.g.  DS_11_e1_state_early.pkl  ->  DS_11_e1_state_early_log.json
        """
        stem    = Path(state_pkl_path).stem
        logpath = os.path.join(self.working_dir, f"{stem}_log.json")
        self._data["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._data["duration_s"]  = round(time.time() - self._start_time, 1)
        with open(logpath, "w") as f:
            json.dump(self._data, f, indent=2, default=str)
        print(f"[INFO] Log saved to {logpath}")
        return logpath