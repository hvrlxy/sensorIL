"""
validate_dataset.py
===================
Pre-flight checks before running any experiment.

Verifies:
  - Dataset directory exists and has the expected structure
  - .npy files have the correct shape (T, DIM, C)
  - All seed activities have data
  - Class balance is reasonable
  - All activities in dataset config exist in the data
  - Sensor hint vector lengths match DIM

Usage
-----
  python validate_dataset.py
  python validate_dataset.py --fix-nan     # auto-delete nan.npy files
"""
import sys
from pathlib import Path
# Ensure repo root (parent of scripts/) is on sys.path so config_loader and
# helpers resolve correctly regardless of where the script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))              # scripts/

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from config_loader import cfg


def check(condition, message, fix=None):
    if condition:
        print(f"  ✓  {message}")
        return True
    else:
        if fix:
            print(f"  ⚠  {message} — {fix}")
        else:
            print(f"  ✗  {message}")
        return False


def validate(fix_nan: bool = False):
    errors   = 0
    warnings = 0

    print("=" * 60)
    print(f"HITL-HAR Dataset Validation")
    print(f"Dataset   : {cfg.DATASET_NAME}")
    print(f"Data dir  : {cfg.DATA_DIR}")
    print(f"DIM       : {cfg.DIM}")
    print("=" * 60)

    data_root = Path(cfg.DATA_DIR)

    # ── 1. Data directory ─────────────────────────────────────────────────────
    print("\n[1] Data directory")
    if not check(data_root.exists(), f"Data directory exists: {data_root}"):
        print("  Cannot continue — data directory missing.")
        return False

    participants = cfg.PARTICIPANTS
    if participants is None:
        participants = sorted([p.name for p in data_root.iterdir() if p.is_dir()])
    check(len(participants) > 0, f"{len(participants)} participant(s) found")

    # ── 2. Per-participant checks ─────────────────────────────────────────────
    print(f"\n[2] Per-participant file structure")
    found_labels   = set()
    shape_errors   = []
    nan_files      = []
    per_class_counts = {}

    expected_sample_shape = (100, cfg.DIM, 3)

    for pid in participants:
        pdir = data_root / pid
        if not pdir.exists():
            print(f"  ✗  Participant '{pid}' directory missing")
            errors += 1
            continue

        npy_files = [f for f in pdir.iterdir()
                     if f.is_file() and f.suffix == ".npy"]

        if not check(len(npy_files) > 0, f"  {pid}: {len(npy_files)} .npy files"):
            errors += 1
            continue

        for f in npy_files:
            if f.name.lower() == "nan.npy":
                nan_files.append(f)
                if fix_nan:
                    f.unlink()
                    print(f"  [FIX] Deleted {f}")
                continue

            try:
                X = np.load(f, allow_pickle=False)
            except Exception as e:
                print(f"  ✗  {pid}/{f.name}: load failed ({e})")
                errors += 1
                continue

            # Fix single-sample saves
            if X.ndim == 3 and tuple(X.shape) == expected_sample_shape:
                X = X[None]

            if X.ndim != 4 or tuple(X.shape[1:]) != expected_sample_shape:
                shape_errors.append((pid, f.name, X.shape))
                errors += 1
                continue

            label = f.stem
            found_labels.add(label)
            per_class_counts[label] = per_class_counts.get(label, 0) + X.shape[0]

    if shape_errors:
        print(f"\n  Shape errors (expected N×{expected_sample_shape}):")
        for pid, fname, shape in shape_errors:
            print(f"    {pid}/{fname}: {shape}")

    if nan_files and not fix_nan:
        print(f"\n  Found {len(nan_files)} nan.npy file(s). "
              f"Run with --fix-nan to delete them.")
        warnings += len(nan_files)

    # ── 3. Seed activities in data ────────────────────────────────────────────
    print(f"\n[3] Seed activities")
    for act in cfg.SEED_ACTIVITIES:
        present = act in found_labels
        n       = per_class_counts.get(act, 0)
        if not check(present and n >= 10,
                     f"'{act}': {n} windows {'(present)' if present else '(MISSING)'}",
                     fix="need ≥10 windows for reliable training"):
            errors += 1

    # ── 4. All configured activities in data ──────────────────────────────────
    print(f"\n[4] Configured activities vs data")
    configured  = set(cfg.ALL_ACTIVITIES)
    missing_acts = configured - found_labels
    extra_acts   = found_labels - configured

    check(len(missing_acts) == 0,
          f"{len(configured) - len(missing_acts)}/{len(configured)} configured activities found in data")
    if missing_acts:
        print(f"  Activities in config but NOT in data ({len(missing_acts)}):")
        for a in sorted(missing_acts):
            print(f"    - {a}")
        warnings += len(missing_acts)

    if extra_acts:
        print(f"  Activities in data but NOT in config ({len(extra_acts)}) — will be ignored:")
        for a in sorted(extra_acts):
            print(f"    + {a}")

    # ── 5. Class balance summary ──────────────────────────────────────────────
    print(f"\n[5] Class balance")
    if per_class_counts:
        counts = sorted(per_class_counts.values())
        print(f"  Min: {counts[0]}  Median: {counts[len(counts)//2]}  "
              f"Max: {counts[-1]}  Total windows: {sum(counts)}")
        sparse = {a: n for a, n in per_class_counts.items() if n < 10}
        if sparse:
            print(f"  Activities with < 10 windows (may cause training issues):")
            for a, n in sorted(sparse.items(), key=lambda x: x[1]):
                print(f"    {a}: {n}")
            warnings += len(sparse)

    # ── 6. Sensor hints ───────────────────────────────────────────────────────
    print(f"\n[6] Sensor hint validation")
    hints   = cfg._dataset["sensor_hints"]["hints"]
    uniform = [1.0] * cfg.DIM  # no gating
    ok = True
    for act, hint in hints.items():
        if len(hint) != cfg.DIM:
            print(f"  ✗  '{act}' hint length {len(hint)} ≠ DIM {cfg.DIM}")
            errors += 1
            ok = False
    if ok:
        check(True, f"All {len(hints)} sensor hints have correct length ({cfg.DIM})")
    check(len(uniform) == cfg.DIM,
          f"uniform_hint length {len(uniform)} matches DIM {cfg.DIM}")

    # ── 7. Encoder model file ─────────────────────────────────────────────────
    print(f"\n[7] Encoder model(s)")
    import os
    encoder_paths = cfg.ENCODER_PATHS
    for key, path in encoder_paths.items():
        if not check(os.path.exists(path),
                     f"Encoder '{key}' found: {path}"):
            errors += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if errors == 0 and warnings == 0:
        print("✓  All checks passed. Ready to run experiments.")
    elif errors == 0:
        print(f"✓  No errors. {warnings} warning(s) — review above.")
    else:
        print(f"✗  {errors} error(s), {warnings} warning(s). Fix before running experiments.")

    return errors == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate dataset and config before running HITL-HAR experiments."
    )
    parser.add_argument("--fix-nan", action="store_true",
                        help="Automatically delete nan.npy files found in data directory.")
    args = parser.parse_args()

    ok = validate(fix_nan=args.fix_nan)
    sys.exit(0 if ok else 1)