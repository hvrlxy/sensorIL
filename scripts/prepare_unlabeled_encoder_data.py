"""
prepare_unlabeled_encoder_data.py
==================================
Stage free-living (FL) .npy files as a single unlabeled dataset with ALL
sensors preserved.  Run this ONCE before any experiments.

The script simply combines per-activity .npy files, shuffles windows, and
saves train/val splits.  It does NOT select sensors or reorder columns —
all 5 sensors are kept in their original FL axis-2 order.

Sensor selection happens INSIDE the experiment runner at each sensor
increment step, where it slices the columns it needs from these files.

FL axis-2 sensor order (PAAWS device — preserved as-is in output):
    0: LeftWrist
    1: RightAnkle
    2: RightThigh
    3: RightWaist
    4: RightWrist

Output
------
    <out-dir>/encoder_train.npy    (N_train, 100, 5, 3)  float32
    <out-dir>/encoder_val.npy      (N_val,   100, 5, 3)  float32
    <out-dir>/encoder_meta.json    provenance + stats

Usage
-----
    python prepare_unlabeled_encoder_data.py \\
        --fl-dir /mnt/storage/hitl_experiments/paaws_fl_tuned/DS_10 \\
        --out-dir /mnt/storage/hitl_experiments/paaws_fl_tuned/DS_10 \\
        --train-frac 0.80

    # Pool multiple participant directories
    python prepare_unlabeled_encoder_data.py \\
        --fl-dir /mnt/storage/.../DS_10 /mnt/storage/.../DS_11 \\
        --out-dir /mnt/storage/.../unlabeled

    # Verify
    python prepare_unlabeled_encoder_data.py \\
        --verify /mnt/storage/hitl_experiments/paaws_fl_tuned/DS_10

Activity files excluded by default
------------------------------------
    Synchronizing_Sensors                     — device calibration artefact
    PA_Type_Too_Complex                       — label quality issue
    PA_Type_Video_Unavailable_Indecipherable  — label quality issue
    PA_Type_Other                             — noisy catch-all
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Physical sensor order in the PAAWS FL .npy files (axis-2).
# Saved in encoder_meta.json so the runner knows how to slice columns.
FL_SENSOR_ORDER = [
    "LeftWrist",    # 0
    "RightAnkle",   # 1
    "RightThigh",   # 2
    "RightWaist",   # 3
    "RightWrist",   # 4
]

DEFAULT_EXCLUDE = {
    "Synchronizing_Sensors",
    "PA_Type_Too_Complex",
    "PA_Type_Video_Unavailable_Indecipherable",
    "PA_Type_Other",
}


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_fl_dir(fl_dir: Path, exclude: set, verbose: bool = True):
    """
    Load all valid .npy files from fl_dir, drop NaN windows, concatenate.
    Returns (X, file_stats) where X is (N, 100, 5, 3) float32.
    """
    npy_files = [f for f in sorted(fl_dir.glob("*.npy"))
                 if f.stem not in exclude]
    if not npy_files:
        raise FileNotFoundError(
            f"No .npy files in {fl_dir} after exclusions.\n"
            f"Excluded: {sorted(exclude)}"
        )

    expected_S = len(FL_SENSOR_ORDER)
    arrays, file_stats = [], {}

    for fpath in npy_files:
        try:
            X = np.load(str(fpath), allow_pickle=False)
        except Exception as e:
            print(f"  [WARN] {fpath.name}: load failed ({e}) — skipping")
            continue

        if X.ndim == 3:
            X = X[None]                 # single window saved as (T, S, C)
        if X.ndim != 4:
            print(f"  [WARN] {fpath.name}: ndim={X.ndim}, expected 4 — skipping")
            continue

        N, T, S, C = X.shape

        if S != expected_S:
            print(f"  [WARN] {fpath.name}: {S} sensors in axis-2, "
                  f"expected {expected_S} — skipping")
            continue
        if T != 100:
            print(f"  [WARN] {fpath.name}: T={T}, expected 100 — skipping")
            continue
        if C != 3:
            print(f"  [WARN] {fpath.name}: C={C}, expected 3 — skipping")
            continue

        nan_mask = np.any(np.isnan(X), axis=(1, 2, 3))
        if nan_mask.any():
            print(f"  [WARN] {fpath.name}: dropping {nan_mask.sum()}/{N} NaN windows")
            X = X[~nan_mask]
        if X.shape[0] == 0:
            print(f"  [WARN] {fpath.name}: 0 valid windows after NaN drop — skipping")
            continue

        arrays.append(X.astype(np.float32))
        file_stats[fpath.stem] = int(X.shape[0])
        if verbose:
            print(f"  {fpath.stem:<58} {X.shape[0]:>5} windows")

    if not arrays:
        raise ValueError(f"All files in {fl_dir} were skipped.")
    return np.concatenate(arrays, axis=0), file_stats


# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────

def build_encoder_dataset(
    fl_dirs: list,
    out_dir: Path,
    train_frac: float = 0.80,
    seed: int = 42,
    exclude: set = DEFAULT_EXCLUDE,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:

    rng = np.random.default_rng(seed)

    print(f"\n{'='*60}")
    print(f"Free-living encoder dataset builder")
    print(f"  Source dirs : {[str(d) for d in fl_dirs]}")
    print(f"  Output dir  : {out_dir}")
    print(f"  Sensors     : {FL_SENSOR_ORDER}  (all kept, no selection)")
    print(f"  Train/val   : {train_frac:.0%} / {1-train_frac:.0%}")
    print(f"  Excluded    : {sorted(exclude)}")
    print(f"{'='*60}")

    # ── Load ─────────────────────────────────────────────────────────────────
    all_arrays, all_stats = [], {}
    for fl_dir in fl_dirs:
        print(f"\n[Loading] {fl_dir.name}")
        X_dir, stats_dir = load_fl_dir(fl_dir, exclude, verbose=verbose)
        all_arrays.append(X_dir)
        for k, v in stats_dir.items():
            all_stats[f"{fl_dir.name}/{k}"] = v
        print(f"  Subtotal: {X_dir.shape[0]:,} windows")

    X_all   = np.concatenate(all_arrays, axis=0)
    N_total = X_all.shape[0]
    print(f"\n{'─'*60}")
    print(f"Combined : {N_total:,} windows   shape={X_all.shape}")
    print(f"Memory   : {X_all.nbytes / 1e6:.1f} MB (float32)")

    # ── Window distribution ───────────────────────────────────────────────────
    print(f"\nWindow counts per activity:")
    for stem, count in sorted(all_stats.items(), key=lambda x: -x[1]):
        pct = count / N_total
        print(f"  {stem:<58} {count:>5}  {pct:>5.1%}  {'█'*int(pct*40)}")

    # ── Shuffle + split ───────────────────────────────────────────────────────
    print(f"\nShuffling {N_total:,} windows (seed={seed})...")
    X_all   = X_all[rng.permutation(N_total)]
    n_train = int(N_total * train_frac)
    n_val   = N_total - n_train
    X_train = X_all[:n_train]
    X_val   = X_all[n_train:]
    print(f"Split:  train={n_train:,}  val={n_val:,}")

    # ── Per-sensor stats ──────────────────────────────────────────────────────
    print(f"\nPer-sensor statistics (train set):")
    print(f"  {'Col':>3}  {'Sensor':<14}  {'mean':>7}  {'std':>7}  {'min':>7}  {'max':>7}")
    for s, name in enumerate(FL_SENSOR_ORDER):
        vals = X_train[:, :, s, :].reshape(-1)
        print(f"  {s:>3}  {name:<14}  {vals.mean():>7.3f}  {vals.std():>7.3f}  "
              f"{vals.min():>7.3f}  {vals.max():>7.3f}")

    # ── Meta ──────────────────────────────────────────────────────────────────
    meta = {
        "created_at":      datetime.now(timezone.utc).isoformat(),
        "seed":            seed,
        "train_frac":      train_frac,
        "n_train":         int(n_train),
        "n_val":           int(n_val),
        "n_total":         int(N_total),
        "sample_shape":    list(X_all.shape[1:]),   # [100, 5, 3]
        "fl_sensor_order": FL_SENSOR_ORDER,          # axis-2 column mapping
        "source_dirs":     [str(d) for d in fl_dirs],
        "excluded":        sorted(exclude),
        "per_file_windows": all_stats,
        "files": {
            "train": "encoder_train.npy",
            "val":   "encoder_val.npy",
            "meta":  "encoder_meta.json",
        },
    }

    if dry_run:
        print(f"\n[Dry run] Would write to {out_dir}:")
        print(f"  encoder_train.npy  {X_train.shape}  {X_train.nbytes/1e6:.1f} MB")
        print(f"  encoder_val.npy    {X_val.shape}  {X_val.nbytes/1e6:.1f} MB")
        print(f"  encoder_meta.json")
        return meta

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / "encoder_train.npy"), X_train)
    np.save(str(out_dir / "encoder_val.npy"),   X_val)
    with open(out_dir / "encoder_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Written to {out_dir}")
    print(f"  encoder_train.npy  {X_train.shape}  {X_train.nbytes/1e6:.1f} MB")
    print(f"  encoder_val.npy    {X_val.shape}  {X_val.nbytes/1e6:.1f} MB")
    print(f"  encoder_meta.json")
    print(f"\nVerify:  python prepare_unlabeled_encoder_data.py --verify {out_dir}")
    print(f"{'='*60}\n")
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Verify
# ─────────────────────────────────────────────────────────────────────────────

def verify_output(out_dir: Path):
    meta_path = out_dir / "encoder_meta.json"
    if not meta_path.exists():
        print(f"No encoder_meta.json in {out_dir}")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    print(f"\n{'='*60}")
    print(f"Verifying {out_dir}")
    print(f"{'='*60}")
    print(f"\n  FL sensor order (axis-2): {meta['fl_sensor_order']}")
    print(f"  Sensor selection happens in the experiment runner at each increment step.")

    ok = True
    for key in ("train", "val"):
        fpath = out_dir / meta["files"][key]
        X = np.load(str(fpath), allow_pickle=False)
        expected = tuple([meta[f"n_{key}"], *meta["sample_shape"]])
        checks = {
            "shape correct": X.shape == expected,
            "dtype float32": X.dtype == np.float32,
            "no NaN":        not bool(np.any(np.isnan(X))),
            "5 sensors":     X.shape[2] == 5,
            "T=100":         X.shape[1] == 100,
        }
        print(f"\n  {key.upper()}  {X.shape}")
        for label, passed in checks.items():
            print(f"    {'✓' if passed else '✗'}  {label}")
            if not passed:
                ok = False

    print(f"\n  {'✓ All checks passed' if ok else '✗ Checks failed'}")
    print(f"\n  Add to paths.json:")
    print(f'    "unlabeled_data_dir": "{out_dir}"')
    print(f"\n{'='*60}")
    sys.exit(0 if ok else 1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage FL .npy files as unlabeled encoder dataset (all 5 sensors).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--fl-dir", nargs="+", type=Path, metavar="DIR",
                        help="One or more FL dataset directories.")
    parser.add_argument("--out-dir", type=Path, required=False,
                        help="Output directory. Defaults to same as --fl-dir if one dir given.")
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude", nargs="*", default=None, metavar="STEM",
                        help="Override default exclusion list.")
    parser.add_argument("--include-all", action="store_true",
                        help="Include all files, no exclusions.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", type=Path, metavar="DIR")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.verify:
        verify_output(args.verify)   # exits

    if not args.fl_dir:
        parser.error("--fl-dir is required unless --verify is used.")

    out_dir = args.out_dir or args.fl_dir[0]

    exclude = (set() if args.include_all
               else set(args.exclude) if args.exclude is not None
               else DEFAULT_EXCLUDE)

    for d in args.fl_dir:
        if not d.is_dir():
            parser.error(f"Not a directory: {d}")
    if not (0.0 < args.train_frac < 1.0):
        parser.error(f"--train-frac must be in (0, 1), got {args.train_frac}")

    build_encoder_dataset(
        fl_dirs=args.fl_dir, out_dir=out_dir,
        train_frac=args.train_frac, seed=args.seed,
        exclude=exclude, dry_run=args.dry_run,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()