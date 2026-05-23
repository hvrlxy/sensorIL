"""
check_signal_stats.py
=====================
Check raw signal statistics per activity per sensor.
Helps diagnose whether amplitude mismatch is due to:
1. Class imbalance in FL data (too much sitting)
2. RightThigh having inherently different magnitude than LeftWrist/RightAnkle

Usage:
    python scripts/check_signal_stats.py \
        --lab-dir /mnt/storage/hitl_experiments/paaws_tuned/DS_10 \
        --fl-raw /mnt/storage/hitl_experiments/paaws_fl_tuned/DS_10/encoder_train_raw_LeftWrist_RightAnkle_RightThigh.npy \
        --sensors LeftWrist RightAnkle RightThigh
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab-dir",   type=Path, required=True)
    parser.add_argument("--fl-raw",    type=Path, required=True)
    parser.add_argument("--sensors",   nargs="+",
                        default=["LeftWrist", "RightAnkle", "RightThigh"])
    parser.add_argument("--lab-sensor-order", nargs="+",
                        default=["LeftAnkle","LeftThigh","LeftWaist","LeftWrist",
                                 "RightAnkle","RightThigh","RightWaist","RightWrist"])
    args = parser.parse_args()

    sensors = args.sensors
    S = len(sensors)

    # ── 1. Lab data: per activity per sensor ─────────────────────────────────
    print("=" * 70)
    print("LAB DATA — signal std per activity per sensor")
    print("=" * 70)

    lab_files = sorted(args.lab_dir.glob("*.npy"))
    sensor_cols = [args.lab_sensor_order.index(s) for s in sensors]

    print(f"\n  {'Activity':<45}", end="")
    for s in sensors:
        print(f"  {s:>12}", end="")
    print(f"  {'windows':>8}")
    print("  " + "-" * (45 + 14 * S + 10))

    activity_stats = {}
    for fpath in lab_files:
        try:
            X = np.load(str(fpath), allow_pickle=False)
        except:
            continue
        if X.ndim != 4 or X.shape[1] != 100:
            continue

        act = fpath.stem
        stds = []
        for col in sensor_cols:
            if col < X.shape[2]:
                stds.append(float(X[:, :, col, :].std()))
            else:
                stds.append(float("nan"))

        activity_stats[act] = {"stds": stds, "n": X.shape[0]}
        print(f"  {act:<45}", end="")
        for std in stds:
            print(f"  {std:>12.4f}", end="")
        print(f"  {X.shape[0]:>8}")

    # ── 2. Sensor-to-sensor ratio per activity ────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"SENSOR RATIO — RightThigh std / LeftWrist std (per activity)")
    print(f"{'='*70}")
    print(f"\n  {'Activity':<45}  {'RThigh/LWrist':>14}  {'RThigh/RAnkle':>14}")
    print("  " + "-" * 76)

    lw_idx = sensors.index("LeftWrist")  if "LeftWrist"  in sensors else None
    ra_idx = sensors.index("RightAnkle") if "RightAnkle" in sensors else None
    rt_idx = sensors.index("RightThigh") if "RightThigh" in sensors else None

    for act, stat in sorted(activity_stats.items()):
        stds = stat["stds"]
        r_lw = stds[rt_idx] / (stds[lw_idx] + 1e-8) if lw_idx is not None else float("nan")
        r_ra = stds[rt_idx] / (stds[ra_idx] + 1e-8) if ra_idx is not None else float("nan")
        print(f"  {act:<45}  {r_lw:>14.3f}  {r_ra:>14.3f}")

    # ── 3. FL data: class distribution ───────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"FL DATA — signal variance distribution")
    print(f"{'='*70}")

    print(f"\nLoading {args.fl_raw.name}...")
    X_fl = np.load(str(args.fl_raw), allow_pickle=False)
    print(f"Shape: {X_fl.shape}  dtype: {X_fl.dtype}")

    # Per-window variance across all sensors and axes
    win_var = X_fl.var(axis=(1, 2, 3))
    print(f"\nFL window variance distribution:")
    for thresh, label in [(0.01, "near-zero (<0.01)"),
                           (0.05, "low (0.01-0.05)"),
                           (0.1,  "medium (0.05-0.1)"),
                           (0.5,  "high (0.1-0.5)"),
                           (999,  "very high (>0.5)")]:
        prev = [0.01, 0.05, 0.1, 0.5, 0][
            [0.01, 0.05, 0.1, 0.5, 999].index(thresh)]
        mask = (win_var >= prev) & (win_var < thresh)
        pct  = mask.mean()
        bar  = "█" * int(pct * 50)
        print(f"  {label:<25}  {mask.sum():>7}  {pct:>6.1%}  {bar}")

    print(f"\nPer-sensor std in FL data:")
    print(f"  {'Sensor':<15}  {'mean_std':>10}  {'median_std':>11}")
    for i, s in enumerate(sensors):
        stds = X_fl[:, :, i, :].reshape(len(X_fl), -1).std(axis=1)
        print(f"  {s:<15}  {stds.mean():>10.4f}  {stds.median():>11.4f}"
              if hasattr(stds, 'median') else
              f"  {s:<15}  {stds.mean():>10.4f}  {float(np.median(stds)):>11.4f}")

    print(f"\nKey question: does FL data contain enough high-movement windows?")
    high_var = (win_var > 0.1).mean()
    print(f"  High-movement windows (var>0.1): {high_var:.1%}")
    print(f"  At 5% = 14554 windows: ~{int(high_var * 14554)} high-movement")
    print(f"  At 10% = 29109 windows: ~{int(high_var * 29109)} high-movement")


if __name__ == "__main__":
    main()
