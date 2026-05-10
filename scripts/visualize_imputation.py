"""
visualize_imputation.py
=======================
Visualize real vs imputed raw signals for each activity.

Usage:
    python scripts/visualize_imputation.py \
        --encoder output/20260506_encoder_sweep_5pct.pt \
        --data-dir /mnt/storage/hitl_experiments/paaws_tuned/ \
        --participant DS_10 \
        --initial-sensors LeftWrist RightAnkle \
        --target-sensor RightThigh \
        --lab-sensor-order LeftAnkle LeftThigh LeftWaist LeftWrist RightAnkle RightThigh RightWaist RightWrist \
        --out-dir output/imputation_viz \
        --n-samples 5
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))


def load_lab_data(data_dir, participant, sensor_order):
    """Load lab .npy files, return (X, y, label_dict)."""
    from helpers import create_dataset_file_split
    from config_loader import cfg

    np_train, np_val, np_test, label_dict = create_dataset_file_split(
        data_dir, [participant], cfg.SEED
    )
    # Use val set for visualization
    X = np_val[0]   # (N, T, S, C)
    y = np_val[1]   # (N,)
    return X, y, label_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--participant", default="DS_10")
    parser.add_argument("--initial-sensors", nargs="+", required=True)
    parser.add_argument("--target-sensor", required=True)
    parser.add_argument("--lab-sensor-order", nargs="+",
                        default=["LeftAnkle","LeftThigh","LeftWaist","LeftWrist",
                                 "RightAnkle","RightThigh","RightWaist","RightWrist"])
    parser.add_argument("--out-dir", type=Path, default=Path("output/imputation_viz"))
    parser.add_argument("--n-samples", type=int, default=5)
    args = parser.parse_args()

    from projector import load_projector, impute_missing_streams

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load encoder
    print(f"Loading encoder: {args.encoder}")
    encoder = load_projector(str(args.encoder))
    encoder.eval()

    # Stream indices in lab data
    all_streams  = args.initial_sensors + [args.target_sensor]
    known_idx    = [args.lab_sensor_order.index(s) for s in args.initial_sensors]
    target_idx   = args.lab_sensor_order.index(args.target_sensor)
    all_idx      = [args.lab_sensor_order.index(s) for s in all_streams]
    n_streams    = len(all_streams)
    known_pos    = list(range(len(args.initial_sensors)))   # positions in all_streams
    target_pos   = len(args.initial_sensors)                # last position

    # Load lab data
    print(f"Loading lab data from {args.data_dir}")
    X_all, y_all, label_dict = load_lab_data(
        str(args.data_dir), args.participant, args.lab_sensor_order
    )
    # Slice to relevant sensors: (N, T, n_streams, C)
    X_sliced = X_all[:, :, all_idx, :]

    # Impute: give only initial sensors, predict target
    X_known   = X_sliced[:, :, known_pos, :]   # (N, T, S_known, C)
    X_imputed = impute_missing_streams(
        encoder, X_known, known_pos, n_streams, T=100, C=3
    )

    inv_label = {v: k for k, v in label_dict.items()}
    activities = sorted(label_dict.keys())

    axis_names = ["x", "y", "z"]
    colors_real  = ["#2196F3", "#4CAF50", "#F44336"]
    colors_imp   = ["#90CAF9", "#A5D6A7", "#EF9A9A"]

    print(f"\nGenerating plots for {len(activities)} activities, "
          f"{args.n_samples} samples each...")
    print(f"Output: {args.out_dir}")

    mse_summary = {}

    for act in activities:
        if act not in label_dict:
            continue
        label_int = label_dict[act]
        mask      = y_all == label_int
        if mask.sum() == 0:
            continue

        idx = np.where(mask)[0]
        np.random.seed(42)
        idx = np.random.choice(idx, min(args.n_samples, len(idx)), replace=False)

        real_target = X_sliced[idx, :, target_pos, :]   # (n, T, C)
        imp_target  = X_imputed[idx, :, target_pos, :]  # (n, T, C)
        mse = float(np.mean((real_target - imp_target) ** 2))
        var_real = float(np.var(real_target))
        var_imp  = float(np.var(imp_target))
        mse_summary[act] = {"mse": mse, "var_real": var_real, "var_imp": var_imp,
                             "ratio": var_imp / (var_real + 1e-8), "n": len(idx)}

        n = len(idx)
        fig = plt.figure(figsize=(18, n * 2.5 + 1.5))
        fig.suptitle(
            f"{act}  |  target: {args.target_sensor}  "
            f"|  MSE={mse:.4f}  var_ratio={var_imp/(var_real+1e-8):.2f}  "
            f"|  known: {args.initial_sensors}",
            fontsize=11, fontweight="bold"
        )

        gs = gridspec.GridSpec(n, 3, figure=fig,
                               hspace=0.5, wspace=0.3,
                               top=0.92, bottom=0.05)

        for i in range(n):
            for c in range(3):
                ax_real = fig.add_subplot(gs[i, c])
                t = np.arange(100)

                r = real_target[i, :, c]
                p = imp_target[i, :, c]

                # Real signal on left axis
                ax_real.plot(t, r, color=colors_real[c], lw=1.5,
                             label="Real", alpha=0.9)
                ax_real.set_xlim(0, 99)
                ax_real.tick_params(axis="y", labelsize=6,
                                    labelcolor=colors_real[c])
                ax_real.tick_params(axis="x", labelsize=6)

                # Imputed signal on independent right axis — own scale
                ax_imp = ax_real.twinx()
                ax_imp.plot(t, p, color=colors_imp[c], lw=1.5,
                            label="Imputed", linestyle="--", alpha=0.9)
                ax_imp.tick_params(axis="y", labelsize=6,
                                   labelcolor=colors_imp[c])
                # Pad both axes by 20% so lines don't hug the edges
                for ax, sig in [(ax_real, r), (ax_imp, p)]:
                    lo, hi = sig.min(), sig.max()
                    pad = max((hi - lo) * 0.2, 0.05)
                    ax.set_ylim(lo - pad, hi + pad)

                if i == 0:
                    ax_real.set_title(f"Axis {axis_names[c]}", fontsize=9)
                if c == 0:
                    ax_real.set_ylabel(f"Sample {i+1}", fontsize=8)

                # Combined legend on first sample, last column only
                if i == 0 and c == 2:
                    lines_r, labs_r = ax_real.get_legend_handles_labels()
                    lines_i, labs_i = ax_imp.get_legend_handles_labels()
                    ax_imp.legend(lines_r + lines_i, labs_r + labs_i,
                                  fontsize=7, loc="upper right")

                # Footer: per-channel cosine similarity of shape
                # (scale-invariant — tells us if shape is correct)
                r_c = r - r.mean();  p_c = p - p.mean()
                denom = (np.linalg.norm(r_c) * np.linalg.norm(p_c) + 1e-8)
                cos = float(np.dot(r_c, p_c) / denom)
                ch_mse = float(np.mean((r - p) ** 2))
                ax_real.set_xlabel(
                    f"cos={cos:.2f}  mse={ch_mse:.3f}", fontsize=6)

        safe = act.replace("/", "_").replace(" ", "_")
        out  = args.out_dir / f"{safe}.png"
        fig.savefig(out, dpi=80, bbox_inches="tight")
        plt.close(fig)
        print(f"  {act:<50} MSE={mse:.4f}  var_ratio={var_imp/(var_real+1e-8):.2f}  → {out.name}")

    # Summary plot
    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, len(mse_summary) * 0.4 + 1)))
    acts   = list(mse_summary.keys())
    mses   = [mse_summary[a]["mse"]   for a in acts]
    ratios = [mse_summary[a]["ratio"] for a in acts]
    colors = ["#F44336" if r < 0.2 else "#FF9800" if r < 0.5 else "#4CAF50"
              for r in ratios]

    axes[0].barh(acts, mses, color=colors)
    axes[0].set_xlabel("MSE (lower=better)")
    axes[0].set_title("Imputation MSE per activity")
    axes[0].axvline(np.mean(mses), color="black", linestyle="--", lw=1, label="mean")
    axes[0].legend(fontsize=8)

    axes[1].barh(acts, ratios, color=colors)
    axes[1].set_xlabel("var(imputed) / var(real)")
    axes[1].set_title("Variance ratio (1.0 = perfect)")
    axes[1].axvline(1.0, color="black", linestyle="--", lw=1)

    # Scatter MSE vs var_ratio
    axes[2].scatter(mses, ratios,
                    c=["#F44336" if r < 0.2 else "#4CAF50" for r in ratios],
                    s=80, zorder=3)
    for a, m, r in zip(acts, mses, ratios):
        axes[2].annotate(a[:15], (m, r), fontsize=6, ha="left")
    axes[2].set_xlabel("MSE")
    axes[2].set_ylabel("Variance ratio")
    axes[2].set_title("MSE vs Variance ratio")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"Imputation quality summary\n"
        f"Encoder: {args.encoder.name}  |  "
        f"Known: {args.initial_sensors}  →  Target: {args.target_sensor}",
        fontsize=10
    )
    plt.tight_layout()
    summary_path = args.out_dir / "summary.pdf"
    fig.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSummary saved: {summary_path}")
    print(f"\n{'Activity':<45} {'MSE':>8} {'var_real':>10} {'var_imp':>9} {'ratio':>6}")
    print("-" * 82)
    for act in sorted(mse_summary, key=lambda a: mse_summary[a]["mse"]):
        d = mse_summary[act]
        print(f"{act:<45} {d['mse']:>8.4f} {d['var_real']:>10.4f} "
              f"{d['var_imp']:>9.4f} {d['ratio']:>6.2f}")


if __name__ == "__main__":
    main()