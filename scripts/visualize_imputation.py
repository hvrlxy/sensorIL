"""
visualize_imputation.py
=======================
Visualize real vs imputed raw signals for each activity.
Shows ALL sensor streams (known + target) so you can see the cross-sensor
physics. Real and imputed signals use the same color per axis, distinguished
by line style and alpha.

Layout per sample row:
  [known_sensor_1: x y z] [known_sensor_2: x y z] ... [target ★: x y z]
  Each group = one sensor, each subplot = one axis.
  Known sensors: solid line only.
  Target sensor: solid=Real, dashed=Imputed (same color, different alpha).

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
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.lines
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))

# Colors: one per sensor stream, consistent across axes
# Real signal: solid, full alpha
# Imputed signal: different color, dashed
REAL_COLOR    = "#2196F3"   # blue — real signal
IMPUTED_COLOR = "#FF6B35"   # orange — imputed signal
AXIS_NAMES    = ["x", "y", "z"]


def load_lab_data(data_dir, participant):
    from helpers import create_dataset_file_split
    from config_loader import cfg
    _, np_val, _, label_dict = create_dataset_file_split(
        data_dir, [participant], cfg.SEED
    )
    return np_val[0], np_val[1], label_dict


def plot_sensor_axes(axes_row, signal, imp_signal, is_target, sensor_name,
                     sample_idx, show_title):
    """
    Plot one sensor (3 axes) into axes_row[0..2].
    signal     : (T, 3) real
    imp_signal : (T, 3) or None — imputed (target sensor only)
    """
    T = signal.shape[0]
    t = np.arange(T)

    for c in range(3):
        ax  = axes_row[c]
        r   = signal[:, c]

        # Determine y-range from both real and imputed
        lo, hi = r.min(), r.max()
        if imp_signal is not None:
            lo = min(lo, imp_signal[:, c].min())
            hi = max(hi, imp_signal[:, c].max())
        pad = max((hi - lo) * 0.15, 0.03)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(0, T - 1)
        ax.tick_params(labelsize=5.5)

        if is_target and imp_signal is not None:
            p = imp_signal[:, c]
            # Imputed first (behind) — orange dashed
            ax.plot(t, p, color=IMPUTED_COLOR, lw=1.5, alpha=0.80,
                    linestyle="--",
                    label="Imputed" if (c == 2 and sample_idx == 0) else "_")
            # Real on top — blue solid
            ax.plot(t, r, color=REAL_COLOR, lw=1.5, alpha=0.90,
                    label="Real" if (c == 2 and sample_idx == 0) else "_")

            # Footer stats
            r_c    = r - r.mean(); p_c = p - p.mean()
            cos    = float(np.dot(r_c, p_c) /
                           (np.linalg.norm(r_c) * np.linalg.norm(p_c) + 1e-8))
            ch_mse = float(np.mean((r - p) ** 2))
            ax.set_xlabel(f"cos={cos:.2f}  mse={ch_mse:.3f}", fontsize=5.5)
        else:
            # Known sensor — real only, blue solid
            ax.plot(t, r, color=REAL_COLOR, lw=1.3, alpha=0.90)
            ax.set_xlabel("")

        if show_title:
            prefix = "★ " if is_target else ""
            ax.set_title(f"{prefix}{sensor_name}\n{AXIS_NAMES[c]}",
                         fontsize=7,
                         fontweight="bold" if is_target else "normal",
                         color="#B71C1C" if is_target else "#444")

        if c == 0 and sample_idx >= 0:
            ax.set_ylabel(f"S{sample_idx+1}", fontsize=7,
                          rotation=0, labelpad=16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder",          required=True, type=Path)
    parser.add_argument("--data-dir",         required=True, type=Path)
    parser.add_argument("--participant",      default="DS_10")
    parser.add_argument("--initial-sensors",  nargs="+", required=True)
    parser.add_argument("--target-sensor",    required=True)
    parser.add_argument("--lab-sensor-order", nargs="+",
                        default=["LeftAnkle","LeftThigh","LeftWaist","LeftWrist",
                                 "RightAnkle","RightThigh","RightWaist","RightWrist"])
    parser.add_argument("--out-dir",   type=Path, default=Path("output/imputation_viz"))
    parser.add_argument("--n-samples", type=int,  default=5)
    args = parser.parse_args()

    from projector import load_projector, impute_missing_streams

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading encoder: {args.encoder}")
    encoder = load_projector(str(args.encoder))
    encoder.eval()

    all_streams = args.initial_sensors + [args.target_sensor]
    all_idx     = [args.lab_sensor_order.index(s) for s in all_streams]
    n_streams   = len(all_streams)
    known_pos   = list(range(len(args.initial_sensors)))
    target_pos  = len(args.initial_sensors)

    print(f"Loading lab data from {args.data_dir}")
    X_all, y_all, label_dict = load_lab_data(str(args.data_dir), args.participant)
    X_sliced  = X_all[:, :, all_idx, :]         # (N, T, n_streams, 3)
    X_known   = X_sliced[:, :, known_pos, :]
    X_imputed = impute_missing_streams(
        encoder, X_known, known_pos, n_streams, T=100, C=3
    )

    activities = sorted(label_dict.keys())
    n_sensors  = len(all_streams)
    # Tight wspace between axes within a sensor, larger gap between sensors
    # We achieve this with nested GridSpec

    print(f"\nGenerating plots — {len(activities)} activities, "
          f"{args.n_samples} samples each")
    print(f"Layout: {n_sensors} sensors × 3 axes per row  |  "
          f"solid=Real  dashed=Imputed  (target ★ only)\n")

    mse_summary = {}

    for act in activities:
        if act not in label_dict:
            continue
        label_int = label_dict[act]
        mask = (y_all == label_int)
        if mask.sum() == 0:
            continue

        idx = np.where(mask)[0]
        np.random.seed(42)
        idx    = np.random.choice(idx, min(args.n_samples, len(idx)), replace=False)
        n      = len(idx)
        r_tgt  = X_sliced[idx, :, target_pos, :]   # (n, T, 3) real target
        i_tgt  = X_imputed[idx, :, target_pos, :]  # (n, T, 3) imputed target
        mse    = float(np.mean((r_tgt - i_tgt) ** 2))
        vr     = float(np.var(r_tgt))
        vi     = float(np.var(i_tgt))
        ratio  = vi / (vr + 1e-8)
        mse_summary[act] = dict(mse=mse, var_real=vr, var_imp=vi, ratio=ratio)

        # Figure layout: one row per sample, sensors stacked left-to-right
        # within each row. Each sensor gets its own tall subplot (all 3 axes
        # stacked vertically), so signals are large enough to read.
        #
        # Layout: rows = samples, cols = sensors
        # Each cell = 3 vertically-stacked axis subplots (x, y, z)

        n_rows_per_sample = 3   # x, y, z stacked
        total_rows = n * n_rows_per_sample
        fig_w = max(5 * n_sensors, 12)
        fig_h = total_rows * 1.8 + 2.0
        fig   = plt.figure(figsize=(fig_w, fig_h))
        fig.suptitle(
            f"{act}   |   target: ★{args.target_sensor}   "
            f"|   MSE={mse:.4f}   var_ratio={ratio:.2f}\n"
            f"known: {args.initial_sensors}     "
            f"[blue=Real  orange=Imputed]",
            fontsize=11, fontweight="bold", y=0.99
        )

        outer = gridspec.GridSpec(
            total_rows, n_sensors,
            figure=fig,
            hspace=0.15, wspace=0.35,
            top=0.95, bottom=0.03,
            left=0.07, right=0.98,
        )

        for i in range(n):
            for j, sensor_name in enumerate(all_streams):
                is_tgt = (j == target_pos)
                for c in range(3):
                    row = i * n_rows_per_sample + c
                    ax  = fig.add_subplot(outer[row, j])
                    r   = X_sliced[idx[i], :, j, c]
                    t   = np.arange(len(r))

                    # Y range — for known sensors only; target uses twin axes
                    lo, hi = r.min(), r.max()
                    pad = max((hi - lo) * 0.15, 0.03)
                    if not is_tgt:
                        ax.set_ylim(lo - pad, hi + pad)
                    ax.set_xlim(0, len(r) - 1)
                    ax.tick_params(labelsize=6)

                    if is_tgt:
                        p = i_tgt[i, :, c]
                        # Real on left axis — blue solid
                        ax.plot(t, r, color=REAL_COLOR, lw=1.4, alpha=0.90,
                                label="Real" if (i==0 and c==0) else "_")
                        # Imputed on real scale — black dashed (shows absolute error)
                        ax.plot(t, p, color="black", lw=1.2, alpha=0.55,
                                linestyle="--",
                                label="Imputed (real scale)" if (i==0 and c==0) else "_")
                        lo_r = min(r.min(), p.min())
                        hi_r = max(r.max(), p.max())
                        pad_r = max((hi_r - lo_r) * 0.15, 0.03)
                        ax.set_ylim(lo_r - pad_r, hi_r + pad_r)
                        ax.tick_params(axis='y', labelsize=6,
                                       labelcolor=REAL_COLOR)

                        # Imputed on independent right axis — orange (shape comparison)
                        ax2 = ax.twinx()
                        ax2.plot(t, p, color=IMPUTED_COLOR, lw=1.4, alpha=0.75,
                                 linestyle="--",
                                 label="Imputed (own scale)" if (i==0 and c==0) else "_")
                        lo_p, hi_p = p.min(), p.max()
                        pad_p = max((hi_p - lo_p) * 0.15, 0.03)
                        ax2.set_ylim(lo_p - pad_p, hi_p + pad_p)
                        ax2.tick_params(axis='y', labelsize=5.5,
                                        labelcolor=IMPUTED_COLOR)

                        r_c = r - r.mean(); p_c = p - p.mean()
                        cos = float(np.dot(r_c, p_c) /
                                    (np.linalg.norm(r_c)*np.linalg.norm(p_c)+1e-8))
                        ch_mse = float(np.mean((r-p)**2))
                        ax.set_xlabel(f"cos={cos:.2f} mse={ch_mse:.3f}",
                                      fontsize=5.5)
                    else:
                        ax.plot(t, r, color=REAL_COLOR, lw=1.3, alpha=0.90)
                        if c < 2:
                            ax.set_xticklabels([])

                    # Axis label on left
                    ax.set_ylabel(AXIS_NAMES[c], fontsize=7,
                                  rotation=0, labelpad=12)

                    # Column title: sensor name on first sample, first axis
                    if i == 0 and c == 0:
                        prefix = "★ " if is_tgt else ""
                        ax.set_title(f"{prefix}{sensor_name}", fontsize=8,
                                     fontweight="bold" if is_tgt else "normal",
                                     color="#B71C1C" if is_tgt else "#333")

                    # Sample label on left edge of first sensor, first axis
                    if j == 0 and c == 0:
                        ax.annotate(f"S{i+1}", xy=(-0.18, 0.5),
                                    xycoords="axes fraction",
                                    fontsize=8, fontweight="bold",
                                    va="center", ha="right")

        # Legend
        handles = [
            matplotlib.lines.Line2D([0],[0], color=REAL_COLOR,    lw=1.5,
                                    label="Real"),
            matplotlib.lines.Line2D([0],[0], color="black",       lw=1.2,
                                    linestyle="--", alpha=0.55,
                                    label="Imputed (real scale)"),
            matplotlib.lines.Line2D([0],[0], color=IMPUTED_COLOR, lw=1.5,
                                    linestyle="--",
                                    label="Imputed (own scale →)"),
        ]
        fig.legend(handles=handles, fontsize=8, loc="upper right",
                   bbox_to_anchor=(0.98, 0.98), framealpha=0.9)

        safe = act.replace("/", "_").replace(" ", "_")
        out  = args.out_dir / f"{safe}.png"
        fig.savefig(out, dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(f"  {act:<50} MSE={mse:.4f}  var_ratio={ratio:.2f}  → {out.name}")

    # ── Summary ───────────────────────────────────────────────────────────────
    acts   = sorted(mse_summary, key=lambda a: mse_summary[a]["mse"])
    mses   = [mse_summary[a]["mse"]   for a in acts]
    ratios = [mse_summary[a]["ratio"] for a in acts]
    clr    = ["#E53935" if r < 0.3 else
              "#FB8C00" if r < 0.7 else
              "#43A047" for r in ratios]

    fig, (ax0, ax1) = plt.subplots(1, 2,
                                    figsize=(16, max(5, len(acts)*0.38+1.5)))
    ax0.barh(acts, mses,   color=clr, edgecolor="white", height=0.7)
    ax0.axvline(np.mean(mses), color="#333", linestyle="--", lw=1.2,
                label=f"mean={np.mean(mses):.3f}")
    ax0.set_xlabel("MSE (lower = better)", fontsize=9)
    ax0.set_title("Imputation MSE per activity", fontsize=10)
    ax0.legend(fontsize=8); ax0.tick_params(labelsize=7)

    ax1.barh(acts, ratios, color=clr, edgecolor="white", height=0.7)
    ax1.axvline(1.0, color="#333", linestyle="--", lw=1.2, label="ideal = 1.0")
    ax1.set_xlabel("var(imputed) / var(real)", fontsize=9)
    ax1.set_title("Variance ratio (1.0 = perfect)", fontsize=10)
    ax1.legend(fontsize=8); ax1.tick_params(labelsize=7)

    fig.suptitle(
        f"Imputation quality summary  |  Encoder: {args.encoder.name}\n"
        f"Known: {args.initial_sensors}  →  Target: {args.target_sensor}",
        fontsize=10
    )
    plt.tight_layout()
    fig.savefig(args.out_dir / "summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSummary: {args.out_dir / 'summary.png'}")
    print(f"\n{'Activity':<45} {'MSE':>8} {'var_real':>10} {'var_imp':>9} {'ratio':>6}")
    print("-" * 82)
    for a in acts:
        d = mse_summary[a]
        print(f"{a:<45} {d['mse']:>8.4f} {d['var_real']:>10.4f} "
              f"{d['var_imp']:>9.4f} {d['ratio']:>6.2f}")


if __name__ == "__main__":
    main()