"""
visualize_translator.py
=======================
Run the trained translator on all activities in the lab dataset
and generate real vs generated signal visualizations.

Usage:
    python scripts/visualize_translator.py \
        --translator output/translator/translator.pt \
        --lab-data-dir /mnt/storage/hitl_experiments/paaws_tuned \
        --participant DS_11 \
        --known-sensors LeftWrist RightAnkle \
        --target-sensor RightThigh \
        --lab-sensor-order LeftAnkle LeftThigh LeftWaist LeftWrist RightAnkle RightThigh RightWaist RightWrist \
        --out-dir output/translator_viz_activities \
        --n-samples 5
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scripts.misc.signal_translator import load_translator, normalize_sample

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_activity(translator, X_samples, known_indices, target_idx,
                        activity_name, out_path, n_samples=5):
    """
    X_samples: (N, T, n_streams, C) — already sliced to [known + target] streams
    known_indices: indices within X_samples for known streams
    target_idx: index within X_samples for target stream
    """
    n = min(n_samples, len(X_samples))
    if n == 0:
        return

    np.random.seed(42)
    idx = np.random.choice(len(X_samples), n, replace=False)
    samples = X_samples[idx]

    T   = samples.shape[1]
    C   = samples.shape[3]
    n_known = len(known_indices)

    translator.generator.eval()
    with torch.no_grad():
        batch    = torch.from_numpy(samples.astype(np.float32)).to(DEVICE)
        X_k      = batch[:, :, known_indices, :]   # (n, T, n_known, C)
        X_tgt    = batch[:, :, target_idx, :]      # (n, T, C)

        # Normalize known streams
        kn_list = []
        for k in range(n_known):
            xk, _, _ = normalize_sample(X_k[:, :, k, :])
            kn_list.append(xk)
        kn_norm = torch.stack(kn_list, dim=2)

        B  = batch.shape[0]
        T  = batch.shape[1]
        C  = batch.shape[3]

        # Weighted proxy for dynamic component
        pw = getattr(translator, 'proxy_weights', None)
        sr = getattr(translator, 'proxy_std_ratios', None)
        if pw is not None:
            pw_t = torch.from_numpy(pw).to(DEVICE)
            sr_t = torch.from_numpy(sr).to(DEVICE)
            proxy_shape = torch.zeros(B, T, C, device=DEVICE)
            proxy_amp   = torch.zeros(B, C, device=DEVICE)
            for k in range(n_known):
                kn_raw  = X_k[:, :, k, :]
                kn_dyn  = kn_raw - kn_raw.mean(dim=1, keepdim=True)
                kn_std  = kn_dyn.std(dim=1).clamp(min=1e-6)
                kn_norm = kn_dyn / kn_std.unsqueeze(1)
                proxy_shape = proxy_shape + pw_t[k].unsqueeze(0).unsqueeze(0) * kn_norm
                proxy_amp   = proxy_amp   + pw_t[k].unsqueeze(0) * kn_std * sr_t[k].unsqueeze(0)
            proxy_dyn = proxy_shape * proxy_amp.unsqueeze(1)
        else:
            proxy_dyn = X_k[:, :, 0, :] - X_k[:, :, 0, :].mean(dim=1, keepdim=True)

        # DC prediction: Ridge > mean known DC
        coef = getattr(translator, 'dc_ridge_coef', None)
        bias = getattr(translator, 'dc_ridge_bias', None)
        if coef is not None:
            known_dc_flat = X_k.mean(dim=1).reshape(X_k.shape[0], -1)
            coef_t = torch.from_numpy(coef).to(DEVICE)
            bias_t = torch.from_numpy(bias).to(DEVICE)
            pred_dc = (known_dc_flat @ coef_t.T) + bias_t
        else:
            pred_dc = X_k.mean(dim=2).mean(dim=1)
        fake = proxy_dyn + pred_dc.unsqueeze(1)

    real_np = X_tgt.cpu().numpy()
    fake_np = fake.cpu().numpy()

    axis_names  = ['x', 'y', 'z']
    axis_colors = ['#2196F3', '#4CAF50', '#E53935']
    t = np.arange(T)

    # ── Plot 1: Shared scale ──────────────────────────────────────────────
    fig, axes = plt.subplots(n, C, figsize=(5*C, 2*n))
    if n == 1: axes = axes[np.newaxis]
    for i in range(n):
        for c in range(C):
            ax = axes[i, c]
            r  = real_np[i, :, c]
            f  = fake_np[i, :, c]
            lo = min(r.min(), f.min())
            hi = max(r.max(), f.max())
            pad = max((hi - lo) * 0.1, 0.05)
            if pad < 0.5: pad = 0.5
            ax.set_ylim(lo - pad, hi + pad)
            ax.plot(t, r, color=axis_colors[c], lw=1.4, alpha=0.9,
                    label='Real' if i == 0 and c == 0 else '_')
            ax.plot(t, f, color='black', lw=1.2, alpha=0.7, linestyle='--',
                    label='Generated' if i == 0 and c == 0 else '_')
            if i == 0: ax.set_title(f"Axis {axis_names[c]}", fontsize=9)
            ax.set_xlim(0, T-1); ax.tick_params(labelsize=6)
            if i == 0 and c == 0: ax.legend(fontsize=7)
    fig.suptitle(f"{activity_name} — shared scale", fontsize=10, fontweight='bold')
    plt.tight_layout()
    shared_dir = os.path.join(os.path.dirname(out_path), 'shared')
    os.makedirs(shared_dir, exist_ok=True)
    fig.savefig(os.path.join(shared_dir, os.path.basename(out_path)), dpi=80, bbox_inches='tight')
    plt.close(fig)

    # ── Plot 2: Dual scale — shape comparison ─────────────────────────────
    fig, axes = plt.subplots(n, C, figsize=(5*C, 2*n))
    if n == 1: axes = axes[np.newaxis]
    for i in range(n):
        for c in range(C):
            ax = axes[i, c]
            r  = real_np[i, :, c]
            f  = fake_np[i, :, c]
            lo_r, hi_r = r.min(), r.max()
            pad_r = max((hi_r - lo_r) * 0.1, 0.05)
            ax.set_ylim(lo_r - pad_r, hi_r + pad_r)
            ax.plot(t, r, color=axis_colors[c], lw=1.4, alpha=0.9)
            ax.tick_params(axis='y', labelcolor=axis_colors[c], labelsize=6)
            ax2 = ax.twinx()
            lo_f, hi_f = f.min(), f.max()
            pad_f = max((hi_f - lo_f) * 0.1, 0.05)
            ax2.set_ylim(lo_f - pad_f, hi_f + pad_f)
            ax2.plot(t, f, color='black', lw=1.2, alpha=0.7, linestyle='--')
            ax2.tick_params(axis='y', labelcolor='black', labelsize=5)
            if i == 0:
                ax.set_title(f"Axis {axis_names[c]}", fontsize=9)
                if c == 0:
                    lines = [plt.Line2D([0],[0], color=axis_colors[c], lw=1.4, label='Real (←)'),
                             plt.Line2D([0],[0], color='black', lw=1.2, ls='--', label='Gen (→)')]
                    ax.legend(handles=lines, fontsize=7)
            ax.set_xlim(0, T-1)
    fig.suptitle(f"{activity_name} — dual scale (shape only)", fontsize=10, fontweight='bold')
    plt.tight_layout()
    dual_dir = os.path.join(os.path.dirname(out_path), 'dual')
    os.makedirs(dual_dir, exist_ok=True)
    fig.savefig(os.path.join(dual_dir, os.path.basename(out_path)), dpi=80, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--translator",      required=True)
    parser.add_argument("--lab-data-dir",    required=True)
    parser.add_argument("--participant",      default="DS_11")
    parser.add_argument("--known-sensors",   nargs="+",
                        default=["LeftWrist", "RightAnkle"])
    parser.add_argument("--target-sensor",   default="RightThigh")
    parser.add_argument("--lab-sensor-order", nargs="+",
                        default=["LeftAnkle","LeftThigh","LeftWaist","LeftWrist",
                                 "RightAnkle","RightThigh","RightWaist","RightWrist"])
    parser.add_argument("--out-dir",         default="output/translator_viz_activities")
    parser.add_argument("--n-samples",       type=int, default=5)
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Load translator
    print(f"Loading translator: {args.translator}")
    translator = load_translator(args.translator)
    translator.generator.eval()

    # Get stream indices in lab data
    known_lab_idx  = [args.lab_sensor_order.index(s) for s in args.known_sensors]
    target_lab_idx = args.lab_sensor_order.index(args.target_sensor)
    all_lab_idx    = known_lab_idx + [target_lab_idx]
    # Reindex within the sliced array
    known_slice_idx  = list(range(len(known_lab_idx)))
    target_slice_idx = len(known_lab_idx)

    print(f"Known sensors: {args.known_sensors} → indices {known_lab_idx}")
    print(f"Target sensor: {args.target_sensor} → index {target_lab_idx}")

    # Load lab data using create_dataset_file_split
    print(f"\nLoading lab data: {args.lab_data_dir} / {args.participant}")
    try:
        from scripts.misc.helpers import create_dataset_file_split
        from scripts.misc.config_loader import cfg
        _, np_val, _, label_dict = create_dataset_file_split(
            args.lab_data_dir, [args.participant], cfg.SEED)
        X_lab = np_val[0]   # (N, T, 8, C)
        y_lab = np_val[1]   # (N,)
        print(f"  Lab data: {X_lab.shape}  labels: {len(label_dict)} activities")
    except Exception as e:
        print(f"  Error loading via create_dataset_file_split: {e}")
        print("  Trying to load activity .npy files directly...")
        # Fallback: load individual activity files
        label_dict = {}
        X_lab      = None
        y_lab      = None

    if X_lab is not None:
        # Slice to relevant streams
        X_sliced = X_lab[:, :, all_lab_idx, :]   # (N, T, n_streams, C)

        activities = sorted(label_dict.keys())
        print(f"\nGenerating visualizations for {len(activities)} activities...")

        for act in activities:
            act_int = label_dict[act]
            mask    = (y_lab == act_int)
            if mask.sum() == 0:
                continue

            X_act   = X_sliced[mask]
            safe    = act.replace("/", "_").replace(" ", "_")
            out_path = os.path.join(args.out_dir, f"{safe}.png")

            visualize_activity(
                translator, X_act,
                known_slice_idx, target_slice_idx,
                activity_name=f"{act}  (n={mask.sum()})",
                out_path=out_path,
                n_samples=args.n_samples,
            )
            print(f"  {act:<50} n={mask.sum():4d}  → {safe}.png")

    else:
        # Load individual .npy files from lab data dir
        lab_dir = Path(args.lab_data_dir) / args.participant
        npy_files = sorted(lab_dir.glob("*.npy"))
        activity_files = [f for f in npy_files
                          if not f.name.startswith("encoder_")
                          and f.name != "activity_mapping.tsv"
                          and f.name != "limb_order.txt"]
        print(f"  Found {len(activity_files)} activity files")
        for f in activity_files:
            act = f.stem
            X   = np.load(str(f)).astype(np.float32)
            if X.ndim != 4 or X.shape[2] < max(all_lab_idx) + 1:
                continue
            X_sliced = X[:, :, all_lab_idx, :]
            safe     = act.replace("/", "_").replace(" ", "_")
            out_path = os.path.join(args.out_dir, f"{safe}.png")
            visualize_activity(
                translator, X_sliced,
                known_slice_idx, target_slice_idx,
                activity_name=f"{act}  (n={len(X)})",
                out_path=out_path,
                n_samples=args.n_samples,
            )
            print(f"  {act:<50} n={len(X):4d}  → {safe}.png")

    print(f"\nDone. Visualizations saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()