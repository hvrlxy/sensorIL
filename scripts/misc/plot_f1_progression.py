"""
plot_f1_progression.py
======================
Plot F1 progression for all activity heads across incremental steps.

Reads the E3 history pkl and produces:
  1. F1 progression for seed heads (old activities) — main result
  2. Birth F1 for new activities (when first added)
  3. Combined view

Usage
-----
  python scripts/plot_f1_progression.py
  python scripts/plot_f1_progression.py --history output/<ts>_e3_history.pkl
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
import glob
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scripts.misc.config_loader import cfg


def load_history(path=None):
    if path is None:
        matches = sorted(glob.glob(os.path.join(cfg.WORKING_DIR, "*_e3_history.pkl")))
        if not matches:
            raise FileNotFoundError(f"No e3_history.pkl found in {cfg.WORKING_DIR}")
        path = matches[-1]
        print(f"Loading: {path}")
    with open(path, "rb") as f:
        return pickle.load(f), path


def plot_f1_progression(history, save_dir):
    steps      = history["steps"]
    step_names = [s["activity"].replace("_Lab", "").replace("_", " ")
                  for s in steps]
    n_steps    = len(steps)
    seed_set   = set(cfg.SEED_ACTIVITIES)

    # ── Collect F1 series for every activity ──────────────────────────────────
    # seed heads: tracked at every step
    seed_f1 = {act: [] for act in cfg.SEED_ACTIVITIES}
    for s in steps:
        for act in cfg.SEED_ACTIVITIES:
            f1 = s.get("seed_f1", {}).get(act, {}).get("test_f1", np.nan)
            seed_f1[act].append(f1)

    # new activity heads: F1 at birth + any subsequent snapshots from test_ood
    new_act_birth = {}   # activity -> (step_idx, f1)
    new_act_series = {}  # activity -> list of (step_idx, f1)
    for i, s in enumerate(steps):
        act = s["activity"]
        if s.get("test_ood") and act in s["test_ood"]:
            f1 = s["test_ood"][act].get("f1", np.nan)
            new_act_birth[act]  = (i, f1)
        # all activities tracked in test_ood at each step
        if s.get("test_ood"):
            for a, m in s["test_ood"].items():
                if a not in seed_set:
                    new_act_series.setdefault(a, []).append((i, m.get("f1", np.nan)))

    # ── Figure 1: Seed head F1 progression ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = cm.tab10(np.linspace(0, 0.8, len(cfg.SEED_ACTIVITIES)))

    e1_f1 = history.get("seed_metrics_e1", {})
    x     = np.arange(n_steps)

    for i, act in enumerate(cfg.SEED_ACTIVITIES):
        series    = seed_f1[act]
        label     = act.replace("_", " ")
        e1        = e1_f1.get(act, {}).get("f1", np.nan)
        ax.plot(x, series, color=colors[i], linewidth=2,
                marker="o", markersize=4, label=label)
        # E1 baseline as dashed horizontal line
        if not np.isnan(e1):
            ax.axhline(e1, color=colors[i], linewidth=1,
                       linestyle="--", alpha=0.5)

    # Mark sensor increment step (step 0 = first activity after sensor add)
    ax.axvline(0, color="red", linewidth=1.5, linestyle=":",
               label="Sensor added + bootstrap")

    ax.set_xticks(x)
    ax.set_xticklabels(step_names, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Activity added")
    ax.set_ylabel("Test F1")
    ax.set_title("Seed Head F1 Progression\n"
                 "(dashed = E1 baseline on 1-stream; solid = after sensor increment)",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    p1 = os.path.join(save_dir, "f1_seed_progression.pdf")
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p1}")

    # ── Figure 2: New activity birth F1 ───────────────────────────────────────
    if new_act_birth:
        acts_sorted = sorted(new_act_birth.items(), key=lambda x: x[1][0])
        birth_names = [a.replace("_Lab","").replace("_"," ") for a, _ in acts_sorted]
        birth_f1s   = [v[1] for _, v in acts_sorted]

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(range(len(birth_f1s)), birth_f1s,
                      color=cm.viridis(np.linspace(0.2, 0.8, len(birth_f1s))),
                      alpha=0.85)
        for bar, val in zip(bars, birth_f1s):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(birth_names)))
        ax.set_xticklabels(birth_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Test F1 at birth")
        ax.set_title("New Activity Head F1 at Birth\n(2-stream, real data)", fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        p2 = os.path.join(save_dir, "f1_new_activity_birth.pdf")
        plt.savefig(p2, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {p2}")

    # ── Figure 3: Combined heatmap — all heads at all steps ───────────────────
    all_acts   = cfg.SEED_ACTIVITIES + [s["activity"] for s in steps]
    matrix     = np.full((len(all_acts), n_steps), np.nan)

    for i, act in enumerate(all_acts):
        for j, s in enumerate(steps):
            if act in seed_set:
                f1 = s.get("seed_f1", {}).get(act, {}).get("test_f1", np.nan)
            else:
                f1 = s.get("test_ood", {}).get(act, {}).get("f1", np.nan)
            matrix[i, j] = f1

    fig, ax = plt.subplots(figsize=(max(10, n_steps * 1.2), max(5, len(all_acts) * 0.5)))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1,
                   cmap="RdYlGn", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Test F1")

    ax.set_xticks(range(n_steps))
    ax.set_xticklabels(step_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(all_acts)))
    act_labels = [a.replace("_Lab","").replace("_"," ") +
                  (" *" if a in seed_set else "")
                  for a in all_acts]
    ax.set_yticklabels(act_labels, fontsize=8)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="black")

    # Gold border for seed heads
    for i, act in enumerate(all_acts):
        if act in seed_set:
            ax.add_patch(plt.Rectangle((-0.5, i-0.5), n_steps, 1,
                                        linewidth=2, edgecolor="gold",
                                        facecolor="none", zorder=3))

    ax.axvline(-0.5, color="red", linewidth=2, label="← sensor added →")
    ax.set_title("All Head F1 Heatmap (Test)\n(* = seed head, gold = retrained with synthetic data)",
                 fontsize=11)
    plt.tight_layout()

    p3 = os.path.join(save_dir, "f1_heatmap_all.pdf")
    plt.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p3}")

    # ── Figure 4: Seed head delta vs E1 baseline ───────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, act in enumerate(cfg.SEED_ACTIVITIES):
        e1   = e1_f1.get(act, {}).get("f1", np.nan)
        if np.isnan(e1):
            continue
        deltas = [f - e1 if not np.isnan(f) else np.nan
                  for f in seed_f1[act]]
        label  = act.replace("_", " ")
        ax.plot(x, deltas, color=colors[i], linewidth=2,
                marker="o", markersize=4, label=label)

    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6,
               label="E1 baseline")
    ax.axvline(0, color="red", linewidth=1.5, linestyle=":",
               label="Sensor added")
    ax.set_xticks(x)
    ax.set_xticklabels(step_names, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Activity added")
    ax.set_ylabel("ΔF1 vs E1 baseline")
    ax.set_title("Seed Head F1 Delta vs E1 Baseline\n"
                 "(positive = improvement over 1-stream E1)",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    p4 = os.path.join(save_dir, "f1_seed_delta.pdf")
    plt.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p4}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, default=None,
                        help="Path to _e3_history.pkl. Auto-detects latest if not set.")
    args = parser.parse_args()

    if cfg is None:
        raise RuntimeError("configs/paths.json not found.")

    history, hist_path = load_history(args.history)
    save_dir = cfg.FIGS_DIR
    print(f"Plotting {len(history['steps'])} steps → {save_dir}")
    plot_f1_progression(history, save_dir)
    print("Done.")
