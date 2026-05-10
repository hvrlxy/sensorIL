"""
helpers.py
==========
Dataset loading and evaluation plotting utilities.

No imports from config_loader — all parameters are passed as arguments
so this module works with any dataset.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def create_dataset_file_split(
    data_root,
    participant_lst=None,
    seed=42,
    expected_sample_shape=None,
    delete_nan_file=True,
):
    """
    Load windowed .npy data files and create train/val/test splits (1/3 each)
    per participant per activity class.

    Directory structure expected:
        data_root/
            <participant_id>/
                <activity_name>.npy   # shape (N, T, DIM, C) or single (T, DIM, C)

    Parameters
    ----------
    data_root : str | Path
    participant_lst : list[str] | None
        Participant subfolder names to include. None = all subfolders.
    seed : int
    expected_sample_shape : tuple | None
        Expected per-sample shape (T, DIM, C), e.g. (100, 8, 3).
        If None, shape is inferred from the first valid file found.
    delete_nan_file : bool
        Automatically delete nan.npy files if found.

    Returns
    -------
    np_train  : [X_train, y_train]
    np_val    : [X_val,   y_val  ]
    np_test   : [X_test,  y_test ]
    label_to_idx : dict  {activity_name: int}
    """
    rng       = np.random.default_rng(seed)
    data_root = Path(data_root)

    if participant_lst is None:
        participant_lst = sorted([p.name for p in data_root.iterdir() if p.is_dir()])

    # ── Pass 1: collect all label names ──────────────────────────────────────
    label_set = set()
    for ds in participant_lst:
        ds_path = data_root / ds
        if not ds_path.exists():
            continue
        for f in ds_path.iterdir():
            if f.is_file() and f.suffix == ".npy" and f.name.lower() != "nan.npy":
                label_set.add(f.stem)

    label_list   = sorted(label_set)
    label_to_idx = {label: i for i, label in enumerate(label_list)}

    print("Label mapping:")
    for k, v in label_to_idx.items():
        print(f"  {v}: {k}")

    # ── Pass 2: load and split ────────────────────────────────────────────────
    X_train, y_train = [], []
    X_val,   y_val   = [], []
    X_test,  y_test  = [], []

    # Will be set from first valid file if not provided
    inferred_shape = expected_sample_shape

    for ds in participant_lst:
        ds_path = data_root / ds
        if not ds_path.exists():
            print(f"[WARN] Missing participant directory: {ds}")
            continue

        nan_path = ds_path / "nan.npy"
        if delete_nan_file and nan_path.exists():
            try:
                nan_path.unlink()
                print(f"[INFO] Deleted {nan_path}")
            except Exception as e:
                print(f"[WARN] Could not delete {nan_path}: {e}")

        print(f"\nProcessing {ds}")

        for f in ds_path.iterdir():
            if not f.is_file() or f.suffix != ".npy":
                continue
            if f.name.lower() == "nan.npy":
                continue

            label_name = f.stem
            if label_name not in label_to_idx:
                continue

            try:
                X = np.load(f, allow_pickle=False)
            except Exception as e:
                print(f"[SKIP] {ds}/{f.name}: load failed ({e})")
                continue

            # ── Infer expected shape from first valid file ────────────────────
            if inferred_shape is None:
                if X.ndim == 4:
                    inferred_shape = tuple(X.shape[1:])
                elif X.ndim == 3:
                    inferred_shape = tuple(X.shape)
                print(f"[INFO] Inferred sample shape: {inferred_shape}")

            # ── Validate / fix shape ──────────────────────────────────────────
            if X.ndim == 3:
                if tuple(X.shape) == tuple(inferred_shape):
                    X = X[None]   # single sample saved as (T, DIM, C)
                else:
                    print(f"[SKIP] {ds}/{f.name}: expected 3D {inferred_shape}, got {X.shape}")
                    continue

            if X.ndim != 4 or tuple(X.shape[1:]) != tuple(inferred_shape):
                print(f"[SKIP] {ds}/{f.name}: expected (N, {inferred_shape}), got {X.shape}")
                continue

            n = X.shape[0]
            if n < 3:
                print(f"[SKIP] {ds}/{label_name}: too few samples ({n})")
                continue

            indices = rng.permutation(n)
            n_train = n // 3
            n_val   = n // 3
            n_test  = n - n_train - n_val

            if n_train == 0 or n_val == 0 or n_test == 0:
                print(f"[SKIP] {ds}/{label_name}: split too small "
                      f"(n={n} → {n_train}/{n_val}/{n_test})")
                continue

            X_train.append(X[indices[:n_train]])
            X_val.append(  X[indices[n_train:n_train + n_val]])
            X_test.append( X[indices[n_train + n_val:]])

            y_train.append(np.full(n_train, label_to_idx[label_name], dtype=np.int64))
            y_val.append(  np.full(n_val,   label_to_idx[label_name], dtype=np.int64))
            y_test.append( np.full(n_test,  label_to_idx[label_name], dtype=np.int64))

    if not X_train:
        raise ValueError(
            "No training data collected. Check data_root, participant_lst, and file shapes."
        )

    _shape = inferred_shape or (0,)

    X_train = np.concatenate(X_train, axis=0)
    X_val   = np.concatenate(X_val,   axis=0) if X_val  else np.empty((0,) + _shape, dtype=X_train.dtype)
    X_test  = np.concatenate(X_test,  axis=0) if X_test else np.empty((0,) + _shape, dtype=X_train.dtype)

    y_train = np.concatenate(y_train, axis=0)
    y_val   = np.concatenate(y_val,   axis=0) if y_val  else np.empty((0,), dtype=np.int64)
    y_test  = np.concatenate(y_test,  axis=0) if y_test else np.empty((0,), dtype=np.int64)

    print(f"\nFinal dataset sizes:")
    print(f"  Train : {X_train.shape}  {y_train.shape}")
    print(f"  Val   : {X_val.shape}    {y_val.shape}")
    print(f"  Test  : {X_test.shape}   {y_test.shape}")

    return [X_train, y_train], [X_val, y_val], [X_test, y_test], label_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_from_model(model, X_test, y_test, label_dict,
                              normalize="true", figsize=(14, 12), cmap="Blues"):
    if y_test.ndim == 2:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test

    y_pred      = np.argmax(model.predict(X_test, verbose=0), axis=1)
    num_classes = len(label_dict)
    cm          = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    if normalize == "true":
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)
    elif normalize == "pred":
        cm = cm / np.maximum(cm.sum(axis=0, keepdims=True), 1e-12)

    idx_to_label = {v: k for k, v in label_dict.items()}
    class_names  = [idx_to_label[i] for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=figsize)
    im      = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=16)
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)

    max_val = cm.max()
    for i in range(num_classes):
        for j in range(num_classes):
            v = cm[i, j]
            if v > 0:
                text  = f"{v:.2f}" if normalize else f"{int(v)}"
                color = "white" if v > max_val * 0.6 else "black"
                ax.text(j, i, text, ha="center", va="center",
                        color=color, fontsize=8, fontweight="bold")
    plt.tight_layout()
    plt.show()
