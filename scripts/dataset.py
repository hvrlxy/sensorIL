"""
dataset.py

Datasets for the active sensor increment pipeline.

  SensorDataset         : labeled lab data, flexible sensor selection
  UnlabeledFLDataset    : unlabeled FL data, flexible sensor selection
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Sensor index maps
# ─────────────────────────────────────────────────────────────────────────────

LAB_SENSOR_ORDER = [
    "LeftAnkle",   # 0
    "LeftThigh",   # 1
    "LeftWaist",   # 2
    "LeftWrist",   # 3
    "RightAnkle",  # 4
    "RightThigh",  # 5
    "RightWaist",  # 6
    "RightWrist",  # 7
]

FL_SENSOR_ORDER = [
    "LeftWrist",   # 0
    "RightAnkle",  # 1
    "RightThigh",  # 2
    "RightWaist",  # 3
    "RightWrist",  # 4
]


def get_sensor_indices(sensor_names, sensor_order):
    return [sensor_order.index(s) for s in sensor_names]


# ─────────────────────────────────────────────────────────────────────────────
# Labeled Lab Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SensorDataset(Dataset):
    """
    Labeled lab dataset with flexible sensor selection.

    Returns (window, label) where:
      window : (n_sensors, 100, 3)
      label  : int class index

    Args:
        data_dir              : path to lab data
        sensors               : list of sensor names to include
        max_samples_per_class : few-shot cap (-1 = all)
        split                 : 'train', 'val', or 'all'
        val_split             : fraction for validation
        seed                  : random seed
        include_classes       : list of class names to include (None = all)
    """

    def __init__(self, data_dir, sensors,
                 max_samples_per_class=50,
                 split="train", val_split=0.2,
                 seed=42, include_classes=None):

        self.sensor_idx = get_sensor_indices(sensors, LAB_SENSOR_ORDER)
        rng = np.random.default_rng(seed)

        windows, labels = [], []
        self.class_names = []
        class_id = 0

        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".npy"):
                continue

            arr = np.load(os.path.join(data_dir, fname))
            if arr.ndim != 4:
                continue

            activity = fname.replace(".npy", "")

            if include_classes is not None and activity not in include_classes:
                continue

            self.class_names.append(activity)

            # Cap samples
            if max_samples_per_class > 0:
                idx = rng.permutation(len(arr))[:max_samples_per_class]
                arr = arr[idx]

            # Split
            if split != "all":
                n_val   = max(1, int(len(arr) * val_split))
                n_train = len(arr) - n_val
                arr = arr[:n_train] if split == "train" else arr[n_train:]

            windows.append(arr)
            labels.extend([class_id] * len(arr))
            class_id += 1

        data = np.concatenate(windows, axis=0)  # (N, 100, 8, 3)

        # Extract sensors, move sensor dim first: (N, n_sensors, 100, 3)
        self.data   = torch.tensor(
            data[:, :, self.sensor_idx, :], dtype=torch.float32
        ).permute(0, 2, 1, 3)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.n_classes = len(self.class_names)

        print(f"[SensorDataset:{split}] {len(self.data)} windows | "
              f"{self.n_classes} classes | sensors={sensors}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Unlabeled FL Dataset
# ─────────────────────────────────────────────────────────────────────────────

class UnlabeledFLDataset(Dataset):
    """
    Unlabeled FL dataset with flexible sensor selection.

    Returns window : (n_sensors, 100, 3)
    """

    def __init__(self, data_dir, sensors):
        self.sensor_idx = get_sensor_indices(sensors, FL_SENSOR_ORDER)

        arrays = []
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".npy"):
                continue
            arr = np.load(os.path.join(data_dir, fname))
            if arr.ndim != 4:
                print(f"  [skip] {fname} — shape {arr.shape}")
                continue
            arrays.append(arr)

        data = np.concatenate(arrays, axis=0)

        self.data = torch.tensor(
            data[:, :, self.sensor_idx, :], dtype=torch.float32
        ).permute(0, 2, 1, 3)

        print(f"[UnlabeledFLDataset] {len(self.data)} windows | sensors={sensors}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factories
# ─────────────────────────────────────────────────────────────────────────────

def get_lab_loaders(config, sensors, split_both=True):
    """Returns train and val loaders for lab data."""
    kwargs = dict(
        data_dir              = config["data"]["labeled_dir"],
        sensors               = sensors,
        max_samples_per_class = config["finetune"]["few_shot_samples_per_class"],
        val_split             = config["finetune"]["val_split"]
    )
    train_ds = SensorDataset(**kwargs, split="train")
    val_ds   = SensorDataset(**kwargs, split="val")

    train_loader = DataLoader(train_ds, batch_size=config["finetune"]["batch_size"],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["finetune"]["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, train_ds.n_classes, train_ds.class_names


def get_fl_loader(config, sensors, batch_size=512):
    """Returns loader for unlabeled FL data."""
    ds = UnlabeledFLDataset(config["data"]["unlabeled_dir"], sensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)
