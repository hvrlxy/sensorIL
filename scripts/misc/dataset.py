"""
dataset.py

Datasets:
  - UnlabeledFLDataset         : FL data for BYOL pretraining (masked vs full)
  - BaselineUnlabeledFLDataset : FL data for baseline pretraining (aug vs aug, n sensors only)
  - LabeledLabDataset          : Lab data for few-shot fine-tuning (n sensors, new sensor zeroed)
  - TestLabDataset             : Lab data for evaluation (n+1 sensors, full)
  - BaselineTestLabDataset     : Lab data for baseline evaluation (n sensors only, no zeroing)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Sensor index helpers
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
    """Map sensor names to indices in a given sensor order list."""
    return [sensor_order.index(s) for s in sensor_names]


# ─────────────────────────────────────────────────────────────────────────────
# Time-series augmentations
# ─────────────────────────────────────────────────────────────────────────────

def augment(x, jitter_std=0.05, scale_range=(0.8, 1.2), crop_ratio=0.8):
    """
    Apply random augmentations to a sensor window.

    Args:
        x           : (n_sensors, T, C) tensor
        jitter_std  : std of Gaussian noise added
        scale_range : (min, max) random amplitude scaling
        crop_ratio  : fraction of window to crop then resize

    Returns augmented (n_sensors, T, C) tensor.
    """
    n_sensors, T, C = x.shape

    # Jitter: add Gaussian noise
    x = x + torch.randn_like(x) * jitter_std

    # Scaling: random per-sensor amplitude scale
    scale = torch.empty(n_sensors, 1, 1).uniform_(*scale_range)
    x = x * scale

    # Temporal crop: crop random subwindow and interpolate back to T
    crop_len = int(T * crop_ratio)
    start    = torch.randint(0, T - crop_len + 1, (1,)).item()
    x_crop   = x[:, start:start + crop_len, :]          # (S, crop_len, C)

    # Interpolate back to T using linear interpolation
    # reshape to (S*C, 1, crop_len) for F.interpolate
    x_crop = x_crop.permute(0, 2, 1).reshape(n_sensors * C, 1, crop_len)
    x_crop = torch.nn.functional.interpolate(
        x_crop, size=T, mode='linear', align_corners=False
    )
    x = x_crop.reshape(n_sensors, C, T).permute(0, 2, 1)  # (S, T, C)

    return x


# ─────────────────────────────────────────────────────────────────────────────
# Unlabeled FL Dataset (BYOL pretraining: masked vs full)
# ─────────────────────────────────────────────────────────────────────────────

class UnlabeledFLDataset(Dataset):
    """
    Returns two views of each window for BYOL:
      - view_masked : known sensors only (new sensor zeroed)
      - view_full   : all sensors (known + new)

    Shape of each view: (n_sensors_full, 100, 3)
    """

    def __init__(self, data_dir, known_sensors, new_sensor):
        self.known_idx = get_sensor_indices(known_sensors, FL_SENSOR_ORDER)
        self.new_idx   = get_sensor_indices(new_sensor,    FL_SENSOR_ORDER)
        self.all_idx   = self.known_idx + self.new_idx

        arrays = []
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".npy"):
                continue
            arr = np.load(os.path.join(data_dir, fname))
            if arr.ndim != 4:
                print(f"  [skip] {fname} — unexpected shape {arr.shape}")
                continue
            arrays.append(arr)

        data = np.concatenate(arrays, axis=0)  # (N, 100, 5, 3)

        # (N, 100, n_sensors, 3) -> (N, n_sensors, 100, 3)
        self.data_full   = torch.tensor(
            data[:, :, self.all_idx, :], dtype=torch.float32
        ).permute(0, 2, 1, 3)

        self.data_masked = torch.tensor(
            data[:, :, self.known_idx, :], dtype=torch.float32
        ).permute(0, 2, 1, 3)

        # Pad masked view with zeros for new sensor slots
        n_new = len(self.new_idx)
        pad   = torch.zeros(len(self.data_masked), n_new,
                            self.data_masked.shape[2], self.data_masked.shape[3])
        self.data_masked = torch.cat([self.data_masked, pad], dim=1)

        print(f"[UnlabeledFLDataset] {len(self.data_full)} windows "
              f"| known={known_sensors} | new={new_sensor}")

    def __len__(self):
        return len(self.data_full)

    def __getitem__(self, idx):
        return self.data_masked[idx], self.data_full[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Unlabeled FL Dataset (BYOL baseline: aug vs aug, n sensors only)
# ─────────────────────────────────────────────────────────────────────────────

class BaselineUnlabeledFLDataset(Dataset):
    """
    Returns two augmented views of each window for baseline BYOL.
    Uses only known sensors — new sensor is never loaded.

    Shape of each view: (n_known, 100, 3)

    Augmentations: jitter + scaling + temporal crop
    """

    def __init__(self, data_dir, known_sensors,
                 jitter_std=0.05, scale_range=(0.8, 1.2), crop_ratio=0.8):
        self.known_idx   = get_sensor_indices(known_sensors, FL_SENSOR_ORDER)
        self.jitter_std  = jitter_std
        self.scale_range = scale_range
        self.crop_ratio  = crop_ratio

        arrays = []
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".npy"):
                continue
            arr = np.load(os.path.join(data_dir, fname))
            if arr.ndim != 4:
                print(f"  [skip] {fname} — unexpected shape {arr.shape}")
                continue
            arrays.append(arr)

        data = np.concatenate(arrays, axis=0)  # (N, 100, 5, 3)

        # Extract known sensors only, move sensor dim first
        self.data = torch.tensor(
            data[:, :, self.known_idx, :], dtype=torch.float32
        ).permute(0, 2, 1, 3)   # (N, n_known, 100, 3)

        print(f"[BaselineUnlabeledFLDataset] {len(self.data)} windows "
              f"| known={known_sensors} | new sensor excluded")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x        = self.data[idx]   # (n_known, 100, 3)
        view_a   = augment(x, self.jitter_std, self.scale_range, self.crop_ratio)
        view_b   = augment(x, self.jitter_std, self.scale_range, self.crop_ratio)
        return view_a, view_b


# ─────────────────────────────────────────────────────────────────────────────
# Labeled Lab Dataset (fine-tuning: n sensors, new sensor zeroed)
# ─────────────────────────────────────────────────────────────────────────────

class LabeledLabDataset(Dataset):
    """
    Returns (window, label) pairs using only known sensors.
    New sensor slot is zeroed for consistent encoder input shape.

    Shape of window: (n_full, 100, 3)  where n_full = n_known + n_new
    """

    def __init__(self, data_dir, known_sensors, new_sensor,
                 max_samples_per_class=50, split="train", val_split=0.2,
                 seed=42):
        self.known_idx = get_sensor_indices(known_sensors, LAB_SENSOR_ORDER)
        self.new_idx   = get_sensor_indices(new_sensor,    LAB_SENSOR_ORDER)
        self.all_idx   = self.known_idx + self.new_idx

        rng = np.random.default_rng(seed)

        windows, labels = [], []
        self.class_names = []
        class_id = 0

        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".npy"):
                continue

            arr = np.load(os.path.join(data_dir, fname))
            if arr.ndim != 4:
                print(f"  [skip] {fname} — unexpected shape {arr.shape}")
                continue

            activity = fname.replace(".npy", "")
            self.class_names.append(activity)

            # Shuffle and cap at max_samples_per_class
            idx = rng.permutation(len(arr))[:max_samples_per_class]
            arr = arr[idx]

            # Train/val split
            n_val   = max(1, int(len(arr) * val_split))
            n_train = len(arr) - n_val

            if split == "train":
                arr = arr[:n_train]
            else:
                arr = arr[n_train:]

            windows.append(arr)
            labels.extend([class_id] * len(arr))
            class_id += 1

        data = np.concatenate(windows, axis=0)  # (N, 100, 8, 3)

        # Extract known sensors, move sensor dim first
        self.data = torch.tensor(
            data[:, :, self.known_idx, :], dtype=torch.float32
        ).permute(0, 2, 1, 3)   # (N, n_known, 100, 3)
        self.labels = torch.tensor(labels, dtype=torch.long)

        # Pad with zeros for new sensor slots
        n_new = len(self.new_idx)
        pad   = torch.zeros(len(self.data), n_new,
                            self.data.shape[2], self.data.shape[3])
        self.data = torch.cat([self.data, pad], dim=1)  # (N, n_full, 100, 3)

        self.n_classes = len(self.class_names)

        print(f"[LabeledLabDataset:{split}] {len(self.data)} windows "
              f"| {self.n_classes} classes "
              f"| known={known_sensors} | new sensor zeroed")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Labeled Lab Dataset (fine-tuning: n sensors only, no zeroing)
# ─────────────────────────────────────────────────────────────────────────────

class BaselineLabeledLabDataset(Dataset):
    """
    Same as LabeledLabDataset but uses ONLY known sensors — no zeroed channel.
    Used for baseline fine-tuning and evaluation.

    Shape of window: (n_known, 100, 3)
    """

    def __init__(self, data_dir, known_sensors,
                 max_samples_per_class=50, split="train", val_split=0.2,
                 seed=42):
        self.known_idx = get_sensor_indices(known_sensors, LAB_SENSOR_ORDER)

        rng = np.random.default_rng(seed)

        windows, labels = [], []
        self.class_names = []
        class_id = 0

        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".npy"):
                continue

            arr = np.load(os.path.join(data_dir, fname))
            if arr.ndim != 4:
                print(f"  [skip] {fname} — unexpected shape {arr.shape}")
                continue

            activity = fname.replace(".npy", "")
            self.class_names.append(activity)

            idx = rng.permutation(len(arr))[:max_samples_per_class]
            arr = arr[idx]

            n_val   = max(1, int(len(arr) * val_split))
            n_train = len(arr) - n_val

            if split == "train":
                arr = arr[:n_train]
            else:
                arr = arr[n_train:]

            windows.append(arr)
            labels.extend([class_id] * len(arr))
            class_id += 1

        data = np.concatenate(windows, axis=0)

        self.data = torch.tensor(
            data[:, :, self.known_idx, :], dtype=torch.float32
        ).permute(0, 2, 1, 3)   # (N, n_known, 100, 3)
        self.labels     = torch.tensor(labels, dtype=torch.long)
        self.n_classes  = len(self.class_names)

        print(f"[BaselineLabeledLabDataset:{split}] {len(self.data)} windows "
              f"| {self.n_classes} classes | known={known_sensors}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Test Dataset (BYOL evaluation: n+1 sensors)
# ─────────────────────────────────────────────────────────────────────────────

class TestLabDataset(Dataset):
    """
    Uses ALL sensors (known + new) for BYOL test time evaluation.
    Shape of window: (n_full, 100, 3)
    """

    def __init__(self, data_dir, known_sensors, new_sensor, seed=42):
        self.known_idx = get_sensor_indices(known_sensors, LAB_SENSOR_ORDER)
        self.new_idx   = get_sensor_indices(new_sensor,    LAB_SENSOR_ORDER)
        self.all_idx   = self.known_idx + self.new_idx

        windows, labels = [], []
        self.class_names = []
        class_id = 0

        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".npy"):
                continue

            arr = np.load(os.path.join(data_dir, fname))
            if arr.ndim != 4:
                print(f"  [skip] {fname} - unexpected shape {arr.shape}")
                continue

            activity = fname.replace(".npy", "")
            self.class_names.append(activity)

            windows.append(arr)
            labels.extend([class_id] * len(arr))
            class_id += 1

        data = np.concatenate(windows, axis=0)

        self.data   = torch.tensor(
            data[:, :, self.all_idx, :], dtype=torch.float32
        ).permute(0, 2, 1, 3)
        self.labels     = torch.tensor(labels, dtype=torch.long)
        self.n_classes  = len(self.class_names)

        print(f"[TestLabDataset] {len(self.data)} windows "
              f"| {self.n_classes} classes "
              f"| sensors={known_sensors + new_sensor}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Test Dataset (baseline evaluation: n sensors only)
# ─────────────────────────────────────────────────────────────────────────────

class BaselineTestLabDataset(Dataset):
    """
    Uses ONLY known sensors for baseline test time evaluation.
    Shape of window: (n_known, 100, 3)
    """

    def __init__(self, data_dir, known_sensors, seed=42):
        self.known_idx = get_sensor_indices(known_sensors, LAB_SENSOR_ORDER)

        windows, labels = [], []
        self.class_names = []
        class_id = 0

        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".npy"):
                continue

            arr = np.load(os.path.join(data_dir, fname))
            if arr.ndim != 4:
                print(f"  [skip] {fname} - unexpected shape {arr.shape}")
                continue

            activity = fname.replace(".npy", "")
            self.class_names.append(activity)

            windows.append(arr)
            labels.extend([class_id] * len(arr))
            class_id += 1

        data = np.concatenate(windows, axis=0)

        self.data   = torch.tensor(
            data[:, :, self.known_idx, :], dtype=torch.float32
        ).permute(0, 2, 1, 3)
        self.labels     = torch.tensor(labels, dtype=torch.long)
        self.n_classes  = len(self.class_names)

        print(f"[BaselineTestLabDataset] {len(self.data)} windows "
              f"| {self.n_classes} classes | sensors={known_sensors}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factories
# ─────────────────────────────────────────────────────────────────────────────

def get_pretrain_loader(config, experiment, mode="byol"):
    """
    mode='byol'     : UnlabeledFLDataset     (masked vs full, n+1 sensors)
    mode='baseline' : BaselineUnlabeledFLDataset (aug vs aug, n sensors only)
    """
    exp   = config["sensors"]["experiments"][experiment]
    known = exp["known_sensors"]
    new   = exp["new_sensor"]
    aug   = config.get("augmentation", {})

    if mode == "byol":
        dataset = UnlabeledFLDataset(config["data"]["unlabeled_dir"], known, new)
    else:
        dataset = BaselineUnlabeledFLDataset(
            data_dir    = config["data"]["unlabeled_dir"],
            known_sensors = known,
            jitter_std  = aug.get("jitter_std",  0.05),
            scale_range = aug.get("scale_range", [0.8, 1.2]),
            crop_ratio  = aug.get("crop_ratio",  0.8)
        )

    return DataLoader(
        dataset,
        batch_size  = config["pretrain"]["batch_size"],
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True,
        drop_last   = True
    )


def get_supcon_loader(config, experiment, mode="byol"):
    """
    Returns labeled lab loader for SupCon loss during pretraining.
    Both byol and baseline use n sensors only (new sensor zeroed for byol,
    absent entirely for baseline).
    """
    exp   = config["sensors"]["experiments"][experiment]
    known = exp["known_sensors"]
    new   = exp["new_sensor"]

    if mode == "byol":
        dataset = LabeledLabDataset(
            data_dir              = config["data"]["labeled_dir"],
            known_sensors         = known,
            new_sensor            = new,
            max_samples_per_class = config["finetune"]["few_shot_samples_per_class"],
            split                 = "train",
            val_split             = config["finetune"]["val_split"]
        )
    else:
        dataset = BaselineLabeledLabDataset(
            data_dir              = config["data"]["labeled_dir"],
            known_sensors         = known,
            max_samples_per_class = config["finetune"]["few_shot_samples_per_class"],
            split                 = "train",
            val_split             = config["finetune"]["val_split"]
        )

    return DataLoader(
        dataset,
        batch_size  = config["pretrain"]["supcon_batch_size"],
        shuffle     = True,
        num_workers = 2,
        pin_memory  = True,
        drop_last   = True
    )


def get_finetune_loaders(config, experiment, mode="byol"):
    """
    mode='byol'     : n sensors + zeroed new sensor
    mode='baseline' : n sensors only
    """
    exp   = config["sensors"]["experiments"][experiment]
    known = exp["known_sensors"]
    new   = exp["new_sensor"]

    if mode == "byol":
        kwargs = dict(
            data_dir              = config["data"]["labeled_dir"],
            known_sensors         = known,
            new_sensor            = new,
            max_samples_per_class = config["finetune"]["few_shot_samples_per_class"],
            val_split             = config["finetune"]["val_split"]
        )
        train_ds = LabeledLabDataset(**kwargs, split="train")
        val_ds   = LabeledLabDataset(**kwargs, split="val")
    else:
        kwargs = dict(
            data_dir              = config["data"]["labeled_dir"],
            known_sensors         = known,
            max_samples_per_class = config["finetune"]["few_shot_samples_per_class"],
            val_split             = config["finetune"]["val_split"]
        )
        train_ds = BaselineLabeledLabDataset(**kwargs, split="train")
        val_ds   = BaselineLabeledLabDataset(**kwargs, split="val")

    train_loader = DataLoader(train_ds, batch_size=config["finetune"]["batch_size"],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["finetune"]["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, train_ds.n_classes, train_ds.class_names


def get_test_loaders(config, experiment, mode="byol"):
    """
    mode='byol'     : returns (byol_loader with n+1 sensors, class info)
    mode='baseline' : returns (baseline_loader with n sensors only, class info)
    """
    exp   = config["sensors"]["experiments"][experiment]
    known = exp["known_sensors"]
    new   = exp["new_sensor"]

    if mode == "byol":
        ds = TestLabDataset(
            data_dir      = config["data"]["labeled_dir"],
            known_sensors = known,
            new_sensor    = new
        )
    else:
        ds = BaselineTestLabDataset(
            data_dir      = config["data"]["labeled_dir"],
            known_sensors = known
        )

    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
    return loader, ds.n_classes, ds.class_names
