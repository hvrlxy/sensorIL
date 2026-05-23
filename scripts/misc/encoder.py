"""
encoder.py
==========
Model-agnostic encoder loading and feature extraction.

Decouples the rest of the pipeline from any specific model architecture.
To use a different encoder (e.g. a transformer, a different CNN, a model
trained with a different framework), implement the StreamEncoder interface
and register it — nothing else in the codebase needs to change.

Supported out of the box
------------------------
  "simclr_pt"   — SimCLRModel from simclr_models_pt.py (default, PyTorch .pt)

Adding your own
---------------
  1. Subclass StreamEncoder and implement encode().
  2. Register it:  EncoderRegistry.register("my_encoder", MyEncoder)
  3. Set "encoder_type": "my_encoder" in paths.json.

Multi-encoder support
---------------------
If different sensor streams should use different encoders (e.g. a wrist-specific
model vs a body-worn model), configure "encoders" and "stream_to_encoder" in
your dataset JSON. load_encoders_from_cfg() handles this automatically.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from scripts.misc.config_loader import Config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class StreamEncoder(ABC):
    """
    Interface every encoder must implement.

    An encoder takes a batch of single-stream windows and returns embeddings.
    """

    @abstractmethod
    def encode(self, X: np.ndarray, batch_size: int = 200) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray  shape (N, T, C)   — N windows, T timesteps, C channels
        batch_size : int

        Returns
        -------
        Z : np.ndarray  shape (N, D)      — N embedding vectors
        """
        ...

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Dimensionality D of the output embeddings."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class EncoderRegistry:
    """Maps encoder type strings to StreamEncoder subclasses."""
    _registry: dict[str, type[StreamEncoder]] = {}

    @classmethod
    def register(cls, name: str, encoder_cls: type[StreamEncoder]):
        cls._registry[name] = encoder_cls

    @classmethod
    def build(cls, encoder_type: str, model_path: str,
              input_shape: list[int], embed_dim: int, **kwargs) -> StreamEncoder:
        if encoder_type not in cls._registry:
            raise ValueError(
                f"Unknown encoder type '{encoder_type}'. "
                f"Available: {sorted(cls._registry.keys())}\n"
                f"Register a new type with EncoderRegistry.register()."
            )
        return cls._registry[encoder_type](
            model_path=model_path,
            input_shape=input_shape,
            embed_dim=embed_dim,
            **kwargs
        )

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry.keys())


# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN IMPLEMENTATION — SimCLR PyTorch
# ─────────────────────────────────────────────────────────────────────────────

class SimCLRPyTorchEncoder(StreamEncoder):
    """
    Wraps SimCLRModel from simclr_models_pt.py.

    Uses model.encoder (BaseEncoder) to extract embeddings.
    The projection head is discarded — only the CNN backbone is used.
    """

    def __init__(self, model_path: str, input_shape: list[int],
                 embed_dim: int, in_channels: int = 3, **kwargs):
        """
        Parameters
        ----------
        model_path  : path to .pt file containing SimCLRModel state_dict
        input_shape : [T, C] — e.g. [100, 3]
        embed_dim   : expected embedding dimension (used for validation)
        in_channels : number of input channels (default 3 for triaxial accel)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Encoder weights not found: {model_path}\n"
                f"Check encoder_paths in paths.json."
            )

        from scripts.misc.simclr_models_pt import SimCLRModel
        full_model = SimCLRModel(in_channels=in_channels)
        full_model.load_state_dict(
            torch.load(model_path, map_location=DEVICE)
        )
        full_model = full_model.to(DEVICE).eval()

        self._encoder   = full_model.encoder   # BaseEncoder only
        self._embed_dim = embed_dim
        self.input_shape = input_shape

        # Validate embed_dim against actual model output
        T, C   = input_shape
        dummy  = torch.zeros(1, T, C, device=DEVICE)
        with torch.no_grad():
            actual_dim = self._encoder(dummy).shape[-1]
        if actual_dim != embed_dim:
            raise ValueError(
                f"embed_dim mismatch: config says {embed_dim}, "
                f"model outputs {actual_dim}. "
                f"Update 'embed_dim' in your dataset config encoders section."
            )
        print(f"  [Encoder] Loaded SimCLR from {model_path}  "
              f"input={input_shape}  embed_dim={actual_dim}")

    def encode(self, X: np.ndarray, batch_size: int = 200) -> np.ndarray:
        """X: (N, T, C) → Z: (N, D)"""
        self._encoder.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.from_numpy(
                    X[i:i + batch_size].astype(np.float32)
                ).to(DEVICE)
                preds.append(self._encoder(xb).cpu().numpy())
        return np.concatenate(preds, axis=0).astype(np.float32)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim


# Register the built-in encoder
EncoderRegistry.register("simclr_pt", SimCLRPyTorchEncoder)


# ─────────────────────────────────────────────────────────────────────────────
# LOADING FROM CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def load_encoders_from_cfg(cfg: "Config") -> dict[str, StreamEncoder]:
    """
    Build encoder objects by combining:
      - dataset config  (configs/<dataset>.json) — defines encoder keys,
                         input_shape, embed_dim, and optionally encoder type
      - paths.json      — defines the actual model file path per encoder key
                         via encoder_paths: {"default": "./models/encoder.pt"}

    paths.json is always the authoritative source for model paths so that
    machine-specific paths never end up committed to the dataset config.

    Example — single encoder (PAAWS Lab):
        paths.json:      "encoder_paths": {"default": "./models/simclr.pt"}
        dataset config:  "encoders": {"default": {...}}

    Example — two encoders:
        paths.json:      "encoder_paths": {"body": "./models/body.pt",
                                           "wrist": "./models/wrist.pt"}
        dataset config:  "encoders": {"body": {...}, "wrist": {...}}
                         "stream_to_encoder": {"LeftWrist": "wrist", ...}
    """
    encoder_type = getattr(cfg, "ENCODER_TYPE", "simclr_pt")
    encoder_paths = cfg.ENCODER_PATHS   # {key: resolved_path} from paths.json

    encoders_cfg = cfg.ENCODERS         # {key: {input_shape, embed_dim, ...}} from dataset config
    if not encoders_cfg:
        # Fallback: single default encoder, shape inferred as [100, 3]
        print("  [Encoder] No 'encoders' section in dataset config — "
              "using single default encoder from encoder_paths.")
        encoders_cfg = {
            "default": {"input_shape": [100, 3], "embed_dim": 96}
        }

    # Validate that every key in encoders_cfg has a path in encoder_paths
    missing = [k for k in encoders_cfg if k not in encoder_paths]
    if missing:
        paths_hint = ", ".join(f'"{k}": "./models/{k}.pt"' for k in missing)
        raise KeyError(
            f"Encoder key(s) {missing} defined in dataset config but not in "
            f"paths.json 'encoder_paths'. Add them: "
            + '{"' + paths_hint + '"}'
        )

    built = {}
    for key, spec in encoders_cfg.items():
        model_path  = encoder_paths[key]
        input_shape = spec["input_shape"]
        embed_dim   = spec["embed_dim"]
        enc_type    = spec.get("type", encoder_type)

        built[key] = EncoderRegistry.build(
            encoder_type=enc_type,
            model_path=model_path,
            input_shape=input_shape,
            embed_dim=embed_dim,
        )

    return built


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  (replaces extract_features in helpers_hitl.py)
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(
    X: np.ndarray,
    encoders: dict[str, StreamEncoder],
    stream_to_encoder: dict[str, str],
    stream_names: list[str],
    batch_size: int = 200,
    stream_indices: list[int] | None = None,
) -> np.ndarray:
    """
    Extract per-stream embeddings and stack into (N, len(stream_names), D).

    Parameters
    ----------
    X                 : np.ndarray  shape (N, T, DIM_raw, C)
    encoders          : dict  {encoder_key: StreamEncoder}
    stream_to_encoder : dict  {stream_name: encoder_key}  from dataset config
    stream_names      : list[str]  names of the streams to extract
    batch_size        : int
    stream_indices    : list[int] | None
        If provided, these column indices (into axis 2 of X) are used to
        select streams. Use this when X has more streams than stream_names
        (e.g. full 8-stream PAAWS data but you only want RightThigh + LeftWrist).
        If None, stream_names must match DIM_raw exactly.

    Returns
    -------
    Z : np.ndarray  shape (N, len(stream_names), D)
    """
    N, T, DIM_raw, C = X.shape

    if stream_indices is not None:
        # Slice to requested streams only
        assert len(stream_indices) == len(stream_names), (
            f"stream_indices length {len(stream_indices)} != "
            f"stream_names length {len(stream_names)}"
        )
        X = X[:, :, stream_indices, :]   # (N, T, len(stream_names), C)
    else:
        assert len(stream_names) == DIM_raw, (
            f"stream_names has {len(stream_names)} entries but X has DIM={DIM_raw}. "
            f"Pass stream_indices to select a subset of streams."
        )

    # Determine per-stream encoder assignment
    default_key = next(iter(encoders))   # first key as fallback
    assignments = [
        stream_to_encoder.get(name, default_key)
        for name in stream_names
    ]

    # Check all assigned keys exist
    for i, key in enumerate(assignments):
        if key not in encoders:
            raise KeyError(
                f"Stream '{stream_names[i]}' mapped to encoder '{key}' "
                f"which is not in the loaded encoders dict. "
                f"Available: {list(encoders.keys())}"
            )

    # Encode each stream
    stream_embeddings = []
    for i, (stream_name, enc_key) in enumerate(zip(stream_names, assignments)):
        X_stream = X[:, :, i, :]                              # (N, T, C)
        Z_stream = encoders[enc_key].encode(X_stream, batch_size=batch_size)  # (N, D)
        stream_embeddings.append(Z_stream)

    # Determine output D — use max embed_dim, zero-pad shorter ones
    D_values = [z.shape[1] for z in stream_embeddings]
    D        = max(D_values)

    if len(set(D_values)) > 1:
        import warnings
        warnings.warn(
            f"Encoders produce different embed_dims {D_values}. "
            f"Shorter embeddings will be zero-padded to D={D}. "
            f"Consider using encoders with the same embed_dim.",
            stacklevel=2,
        )
        padded = []
        for z in stream_embeddings:
            if z.shape[1] < D:
                pad = np.zeros((N, D - z.shape[1]), dtype=np.float32)
                z   = np.concatenate([z, pad], axis=1)
            padded.append(z)
        stream_embeddings = padded

    # Stack: list of DIM arrays (N, D) → (N, DIM, D)
    return np.stack(stream_embeddings, axis=1).astype(np.float32)