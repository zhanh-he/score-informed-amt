"""Minimal configuration shim for the vendored FiLM U-Net blocks.

Only exposes the attributes accessed inside ``model.py`` and ``sub_models.py``.
`model_FilmUnet.py` mutates a few of these values (e.g., ``condition_check``),
so keep everything as module-level variables.
"""

from __future__ import annotations

import torch

# Audio / feature extraction parameters
sample_rate: int = 16000
frames_per_second: int = 100
classes_num: int = 88
spec_feat: str = "mel"  # {"power", "mel", "bark", "sone"}

# Conditioning setup (can be overridden by our wrapper)
condition_check: bool = False
condition_type: str = "frame"
condition_net: str = "DynamicFCInitFirst"  # or "DynamicFC"

# Misc constants occasionally referenced in helper layers
begin_note: int = 21
velocity_scale: int = 128
cvx_theta: float = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_overrides(**kwargs) -> None:
    """Utility for tests to override defaults without importing Hydra configs."""
    for key, value in kwargs.items():
        if not hasattr(__import__(__name__), key):
            raise AttributeError(f"Unknown FiLM config key: {key}")
        globals()[key] = value
