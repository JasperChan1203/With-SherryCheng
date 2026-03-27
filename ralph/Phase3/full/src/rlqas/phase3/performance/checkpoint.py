"""Checkpoint utilities for long-running Phase 3 training jobs.

Provides JSON-based save/load that handles numpy arrays transparently.
Checkpoints are human-readable (JSON) for ease of inspection.
"""

import json
import os
from typing import Any, Dict

import numpy as np


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """Save training state checkpoint to disk as JSON.

    Numpy arrays are encoded as ``{"__ndarray__": true, "data": [...], "dtype": "..."}``
    dicts so they survive the JSON round-trip.

    Args:
        state: Dictionary containing training state.  May contain:
          - Python scalars (int, float, str, bool, None)
          - Lists and nested dicts thereof
          - numpy ndarrays
          - numpy scalar types (np.integer, np.floating)
        path: Destination file path.  Parent directory is created if needed.
    """
    dir_part = os.path.dirname(path)
    if dir_part:
        os.makedirs(dir_part, exist_ok=True)

    with open(path, "w") as fh:
        json.dump(_convert(state), fh, indent=2)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load training state from a checkpoint file.

    Args:
        path: Source file path.

    Returns:
        Dictionary with numpy arrays restored from their encoded form.
    """
    with open(path, "r") as fh:
        data = json.load(fh)
    return _restore(data)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _convert(obj: Any) -> Any:
    """Recursively convert obj to a JSON-serialisable form."""
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "data": obj.tolist(),
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
        }
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert(x) for x in obj]
    # Assume already JSON-serialisable (int, float, str, bool, None)
    return obj


def _restore(obj: Any) -> Any:
    """Recursively restore obj from JSON-decoded form."""
    if isinstance(obj, dict):
        if obj.get("__ndarray__") is True:
            arr = np.array(obj["data"], dtype=obj.get("dtype", "float64"))
            if "shape" in obj:
                arr = arr.reshape(obj["shape"])
            return arr
        return {k: _restore(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore(x) for x in obj]
    return obj
