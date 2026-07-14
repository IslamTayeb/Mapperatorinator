from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def validate_preloaded_audio(samples: Any) -> np.ndarray:
    """Validate an already-decoded audio array without changing its values."""

    if not isinstance(samples, np.ndarray):
        raise TypeError("preloaded audio must be a numpy.ndarray")
    if samples.dtype != np.float32:
        raise TypeError(
            f"preloaded audio must use float32 storage, got {samples.dtype}"
        )
    if samples.ndim != 1:
        raise ValueError(
            f"preloaded audio must be a mono one-dimensional array, got {samples.shape}"
        )
    if samples.size == 0:
        raise ValueError("preloaded audio must contain at least one sample")
    if not np.isfinite(samples).all():
        raise ValueError("preloaded audio contains non-finite samples")
    return samples


def audio_array_metadata(samples: Any) -> dict[str, Any]:
    """Return a canonical value-and-storage hash for decoded mono audio."""

    samples = validate_preloaded_audio(samples)
    shape = [int(size) for size in samples.shape]
    header = json.dumps(
        {"dtype": samples.dtype.str, "shape": shape},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(samples.tobytes(order="C"))
    return {
        "audio_array_sha256": digest.hexdigest(),
        "audio_array_dtype": str(samples.dtype),
        "audio_array_shape": shape,
        "audio_array_nbytes": int(samples.nbytes),
    }


def resolve_audio_samples(preprocessor, path: str, preloaded_audio=None) -> np.ndarray:
    """Use the unchanged loader unless an opt-in caller supplies its exact result."""

    if preloaded_audio is None:
        return preprocessor.load(path)
    return validate_preloaded_audio(preloaded_audio)


__all__ = [
    "audio_array_metadata",
    "resolve_audio_samples",
    "validate_preloaded_audio",
]
