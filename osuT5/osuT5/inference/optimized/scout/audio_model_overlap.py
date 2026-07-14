from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ...audio_preparation import audio_array_metadata, validate_preloaded_audio


@dataclass(frozen=True, slots=True)
class PreparedAudioResult:
    samples: np.ndarray
    metadata: dict[str, Any]
    worker_started_at_perf_counter_seconds: float
    worker_finished_at_perf_counter_seconds: float
    worker_thread_name: str

    @property
    def worker_wall_seconds(self) -> float:
        return (
            self.worker_finished_at_perf_counter_seconds
            - self.worker_started_at_perf_counter_seconds
        )


def _load_audio(
    loader: Callable[[str], np.ndarray],
    path: str,
) -> PreparedAudioResult:
    started = time.perf_counter()
    thread_name = threading.current_thread().name
    samples = validate_preloaded_audio(loader(path))
    finished = time.perf_counter()
    return PreparedAudioResult(
        samples=samples,
        metadata=audio_array_metadata(samples),
        worker_started_at_perf_counter_seconds=started,
        worker_finished_at_perf_counter_seconds=finished,
        worker_thread_name=thread_name,
    )


class AudioPreparationTask:
    """Own one opt-in audio worker and guarantee deterministic cleanup."""

    def __init__(
        self,
        loader: Callable[[str], np.ndarray],
        path: str,
    ) -> None:
        if not callable(loader):
            raise TypeError("audio loader must be callable")
        if not isinstance(path, str) or not path:
            raise ValueError("audio path must be a non-empty string")
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mapperatorinator-audio",
        )
        self._future: Future[PreparedAudioResult] = self._executor.submit(
            _load_audio,
            loader,
            path,
        )
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def done(self) -> bool:
        return self._future.done()

    def result(self) -> PreparedAudioResult:
        if self._closed:
            raise RuntimeError("audio preparation task is already closed")
        return self._future.result()

    def close(self) -> None:
        if self._closed:
            return
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._closed = True

    def __enter__(self) -> "AudioPreparationTask":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def preload_accepted_native_extensions() -> None:
    """Load the accepted opt-in native extensions without installing hooks."""

    from ..kernels import decoder_layer, q1_attention

    q1_attention.preload_native_q1_attention()
    decoder_layer.preload_native_decoder_layer()


__all__ = [
    "AudioPreparationTask",
    "PreparedAudioResult",
    "preload_accepted_native_extensions",
]
