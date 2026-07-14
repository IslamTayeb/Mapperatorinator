"""Opt-in runtime composition for final confirmation jobs.

The confirmation harness owns this interface.  Production inference selectors
remain unchanged: a selected plugin may only install temporary process hooks,
initialize an already loaded optimized runtime, and attach benchmark metadata.
"""

from __future__ import annotations

import importlib
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from osuT5.osuT5.inference.engine_binding import InferenceEngineBinding
from utils.run_approximate_weight_only import _initialize_with_evidence


RUNTIME_SPEC_SCHEMA_VERSION = "mapperatorinator.final-runtime-spec.v1"


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _import_factory(spec: str):
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise ValueError("runtime factory must use module:callable syntax")
    module_name, callable_name = spec.split(":", 1)
    if not module_name or not callable_name:
        raise ValueError("runtime factory module and callable must be non-empty")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(f"cannot import runtime factory module {module_name!r}") from exc
    factory = getattr(module, callable_name, None)
    if not callable(factory):
        raise RuntimeError(f"runtime factory {spec!r} is not callable")
    return factory


def load_runtime_plugin(path: Path | None):
    if path is None:
        return AcceptedRuntimePlugin()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read runtime spec {path}: {exc}") from exc
    root = _object(payload, name="runtime spec")
    if root.get("schema_version") != RUNTIME_SPEC_SCHEMA_VERSION:
        raise ValueError("runtime spec schema is missing or unsupported")
    name = root.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("runtime spec name must be non-empty")
    factory_spec = root.get("factory")
    kwargs = _object(root.get("kwargs", {}), name="runtime spec kwargs")
    factory = _import_factory(factory_spec)
    try:
        plugin = factory(name=name, spec=dict(root), **kwargs)
    except Exception as exc:
        raise RuntimeError(f"runtime factory {factory_spec!r} failed") from exc
    for method in ("__enter__", "__exit__", "transform_binding", "evidence"):
        if not callable(getattr(plugin, method, None)):
            raise TypeError(f"runtime plugin lacks callable {method}")
    return plugin


class AcceptedRuntimePlugin:
    """No-op accepted production runtime control."""

    name = "accepted-optimized"

    def __init__(self) -> None:
        self._bindings = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def transform_binding(
        self,
        binding: InferenceEngineBinding,
        *,
        binding_index: int,
    ) -> InferenceEngineBinding:
        if not isinstance(binding, InferenceEngineBinding):
            raise TypeError("final confirmation requires an optimized engine binding")
        if binding_index != self._bindings:
            raise RuntimeError("runtime binding indices must be contiguous")
        self._bindings += 1
        return binding

    def evidence(self) -> dict[str, Any]:
        if self._bindings <= 0:
            raise RuntimeError("accepted runtime plugin observed no model binding")
        return {
            "name": self.name,
            "factory": None,
            "binding_count": self._bindings,
            "initialization": None,
            "temporary_hooks": [],
        }


class KBlockSharedRopeWeightPlugin:
    """Compose block decoding, exact shared RoPE, and a runtime initializer."""

    def __init__(
        self,
        *,
        name: str,
        spec: dict[str, Any],
        block_size: int,
        graph_remainders: bool,
        initializer_name: str,
        shared_rope_binding_index: int = 0,
        initializer_binding_index: int = 0,
        minimum_bindings: int = 1,
    ) -> None:
        if isinstance(block_size, bool) or block_size not in {1, 2, 4, 8}:
            raise ValueError("runtime block_size must be one of 1, 2, 4, or 8")
        if not isinstance(graph_remainders, bool):
            raise TypeError("graph_remainders must be boolean")
        if not isinstance(initializer_name, str) or not initializer_name:
            raise ValueError("initializer_name must be non-empty")
        for label, value in (
            ("shared_rope_binding_index", shared_rope_binding_index),
            ("initializer_binding_index", initializer_binding_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if isinstance(minimum_bindings, bool) or not isinstance(minimum_bindings, int) or minimum_bindings <= 0:
            raise ValueError("minimum_bindings must be a positive integer")
        self.name = name
        self._spec = spec
        self._block_size = block_size
        self._graph_remainders = graph_remainders
        self._initializer_name = initializer_name
        self._shared_rope_binding_index = shared_rope_binding_index
        self._initializer_binding_index = initializer_binding_index
        self._minimum_bindings = minimum_bindings
        self._stack = ExitStack()
        self._entered = False
        self._closed = False
        self._bindings = 0
        self._initialization: dict[str, Any] | None = None
        self._shared_stats = None

    def __enter__(self):
        if self._entered:
            raise RuntimeError("runtime plugin cannot be entered twice")
        from osuT5.osuT5.inference.optimized.single.k8_runtime import (
            install_k8_candidate,
        )

        self._entered = True
        self._stack.enter_context(
            install_k8_candidate(
                block_size=self._block_size,
                graph_remainders=self._graph_remainders,
            )
        )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._closed = True
        self._stack.close()

    def transform_binding(
        self,
        binding: InferenceEngineBinding,
        *,
        binding_index: int,
    ) -> InferenceEngineBinding:
        if not self._entered or self._closed:
            raise RuntimeError("runtime binding arrived outside plugin lifetime")
        if not isinstance(binding, InferenceEngineBinding):
            raise TypeError("final confirmation requires an optimized engine binding")
        if binding_index != self._bindings:
            raise RuntimeError("runtime binding indices must be contiguous")
        if binding_index == self._shared_rope_binding_index:
            from osuT5.osuT5.inference.optimized.scout.shared_rope import (
                SharedRopeStats,
                shared_decoder_rope_context,
            )

            if self._shared_stats is not None:
                raise RuntimeError("shared RoPE binding was installed twice")
            self._shared_stats = SharedRopeStats()
            self._stack.enter_context(
                shared_decoder_rope_context(
                    binding.raw_model,
                    stats=self._shared_stats,
                )
            )
        if binding_index == self._initializer_binding_index:
            if self._initialization is not None:
                raise RuntimeError("runtime initializer binding was initialized twice")
            initializer = getattr(binding.runtime, self._initializer_name, None)
            if not callable(initializer):
                raise RuntimeError(
                    "loaded runtime does not expose requested initializer "
                    f"{self._initializer_name!r}"
                )
            self._initialization = _initialize_with_evidence(
                initializer,
                binding.raw_model,
            )
        self._bindings += 1
        return binding

    def evidence(self) -> dict[str, Any]:
        if self._bindings < self._minimum_bindings:
            raise RuntimeError(
                f"runtime plugin observed {self._bindings} bindings, expected at least "
                f"{self._minimum_bindings}"
            )
        if self._initialization is None:
            raise RuntimeError("runtime plugin did not initialize its target binding")
        if self._shared_stats is None:
            raise RuntimeError("runtime plugin did not install shared RoPE")
        shared = self._shared_stats.as_dict()
        if shared["forwards"] <= 0 or shared["reuses"] <= 0:
            raise RuntimeError("shared RoPE did not observe and reuse decoder forwards")
        if shared["computes"] != shared["expected_computes"]:
            raise RuntimeError("shared RoPE compute accounting is inconsistent")
        if shared["reuses"] != shared["expected_reuses"]:
            raise RuntimeError("shared RoPE reuse accounting is inconsistent")
        return {
            "name": self.name,
            "factory": self._spec["factory"],
            "spec": self._spec,
            "binding_count": self._bindings,
            "initialization": self._initialization,
            "temporary_hooks": [
                {
                    "kind": "block_decode",
                    "block_size": self._block_size,
                    "graph_remainders": self._graph_remainders,
                },
                {
                    "kind": "shared_decoder_rope",
                    "binding_index": self._shared_rope_binding_index,
                    "stats": shared,
                },
                {
                    "kind": "runtime_initializer",
                    "binding_index": self._initializer_binding_index,
                    "name": self._initializer_name,
                },
            ],
        }


def kblock_shared_rope_weight_plugin(**kwargs: Any) -> KBlockSharedRopeWeightPlugin:
    return KBlockSharedRopeWeightPlugin(**kwargs)


__all__ = [
    "RUNTIME_SPEC_SCHEMA_VERSION",
    "AcceptedRuntimePlugin",
    "KBlockSharedRopeWeightPlugin",
    "kblock_shared_rope_weight_plugin",
    "load_runtime_plugin",
]
