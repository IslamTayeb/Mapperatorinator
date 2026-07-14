"""Opt-in same-process CUDA-graph workspace persistence scout."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import threading
from typing import Any, Iterator
import weakref

import torch

from ..single.k8_runtime import _K8GraphLifecycle
from ..single.state import ProductionDecodeSession


def _tensor_fingerprint(name: str, value: torch.Tensor) -> tuple[Any, ...]:
    return (
        name,
        int(value.data_ptr()),
        tuple(value.shape),
        tuple(value.stride()),
        str(value.dtype),
        str(value.device),
        int(getattr(value, "_version", 0)),
    )


def _owned_tensor_fingerprint(value: Any) -> tuple[tuple[Any, ...], ...]:
    """Fingerprint tensors owned by a packed runtime state without copying them."""

    rows: list[tuple[Any, ...]] = []
    visited: set[int] = set()

    def visit(path: str, item: Any) -> None:
        if isinstance(item, torch.Tensor):
            rows.append(_tensor_fingerprint(path, item))
            return
        if item is None or isinstance(item, (bool, int, float, str, bytes)):
            return
        identity = id(item)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(item, weakref.ReferenceType):
            return
        if isinstance(item, dict):
            for key, child in sorted(item.items(), key=lambda pair: str(pair[0])):
                visit(f"{path}[{key!r}]", child)
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(f"{path}[{index}]", child)
            return
        if is_dataclass(item):
            for field in fields(item):
                visit(f"{path}.{field.name}", getattr(item, field.name))

    visit("owner", value)
    return tuple(sorted(rows))


def _model_fingerprint(model: torch.nn.Module) -> tuple[tuple[Any, ...], ...]:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("persistent graph workspace owner must be a torch module")
    rows = [
        _tensor_fingerprint(f"parameter:{name}", value)
        for name, value in model.named_parameters()
    ]
    rows.extend(
        _tensor_fingerprint(f"buffer:{name}", value)
        for name, value in model.named_buffers()
    )
    return tuple(rows)


class _PersistentWorkspace:
    def __init__(self, signature: tuple[Any, ...]):
        self.signature = signature
        self.session = ProductionDecodeSession()
        self.lifecycle = _K8GraphLifecycle()
        self.in_use = False
        self.closed = False
        self.last_used = 0

    def close(self) -> None:
        if self.closed:
            return
        if self.in_use:
            raise RuntimeError("cannot close an active persistent graph workspace")
        self.lifecycle.close_all()
        self.session.graph_cache.clear()
        self.session.stable_encoder_holders.clear()
        self.session.caches.clear()
        self.session.active_state_signature = None
        self.closed = True

    def summary(self) -> dict[str, Any]:
        cross_request_hits = sum(
            int(entry.get("cross_request_reuse_hits", 0))
            for entry in self.session.graph_cache.values()
            if isinstance(entry, dict)
        )
        encoder_slots = sum(
            len(holder.get("__persistent_encoder_slots__", {}))
            for holder in self.session.stable_encoder_holders.values()
        )
        return {
            "signature": repr(self.signature),
            "graph_count": self.session.graph_count,
            "cross_request_graph_hits": cross_request_hits,
            "encoder_slot_count": encoder_slots,
            "cache_count": len(self.session.caches),
            "in_use": self.in_use,
            "closed": self.closed,
        }


class PersistentGraphWorkspacePool:
    """Bounded model-owned workspace pool with fail-loud serialized borrowing."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        topology_signature: tuple[Any, ...],
        packed_state: Any | None,
        max_slots: int = 2,
    ):
        if not isinstance(topology_signature, tuple) or not topology_signature:
            raise ValueError("persistent graph topology signature must be non-empty")
        if isinstance(max_slots, bool) or not isinstance(max_slots, int) or max_slots <= 0:
            raise ValueError("persistent graph max_slots must be a positive integer")
        self._owner_ref = weakref.ref(model)
        self._model_fingerprint = _model_fingerprint(model)
        self._packed_state_ref = packed_state
        self._packed_fingerprint = _owned_tensor_fingerprint(packed_state)
        self.topology_signature = topology_signature
        self.max_slots = max_slots
        self._workspaces: OrderedDict[tuple[Any, ...], _PersistentWorkspace] = (
            OrderedDict()
        )
        self._lock = threading.RLock()
        self._clock = 0
        self._request_serial = 0
        self._closed = False
        self._workspaces_created = 0
        self._workspaces_evicted = 0
        self._borrow_count = 0
        self._max_resident_slots = 0

    def _validate_owner(self) -> torch.nn.Module:
        if self._closed:
            raise RuntimeError("persistent graph workspace pool is closed")
        model = self._owner_ref()
        if model is None:
            raise RuntimeError("persistent graph workspace model was destroyed")
        if _model_fingerprint(model) != self._model_fingerprint:
            raise RuntimeError("persistent graph model tensor addresses or versions changed")
        if _owned_tensor_fingerprint(self._packed_state_ref) != self._packed_fingerprint:
            raise RuntimeError("persistent packed-weight tensor addresses or versions changed")
        return model

    def new_request(self) -> "PersistentDecodeRequest":
        with self._lock:
            self._validate_owner()
            self._request_serial += 1
            return PersistentDecodeRequest(self, request_serial=self._request_serial)

    def _workspace(self, signature: tuple[Any, ...]) -> _PersistentWorkspace:
        workspace = self._workspaces.get(signature)
        if workspace is not None:
            if workspace.closed:
                raise RuntimeError("persistent workspace cache contains a closed slot")
            self._workspaces.move_to_end(signature)
            return workspace
        if len(self._workspaces) >= self.max_slots:
            evict_signature = next(
                (
                    key
                    for key, candidate in self._workspaces.items()
                    if not candidate.in_use
                ),
                None,
            )
            if evict_signature is None:
                raise RuntimeError("all persistent graph workspace slots are active")
            evicted = self._workspaces.pop(evict_signature)
            evicted.close()
            self._workspaces_evicted += 1
        workspace = _PersistentWorkspace(signature)
        self._workspaces[signature] = workspace
        self._workspaces_created += 1
        self._max_resident_slots = max(
            self._max_resident_slots,
            len(self._workspaces),
        )
        return workspace

    @contextmanager
    def borrow(
        self,
        signature: tuple[Any, ...],
        request: "PersistentDecodeRequest",
    ) -> Iterator[_PersistentWorkspace]:
        with self._lock:
            self._validate_owner()
            workspace = self._workspace(signature)
            if workspace.in_use:
                raise RuntimeError("persistent graph workspace does not allow concurrent use")
            workspace.in_use = True
            self._clock += 1
            workspace.last_used = self._clock
            self._borrow_count += 1
            request._active_workspace = workspace
        try:
            yield workspace
        finally:
            with self._lock:
                if request._active_workspace is not workspace:
                    raise RuntimeError("persistent graph workspace lease was corrupted")
                request._active_workspace = None
                workspace.in_use = False

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if any(workspace.in_use for workspace in self._workspaces.values()):
                raise RuntimeError("cannot close persistent graph pool with active leases")
            for workspace in self._workspaces.values():
                workspace.close()
            self._workspaces.clear()
            self._closed = True

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": True,
                "topology_signature": repr(self.topology_signature),
                "max_slots": self.max_slots,
                "resident_slots": len(self._workspaces),
                "workspaces_created": self._workspaces_created,
                "workspaces_evicted": self._workspaces_evicted,
                "borrow_count": self._borrow_count,
                "max_resident_slots": self._max_resident_slots,
                "request_count": self._request_serial,
                "closed": self._closed,
                "workspaces": [
                    workspace.summary() for workspace in self._workspaces.values()
                ],
            }


class PersistentDecodeRequest(ProductionDecodeSession):
    """Request-local counters backed by one borrowed persistent workspace."""

    def __init__(
        self,
        pool: PersistentGraphWorkspacePool,
        *,
        request_serial: int,
    ):
        super().__init__()
        self.pool = pool
        self.request_serial = request_serial
        self.request_state: dict[str, Any] = {
            "__persistent_request_serial__": request_serial,
            "__k8_window_serial__": 0,
        }
        self._active_workspace: _PersistentWorkspace | None = None

    def window_lease(
        self,
        model,
        *,
        batch_size: int,
        num_beams: int,
        cfg_scale: float,
    ):
        signature = ProductionDecodeSession._state_signature(
            model,
            batch_size=batch_size,
            num_beams=num_beams,
            cfg_scale=cfg_scale,
        ) + self.pool.topology_signature
        return self.pool.borrow(signature, self)

    def _workspace(self) -> _PersistentWorkspace:
        if self._active_workspace is None:
            raise RuntimeError("persistent request used outside its workspace lease")
        return self._active_workspace

    def cache_for_window(self, model, **kwargs):
        workspace = self._workspace()
        cache = workspace.session.cache_for_window(model, **kwargs)
        holder = workspace.session.stable_encoder_holders[
            workspace.session.active_state_signature
        ]
        holder.setdefault("__persistent_encoder_slots__", {})
        return cache

    def active_prefix_decode_kwargs(self) -> dict[str, Any]:
        workspace = self._workspace()
        return {
            **workspace.session.active_prefix_decode_kwargs(),
            "k8_request_state": self.request_state,
            "k8_graph_lifecycle": workspace.lifecycle,
        }

    @property
    def graph_count(self) -> int:
        return self._workspace().session.graph_count

    @property
    def graph_capture_seconds(self) -> float:
        return self._workspace().session.graph_capture_seconds

    @property
    def graph_decode_replays(self) -> int:
        return self._workspace().session.graph_decode_replays

    def graph_profile_summary(self) -> dict[str, Any]:
        workspace = self._workspace()
        result = workspace.session.graph_profile_summary()
        result["persistent_workspace"] = {
            **workspace.summary(),
            "pool": self.pool.summary(),
            "request_serial": self.request_serial,
            "request_window_serial": int(
                self.request_state.get("__k8_window_serial__", 0)
            ),
        }
        return result


__all__ = [
    "PersistentDecodeRequest",
    "PersistentGraphWorkspacePool",
]
