"""Opt-in direct loading for exactly matched native extension builds."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import sysconfig
import time
from typing import Any

import torch


MANIFEST_ENV = "MAPPERATORINATOR_NATIVE_EXTENSION_MANIFEST"
SCHEMA_VERSION = 1
_NON_CODEGEN_KEYS = frozenset({"build_directory", "keep_intermediates", "verbose"})
_LOADED_EXTENSIONS: dict[str, dict[str, Any]] = {}


def _json_value(value: Any, *, name: str) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, name=name) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item, name=name)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    raise TypeError(f"native extension {name} has non-deterministic build value {value!r}")


def extension_source_hash(kwargs: Mapping[str, Any]) -> str:
    if not isinstance(kwargs.get("name"), str) or not kwargs["name"]:
        raise ValueError("native extension build requires a nonempty name")
    codegen = {
        key: _json_value(value, name=str(kwargs["name"]))
        for key, value in sorted(kwargs.items())
        if key not in _NON_CODEGEN_KEYS
    }
    encoded = json.dumps(codegen, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalize_arches(raw: str) -> list[str]:
    arches = []
    for item in raw.replace(";", " ").split():
        value = item.strip().lower().removeprefix("sm_").removeprefix("compute_")
        value = value.removesuffix("+ptx")
        if value.isdigit() and len(value) >= 2:
            value = f"{value[:-1]}.{value[-1]}"
        if not value or any(part and not part.isdigit() for part in value.split(".")):
            raise ValueError(f"invalid TORCH_CUDA_ARCH_LIST entry {item!r}")
        if value not in arches:
            arches.append(value)
    return sorted(arches)


def requested_cuda_arches() -> list[str]:
    return _normalize_arches(os.environ.get("TORCH_CUDA_ARCH_LIST", ""))


def runtime_abi() -> dict[str, Any]:
    return {
        "python_cache_tag": sys.implementation.cache_tag,
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "torch_pybind_compiler_type": getattr(
            torch._C, "_PYBIND11_COMPILER_TYPE", None
        ),
        "torch_pybind_stdlib": getattr(torch._C, "_PYBIND11_STDLIB", None),
        "torch_pybind_build_abi": getattr(torch._C, "_PYBIND11_BUILD_ABI", None),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read native extension manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"native extension manifest {path} must use schema {SCHEMA_VERSION}"
        )
    source_commit = manifest.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise RuntimeError("native extension manifest requires a full source commit")
    if manifest.get("abi") != runtime_abi():
        raise RuntimeError(
            "native extension manifest ABI mismatch: "
            f"expected {runtime_abi()}, got {manifest.get('abi')}"
        )
    raw_arches = manifest.get("cuda_arches")
    if not isinstance(raw_arches, list) or not raw_arches or not all(
        isinstance(value, str) for value in raw_arches
    ):
        raise RuntimeError("native extension manifest requires CUDA architecture targets")
    manifest_arches = sorted(set(raw_arches))
    requested = requested_cuda_arches()
    if requested and requested != manifest_arches:
        raise RuntimeError(
            "native extension manifest CUDA target mismatch: "
            f"requested {requested}, built {manifest_arches}"
        )
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        live = f"{major}.{minor}"
        if live not in manifest_arches:
            raise RuntimeError(
                f"native extension manifest does not contain live CUDA capability {live}"
            )
    return manifest


def _load_direct(kwargs: Mapping[str, Any], manifest_path: Path):
    name = str(kwargs["name"])
    manifest = _load_manifest(manifest_path)
    extensions = manifest.get("extensions")
    if not isinstance(extensions, dict) or name not in extensions:
        raise RuntimeError(f"native extension manifest is missing {name}")
    entry = extensions[name]
    if not isinstance(entry, dict):
        raise RuntimeError(f"native extension manifest entry {name} must be an object")
    expected_source = extension_source_hash(kwargs)
    if entry.get("source_sha256") != expected_source:
        raise RuntimeError(
            f"native extension source mismatch for {name}: "
            f"expected {expected_source}, got {entry.get('source_sha256')}"
        )
    functions = list(kwargs.get("functions") or [])
    if entry.get("functions") != functions:
        raise RuntimeError(
            f"native extension exported-symbol mismatch for {name}: "
            f"expected {functions}, got {entry.get('functions')}"
        )
    relative = entry.get("library")
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"native extension manifest entry {name} lacks a library")
    root = manifest_path.parent.resolve()
    library = (root / relative).resolve()
    if not library.is_relative_to(root):
        raise RuntimeError(f"native extension library for {name} escapes manifest root")
    if not library.is_file():
        raise RuntimeError(f"native extension library for {name} is missing: {library}")
    actual_library_hash = _sha256_file(library)
    if entry.get("library_sha256") != actual_library_hash:
        raise RuntimeError(
            f"native extension library hash mismatch for {name}: "
            f"expected {entry.get('library_sha256')}, got {actual_library_hash}"
        )
    previous = sys.modules.get(name)
    if previous is not None:
        previous_path = Path(getattr(previous, "__file__", "")).resolve()
        if previous_path != library:
            raise RuntimeError(
                f"native extension module {name} is already loaded from {previous_path}"
            )
        module = previous
    else:
        spec = importlib.util.spec_from_file_location(name, library)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot create import spec for native extension {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    missing = [symbol for symbol in functions if not callable(getattr(module, symbol, None))]
    if missing:
        raise RuntimeError(f"native extension {name} is missing callables {missing}")
    _LOADED_EXTENSIONS[name] = {
        "mode": "direct",
        "source_sha256": expected_source,
        "library": str(library),
        "library_sha256": actual_library_hash,
        "functions": functions,
    }
    return module


def load_inline_or_prebuilt(**kwargs):
    """Use normal lazy JIT resolution unless an explicit manifest is selected."""

    started = time.perf_counter()
    manifest_value = os.environ.get(MANIFEST_ENV)
    if manifest_value:
        module = _load_direct(kwargs, Path(manifest_value).expanduser().resolve())
        _LOADED_EXTENSIONS[str(kwargs["name"])]["load_seconds"] = (
            time.perf_counter() - started
        )
        return module
    from torch.utils.cpp_extension import load_inline

    module = load_inline(**kwargs)
    name = str(kwargs["name"])
    library = Path(module.__file__).resolve()
    _LOADED_EXTENSIONS[name] = {
        "mode": "load_inline",
        "source_sha256": extension_source_hash(kwargs),
        "library": str(library),
        "library_sha256": _sha256_file(library),
        "functions": list(kwargs.get("functions") or []),
        "load_seconds": time.perf_counter() - started,
    }
    return module


def loaded_extension_records() -> dict[str, dict[str, Any]]:
    return json.loads(json.dumps(_LOADED_EXTENSIONS, sort_keys=True))


def write_loaded_extension_manifest(
    output_path: Path,
    *,
    expected_names: tuple[str, ...],
    source_commit: str,
) -> dict[str, Any]:
    arches = requested_cuda_arches()
    if not arches:
        raise RuntimeError("building a direct-load manifest requires TORCH_CUDA_ARCH_LIST")
    if os.environ.get(MANIFEST_ENV):
        raise RuntimeError(f"unset {MANIFEST_ENV} before building a manifest")
    if (
        len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise RuntimeError("manifest source_commit must be a full lowercase Git hash")
    if set(_LOADED_EXTENSIONS) != set(expected_names):
        raise RuntimeError(
            "loaded native extension set mismatch: "
            f"expected {sorted(expected_names)}, got {sorted(_LOADED_EXTENSIONS)}"
        )
    output_path = output_path.resolve()
    library_dir = output_path.parent / "libraries"
    library_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    for name in expected_names:
        record = _LOADED_EXTENSIONS[name]
        if record["mode"] != "load_inline":
            raise RuntimeError(f"cannot package non-JIT extension record for {name}")
        source = Path(record["library"])
        suffix = "".join(source.suffixes) or ".so"
        destination = library_dir / f"{name}{suffix}"
        shutil.copy2(source, destination)
        copied_hash = _sha256_file(destination)
        if copied_hash != record["library_sha256"]:
            raise RuntimeError(f"copied native extension changed bytes for {name}")
        entries[name] = {
            "source_sha256": record["source_sha256"],
            "library": destination.relative_to(output_path.parent).as_posix(),
            "library_sha256": copied_hash,
            "functions": record["functions"],
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_commit": source_commit,
        "abi": runtime_abi(),
        "cuda_arches": arches,
        "extensions": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def validate_packaged_manifest(
    manifest_path: Path,
    *,
    expected_source_commit: str,
    expected_manifest_sha256: str,
    extension_cache_root: Path,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    actual_manifest_hash = _sha256_file(manifest_path)
    if actual_manifest_hash != expected_manifest_sha256:
        raise RuntimeError(
            "native extension manifest hash mismatch: "
            f"expected {expected_manifest_sha256}, got {actual_manifest_hash}"
        )
    manifest = _load_manifest(manifest_path)
    if manifest["source_commit"] != expected_source_commit:
        raise RuntimeError(
            "native extension manifest commit mismatch: "
            f"expected {expected_source_commit}, got {manifest['source_commit']}"
        )
    extensions = manifest.get("extensions")
    if not isinstance(extensions, dict) or not extensions:
        raise RuntimeError("native extension manifest has no extensions")
    cache_root = extension_cache_root.expanduser().resolve()
    if not cache_root.is_dir():
        raise RuntimeError(f"native extension cache root is missing: {cache_root}")
    validated: dict[str, Any] = {}
    for name, entry in sorted(extensions.items()):
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise RuntimeError("native extension manifest entries must be named objects")
        relative = entry.get("library")
        if not isinstance(relative, str):
            raise RuntimeError(f"native extension {name} has no packaged library")
        package_library = (manifest_path.parent / relative).resolve()
        if not package_library.is_relative_to(manifest_path.parent.resolve()):
            raise RuntimeError(f"native extension package library for {name} escapes root")
        cache_library = (cache_root / name / f"{name}.so").resolve()
        if not package_library.is_file():
            raise RuntimeError(
                f"native extension package library for {name} is missing: "
                f"{package_library}"
            )
        if not cache_library.is_file():
            raise RuntimeError(
                f"native extension cached library for {name} is missing: "
                f"{cache_library}"
            )
        expected_hash = entry.get("library_sha256")
        package_hash = _sha256_file(package_library)
        cache_hash = _sha256_file(cache_library)
        if package_hash != expected_hash or cache_hash != expected_hash:
            raise RuntimeError(
                f"native extension artifact hash mismatch for {name}: "
                f"manifest={expected_hash}, package={package_hash}, cache={cache_hash}"
            )
        functions = entry.get("functions")
        source_hash = entry.get("source_sha256")
        if not isinstance(functions, list) or not functions or not all(
            isinstance(function, str) and function for function in functions
        ):
            raise RuntimeError(f"native extension {name} has invalid exported functions")
        if (
            not isinstance(source_hash, str)
            or len(source_hash) != 64
            or any(character not in "0123456789abcdef" for character in source_hash)
        ):
            raise RuntimeError(f"native extension {name} has invalid source hash")
        validated[name] = {
            "source_sha256": source_hash,
            "library_sha256": expected_hash,
            "functions": functions,
            "package_library": str(package_library),
            "cache_library": str(cache_library),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "source_commit": manifest["source_commit"],
        "manifest": str(manifest_path),
        "manifest_sha256": actual_manifest_hash,
        "extension_cache_root": str(cache_root),
        "extensions": validated,
        "pass": True,
    }


__all__ = [
    "MANIFEST_ENV",
    "extension_source_hash",
    "load_inline_or_prebuilt",
    "loaded_extension_records",
    "requested_cuda_arches",
    "runtime_abi",
    "validate_packaged_manifest",
    "write_loaded_extension_manifest",
]
