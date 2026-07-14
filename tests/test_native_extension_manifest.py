from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from osuT5.osuT5.inference.optimized.kernels import native_extension
from utils.benchmark_native_extension_loading import summarize


REPO_ROOT = Path(__file__).resolve().parents[1]


def _kwargs(source: str = "source") -> dict[str, object]:
    return {
        "name": "example_extension",
        "cpp_sources": source,
        "cuda_sources": "cuda",
        "functions": ["first", "second"],
        "extra_cuda_cflags": ["-O3"],
        "verbose": False,
    }


def test_source_hash_is_deterministic_and_excludes_non_codegen_controls() -> None:
    baseline = native_extension.extension_source_hash(_kwargs())
    reordered = dict(reversed(list(_kwargs().items())))
    reordered["verbose"] = True
    reordered["build_directory"] = "/tmp/unrelated"

    assert native_extension.extension_source_hash(reordered) == baseline
    assert native_extension.extension_source_hash(_kwargs("changed")) != baseline


def test_source_hash_rejects_nondeterministic_values() -> None:
    kwargs = _kwargs()
    kwargs["functions"] = [object()]
    with pytest.raises(TypeError, match="non-deterministic build value"):
        native_extension.extension_source_hash(kwargs)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("7.5", ["7.5"]),
        ("7.5;8.0+PTX", ["7.5", "8.0"]),
        ("sm_75 compute_80", ["7.5", "8.0"]),
    ],
)
def test_cuda_arch_normalization(raw: str, expected: list[str]) -> None:
    assert native_extension._normalize_arches(raw) == expected


def test_manifest_source_mismatch_fails_before_library_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {
        "schema_version": 1,
        "abi": native_extension.runtime_abi(),
        "cuda_arches": ["7.5"],
        "extensions": {
            "example_extension": {
                "source_sha256": "wrong",
                "library": "missing.so",
                "library_sha256": "wrong",
                "functions": ["first", "second"],
            }
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setenv(native_extension.MANIFEST_ENV, str(manifest_path))
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "7.5")
    monkeypatch.setattr(native_extension.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="source mismatch"):
        native_extension.load_inline_or_prebuilt(**_kwargs())


def test_manifest_abi_mismatch_fails_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {
        "schema_version": 1,
        "abi": {**native_extension.runtime_abi(), "torch_version": "wrong"},
        "cuda_arches": ["7.5"],
        "extensions": {},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setenv(native_extension.MANIFEST_ENV, str(manifest_path))

    with pytest.raises(RuntimeError, match="ABI mismatch"):
        native_extension.load_inline_or_prebuilt(**_kwargs())


def test_manifest_writer_copies_exact_libraries_and_records_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library = tmp_path / "cached.so"
    library.write_bytes(b"exact shared library bytes")
    kwargs = _kwargs()
    record = {
        "mode": "load_inline",
        "source_sha256": native_extension.extension_source_hash(kwargs),
        "library": str(library),
        "library_sha256": native_extension._sha256_file(library),
        "functions": kwargs["functions"],
    }
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "7.5")
    monkeypatch.delenv(native_extension.MANIFEST_ENV, raising=False)
    monkeypatch.setattr(
        native_extension,
        "_LOADED_EXTENSIONS",
        {"example_extension": record},
    )
    output = tmp_path / "package" / "manifest.json"

    manifest = native_extension.write_loaded_extension_manifest(
        output,
        expected_names=("example_extension",),
    )

    entry = manifest["extensions"]["example_extension"]
    copied = output.parent / entry["library"]
    assert copied.read_bytes() == library.read_bytes()
    assert entry["library_sha256"] == native_extension._sha256_file(copied)
    assert json.loads(output.read_text()) == manifest


def _run(mode: str, seconds: float, source: str = "same") -> dict[str, object]:
    return {
        "mode": mode,
        "seconds": seconds,
        "records": {
            "extension": {
                "source_sha256": source,
                "library_sha256": "library",
                "functions": ["call"],
            }
        },
        "probes": {
            "extension": {
                "call": {
                    "kind": "exception",
                    "type": "TypeError",
                    "message": "call(): incompatible arguments",
                }
            }
        },
    }


def test_loading_summary_requires_parity_and_half_second_saving() -> None:
    passed = summarize(
        [_run("cached", 1.2), _run("direct", 0.5)],
        minimum_saving_seconds=0.5,
    )
    assert passed["pass"] is True
    assert passed["saving_seconds"] == pytest.approx(0.7)

    drift = summarize(
        [_run("cached", 1.2), _run("direct", 0.5, source="different")],
        minimum_saving_seconds=0.5,
    )
    assert drift["saving_pass"] is True
    assert drift["parity_pass"] is False
    assert drift["pass"] is False


def test_importing_native_extension_policy_does_not_import_cpp_extension() -> None:
    source = """
import sys
import osuT5.osuT5.inference.optimized.kernels.native_extension
assert "torch.utils.cpp_extension" not in sys.modules
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
