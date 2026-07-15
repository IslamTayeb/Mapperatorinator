from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from osuT5.osuT5.inference.optimized.kernels import native_extension


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "1" * 40


def _kwargs(source: str = "source") -> dict[str, object]:
    return {
        "name": "example_extension",
        "cpp_sources": source,
        "cuda_sources": "cuda",
        "functions": ["first", "second"],
        "extra_cuda_cflags": ["-O3"],
        "verbose": False,
    }


def _manifest(root: Path, *, library_bytes: bytes = b"extension") -> Path:
    library = root / "libraries" / "example_extension.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(library_bytes)
    payload = {
        "schema_version": 1,
        "source_commit": SOURCE_COMMIT,
        "abi": native_extension.runtime_abi(),
        "cuda_arches": ["7.5"],
        "extensions": {
            "example_extension": {
                "source_sha256": native_extension.extension_source_hash(_kwargs()),
                "library": "libraries/example_extension.so",
                "library_sha256": native_extension._sha256_file(library),
                "functions": ["first", "second"],
            }
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(payload))
    return path


def test_source_hash_is_deterministic_and_excludes_runtime_controls() -> None:
    baseline = native_extension.extension_source_hash(_kwargs())
    reordered = dict(reversed(list(_kwargs().items())))
    reordered["verbose"] = True
    reordered["build_directory"] = "/tmp/unrelated"

    assert native_extension.extension_source_hash(reordered) == baseline
    assert native_extension.extension_source_hash(_kwargs("changed")) != baseline


def test_manual_pybind_symbols_are_stripped_from_jit_and_recorded(
    tmp_path: Path,
) -> None:
    library = tmp_path / "manual.so"
    library.write_bytes(b"manual pybind extension")
    module = SimpleNamespace(__file__=str(library))
    kwargs = {
        "name": "manual_extension",
        "cpp_sources": "manual binding",
        "cuda_sources": "cuda",
        "prebuilt_functions": ["manual_call"],
    }
    records = {}
    with mock.patch.dict(
        os.environ,
        {native_extension.MANIFEST_ENV: ""},
        clear=False,
    ), mock.patch(
        "torch.utils.cpp_extension.load_inline",
        return_value=module,
    ) as load_inline, mock.patch.object(
        native_extension,
        "_LOADED_EXTENSIONS",
        records,
    ):
        native_extension.load_inline_or_prebuilt(**kwargs)

    assert "prebuilt_functions" not in load_inline.call_args.kwargs
    assert records["manual_extension"]["functions"] == ["manual_call"]


def test_direct_manifest_validates_exact_commit_manifest_and_library_hashes(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    manifest_hash = native_extension._sha256_file(manifest)
    environment = {"TORCH_CUDA_ARCH_LIST": "7.5"}
    with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
        native_extension.torch.cuda,
        "is_available",
        return_value=False,
    ):
        report = native_extension.validate_direct_manifest(
            manifest,
            expected_source_commit=SOURCE_COMMIT,
            expected_manifest_sha256=manifest_hash,
        )
        assert report["pass"] is True
        with pytest.raises(RuntimeError, match="manifest hash mismatch"):
            native_extension.validate_direct_manifest(
                manifest,
                expected_source_commit=SOURCE_COMMIT,
                expected_manifest_sha256="0" * 64,
            )
        with pytest.raises(RuntimeError, match="commit mismatch"):
            native_extension.validate_direct_manifest(
                manifest,
                expected_source_commit="2" * 40,
                expected_manifest_sha256=manifest_hash,
            )
        library = tmp_path / "libraries" / "example_extension.so"
        library.write_bytes(b"corrupted")
        with pytest.raises(RuntimeError, match="artifact hash mismatch"):
            native_extension.validate_direct_manifest(
                manifest,
                expected_source_commit=SOURCE_COMMIT,
                expected_manifest_sha256=manifest_hash,
            )


def test_explicit_manifest_missing_extension_never_falls_back_to_jit(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["extensions"] = {}
    manifest.write_text(json.dumps(payload))
    environment = {
        native_extension.MANIFEST_ENV: str(manifest),
        "TORCH_CUDA_ARCH_LIST": "7.5",
    }
    with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
        native_extension.torch.cuda,
        "is_available",
        return_value=False,
    ), mock.patch("torch.utils.cpp_extension.load_inline") as load_inline:
        with pytest.raises(RuntimeError, match="manifest is missing"):
            native_extension.load_inline_or_prebuilt(**_kwargs())
    load_inline.assert_not_called()


def test_native_extension_policy_imports_remain_cold_and_quiet() -> None:
    source = """
import os
import sys
import warnings
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")
os.environ.pop("MAPPERATORINATOR_NATIVE_EXTENSION_MANIFEST", None)
from osuT5.osuT5.inference.optimized.kernels import decoder_layer, q1_attention, weight_only
from osuT5.osuT5.inference.optimized.scout import int8_mlp
from osuT5.osuT5.inference.optimized.kernels.native_extension import loaded_extension_records
assert "torch.utils.cpp_extension" not in sys.modules
assert loaded_extension_records() == {}
assert decoder_layer._NATIVE_DECODER_LAYER is None
assert q1_attention._NATIVE_Q1_ATTENTION is None
assert weight_only._WEIGHT_ONLY_EXTENSION is None
assert int8_mlp._INT8_MLP_EXTENSION is None
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""
