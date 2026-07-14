from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from osuT5.osuT5.inference.optimized.kernels import native_extension
from utils.benchmark_native_extension_loading import summarize


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


class NativeExtensionManifestTest(unittest.TestCase):
    def test_source_hash_is_deterministic_and_excludes_runtime_controls(self) -> None:
        baseline = native_extension.extension_source_hash(_kwargs())
        reordered = dict(reversed(list(_kwargs().items())))
        reordered["verbose"] = True
        reordered["build_directory"] = "/tmp/unrelated"

        self.assertEqual(native_extension.extension_source_hash(reordered), baseline)
        self.assertNotEqual(
            native_extension.extension_source_hash(_kwargs("changed")), baseline
        )

    def test_source_hash_rejects_nondeterministic_values(self) -> None:
        kwargs = _kwargs()
        kwargs["functions"] = [object()]
        with self.assertRaisesRegex(TypeError, "non-deterministic build value"):
            native_extension.extension_source_hash(kwargs)

    def test_cuda_arch_normalization(self) -> None:
        cases = (
            ("7.5", ["7.5"]),
            ("7.5;8.0+PTX", ["7.5", "8.0"]),
            ("sm_75 compute_80", ["7.5", "8.0"]),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(native_extension._normalize_arches(raw), expected)

    def test_manifest_source_mismatch_fails_before_library_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "schema_version": 1,
                "source_commit": SOURCE_COMMIT,
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
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            environment = {
                native_extension.MANIFEST_ENV: str(manifest_path),
                "TORCH_CUDA_ARCH_LIST": "7.5",
            }
            with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
                native_extension.torch.cuda, "is_available", return_value=False
            ):
                with self.assertRaisesRegex(RuntimeError, "source mismatch"):
                    native_extension.load_inline_or_prebuilt(**_kwargs())

    def test_manifest_abi_mismatch_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "schema_version": 1,
                "source_commit": SOURCE_COMMIT,
                "abi": {
                    **native_extension.runtime_abi(),
                    "torch_version": "wrong",
                },
                "cuda_arches": ["7.5"],
                "extensions": {},
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            with mock.patch.dict(
                os.environ,
                {native_extension.MANIFEST_ENV: str(manifest_path)},
                clear=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "ABI mismatch"):
                    native_extension.load_inline_or_prebuilt(**_kwargs())

    def test_manifest_writer_copies_exact_libraries_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "cached.so"
            library.write_bytes(b"exact shared library bytes")
            kwargs = _kwargs()
            record = {
                "mode": "load_inline",
                "source_sha256": native_extension.extension_source_hash(kwargs),
                "library": str(library),
                "library_sha256": native_extension._sha256_file(library),
                "functions": kwargs["functions"],
            }
            output = root / "package" / "manifest.json"
            with mock.patch.dict(
                os.environ,
                {"TORCH_CUDA_ARCH_LIST": "7.5"},
                clear=False,
            ), mock.patch.object(
                native_extension,
                "_LOADED_EXTENSIONS",
                {"example_extension": record},
            ):
                os.environ.pop(native_extension.MANIFEST_ENV, None)
                manifest = native_extension.write_loaded_extension_manifest(
                    output,
                    expected_names=("example_extension",),
                    source_commit=SOURCE_COMMIT,
                )

            entry = manifest["extensions"]["example_extension"]
            copied = output.parent / entry["library"]
            self.assertEqual(copied.read_bytes(), library.read_bytes())
            self.assertEqual(
                entry["library_sha256"], native_extension._sha256_file(copied)
            )
            self.assertEqual(json.loads(output.read_text()), manifest)

    def test_packaged_manifest_validates_exact_commit_manifest_and_cache_hashes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            libraries = package / "libraries"
            cache = root / "cache" / "example_extension"
            libraries.mkdir(parents=True)
            cache.mkdir(parents=True)
            library = libraries / "example_extension.so"
            cached = cache / "example_extension.so"
            library.write_bytes(b"same extension")
            cached.write_bytes(library.read_bytes())
            library_hash = native_extension._sha256_file(library)
            manifest = {
                "schema_version": 1,
                "source_commit": SOURCE_COMMIT,
                "abi": native_extension.runtime_abi(),
                "cuda_arches": ["7.5"],
                "extensions": {
                    "example_extension": {
                        "source_sha256": "a" * 64,
                        "library": "libraries/example_extension.so",
                        "library_sha256": library_hash,
                        "functions": ["call"],
                    }
                },
            }
            manifest_path = package / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            manifest_hash = native_extension._sha256_file(manifest_path)
            with mock.patch.dict(
                os.environ,
                {"TORCH_CUDA_ARCH_LIST": "7.5"},
                clear=False,
            ), mock.patch.object(
                native_extension.torch.cuda, "is_available", return_value=False
            ):
                report = native_extension.validate_packaged_manifest(
                    manifest_path,
                    expected_source_commit=SOURCE_COMMIT,
                    expected_manifest_sha256=manifest_hash,
                    extension_cache_root=root / "cache",
                )
                self.assertTrue(report["pass"])
                with self.assertRaisesRegex(RuntimeError, "manifest hash mismatch"):
                    native_extension.validate_packaged_manifest(
                        manifest_path,
                        expected_source_commit=SOURCE_COMMIT,
                        expected_manifest_sha256="0" * 64,
                        extension_cache_root=root / "cache",
                    )
                with self.assertRaisesRegex(RuntimeError, "commit mismatch"):
                    native_extension.validate_packaged_manifest(
                        manifest_path,
                        expected_source_commit="2" * 40,
                        expected_manifest_sha256=manifest_hash,
                        extension_cache_root=root / "cache",
                    )
                cached.write_bytes(b"corrupted")
                with self.assertRaisesRegex(RuntimeError, "artifact hash mismatch"):
                    native_extension.validate_packaged_manifest(
                        manifest_path,
                        expected_source_commit=SOURCE_COMMIT,
                        expected_manifest_sha256=manifest_hash,
                        extension_cache_root=root / "cache",
                    )

    def test_loading_summary_requires_parity_and_half_second_saving(self) -> None:
        passed = summarize(
            [_run("cached", 1.2), _run("direct", 0.5)],
            minimum_saving_seconds=0.5,
        )
        self.assertTrue(passed["pass"])
        self.assertAlmostEqual(passed["saving_seconds"], 0.7)

        drift = summarize(
            [_run("cached", 1.2), _run("direct", 0.5, source="different")],
            minimum_saving_seconds=0.5,
        )
        self.assertTrue(drift["saving_pass"])
        self.assertFalse(drift["parity_pass"])
        self.assertFalse(drift["pass"])

    def test_importing_policy_does_not_import_cpp_extension(self) -> None:
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
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
