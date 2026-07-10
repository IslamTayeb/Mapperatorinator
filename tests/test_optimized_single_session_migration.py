from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_fresh_python(source: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_legacy_session_module_is_lazy_until_an_export_is_requested():
    completed = _run_fresh_python(
        """
import importlib
import sys

legacy = importlib.import_module("osuT5.osuT5.inference.direct_decode")
assert "osuT5.osuT5.inference.optimized.single.session" not in sys.modules
legacy.DecodeSession
assert "osuT5.osuT5.inference.optimized.single.session" in sys.modules
"""
    )
    assert completed.returncode == 0, completed.stderr


def test_legacy_session_exports_are_identical_to_optimized_source():
    legacy = importlib.import_module("osuT5.osuT5.inference.direct_decode")
    optimized = importlib.import_module(
        "osuT5.osuT5.inference.optimized.single.session"
    )
    for name in legacy.__all__:
        assert getattr(legacy, name) is getattr(optimized, name)


def test_optimized_packages_do_not_import_session_through_legacy_shim():
    offenders = []
    optimized_root = REPO_ROOT / "osuT5/osuT5/inference/optimized"
    for path in optimized_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "inference.direct_decode" in source or "...direct_decode" in source:
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert offenders == []
