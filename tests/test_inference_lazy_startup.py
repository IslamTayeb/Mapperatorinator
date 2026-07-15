from __future__ import annotations

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


def test_default_inference_import_skips_optional_runtime_stacks() -> None:
    completed = _run_fresh_python(
        """
import sys
import inference

for name in (
    "diffusion_pipeline",
    "osu_diffusion",
    "osuT5.osuT5.inference.super_timing_generator",
    "osuT5.osuT5.utils.train_utils",
    "datasets",
    "wandb",
):
    assert name not in sys.modules, name
"""
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""


def test_utils_exports_load_only_the_owning_module() -> None:
    completed = _run_fresh_python(
        """
import sys
import osuT5.osuT5.utils as inference_utils

assert "osuT5.osuT5.utils.model_utils" not in sys.modules
assert "osuT5.osuT5.utils.train_utils" not in sys.modules
assert callable(inference_utils.load_model_loaders)
assert "osuT5.osuT5.utils.model_utils" in sys.modules
assert "osuT5.osuT5.utils.train_utils" not in sys.modules
assert callable(inference_utils.train)
assert "osuT5.osuT5.utils.train_utils" in sys.modules
"""
    )
    assert completed.returncode == 0, completed.stderr


def test_lazy_utils_unknown_names_fail_loudly() -> None:
    completed = _run_fresh_python(
        """
import osuT5.osuT5.utils as inference_utils

try:
    inference_utils.not_a_real_export
except AttributeError as exc:
    assert "not_a_real_export" in str(exc)
else:
    raise AssertionError("unknown lazy export did not fail")
"""
    )
    assert completed.returncode == 0, completed.stderr
