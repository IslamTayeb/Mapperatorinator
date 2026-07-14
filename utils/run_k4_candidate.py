"""Run normal inference with the opt-in K=4 counter-RNG candidate installed."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.optimized.single.k8_runtime import (  # noqa: E402
    install_k8_candidate,
)


def main() -> None:
    with install_k8_candidate(block_size=4):
        runpy.run_path(str(REPO_ROOT / "inference.py"), run_name="__main__")


if __name__ == "__main__":
    main()
