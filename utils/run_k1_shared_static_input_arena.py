"""Run normal inference with the opt-in K1 shared-input arena installed."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from osuT5.osuT5.inference.optimized.single.shared_static_input_arena import (
        install_shared_static_input_arena_candidate,
    )
    from inference import main as inference_main

    with install_shared_static_input_arena_candidate():
        inference_main()


if __name__ == "__main__":
    main()
