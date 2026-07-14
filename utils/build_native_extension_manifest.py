from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EXPECTED_EXTENSIONS = (
    "mapperatorinator_q1_attention",
    "mapperatorinator_native_decoder_layer",
    "mapperatorinator_weight_only_fp16_v1",
    "mapperatorinator_int8_mlp_scout_v1",
)


def build(output_path: Path) -> dict[str, object]:
    from osuT5.osuT5.inference.optimized.kernels import (
        decoder_layer,
        q1_attention,
        weight_only,
    )
    from osuT5.osuT5.inference.optimized.kernels.native_extension import (
        MANIFEST_ENV,
        loaded_extension_records,
        write_loaded_extension_manifest,
    )
    from osuT5.osuT5.inference.optimized.scout import int8_mlp

    if os.environ.get(MANIFEST_ENV):
        raise RuntimeError(f"unset {MANIFEST_ENV} before building extensions")
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout:
        raise RuntimeError("native extension manifest requires a clean worktree")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    started = time.perf_counter()
    modules = (
        q1_attention.preload_native_q1_attention(),
        decoder_layer.preload_native_decoder_layer(),
        weight_only.preload_weight_only_extension(),
        int8_mlp.preload_int8_mlp_extension(),
    )
    build_seconds = time.perf_counter() - started
    for expected, module in zip(EXPECTED_EXTENSIONS, modules, strict=True):
        if module.__name__ != expected:
            raise RuntimeError(
                f"native extension name mismatch: expected {expected}, got {module.__name__}"
            )
    manifest = write_loaded_extension_manifest(
        output_path,
        expected_names=EXPECTED_EXTENSIONS,
        source_commit=source_commit,
    )
    return {
        "manifest": str(output_path.resolve()),
        "source_commit": source_commit,
        "build_seconds": build_seconds,
        "extensions": loaded_extension_records(),
        "packaged": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.manifest)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
