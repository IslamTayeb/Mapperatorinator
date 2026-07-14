from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from osuT5.osuT5.inference.optimized.scout.int8_mlp import (  # noqa: E402
    preload_int8_mlp_extension,
)


def compile_extension() -> dict[str, object]:
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only compile smoke must not receive a visible CUDA device")
    started = time.perf_counter()
    extension = preload_int8_mlp_extension()
    elapsed = time.perf_counter() - started
    bindings = (
        "int8_weight_mlp_residual",
        "int8_weight_rmsnorm_linear",
        "int8_weight_linear_residual",
    )
    missing = [name for name in bindings if not callable(getattr(extension, name, None))]
    if missing:
        raise RuntimeError(f"compiled extension is missing bindings: {missing}")
    module_path = Path(getattr(extension, "__file__", ""))
    if not module_path.is_file() or module_path.stat().st_size <= 0:
        raise RuntimeError(f"compiled extension path is missing or empty: {module_path}")
    return {
        "schema_version": 1,
        "cuda_visible": False,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "extension_path": str(module_path),
        "extension_bytes": module_path.stat().st_size,
        "exported_bindings": list(bindings),
        "compile_load_seconds": elapsed,
        "pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=Path, required=True)
    parser.add_argument("--text-path", type=Path, required=True)
    args = parser.parse_args()
    result = compile_extension()
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    text = "\n".join(f"{key}={value}" for key, value in result.items()) + "\n"
    args.text_path.parent.mkdir(parents=True, exist_ok=True)
    args.text_path.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
