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
    _CPP_SOURCE,
    _CUDA_SOURCE,
    preload_int8_mlp_extension,
    symmetric_int8_per_row,
)
from utils.profile_int8_mlp_component import summarize_component  # noqa: E402


def compile_extension() -> dict[str, object]:
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only compile smoke must not receive a visible CUDA device")
    started = time.perf_counter()
    extension = preload_int8_mlp_extension()
    elapsed = time.perf_counter() - started
    exported = getattr(extension, "int8_weight_mlp_residual", None)
    if not callable(exported):
        raise RuntimeError("compiled extension is missing int8_weight_mlp_residual binding")
    module_path = Path(getattr(extension, "__file__", ""))
    if not module_path.is_file() or module_path.stat().st_size <= 0:
        raise RuntimeError(f"compiled extension path is missing or empty: {module_path}")
    if "PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)" not in _CPP_SOURCE:
        raise RuntimeError("INT8 MLP extension is missing its explicit manual binding")
    if "&int8_weight_mlp_residual" not in _CPP_SOURCE:
        raise RuntimeError("INT8 MLP extension binding does not reference its CUDA export")
    required_cuda_contract = (
        "const char4* weight4",
        "float sum = 0.0f",
        "sum *= scale[out]",
        "input activations must remain contiguous FP32",
        "weight must be contiguous INT8",
    )
    missing_contract = [value for value in required_cuda_contract if value not in _CUDA_SOURCE]
    if missing_contract:
        raise RuntimeError(f"INT8 MLP CUDA source is missing contract markers: {missing_contract}")
    quantized, scale = symmetric_int8_per_row(
        torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [-2.0, -1.0, 1.0, 2.0]],
            dtype=torch.float32,
        )
    )
    if quantized.dtype != torch.int8 or scale.dtype != torch.float32:
        raise RuntimeError("per-row quantizer returned the wrong storage dtypes")
    if int(quantized[1].min()) != -127 or int(quantized[1].max()) != 127:
        raise RuntimeError("per-row quantizer did not use the symmetric INT8 range")
    synthetic_entry = {
        "decode_replays": 100,
        "ms_per_call": {"fp32": 0.3, "fp16_weight": 0.28, "int8_weight": 0.18},
        "checks": {"finite": True, "repeat_deterministic": True},
    }
    summary = summarize_component({"128": synthetic_entry}, total_replays=100)
    if not summary["promotion_pass"] or summary["int8_vs_fp16_local_speedup"] < 1.4:
        raise RuntimeError("component sizing gate contract failed its synthetic control")
    return {
        "schema_version": 1,
        "cuda_visible": False,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "extension_path": str(module_path),
        "extension_bytes": module_path.stat().st_size,
        "exported_binding": "int8_weight_mlp_residual",
        "manual_binding_verified": True,
        "storage_contract_verified": True,
        "quantization_contract_verified": True,
        "sizing_gate_contract_verified": True,
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
