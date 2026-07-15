from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


POSITIVE = "EXACT_CONDITIONAL_WHILE_FEASIBLE"
VALID_NEGATIVE = "VALID_NEGATIVE_CONDITIONAL_WHILE_RNG"
INVALID_CONTROL = "INVALID_ORDINARY_GRAPH_CONTROL"


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _boolean(root: dict[str, Any], name: str) -> bool:
    value = root.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _integer(root: dict[str, Any], name: str) -> int:
    value = root.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _samples(root: dict[str, Any], name: str, iterations: int) -> list[int]:
    value = root.get(name)
    if not isinstance(value, list) or len(value) != iterations:
        raise ValueError(f"{name} must contain exactly {iterations} samples")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{name} must contain only integer tokens")
    return value


def _sha(root: dict[str, Any], name: str) -> str:
    value = root.get(name)
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def analyze(payload: Any, *, probe_exit_code: int) -> tuple[dict[str, Any], int]:
    root = _object(payload, name="probe result")
    if root.get("backend") != "cuda_python_conditional_while_child_graphs":
        raise ValueError("unexpected probe backend")
    if root.get("rng_policy") != "torch_multinomial_philox":
        raise ValueError("unexpected RNG policy")
    if root.get("float32_matmul_precision") != "highest":
        raise ValueError("strict FP32 matmul precision was not highest")
    if root.get("cuda_matmul_allow_tf32") is not False:
        raise ValueError("CUDA matmul TF32 must be disabled")
    if root.get("cudnn_allow_tf32") is not False:
        raise ValueError("cuDNN TF32 must be disabled")
    if root.get("nvidia_tf32_override") != "0":
        raise ValueError("NVIDIA_TF32_OVERRIDE must be 0")
    if root.get("device_capability") not in ([7, 5], (7, 5)):
        raise ValueError("probe result is not from SM75")
    device_name = root.get("device_name")
    if not isinstance(device_name, str) or "2080 Ti" not in device_name:
        raise ValueError("probe result is not from an RTX 2080 Ti")

    iterations = _integer(root, "iterations")
    if not 2 <= iterations <= 32:
        raise ValueError("iterations must be between 2 and 32")
    eager = _samples(root, "eager_samples", iterations)
    ordinary = _samples(root, "ordinary_graph_samples", iterations)
    conditional = _samples(root, "conditional_while_samples", iterations)
    eager_sha = _sha(root, "eager_generator_state_sha256")
    ordinary_sha = _sha(root, "ordinary_graph_generator_state_sha256")
    conditional_sha = _sha(root, "conditional_generator_state_sha256")

    observed = {
        "ordinary_matches_eager": ordinary == eager,
        "ordinary_generator_matches_eager": ordinary_sha == eager_sha,
        "conditional_matches_ordinary": conditional == ordinary,
        "conditional_generator_matches_ordinary": conditional_sha == ordinary_sha,
    }
    for name, expected in observed.items():
        if _boolean(root, name) is not expected:
            raise ValueError(
                f"{name} disagrees with recorded samples or generator state"
            )
    expected_feasible = all(observed.values())
    if _boolean(root, "exact_conditional_while_feasible") is not expected_feasible:
        raise ValueError("exact_conditional_while_feasible is internally inconsistent")

    ordinary_control_pass = (
        observed["ordinary_matches_eager"]
        and observed["ordinary_generator_matches_eager"]
    )
    if not ordinary_control_pass:
        classification = INVALID_CONTROL
        analyzer_exit_code = 3
    elif expected_feasible:
        classification = POSITIVE
        analyzer_exit_code = 0
    else:
        classification = VALID_NEGATIVE
        analyzer_exit_code = 0

    expected_probe_exit = 0 if expected_feasible else 2
    if probe_exit_code != expected_probe_exit:
        raise ValueError(
            f"probe exit code {probe_exit_code} does not match result; "
            f"expected {expected_probe_exit}"
        )

    analysis = {
        "schema_version": 1,
        "classification": classification,
        "ordinary_control_pass": ordinary_control_pass,
        "exact_conditional_while_feasible": expected_feasible,
        "conditional_token_match": observed["conditional_matches_ordinary"],
        "conditional_generator_match": observed[
            "conditional_generator_matches_ordinary"
        ],
        "probe_exit_code": probe_exit_code,
        "iterations": iterations,
        "seed": _integer(root, "seed"),
        "result_sha256": hashlib.sha256(
            (json.dumps(root, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
    }
    return analysis, analyzer_exit_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--probe-exit-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    analysis, exit_code = analyze(payload, probe_exit_code=args.probe_exit_code)
    args.output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    args.text_output.write_text(
        "\n".join(
            (
                f"classification={analysis['classification']}",
                f"ordinary_control_pass={str(analysis['ordinary_control_pass']).lower()}",
                "exact_conditional_while_feasible="
                f"{str(analysis['exact_conditional_while_feasible']).lower()}",
                f"conditional_token_match={str(analysis['conditional_token_match']).lower()}",
                "conditional_generator_match="
                f"{str(analysis['conditional_generator_match']).lower()}",
                f"iterations={analysis['iterations']}",
                f"seed={analysis['seed']}",
                "",
            )
        )
    )
    print(json.dumps(analysis, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
