"""Thin CLI for the single approved v32-mini speculative feasibility scout."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import transformers  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

from inference import compile_args, load_model_with_server, setup_inference_environment  # noqa: E402
from osuT5.osuT5.inference.optimized.speculative.mini_draft_gpu import (  # noqa: E402
    MiniDraftGpuScoutConfig,
    run_mini_draft_gpu_scout,
)
from utils.verify_one_token_decode import (  # noqa: E402
    _assert_supported_probe,
    _build_probe_inputs,
    _load_args,
    _move_kwargs_for_model,
)


TARGET_REPO = "OliBomby/Mapperatorinator-v32"
MINI_REPO = "OliBomby/Mapperatorinator-v32-mini"
TARGET_REVISION = "74f22583400d259bf424819e11027c17933efe54"
MINI_REVISION = "7807f0dc70cab671be012e1f5ddf945b0b8b7278"
EXPECTED_GAMEMODE = 0
EXPECTED_GAMEMODE0_TOKENIZER_SHA256 = (
    "6b98be0fc04a95a9e9d4feb8e8b67cc48728a6667e3091dcd5cc528baeca18bd"
)
CAPACITY_CONFIG_KEYS = {
    "_name_or_path",
    "backbone_model_name",
    "d_model",
    "decoder_attention_heads",
    "decoder_ffn_dim",
    "decoder_layers",
    "encoder_attention_heads",
    "encoder_ffn_dim",
    "encoder_layers",
    "hidden_size",
    "num_attention_heads",
    "num_hidden_layers",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one sampled SALVALAI target transcript against greedy v32-mini K4 proposals. "
            "This is a verifier/cost scout only and self-rejects unless the rebuild-inclusive "
            "finite-window projection strictly exceeds 5%."
        )
    )
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--sequence-index", type=int, default=9)
    parser.add_argument("--speculation-k", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--target-revision", default=TARGET_REVISION)
    parser.add_argument("--mini-revision", default=MINI_REVISION)
    parser.add_argument("--encoder-warmup", type=int, default=2)
    parser.add_argument("--encoder-timing-iters", type=int, default=8)
    parser.add_argument("--steady-cache-warmup", type=int, default=2)
    parser.add_argument("--steady-cache-timing-iters", type=int, default=8)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides; only the audio/output paths vary.")
    return parser.parse_args()


def _validate_approved_contract(cli: argparse.Namespace, args) -> None:
    expected_cli = {
        "config_name": "profile_salvalai_smoke15",
        "sequence_index": 9,
        "speculation_k": 4,
        "max_new_tokens": 256,
        "target_revision": TARGET_REVISION,
        "mini_revision": MINI_REVISION,
    }
    mismatches = {
        name: {"expected": expected, "actual": getattr(cli, name)}
        for name, expected in expected_cli.items()
        if getattr(cli, name) != expected
    }
    expected_runtime = {
        "model_path": TARGET_REPO,
        "seed": 12345,
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "inference_engine": "v32",
        "optimized_inference_mode": "single",
        "gamemode": EXPECTED_GAMEMODE,
        "use_server": False,
        "parallel": False,
        "cfg_scale": 1.0,
        "num_beams": 1,
        "do_sample": True,
        "temperature": 0.9,
        "top_p": 0.9,
        "top_k": 0,
        "inference_generation_compile": False,
        "inference_active_prefix_decode_loop": True,
        "inference_stateful_monotonic_logits_processor": True,
        "start_time": 71000,
        "end_time": 86000,
    }
    mismatches.update({
        name: {"expected": expected, "actual": getattr(args, name)}
        for name, expected in expected_runtime.items()
        if getattr(args, name) != expected
    })
    if mismatches:
        raise ValueError(f"The approved v32-mini scout contract changed: {mismatches}")
    _assert_supported_probe(args)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_capacity_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_capacity_fields(item)
            for key, item in value.items()
            if key not in CAPACITY_CONFIG_KEYS
        }
    if isinstance(value, list):
        return [_strip_capacity_fields(item) for item in value]
    return value


def _snapshot_artifact_manifest(repo_id: str, revision: str, *, gamemode: int) -> dict[str, Any]:
    subfolder = f"gamemode={gamemode}"
    snapshot = Path(snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=[
            f"{subfolder}/config.json",
            f"{subfolder}/generation_config.json",
            f"{subfolder}/model.safetensors",
            f"{subfolder}/tokenizer.json",
        ],
    ))
    resolved = snapshot / subfolder
    required = {
        name: resolved / name
        for name in ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json")
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Pinned model snapshot is missing required artifacts: {missing}")
    hashes = {name: _sha256_file(path) for name, path in required.items()}
    sizes = {name: path.stat().st_size for name, path in required.items()}
    return {
        "repo_id": repo_id,
        "requested_revision": revision,
        "snapshot_revision": snapshot.name,
        "snapshot_path": str(snapshot),
        "resolved_subfolder": subfolder,
        "resolved_path": str(resolved),
        "sha256": hashes,
        "size_bytes": sizes,
        "config": json.loads(required["config.json"].read_text(encoding="utf-8")),
    }


def _artifact_compatibility(target: dict[str, Any], mini: dict[str, Any]) -> dict[str, Any]:
    target_common = _strip_capacity_fields(copy.deepcopy(target["config"]))
    mini_common = _strip_capacity_fields(copy.deepcopy(mini["config"]))
    tokenizer_match = (
        target["sha256"]["tokenizer.json"]
        == mini["sha256"]["tokenizer.json"]
        == EXPECTED_GAMEMODE0_TOKENIZER_SHA256
    )
    generation_match = (
        target["sha256"]["generation_config.json"]
        == mini["sha256"]["generation_config.json"]
    )
    config_common_match = target_common == mini_common
    return {
        "pass": tokenizer_match and generation_match and config_common_match,
        "tokenizer_sha256_match_expected": tokenizer_match,
        "generation_config_sha256_match": generation_match,
        "non_capacity_config_match": config_common_match,
        "target_capacity": {
            "backbone_model_name": target["config"].get("backbone_model_name"),
            "hidden_size": target["config"].get("hidden_size"),
            "num_hidden_layers": target["config"].get("num_hidden_layers"),
            "num_attention_heads": target["config"].get("num_attention_heads"),
        },
        "mini_capacity": {
            "backbone_model_name": mini["config"].get("backbone_model_name"),
            "hidden_size": mini["config"].get("hidden_size"),
            "num_hidden_layers": mini["config"].get("num_hidden_layers"),
            "num_attention_heads": mini["config"].get("num_attention_heads"),
        },
    }


def _public_artifact_manifest(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "config"}


def main() -> int:
    cli = _parse_args()
    started = time.perf_counter()
    metadata: dict[str, Any] = {
        "config_name": cli.config_name,
        "sequence_index": cli.sequence_index,
        "speculation_k": cli.speculation_k,
        "max_new_tokens": cli.max_new_tokens,
        "target_repo": TARGET_REPO,
        "target_revision": cli.target_revision,
        "mini_repo": MINI_REPO,
        "mini_revision": cli.mini_revision,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR"),
        "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
    }
    try:
        args = _load_args(cli.config_name, cli.overrides)
        _validate_approved_contract(cli, args)
        compile_args(args, verbose=False)
        setup_inference_environment(args.seed)

        target_artifact = _snapshot_artifact_manifest(
            TARGET_REPO,
            cli.target_revision,
            gamemode=args.gamemode,
        )
        mini_artifact = _snapshot_artifact_manifest(
            MINI_REPO,
            cli.mini_revision,
            gamemode=args.gamemode,
        )
        compatibility = _artifact_compatibility(target_artifact, mini_artifact)
        if not compatibility["pass"]:
            raise ValueError(f"Pinned target/mini artifacts are incompatible: {compatibility}")
        artifact_manifest = {
            "target": _public_artifact_manifest(target_artifact),
            "mini": _public_artifact_manifest(mini_artifact),
            "compatibility": compatibility,
        }

        target_model, target_tokenizer = load_model_with_server(
            Path(target_artifact["snapshot_path"]),
            args.train,
            args.device,
            max_batch_size=args.max_batch_size,
            use_server=False,
            precision=args.precision,
            attn_implementation=args.attn_implementation,
            lora_path=None,
            gamemode=args.gamemode,
            auto_select_gamemode_model=True,
            generation_compile=False,
        )
        mini_model, mini_tokenizer = load_model_with_server(
            Path(mini_artifact["snapshot_path"]),
            args.train,
            args.device,
            max_batch_size=args.max_batch_size,
            use_server=False,
            precision=args.precision,
            attn_implementation=args.attn_implementation,
            lora_path=None,
            gamemode=args.gamemode,
            auto_select_gamemode_model=True,
            generation_compile=False,
        )
        runtime_tokenizer_ids = {
            name: {
                "target": int(getattr(target_tokenizer, name)),
                "mini": int(getattr(mini_tokenizer, name)),
            }
            for name in ("pad_id", "sos_id", "eos_id")
        }
        if any(item["target"] != item["mini"] for item in runtime_tokenizer_ids.values()):
            raise ValueError(f"Runtime tokenizer special IDs differ: {runtime_tokenizer_ids}")
        artifact_manifest["runtime_tokenizer_special_ids"] = runtime_tokenizer_ids
        model_inputs = _move_kwargs_for_model(
            target_model,
            _build_probe_inputs(
                args,
                target_model,
                target_tokenizer,
                sequence_index=cli.sequence_index,
            ),
        )
        gate_config = MiniDraftGpuScoutConfig(
            speculation_k=cli.speculation_k,
            max_new_tokens=cli.max_new_tokens,
            encoder_warmup=cli.encoder_warmup,
            encoder_timing_iters=cli.encoder_timing_iters,
            steady_cache_warmup=cli.steady_cache_warmup,
            steady_cache_timing_iters=cli.steady_cache_timing_iters,
        )
        result = run_mini_draft_gpu_scout(
            args,
            target_model,
            target_tokenizer,
            mini_model,
            mini_tokenizer,
            model_inputs,
            gate_config,
            artifact_manifest=artifact_manifest,
        )
        metadata.update({
            "precision": args.precision,
            "attn_implementation": args.attn_implementation,
            "profile_sdpa_backend": args.profile_sdpa_backend,
            "seed": args.seed,
            "gamemode": args.gamemode,
            "start_time": args.start_time,
            "end_time": args.end_time,
            "cuda_device_name": torch.cuda.get_device_name(target_model.device),
            "cuda_device_capability": list(torch.cuda.get_device_capability(target_model.device)),
            "generation_compile_requested": args.inference_generation_compile,
        })
        exit_code = 0 if result["pass"] else 1
    except Exception as exc:
        traceback.print_exc()
        result = {
            "gate": "v32_mini_closed_loop_feasibility",
            "pass": False,
            "result_class": "bounded_v32_mini_feasibility_blocked",
            "blocker": {"type": type(exc).__name__, "message": str(exc)},
        }
        exit_code = 1
    metadata["wall_seconds"] = time.perf_counter() - started
    result["metadata"] = metadata
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
