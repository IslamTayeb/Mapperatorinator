from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_GUARDRAILS = (
    "batch_size_1",
    "fp32_hidden_states",
    "has_static_cache_layers",
    "cross_kv_expected_reused",
    "native_candidate_must_preserve_cache_write",
    "native_candidate_must_match_output",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fail(failures: list[str], path: str, message: str) -> None:
    failures.append(f"{path}: {message}")


def _warn(warnings: list[str], path: str, message: str) -> None:
    warnings.append(f"{path}: {message}")


def _shape(tensor_meta: dict[str, Any] | None) -> list[int] | None:
    shape = tensor_meta.get("shape") if isinstance(tensor_meta, dict) else None
    if not isinstance(shape, list):
        return None
    return [int(item) for item in shape]


def _stride(tensor_meta: dict[str, Any] | None) -> list[int] | None:
    stride = tensor_meta.get("stride") if isinstance(tensor_meta, dict) else None
    if not isinstance(stride, list):
        return None
    return [int(item) for item in stride]


def _check_tensor(
        tensor_meta: dict[str, Any] | None,
        *,
        path: str,
        failures: list[str],
        expected_dtype: str | None = None,
        expected_shape: list[int] | None = None,
        require_cuda: bool = True,
        require_contiguous: bool | None = None,
        require_last_stride_one: bool = True,
) -> list[int] | None:
    if not isinstance(tensor_meta, dict):
        _fail(failures, path, "missing tensor metadata")
        return None
    shape = _shape(tensor_meta)
    if shape is None:
        _fail(failures, path, "missing shape")
    elif expected_shape is not None and shape != expected_shape:
        _fail(failures, path, f"shape {shape} != expected {expected_shape}")
    dtype = tensor_meta.get("dtype")
    if expected_dtype is not None and dtype != expected_dtype:
        _fail(failures, path, f"dtype {dtype!r} != expected {expected_dtype!r}")
    if require_cuda and tensor_meta.get("is_cuda") is not True:
        _fail(failures, path, "expected CUDA tensor")
    if require_contiguous is not None and bool(tensor_meta.get("is_contiguous")) is not require_contiguous:
        _fail(failures, path, f"is_contiguous={tensor_meta.get('is_contiguous')} != {require_contiguous}")
    stride = _stride(tensor_meta)
    if require_last_stride_one and stride is not None and stride and stride[-1] != 1:
        _fail(failures, path, f"last stride {stride[-1]} != 1")
    return shape


def _check_linear(
        module_meta: dict[str, Any] | None,
        *,
        path: str,
        failures: list[str],
        in_features: int,
        out_features: int,
        expected_dtype: str = "torch.float32",
) -> None:
    if not isinstance(module_meta, dict):
        _fail(failures, path, "missing Linear metadata")
        return
    if module_meta.get("type") != "Linear":
        _fail(failures, path, f"type {module_meta.get('type')!r} != 'Linear'")
    if int(module_meta.get("in_features", -1)) != in_features:
        _fail(failures, path, f"in_features {module_meta.get('in_features')} != {in_features}")
    if int(module_meta.get("out_features", -1)) != out_features:
        _fail(failures, path, f"out_features {module_meta.get('out_features')} != {out_features}")
    _check_tensor(
        module_meta.get("weight"),
        path=f"{path}.weight",
        failures=failures,
        expected_dtype=expected_dtype,
        expected_shape=[out_features, in_features],
        require_contiguous=True,
    )
    if module_meta.get("has_bias"):
        _check_tensor(
            module_meta.get("bias"),
            path=f"{path}.bias",
            failures=failures,
            expected_dtype=expected_dtype,
            expected_shape=[out_features],
            require_contiguous=True,
        )


def _check_norm(
        module_meta: dict[str, Any] | None,
        *,
        path: str,
        failures: list[str],
        model_dim: int,
        expected_dtype: str = "torch.float32",
) -> None:
    if not isinstance(module_meta, dict):
        _fail(failures, path, "missing norm metadata")
        return
    if module_meta.get("type") != "RMSNorm":
        _fail(failures, path, f"type {module_meta.get('type')!r} != 'RMSNorm'")
    if module_meta.get("normalized_shape") != [model_dim]:
        _fail(failures, path, f"normalized_shape {module_meta.get('normalized_shape')} != {[model_dim]}")
    _check_tensor(
        module_meta.get("weight"),
        path=f"{path}.weight",
        failures=failures,
        expected_dtype=expected_dtype,
        expected_shape=[model_dim],
        require_contiguous=True,
    )


def _check_cache_layer(
        layer_meta: dict[str, Any] | None,
        *,
        path: str,
        failures: list[str],
        heads: int,
        cache_len: int,
        head_dim: int,
        expected_dtype: str = "torch.float32",
) -> None:
    if not isinstance(layer_meta, dict):
        _fail(failures, path, "missing cache layer metadata")
        return
    if layer_meta.get("type") != "StaticLayer":
        _fail(failures, path, f"type {layer_meta.get('type')!r} != 'StaticLayer'")
    if layer_meta.get("is_initialized") is not True:
        _fail(failures, path, "cache layer is not initialized")
    expected_shape = [1, heads, cache_len, head_dim]
    for name in ("keys", "values"):
        _check_tensor(
            layer_meta.get(name),
            path=f"{path}.{name}",
            failures=failures,
            expected_dtype=expected_dtype,
            expected_shape=expected_shape,
            require_contiguous=True,
        )


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value)


def _check_tensor_fingerprint(
        fingerprint: dict[str, Any] | None,
        *,
        path: str,
        failures: list[str],
        expected_dtype: str,
        expected_shape: list[int],
) -> None:
    if not isinstance(fingerprint, dict):
        _fail(failures, path, "missing tensor fingerprint")
        return
    if not _is_sha256(fingerprint.get("sha256")):
        _fail(failures, f"{path}.sha256", "missing or invalid SHA256")
    num_bytes = fingerprint.get("num_bytes")
    expected_bytes = expected_shape[0] * expected_shape[1] * expected_shape[2] * expected_shape[3] * 4
    if not isinstance(num_bytes, int) or num_bytes != expected_bytes:
        _fail(failures, f"{path}.num_bytes", f"expected {expected_bytes}, got {num_bytes!r}")
    if fingerprint.get("hash_layout") != "contiguous_tensor_bytes":
        _fail(failures, f"{path}.hash_layout", "expected contiguous_tensor_bytes")
    if fingerprint.get("hash_dtype") != expected_dtype:
        _fail(failures, f"{path}.hash_dtype", f"expected {expected_dtype!r}")
    _check_tensor(
        fingerprint.get("tensor"),
        path=f"{path}.tensor",
        failures=failures,
        expected_dtype=expected_dtype,
        expected_shape=expected_shape,
        require_cuda=True,
        require_contiguous=None,
    )


def _check_cache_write_fingerprint(
        fingerprint: dict[str, Any] | None,
        *,
        path: str,
        failures: list[str],
        report_cache_position: list[int] | None,
        active_prefix_len: int,
        self_cache_len: int,
        heads: int,
        head_dim: int,
        expected_dtype: str = "torch.float32",
) -> None:
    if not isinstance(fingerprint, dict):
        _fail(failures, path, "missing cache-write fingerprint")
        return
    if fingerprint.get("schema_version") != 1:
        _fail(failures, f"{path}.schema_version", f"expected 1, got {fingerprint.get('schema_version')!r}")
    if fingerprint.get("available") is not True:
        _fail(failures, f"{path}.available", f"expected true, got {fingerprint.get('available')!r}")
        return
    cache_position = fingerprint.get("cache_position")
    if not isinstance(cache_position, int):
        _fail(failures, f"{path}.cache_position", f"expected int, got {cache_position!r}")
    elif report_cache_position is not None and report_cache_position != [cache_position]:
        _fail(
            failures,
            f"{path}.cache_position",
            f"{cache_position} does not match report cache_position {report_cache_position}",
        )
    if fingerprint.get("active_prefix_length") != active_prefix_len:
        _fail(
            failures,
            f"{path}.active_prefix_length",
            f"{fingerprint.get('active_prefix_length')} != {active_prefix_len}",
        )
    if fingerprint.get("max_cache_len") != self_cache_len:
        _fail(failures, f"{path}.max_cache_len", f"{fingerprint.get('max_cache_len')} != {self_cache_len}")
    if fingerprint.get("position_within_cache") is not True:
        _fail(failures, f"{path}.position_within_cache", "expected true")
    if fingerprint.get("position_within_active_prefix") is not True:
        _fail(failures, f"{path}.position_within_active_prefix", "expected true")
    if fingerprint.get("slot_shape_contract") != "[batch, heads, 1, head_dim]":
        _fail(failures, f"{path}.slot_shape_contract", "unexpected slot shape contract")
    expected_slot_shape = [1, heads, 1, head_dim]
    _check_tensor_fingerprint(
        fingerprint.get("keys"),
        path=f"{path}.keys",
        failures=failures,
        expected_dtype=expected_dtype,
        expected_shape=expected_slot_shape,
    )
    _check_tensor_fingerprint(
        fingerprint.get("values"),
        path=f"{path}.values",
        failures=failures,
        expected_dtype=expected_dtype,
        expected_shape=expected_slot_shape,
    )


def _check_candidate_cache_write_checks(
        signature: str,
        signature_report: dict[str, Any],
        *,
        failures: list[str],
) -> None:
    results = signature_report.get("results")
    if not isinstance(results, dict):
        _fail(failures, f"{signature}.results", "missing results for candidate cache-write checks")
        return
    required_variants = [
        name
        for name in (
            "repo_decoder_layer",
            "self_attn_residual_segment",
            "manual_decoder_runtime_island",
            "compiled_decoder_layer",
            "native_self_cross_prefix_warp2",
            "native_self_cross_prefix_warp4",
            "native_self_cross_prefix_warp8",
            "native_decoder_layer_mlp_tail_warp2",
            "native_decoder_layer_mlp_tail_warp4",
            "native_decoder_layer_mlp_tail_warp8",
            "native_cross_mlp_tail_warp2",
            "native_cross_mlp_tail_warp4",
            "native_cross_mlp_tail_warp8",
        )
        if name in results
    ]
    if not required_variants:
        _fail(failures, f"{signature}.results", "no cache-writing variants present")
        return
    for variant in required_variants:
        variant_result = results.get(variant)
        if not isinstance(variant_result, dict):
            _fail(failures, f"{signature}.results.{variant}", "missing variant result")
            continue
        cache_check = variant_result.get("cache_write_check")
        if not isinstance(cache_check, dict):
            _fail(failures, f"{signature}.results.{variant}.cache_write_check", "missing cache-write check")
            continue
        for key in ("checked", "pass", "matches_expected", "keys_match", "values_match", "output_allclose"):
            if cache_check.get(key) is not True:
                _fail(
                    failures,
                    f"{signature}.results.{variant}.cache_write_check.{key}",
                    f"expected true, got {cache_check.get(key)!r}",
                )
        actual = cache_check.get("actual")
        if not isinstance(actual, dict) or actual.get("available") is not True:
            _fail(failures, f"{signature}.results.{variant}.cache_write_check.actual", "missing actual fingerprint")
        else:
            for name in ("keys", "values"):
                fingerprint = actual.get(name)
                if not isinstance(fingerprint, dict) or not _is_sha256(fingerprint.get("sha256")):
                    _fail(
                        failures,
                        f"{signature}.results.{variant}.cache_write_check.actual.{name}.sha256",
                        "missing or invalid SHA256",
                    )


def _validate_signature(
        signature: str,
        signature_report: dict[str, Any],
        *,
        require_cache_write_fingerprint: bool,
        require_candidate_cache_write_checks: bool,
        report_cache_position: list[int] | None,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    abi = signature_report.get("native_decoder_layer_abi")
    if not isinstance(abi, dict):
        _fail(failures, signature, "missing native_decoder_layer_abi")
        return {"signature": signature, "pass": False, "failures": failures, "warnings": warnings}

    if abi.get("schema_version") != 1:
        _fail(failures, f"{signature}.schema_version", f"expected 1, got {abi.get('schema_version')!r}")

    guardrails = abi.get("guardrails") or {}
    for name in REQUIRED_GUARDRAILS:
        if guardrails.get(name) is not True:
            _fail(failures, f"{signature}.guardrails.{name}", f"expected true, got {guardrails.get(name)!r}")

    layer = abi.get("layer") or {}
    model_dim = int(layer.get("embed_dim", 0) or 0)
    active_prefix_len = int(layer.get("active_prefix_length", 0) or 0)
    if model_dim <= 0:
        _fail(failures, f"{signature}.layer.embed_dim", "missing or non-positive")
    if active_prefix_len <= 0:
        _fail(failures, f"{signature}.layer.active_prefix_length", "missing or non-positive")
    if layer.get("training") is not False:
        _fail(failures, f"{signature}.layer.training", "expected false")
    if float(layer.get("dropout", 0.0) or 0.0) != 0.0:
        _warn(warnings, f"{signature}.layer.dropout", "nonzero dropout is okay only because eval mode disables it")
    if float(layer.get("activation_dropout", 0.0) or 0.0) != 0.0:
        _warn(warnings, f"{signature}.layer.activation_dropout", "nonzero activation dropout is okay only because eval mode disables it")

    runtime_inputs = abi.get("runtime_inputs") or {}
    hidden_shape = _check_tensor(
        runtime_inputs.get("hidden_states"),
        path=f"{signature}.runtime_inputs.hidden_states",
        failures=failures,
        expected_dtype="torch.float32",
        expected_shape=[1, 1, model_dim] if model_dim > 0 else None,
        require_contiguous=True,
    )
    _check_tensor(
        runtime_inputs.get("output"),
        path=f"{signature}.runtime_inputs.output",
        failures=failures,
        expected_dtype="torch.float32",
        expected_shape=hidden_shape,
        require_contiguous=True,
    )
    encoder_shape = _check_tensor(
        runtime_inputs.get("encoder_hidden_states"),
        path=f"{signature}.runtime_inputs.encoder_hidden_states",
        failures=failures,
        expected_dtype="torch.float32",
        require_contiguous=True,
    )
    encoder_len = encoder_shape[1] if isinstance(encoder_shape, list) and len(encoder_shape) == 3 else None
    if isinstance(encoder_shape, list) and (encoder_shape[0] != 1 or encoder_shape[2] != model_dim):
        _fail(failures, f"{signature}.runtime_inputs.encoder_hidden_states", "expected [1, encoder_len, model_dim]")
    mask_shape = _check_tensor(
        runtime_inputs.get("attention_mask"),
        path=f"{signature}.runtime_inputs.attention_mask",
        failures=failures,
        expected_dtype="torch.float32",
        require_contiguous=True,
    )
    if isinstance(mask_shape, list):
        if len(mask_shape) != 4 or mask_shape[:3] != [1, 1, 1]:
            _fail(failures, f"{signature}.runtime_inputs.attention_mask", "expected shape [1, 1, 1, cache_len]")
        if active_prefix_len > 0 and mask_shape[-1] < active_prefix_len:
            _fail(failures, f"{signature}.runtime_inputs.attention_mask", "mask shorter than active prefix")
    _check_tensor(
        runtime_inputs.get("cache_position"),
        path=f"{signature}.runtime_inputs.cache_position",
        failures=failures,
        expected_dtype="torch.int64",
        expected_shape=[1],
        require_contiguous=True,
    )

    modules = abi.get("modules") or {}
    for name in ("self_attn_layer_norm", "cross_attn_layer_norm", "final_layer_norm"):
        _check_norm(modules.get(name), path=f"{signature}.modules.{name}", failures=failures, model_dim=model_dim)

    self_attn = modules.get("self_attn") or {}
    cross_attn = modules.get("cross_attn") or {}
    heads = int(self_attn.get("num_heads", 0) or 0)
    head_dim = int(self_attn.get("head_dim", 0) or 0)
    if heads <= 0 or head_dim <= 0 or heads * head_dim != model_dim:
        _fail(failures, f"{signature}.modules.self_attn", "num_heads * head_dim must equal model_dim")
    if int(self_attn.get("all_head_size", 0) or 0) != model_dim:
        _fail(failures, f"{signature}.modules.self_attn.all_head_size", "expected model_dim")
    if int(cross_attn.get("num_heads", 0) or 0) != heads or int(cross_attn.get("head_dim", 0) or 0) != head_dim:
        _fail(failures, f"{signature}.modules.cross_attn", "head layout must match self attention")
    if int(cross_attn.get("all_head_size", 0) or 0) != model_dim:
        _fail(failures, f"{signature}.modules.cross_attn.all_head_size", "expected model_dim")

    _check_linear(self_attn.get("Wqkv"), path=f"{signature}.modules.self_attn.Wqkv", failures=failures, in_features=model_dim, out_features=3 * model_dim)
    _check_linear(self_attn.get("Wo"), path=f"{signature}.modules.self_attn.Wo", failures=failures, in_features=model_dim, out_features=model_dim)
    _check_linear(cross_attn.get("Wq"), path=f"{signature}.modules.cross_attn.Wq", failures=failures, in_features=model_dim, out_features=model_dim)
    _check_linear(cross_attn.get("Wkv"), path=f"{signature}.modules.cross_attn.Wkv", failures=failures, in_features=model_dim, out_features=2 * model_dim)
    _check_linear(cross_attn.get("Wo"), path=f"{signature}.modules.cross_attn.Wo", failures=failures, in_features=model_dim, out_features=model_dim)

    fc1 = modules.get("fc1") or {}
    ffn_dim = int(fc1.get("out_features", 0) or 0)
    _check_linear(fc1, path=f"{signature}.modules.fc1", failures=failures, in_features=model_dim, out_features=ffn_dim)
    _check_linear(modules.get("fc2"), path=f"{signature}.modules.fc2", failures=failures, in_features=ffn_dim, out_features=model_dim)

    cache = abi.get("cache") or {}
    if cache.get("type") != "MapperatorinatorCache":
        _fail(failures, f"{signature}.cache.type", f"expected MapperatorinatorCache, got {cache.get('type')!r}")
    if cache.get("self_attention_cache_type") != "StaticCache":
        _fail(failures, f"{signature}.cache.self_attention_cache_type", "expected StaticCache")
    if cache.get("cross_attention_cache_type") != "StaticCache":
        _fail(failures, f"{signature}.cache.cross_attention_cache_type", "expected StaticCache")
    if cache.get("cross_attention_cache_updated_for_layer") is not True:
        _fail(failures, f"{signature}.cache.cross_attention_cache_updated_for_layer", "expected true")
    self_cache_shape = _shape((cache.get("self_layer") or {}).get("keys"))
    self_cache_len = self_cache_shape[2] if isinstance(self_cache_shape, list) and len(self_cache_shape) == 4 else None
    if not isinstance(self_cache_len, int):
        _fail(failures, f"{signature}.cache.self_layer.keys", "cannot infer self cache length")
        self_cache_len = 0
    if active_prefix_len > 0 and self_cache_len < active_prefix_len:
        _fail(failures, f"{signature}.cache.self_layer.keys", "self cache shorter than active prefix")
    _check_cache_layer(
        cache.get("self_layer"),
        path=f"{signature}.cache.self_layer",
        failures=failures,
        heads=heads,
        cache_len=self_cache_len,
        head_dim=head_dim,
    )
    if not isinstance(encoder_len, int):
        encoder_len = 0
    _check_cache_layer(
        cache.get("cross_layer"),
        path=f"{signature}.cache.cross_layer",
        failures=failures,
        heads=heads,
        cache_len=encoder_len,
        head_dim=head_dim,
    )
    if require_cache_write_fingerprint:
        _check_cache_write_fingerprint(
            abi.get("cache_write_fingerprint"),
            path=f"{signature}.cache_write_fingerprint",
            failures=failures,
            report_cache_position=report_cache_position,
            active_prefix_len=active_prefix_len,
            self_cache_len=self_cache_len,
            heads=heads,
            head_dim=head_dim,
        )
    if require_candidate_cache_write_checks:
        _check_candidate_cache_write_checks(signature, signature_report, failures=failures)

    return {
        "signature": signature,
        "pass": not failures,
        "failures": failures,
        "warnings": warnings,
        "member_count": signature_report.get("member_count"),
        "model_dim": model_dim,
        "ffn_dim": ffn_dim,
        "heads": heads,
        "head_dim": head_dim,
        "active_prefix_length": active_prefix_len,
        "encoder_len": encoder_len,
        "self_cache_len": self_cache_len,
        "cache_write_fingerprint": (
            isinstance(abi.get("cache_write_fingerprint"), dict)
            and abi.get("cache_write_fingerprint", {}).get("available") is True
        ),
    }


def validate_decoder_layer_abi(
        report: dict[str, Any],
        *,
        require_logits_replay: bool = True,
        require_cache_write_fingerprint: bool = False,
        require_candidate_cache_write_checks: bool = False,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    if require_logits_replay:
        if report.get("pass") is not True:
            _fail(failures, "report.pass", f"expected true, got {report.get('pass')!r}")
        if report.get("logits_replay_allclose") is not True:
            _fail(
                failures,
                "report.logits_replay_allclose",
                f"expected true, got {report.get('logits_replay_allclose')!r}",
            )
        max_abs = report.get("logits_replay_max_abs")
        if not isinstance(max_abs, (int, float)) or float(max_abs) > 1e-4:
            _fail(failures, "report.logits_replay_max_abs", f"expected <= 1e-4, got {max_abs!r}")

    signatures = report.get("signature_reports")
    if not isinstance(signatures, dict) or not signatures:
        _fail(failures, "signature_reports", "missing or empty")
        signature_results: list[dict[str, Any]] = []
    else:
        report_cache_position = report.get("cache_position")
        if report_cache_position is not None:
            if not (
                    isinstance(report_cache_position, list)
                    and len(report_cache_position) == 1
                    and isinstance(report_cache_position[0], int)
            ):
                _fail(failures, "report.cache_position", "expected one integer cache position")
                report_cache_position = None
        signature_results = [
            _validate_signature(
                signature,
                signature_report,
                require_cache_write_fingerprint=require_cache_write_fingerprint,
                require_candidate_cache_write_checks=require_candidate_cache_write_checks,
                report_cache_position=report_cache_position,
            )
            for signature, signature_report in sorted(signatures.items())
            if isinstance(signature_report, dict)
        ]
        for item in signature_results:
            failures.extend(item["failures"])
            warnings.extend(item["warnings"])

    result = {
        "pass": not failures,
        "failures": failures,
        "warnings": warnings,
        "signature_count": len(signature_results),
        "require_cache_write_fingerprint": bool(require_cache_write_fingerprint),
        "require_candidate_cache_write_checks": bool(require_candidate_cache_write_checks),
        "signatures": signature_results,
    }
    return result


def print_summary(result: dict[str, Any]) -> None:
    print("Decoder Layer ABI Validation")
    print(f"  pass: {result['pass']}")
    print(f"  signatures: {result['signature_count']}")
    print(f"  failures: {len(result['failures'])}")
    print(f"  warnings: {len(result['warnings'])}")
    for signature in result["signatures"]:
        print(
            "  "
            f"{signature['signature']}: pass={signature['pass']} "
            f"members={signature.get('member_count')} "
            f"D={signature.get('model_dim')} "
            f"ffn={signature.get('ffn_dim')} "
            f"heads={signature.get('heads')} "
            f"prefix={signature.get('active_prefix_length')} "
            f"encoder={signature.get('encoder_len')} "
            f"cache_fingerprint={signature.get('cache_write_fingerprint')}"
        )
    if result["failures"]:
        print("Failures:")
        for failure in result["failures"]:
            print(f"  - {failure}")
    if result["warnings"]:
        print("Warnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate native_decoder_layer_abi metadata emitted by "
            "profile_decode_decoder_layer_island.py. Diagnostic only; not a speed claim."
        )
    )
    parser.add_argument("report", type=Path)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--allow-missing-logits-replay", action="store_true")
    parser.add_argument(
        "--require-cache-write-fingerprint",
        action="store_true",
        help="Require q_len=1 self-cache K/V slot fingerprints in native_decoder_layer_abi.",
    )
    parser.add_argument(
        "--require-candidate-cache-write-checks",
        action="store_true",
        help="Require cache-writing decoder-layer benchmark variants to reproduce the reference K/V slot.",
    )
    args = parser.parse_args()

    result = validate_decoder_layer_abi(
        _load_json(args.report),
        require_logits_replay=not args.allow_missing_logits_replay,
        require_cache_write_fingerprint=args.require_cache_write_fingerprint,
        require_candidate_cache_write_checks=args.require_candidate_cache_write_checks,
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_summary(result)
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
