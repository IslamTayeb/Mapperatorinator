from unittest.mock import patch

import inference
from config import InferenceConfig
from inference import load_model_with_engine, validate_reserved_runtime_flags


def _assert_raises(error_type, message_fragment, callback):
    try:
        callback()
    except error_type as exc:
        assert message_fragment in str(exc)
    else:
        raise AssertionError(f"Expected {error_type.__name__}")


def test_inference_engine_defaults_preserve_v32():
    args = InferenceConfig()

    assert args.inference_engine == "v32"
    assert args.optimized_inference_mode == "single"
    validate_reserved_runtime_flags(args)


def test_invalid_inference_engine_fails_loudly():
    args = InferenceConfig(inference_engine="mystery")

    _assert_raises(
        ValueError,
        "inference_engine must be one of",
        lambda: validate_reserved_runtime_flags(args),
    )


def test_invalid_optimized_mode_fails_loudly():
    args = InferenceConfig(optimized_inference_mode="continuous")

    _assert_raises(
        ValueError,
        "optimized_inference_mode must be one of",
        lambda: validate_reserved_runtime_flags(args),
    )


def test_optimized_engine_is_fp32_only():
    args = InferenceConfig(inference_engine="optimized", precision="bf16")

    _assert_raises(
        ValueError,
        "requires precision=fp32",
        lambda: validate_reserved_runtime_flags(args),
    )


def test_optimized_engine_rejects_legacy_experiment_flags():
    args = InferenceConfig(
        inference_engine="optimized",
        use_server=False,
        inference_generation_compile=True,
    )

    _assert_raises(
        ValueError,
        "cannot be combined with legacy experimental inference flags",
        lambda: validate_reserved_runtime_flags(args),
    )


def test_v32_loader_never_imports_or_dispatches_to_optimized_adapter():
    with patch.object(inference, "load_model_with_server", return_value=("model", "tokenizer")) as legacy_loader:
        with patch.object(inference.importlib, "import_module") as optimized_import:
            result = load_model_with_engine(
                ckpt_path=None,
                t5_args=None,
                device="cpu",
                inference_engine="v32",
            )

    assert result == ("model", "tokenizer")
    legacy_loader.assert_called_once_with(
        ckpt_path=None,
        t5_args=None,
        device="cpu",
        max_batch_size=8,
        use_server=False,
        precision="fp32",
        attn_implementation="sdpa",
        eval_mode=True,
        lora_path=None,
        gamemode=None,
        auto_select_gamemode_model=True,
        generation_compile=False,
        server_allow_auto_start=True,
        server_connect_timeout=60.0,
        server_request_timeout=None,
        server_idle_timeout=20,
        server_batch_timeout=0.2,
    )
    optimized_import.assert_not_called()


def test_optimized_loader_fails_before_model_loading():
    with patch.object(inference, "load_model_with_server") as legacy_loader:
        _assert_raises(
            NotImplementedError,
            "control-plane scaffolding only",
            lambda: load_model_with_engine(
                ckpt_path=None,
                t5_args=None,
                device="cpu",
                inference_engine="optimized",
                optimized_inference_mode="offline_batch",
            ),
        )

    legacy_loader.assert_not_called()
