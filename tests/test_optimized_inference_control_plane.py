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
    validate_reserved_runtime_flags(args)


def test_invalid_inference_engine_fails_loudly():
    args = InferenceConfig(inference_engine="mystery")

    _assert_raises(
        ValueError,
        "inference_engine must be one of",
        lambda: validate_reserved_runtime_flags(args),
    )


def test_optimized_engine_accepts_only_the_two_frozen_precision_presets():
    for precision in ("fp32", "fp16"):
        validate_reserved_runtime_flags(
            InferenceConfig(
                inference_engine="optimized",
                precision=precision,
                use_server=False,
                parallel=False,
                device="cuda",
                attn_implementation="sdpa",
                cfg_scale=1.0,
                num_beams=1,
                super_timing=False,
            )
        )

    args = InferenceConfig(inference_engine="optimized", precision="bf16")

    _assert_raises(
        ValueError,
        "requires precision=fp32 or precision=fp16",
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
    )
    optimized_import.assert_not_called()
