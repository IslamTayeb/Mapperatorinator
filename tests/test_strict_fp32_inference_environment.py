from __future__ import annotations

import torch

import inference


def test_strict_fp32_environment_disables_every_tf32_switch() -> None:
    original_precision = torch.get_float32_matmul_precision()
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        inference.setup_inference_environment(12345, strict_fp32=True)

        assert torch.get_float32_matmul_precision() == "highest"
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    finally:
        torch.set_float32_matmul_precision(original_precision)
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn


def test_profile_metadata_records_strict_fp32_backend_state(monkeypatch) -> None:
    original_precision = torch.get_float32_matmul_precision()
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    try:
        monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
        inference.setup_inference_environment(12345, strict_fp32=True)

        metadata = inference.get_profile_runtime_metadata()

        assert metadata["float32_matmul_precision"] == "highest"
        assert metadata["cuda_matmul_allow_tf32"] is False
        assert metadata["cudnn_allow_tf32"] is False
        assert metadata["nvidia_tf32_override"] == "0"
    finally:
        torch.set_float32_matmul_precision(original_precision)
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn
