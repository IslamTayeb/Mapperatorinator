from __future__ import annotations

from pathlib import Path

import pytest
import torch

from osuT5.osuT5.inference.optimized.scout import conditional_while_multinomial as probe


def test_import_is_cold_and_probe_uses_only_opt_in_extension(monkeypatch) -> None:
    calls = []
    extension = object()

    def fake_load_inline(**kwargs):
        calls.append(kwargs)
        return extension

    monkeypatch.setattr(probe, "_CONTROL_EXTENSION", None)
    monkeypatch.setattr(probe, "_load_inline", fake_load_inline)

    assert probe._CONTROL_EXTENSION is None
    assert probe._load_control_extension() is extension
    assert probe._load_control_extension() is extension
    assert len(calls) == 1
    assert calls[0]["name"] == "mapperatorinator_conditional_while_multinomial_probe"
    assert calls[0]["functions"] == ["conditional_probe_record_and_continue"]


def test_control_kernel_sets_conditional_from_device_code() -> None:
    source = probe._CUDA_SOURCE
    assert "cudaGraphConditionalHandle" in source
    assert "cudaGraphSetConditional" in source
    assert "samples[current] = sampled[0]" in source
    assert "next < iterations ? 1 : 0" in source


def test_graph_builder_uses_public_while_and_child_graph_apis() -> None:
    source = Path(probe.__file__).read_text()
    assert "torch.cuda.CUDAGraph(keep_graph=True)" in source
    assert "graph.register_generator_state(generator)" in source
    assert "torch.multinomial(" in source
    assert "cudaGraphCondTypeWhile" in source
    assert "cudaGraphAddChildGraphNode" in source
    assert "raw_cuda_graph()" in source
    assert "(runtime.cudaGraphLaunch" not in source
    assert "(runtime.cudaGraphExecDestroy" not in source
    assert "(runtime.cudaGraphDestroy" not in source
    assert "install" not in probe.__all__


def test_version_parser_and_bounds_fail_loudly() -> None:
    assert probe._cuda_version_tuple("12.8") == (12, 8)
    assert probe._cuda_version_tuple("13.0.1") == (13, 0)
    with pytest.raises(RuntimeError, match="does not report"):
        probe._cuda_version_tuple(None)
    with pytest.raises(RuntimeError, match="unrecognized"):
        probe._cuda_version_tuple("nightly")


def test_probe_rejects_invalid_iterations_before_cuda_initialization() -> None:
    with pytest.raises(TypeError, match="integer"):
        probe.run_conditional_while_multinomial_probe(iterations=True)
    with pytest.raises(ValueError, match="between 2 and 32"):
        probe.run_conditional_while_multinomial_probe(iterations=1)
    with pytest.raises(ValueError, match="between 2 and 32"):
        probe.run_conditional_while_multinomial_probe(iterations=33)


def test_state_hash_requires_generator_state_storage() -> None:
    state = torch.arange(16, dtype=torch.uint8)
    assert len(probe._state_sha256(state)) == 64
    with pytest.raises(TypeError, match="uint8"):
        probe._state_sha256(state.long())


def test_probe_never_enters_production_selector_surface() -> None:
    runner = Path("utils/probe_conditional_while_multinomial.py").read_text()
    assert "inference_engine" not in runner
    assert "active_prefix_decode_generate" not in runner
    assert "install_" not in runner
    assert probe.RNG_POLICY == "torch_multinomial_philox"
