from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from osuT5.osuT5.inference.engine_binding import InferenceEngineBinding
from osuT5.osuT5.inference.optimized.single.engine import (
    OPTIMIZED_PRESETS,
    OptimizedSingleRuntime,
)
from utils import run_persistent_graph_workspace_scout as runner


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    @property
    def dtype(self):
        return self.weight.dtype


class _Processor:
    def generate(self, *args, **kwargs):
        return None


def _binding() -> InferenceEngineBinding:
    return InferenceEngineBinding(
        raw_model=_Model(),
        runtime=OptimizedSingleRuntime(OPTIMIZED_PRESETS["fp32"]),
    )


def test_runner_reuses_two_bindings_and_closes_pools(monkeypatch, tmp_path: Path) -> None:
    bindings = [_binding(), _binding()]
    source_loads = []

    def source_loader(*args, **kwargs):
        binding = bindings[len(source_loads)]
        source_loads.append(kwargs["auto_select_gamemode_model"])
        return binding, object()

    calls = 0

    def fake_main(args):
        nonlocal calls
        inference.load_model_with_engine(auto_select_gamemode_model=True)
        inference.load_model_with_engine(auto_select_gamemode_model=False)
        result = Path(args.output_path) / "map.osu"
        result.write_text("map\n", encoding="utf-8")
        Path(f"{result}.profile.json").write_text("{}\n", encoding="utf-8")
        calls += 1
        return object(), result

    inference = SimpleNamespace(
        load_model_with_engine=source_loader,
        main=fake_main,
        Processor=_Processor,
    )
    monkeypatch.setitem(sys.modules, "inference", inference)
    monkeypatch.setattr(
        runner,
        "_load_args",
        lambda *args, **kwargs: SimpleNamespace(
            inference_engine="optimized",
            precision="fp32",
            use_server=False,
            profile_inference=True,
            super_timing=False,
            generate_positions=False,
            seed=12345,
            output_path=None,
        ),
    )
    monkeypatch.setattr(
        runner,
        "_initialize_with_evidence",
        lambda *args, **kwargs: {"initialized": True},
    )
    monkeypatch.setattr(
        runner,
        "shared_decoder_rope_context",
        lambda *args, **kwargs: nullcontext(),
    )
    install_calls = []
    original_install = runner.install_k8_candidate

    def fake_install(**kwargs):
        install_calls.append(kwargs)
        return original_install(**kwargs)

    monkeypatch.setattr(runner, "install_k8_candidate", fake_install)
    cuda_calls = []
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        runner.torch.cuda,
        "synchronize",
        lambda: cuda_calls.append("synchronize"),
    )
    monkeypatch.setattr(runner.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        runner.torch.cuda,
        "reset_peak_memory_stats",
        lambda device: cuda_calls.append(("reset", device)),
    )
    monkeypatch.setattr(runner.torch.cuda, "memory_allocated", lambda device: 100)
    monkeypatch.setattr(runner.torch.cuda, "memory_reserved", lambda device: 200)
    monkeypatch.setattr(
        runner.torch.cuda,
        "max_memory_allocated",
        lambda device: 300,
    )
    monkeypatch.setattr(runner.torch.cuda, "max_memory_reserved", lambda device: 400)

    manifest = tmp_path / "manifest.json"
    initialization = tmp_path / "initialization.json"
    runner.run(
        "profile_salvalai",
        ["seed=12345"],
        output_init_json=initialization,
        output_manifest=manifest,
    )

    assert calls == 3
    assert source_loads == [True, False]
    assert inference.load_model_with_engine is source_loader
    assert all(
        binding.runtime._persistent_graph_workspace_pool is None
        for binding in bindings
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["close_completed"] is True
    assert set(payload["results"]) == {"cold", "warm1", "warm2"}
    assert payload["results"]["cold"]["pool_summary"]["main"]["request_count"] == 0
    assert payload["results"]["warm1"]["pool_summary"]["main"]["request_count"] == 0
    assert payload["results"]["warm2"]["pool_summary"]["main"]["request_count"] == 0
    assert payload["results"]["cold"]["cuda_memory"]["max_allocated_bytes"] == 300
    assert payload["results"]["warm1"]["cuda_memory"]["max_reserved_bytes"] == 400
    assert payload["results"]["warm2"]["cuda_memory"]["allocated_bytes"] == 100
    assert payload["results"]["cold"]["cuda_peak_stats_reset"] is True
    assert payload["results"]["warm1"]["cuda_peak_stats_reset"] is True
    assert payload["results"]["warm2"]["cuda_peak_stats_reset"] is True
    assert cuda_calls.count(("reset", 0)) == 3
    assert install_calls == [
        {
            "block_size": 4,
            "graph_remainders": True,
            "shared_static_input_arena": True,
        }
    ]
    initialization_payload = json.loads(initialization.read_text(encoding="utf-8"))
    assert "shared-arena" in initialization_payload["topology_version"]
