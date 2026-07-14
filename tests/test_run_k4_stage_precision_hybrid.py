from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch

import inference
from osuT5.osuT5.inference.engine_binding import InferenceEngineBinding
from osuT5.osuT5.inference.optimized.single import engine
from utils import run_k4_stage_precision_hybrid as runner


class _Model(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.register_parameter("marker", torch.nn.Parameter(torch.zeros((), dtype=dtype)))

    @property
    def dtype(self):
        return self.marker.dtype


class _State:
    def __init__(self, owner):
        self.owner = owner

    def validate_owner(self, model):
        assert model is self.owner

    @staticmethod
    def metadata():
        return {
            "version": "approximate-fp16-weights-fp32-state-v2",
            "exactness_claim": False,
        }


def test_runner_loads_main_then_fp16_timing_and_restores_every_patch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fake_args = SimpleNamespace(
        inference_engine="optimized",
        precision="fp32",
        use_server=False,
        profile_inference=True,
        super_timing=False,
        generate_positions=False,
        seed=12345,
    )
    monkeypatch.setattr(
        runner,
        "_load_args",
        lambda *unused_args, **unused_kwargs: fake_args,
    )
    monkeypatch.setattr(inference, "should_load_separate_timing_model", lambda args: True)
    seen = []

    def original_loader(*loader_args, **kwargs):
        precision = kwargs["precision"]
        seen.append((precision, kwargs["auto_select_gamemode_model"]))
        model = _Model(engine.OPTIMIZED_PRESETS[precision].torch_dtype)
        runtime = engine.OptimizedSingleRuntime(engine.OPTIMIZED_PRESETS[precision])
        if precision == "fp32":
            object.__setattr__(runtime, "_approximate_weight_only_state", _State(model))
        return InferenceEngineBinding(model, runtime), object()

    monkeypatch.setattr(inference, "load_model_with_engine", original_loader)
    monkeypatch.setattr(
        runner,
        "_initialize_with_evidence",
        lambda initializer, model: {
            "version": "approximate-fp16-weights-fp32-state-v2",
            "result_class": "documented-drift",
            "exactness_claim": False,
            "extension_init_seconds": 0.0,
            "weight_pack_seconds": 0.0,
        },
    )
    events = []

    @contextmanager
    def fake_k4(*, block_size):
        events.append(("k4-enter", block_size))
        yield
        events.append(("k4-exit", block_size))

    @contextmanager
    def fake_seed(module, *, base_seed):
        events.append(("seed-enter", base_seed))
        yield
        events.append(("seed-exit", base_seed))

    monkeypatch.setattr(runner, "install_k8_candidate", fake_k4)
    monkeypatch.setattr(runner, "fixed_seed_processor_generation", fake_seed)

    def fake_main(received):
        assert received is fake_args
        main, _ = inference.load_model_with_engine(
            precision="fp32", auto_select_gamemode_model=True
        )
        timing, _ = inference.load_model_with_engine(
            precision="fp32", auto_select_gamemode_model=False
        )
        assert main.raw_model is not timing.raw_model
        assert main.runtime.profile_metadata()["optimized_stage_precision_hybrid"][
            "timing"
        ]["precision"] == "fp16"

    monkeypatch.setattr(inference, "main", fake_main)
    output = tmp_path / "init.json"
    runner.run("profile_salvalai", [], output)

    assert seen == [("fp32", True), ("fp16", False)]
    assert events == [
        ("k4-enter", 4),
        ("seed-enter", 12345),
        ("seed-exit", 12345),
        ("k4-exit", 4),
    ]
    assert inference.load_model_with_engine is original_loader
    assert '"timing"' in output.read_text(encoding="utf-8")
