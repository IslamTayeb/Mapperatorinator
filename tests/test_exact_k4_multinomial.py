from __future__ import annotations

from pathlib import Path

import pytest
import torch

from osuT5.osuT5.inference.optimized.scout import exact_k4_multinomial as probe


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE = (
    REPO_ROOT
    / "osuT5/osuT5/inference/optimized/scout/exact_k4_multinomial.py"
)
RUNNER = REPO_ROOT / "utils/probe_exact_k4_multinomial.py"
SCOUT_INIT = REPO_ROOT / "osuT5/osuT5/inference/optimized/scout/__init__.py"


def _cpu_state() -> probe._ToyState:
    state = probe._ToyState(
        tokens=torch.tensor([1, 3, 4, 5, 6], dtype=torch.long),
        cache=torch.arange(40, dtype=torch.float32).reshape(4, 10),
        processor_state=torch.tensor([18.0], dtype=torch.float32),
        processor_snapshots=torch.tensor(
            [[3.0], [7.0], [12.0], [18.0]], dtype=torch.float32
        ),
        current_length=torch.tensor([5], dtype=torch.long),
        length_snapshots=torch.tensor([[2], [3], [4], [5]], dtype=torch.long),
        unfinished=torch.tensor([0], dtype=torch.long),
        unfinished_snapshots=torch.tensor([[1], [0], [0], [0]], dtype=torch.long),
        stop_flags=torch.tensor([False, True, False, False]),
    )
    return state


def test_terminal_restore_keeps_only_valid_writes_and_state() -> None:
    state = _cpu_state()
    expected_cache = state.cache[:2].clone()

    probe.restore_terminal_state(state, valid_steps=2)

    assert state.tokens.tolist() == [1, 3, 4, -1, -1]
    assert torch.equal(state.cache[:2], expected_cache)
    assert torch.count_nonzero(state.cache[2:]) == 0
    assert state.processor_state.tolist() == [7.0]
    assert state.current_length.tolist() == [3]
    assert state.unfinished.tolist() == [0]
    assert state.stop_flags.tolist() == [False, True, False, False]
    assert torch.count_nonzero(state.processor_snapshots[2:]) == 0
    assert torch.count_nonzero(state.length_snapshots[2:]) == 0
    assert torch.count_nonzero(state.unfinished_snapshots[2:]) == 0


@pytest.mark.parametrize("valid_steps", (0, 5, True, 1.5))
def test_terminal_restore_rejects_invalid_step_counts(valid_steps) -> None:
    with pytest.raises((TypeError, ValueError)):
        probe.restore_terminal_state(_cpu_state(), valid_steps=valid_steps)


def test_offset_schedule_requires_fixed_positive_multinomial_increment() -> None:
    assert probe._validate_offset_schedule([100, 104, 108, 112, 116]) == [4] * 4
    with pytest.raises(RuntimeError, match="positively"):
        probe._validate_offset_schedule([100, 104, 104, 108, 112])
    with pytest.raises(RuntimeError, match="not constant"):
        probe._validate_offset_schedule([100, 104, 112, 116, 120])
    with pytest.raises(ValueError, match="boundaries"):
        probe._validate_offset_schedule([100, 104])


def test_probe_matrix_covers_every_terminal_position() -> None:
    seeds, positions = probe._validate_probe_inputs(
        (7, 12345, 987654, 42),
        (1, 2, 3, 4),
    )
    assert seeds == (7, 12345, 987654, 42)
    assert positions == (1, 2, 3, 4)

    with pytest.raises(ValueError, match="equally sized"):
        probe._validate_probe_inputs((1, 2), (1,))
    with pytest.raises(ValueError, match="distinct"):
        probe._validate_probe_inputs((1, 2), (1, 1))
    with pytest.raises(ValueError, match=r"\[1, 4\]"):
        probe._validate_probe_inputs((1,), (5,))


def test_probe_is_opt_in_and_uses_one_ordinary_graph_with_distinct_steps() -> None:
    source = MODULE.read_text(encoding="utf-8")
    scout_init = SCOUT_INIT.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert "exact_k4_multinomial" not in scout_init
    assert "torch.cuda.CUDAGraph()" in source
    assert "graph.register_generator_state(generator)" in source
    assert "for step in range(BLOCK_SIZE):" in source
    assert "torch.multinomial(probabilities, num_samples=1).squeeze(1)" in source
    assert "torch.gather(eos_mask, 0, sampled_scalar.reshape(1))[0]" in source
    assert "eos_mask[sampled_scalar]" not in source
    assert "raw_cuda_graph" not in source
    assert "cudaGraphAddChildGraphNode" not in source
    assert "generator.set_offset(corrected_offset)" in source
    assert "NVIDIA_TF32_OVERRIDE" in source
    assert "torch.set_float32_matmul_precision(\"highest\")" in source
    assert "--eos-positions" in runner
    assert "exact_k4_feasible" in runner


def test_strict_policy_rejects_missing_process_override(monkeypatch) -> None:
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    with pytest.raises(RuntimeError, match="before process start"):
        probe._enforce_strict_fp32_policy()
