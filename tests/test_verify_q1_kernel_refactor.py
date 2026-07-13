from __future__ import annotations

import json

import pytest
import torch

from utils.verify_q1_kernel_refactor import _assert_untouched_cache, compare


def _payload() -> dict:
    return {
        "schema_version": 1,
        "precision": "fp32",
        "states": {
            "128": {
                "plain_output": torch.arange(4, dtype=torch.float32),
                "cache_keys": torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2),
            }
        },
    }


def test_compare_requires_byte_identical_tensor_state(tmp_path) -> None:
    reference = tmp_path / "reference.pt"
    candidate = tmp_path / "candidate.pt"
    report = tmp_path / "report.json"
    torch.save(_payload(), reference)
    torch.save(_payload(), candidate)

    compare(reference, candidate, report)

    assert json.loads(report.read_text(encoding="utf-8"))["exact"] is True

    changed = _payload()
    changed["states"]["128"]["cache_keys"][0, 0, 2, 1] += 1
    torch.save(changed, candidate)
    with pytest.raises(SystemExit):
        compare(reference, candidate, report)
    parsed = json.loads(report.read_text(encoding="utf-8"))
    assert parsed["exact"] is False
    assert parsed["failures"] == [
        "capture.states.128.cache_keys: tensor differs (max_abs=1.0)"
    ]


def test_cache_ownership_check_allows_only_the_active_slot() -> None:
    before = torch.zeros((1, 1, 4, 2))
    after = before.clone()
    after[..., 2, :] = 1
    _assert_untouched_cache(before, after, write_position=2)

    after[..., 1, 0] = 1
    with pytest.raises(RuntimeError, match="before the active slot"):
        _assert_untouched_cache(before, after, write_position=2)
