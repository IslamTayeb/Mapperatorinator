"""Unit tests for turbo Leviathan rejection sampling (TIER1 primitive)."""

import torch

from osuT5.osuT5.inference.turbo.rejection import (
    acceptance_alpha,
    apply_temp_top_p,
    expected_accepted,
    reject_sample_prefix,
    residual_distribution,
)


def test_expected_accepted_bounds():
    assert expected_accepted(0.0, 5) == 1.0
    assert expected_accepted(1.0, 5) == 6.0
    e = expected_accepted(0.6969, 5)
    assert 2.9 < e < 3.0


def test_acceptance_alpha_identical_is_one():
    p = torch.tensor([[0.25, 0.25, 0.5]])
    assert float(acceptance_alpha(p, p).item()) == 1.0


def test_residual_distribution_normalizes():
    p = torch.tensor([0.7, 0.2, 0.1])
    q = torch.tensor([0.1, 0.5, 0.4])
    r = residual_distribution(p, q)
    assert torch.allclose(r.sum(), torch.tensor(1.0), atol=1e-5)
    assert float(r[0]) > float(r[1])


def test_reject_sample_accepts_all_when_p_dominates_draft_tokens():
    # Draft always picks token 0; p puts almost all mass on 0.
    gamma, v = 3, 4
    p = torch.zeros(gamma, v)
    q = torch.zeros(gamma, v)
    p[:, 0] = 0.99
    p[:, 1:] = 0.01 / (v - 1)
    q[:, 0] = 0.5
    q[:, 1:] = 0.5 / (v - 1)
    n, resid = reject_sample_prefix(
        p_probs=p,
        q_probs=q,
        draft_token_ids=[0, 0, 0],
        rng=torch.Generator().manual_seed(0),
    )
    assert n == gamma
    assert resid is None


def test_reject_sample_rejects_when_q_overconfident():
    gamma, v = 2, 3
    p = torch.tensor([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1]])
    q = torch.tensor([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]])
    # Force rejection at first token by using a generator that draws u≈1
    # via many trials with seed that rejects; or check residual path exists.
    rejected = False
    for seed in range(50):
        n, resid = reject_sample_prefix(
            p_probs=p,
            q_probs=q,
            draft_token_ids=[0, 0],
            rng=torch.Generator().manual_seed(seed),
        )
        if n < gamma:
            rejected = True
            assert resid is not None
            assert 0 <= resid < v
            break
    assert rejected


def test_apply_temp_top_p_normalizes():
    logits = torch.tensor([[2.0, 1.0, 0.0, -4.0]])
    probs = apply_temp_top_p(logits, temperature=0.9, top_p=0.9)
    assert probs.shape == logits.shape
    assert torch.isfinite(probs).all()
    assert abs(float(probs.sum().item()) - 1.0) < 1e-5
