#!/usr/bin/env python3
"""§45 / Track C speculative structural bound (no GPU).

Prints whether K=1 draft/verify can beat tip 366.11 / ≥384 under measured
c_draft, c_verify, and §43 E[acc]. Exit 0 always; prints DEAD_END or OPEN.
"""
from __future__ import annotations

TIP_TPS = 366.11
TIP_MS = 1000.0 / TIP_TPS
GATE_384_TPS = 384.0
GATE_384_MS = 1000.0 / GATE_384_TPS

# Measured components
C_DRAFT_MS = 1.08  # §42 per draft token
C_VERIFY_MS = 3.125  # §41 best K cudagraph verify
C_VERIFY_RATIO = 1.686
E_ACC = 1.974  # §43 one_layer γ=3 temp=0.9 held-out
GAMMA = 3


def ceiling(c_draft_ms: float, c_verify_ms: float, e_acc: float, *, rebuild: bool) -> float:
    step = GAMMA * c_draft_ms + c_verify_ms * (2.0 if rebuild else 1.0)
    return 1000.0 / (step / e_acc)


def main() -> None:
    ideal = ceiling(C_DRAFT_MS, C_VERIFY_MS, E_ACC, rebuild=False)
    with_rebuild = ceiling(C_DRAFT_MS, C_VERIFY_MS, E_ACC, rebuild=True)
    print("tip_tps", TIP_TPS, "tip_ms", round(TIP_MS, 3))
    print("gate_384_tps", GATE_384_TPS, "gate_384_ms", round(GATE_384_MS, 3))
    print("inputs", {"c_draft_ms": C_DRAFT_MS, "c_verify_ms": C_VERIFY_MS,
                     "c_verify_ratio": C_VERIFY_RATIO, "E_acc": E_ACC, "gamma": GAMMA})
    print("ceiling_no_rebuild_tps", round(ideal, 2))
    print("ceiling_with_rebuild_tps", round(with_rebuild, 2))
    print("scout_50148770_main_tps", 44.52)
    dead = ideal < GATE_384_TPS
    print("decision", "DEAD_END" if dead else "OPEN")
    print(
        "note",
        "Optimistic no-rebuild ceiling still below ≥384 and tip at measured gate misses.",
    )


if __name__ == "__main__":
    main()
