"""Opt-in turbo engine — §38 TIER2 relaxed fused decoder step.

Immutable preset behind ``inference_engine=turbo``. Not bit-exact.
Does not alter the ``optimized`` bit-exact default. TIER2 evidence pack
required before any 500 / ship claim. Campaign tip remains ``55949274``.
"""

from .adapter import load_turbo_engine

__all__ = ["load_turbo_engine"]
