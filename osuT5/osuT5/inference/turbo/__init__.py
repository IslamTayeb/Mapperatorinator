"""Opt-in turbo-on-tiger scaffold (strict rejection-sampling; stacks on PR #120).

Teacher verify uses tiger ``CUDAGraphDecoder`` at q_len=K — not the optimized
fused/active-prefix path. Campaign tip ``55949274`` stays frozen elsewhere.
"""

from .adapter import load_turbo_engine

__all__ = ["load_turbo_engine"]
