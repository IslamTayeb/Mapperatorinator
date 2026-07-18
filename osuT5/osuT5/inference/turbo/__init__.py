"""Opt-in turbo-on-tiger (strict rejection-sampling; stacks on PR #120).

Teacher verify uses tiger ``CUDAGraphDecoder`` at q_len=K — not the optimized
fused/active-prefix path. Campaign tip ``55949274`` stays frozen elsewhere.
"""

from .adapter import load_turbo_engine
from .engine import attach_turbo_to_processor, turbo_env_enabled

__all__ = ["attach_turbo_to_processor", "load_turbo_engine", "turbo_env_enabled"]
