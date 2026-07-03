# Main Branch And Runtime Control Refresh

## Summary

The accepted fast inference path is already integrated through the intended control plane:

- `inference.py` validates user-facing optimization flags before model load.
- `InferenceProcessor.model_generate()` forwards normalized generation/runtime flags.
- `osuT5/osuT5/inference/server.py:model_generate()` builds logits processors/cache state and selects the active-prefix custom generation hook.
- Low-level direct-loop, DecodeSession, and native CUDA helpers remain helper modules rather than a second public routing surface.

A follow-up audit found one concrete routing cleanup: `inference_generation_compile` was correctly passed by the main `inference.py` path, but not by `web-ui.py`, `mai_mod.py`, or `calc_fid.py`, and local/custom checkpoint loads could leave `generation_config.disable_compile=True` even when compile was requested. The control-plane compile-routing branch fixes those cases and adds explicit profile metadata when native q1 self-attention is requested but disabled for timing contexts.

## Workflow Rule

Use `main` for accepted work. Start risky profiler/runtime/kernel probes from current `main` on short-lived experiment branches, then merge or cherry-pick them back only after token/output equivalence, no-regression checks, documentation, and push gates pass. Rejected experiment branches should leave a note/result and should not become unmerged shadow production paths.

## Batching Rule

Keep single-song TPS separate from warm-repeat, static server/window batching, and any future continuous batching. Continuous batching remains a future throughput-mode project with explicit flags and per-request token/RNG/cache/logits-processor equivalence gates; it must not be mixed into exact single-song TPS claims.
