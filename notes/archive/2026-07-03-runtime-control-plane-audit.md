# Runtime Control Plane And Fast-Prepare Audit

## Summary

The current accepted fast path should remain integrated through the existing inference routing surface:

- `inference.py` owns user-facing config validation and profile metadata.
- `osuT5/osuT5/inference/server.py:model_generate()` owns generation/runtime integration.
- `osuT5/osuT5/inference/processor.py` prepares per-window/per-batch inputs and forwards normalized generation kwargs into `server.py:model_generate()`.
- Low-level helpers such as `direct_decode.py`, `decode_loop.py`, and native kernels should stay helper/runtime modules, not independent public mode selectors.

This pass tightened `inference.py` validation for active-prefix, stateful monotonic, q1 BMM, DecodeSession, native q1 self-attention, and fused RoPE/cache self-attention. It also added missing profile metadata for `inference_native_q1_rope_cache_self_attention`.

The validation now runs after `compile_device_and_seed()` so `device=auto` and `attn_implementation=auto` are normalized before runtime-mode constraints check precision/backend/device compatibility. This matters because the accepted active-prefix CUDA graph/native q1 flags require an effective CUDA device plus `attn_implementation=sdpa`, and the user-facing defaults are `device=auto` and `attn_implementation=auto`.

Guardrail validation:

- Job: `49231914`
- Commit: `12cfdf3`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, UUID `GPU-de50b870-b708-690d-74a4-6e855163a133`
- Run dir: `/work/imt11/Mapperatorinator/runs/runtime-guardrails-49231914-12cfdf3`
- Result: PASS
- Checks:
  - retained fast stack passes config validation on GPU with `attn_implementation=auto` normalized to `sdpa`
  - `use_server=true` fails early with `inference_active_prefix_decode_loop requires use_server=false`
  - `parallel=true` fails early with `inference_active_prefix_decode_loop currently supports sequential inference only`
  - `inference_native_q1_rope_cache_self_attention=true` without native q1 self-attention fails early
  - backed-out `inference_active_prefix_fast_prepare=true` fails as an unknown Hydra key

## Fast-Prepare Result

Direct verifier gate:

- Job: `49231798`
- Commit: `912bdf7`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, UUID `GPU-ba708720-cba9-538c-6e0f-ecaea3486d09`
- Report: `/work/imt11/Mapperatorinator/runs/fast-prepare-gate-49231798-912bdf7/verify_fast_prepare.json`
- Config: `profile_salvalai_smoke15`, sequence `9`, `max_new_tokens=256`, fp32, SDPA, active-prefix bucket64 CUDA graph, q1 BMM, native q1 self-attention, fused RoPE/cache self-attention, DecodeSession, generation compile
- Result: PASS. Generated tokens matched, raw logits/top-k matched, and final RNG state matched for 256 sampled steps.

Production smoke:

- Job: `49231829`
- Commit: `912bdf7`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, UUID `GPU-ba708720-cba9-538c-6e0f-ecaea3486d09`
- Run dir: `/work/imt11/Mapperatorinator/runs/fast-prepare-smoke15-49231829-912bdf7`
- Control: `control_a` completed and wrote `control_a.profile.json`
- Candidate: `fast_a` failed before writing a complete profile
- Failure: timing generation crashed during active-prefix CUDA graph capture because `prepare_one_token_decode_inputs_fast` copied wrapper-level `negative_prompt` into `VarWhisperForConditionalGeneration.forward()`, which does not accept it.

Decision: reject production use for now. The attempted `inference_active_prefix_fast_prepare` production flag was backed out rather than left as dead user-facing surface area. Do not run more fast-prepare speed tests until the input-builder contract is fixed through the normal `inference.py`/`server.py` path and re-gated from direct-loop through smoke/full-song.

## Accepted Fast-Path Integration Audit

The accepted fast stack is mostly cleanly routed through the existing control plane:

- `Processor.model_generate()` forwards normalized optimization kwargs into `server.py:model_generate()`.
- `server.py:model_generate()` builds cache/logits processors, applies runtime contexts, calls `model.generate()`, and records runtime stats/metadata.
- `active_prefix_decode_generate()` is still a custom generation loop, but it is selected only through `server.py:model_generate()` rather than by a separate top-level runtime.

The main gap was validation drift: some unsupported combinations were rejected only in `processor.py`, `server.py`, or lower-level helpers. This pass moved the user-facing guardrails into `inference.py` so bad mode combinations fail during `compile_args()`, after effective device/backend normalization and before model load.

## Batching/Throughput Design Notes

Existing static batching modes:

- `use_server=true`: `InferenceServer` groups IPC requests by identical generation kwargs, slices up to `max_batch_size`, pads/collates tensors, calls `server.py:model_generate()`, and splits outputs back per request. This is static request batching, not continuous decoding.
- `parallel=true`: `Processor.generate_parallel()` prepares all song windows up front and uses `_batched_inference()` to stack windows into fixed batches. This is static window batching inside a song/context.
- Current DecodeSession/native fast path is batch-1 only and explicitly rejects both modes.

Future continuous batching should be a separate mode with explicit flags and result tables. Exact-equivalent claims must preserve each request's generated-token IDs/output behavior, stopping state, cache state, logits-processor state, RNG behavior, generated-token counts, and output policy. If per-request RNG/order equivalence cannot be proven, label the mode `documented-drift` or non-equivalent.

Recommended reporting split for any future throughput mode:

- cold single-song retained baseline
- same-process warm repeat
- static server batch
- static `parallel=true` window batch
- continuous server batch

Do not mix continuous/static batch aggregate TPS into single-song TPS claims.

Suggested future flags:

- `inference_continuous_batching`
- `inference_continuous_batching_mode`
- `continuous_batch_max_active_sequences`
- `continuous_batch_max_wait_ms`
- `continuous_batch_prefill_policy`
- `continuous_batch_decode_order_policy`
- `continuous_batch_rng_policy`
- `inference_batch_decode_session_runtime`
- `inference_batch_native_decode_kernels`

Suggested future gates:

- per-request/window generated-token identity
- raw-logit/top-k equality for sampled steps
- final RNG or per-sequence RNG-ledger equality
- stop-reason and generated-token-count equality
- output hash/byte equivalence
- per-song/per-request no-regression inside the same batching result class

Suggested future metadata:

- `result_class`
- `throughput_claim_scope`
- `batching_mode`
- scheduler policy
- request/window IDs and batch IDs
- active batch-size histogram
- queue wait and per-request latency
- prefill/decode timing split
- per-sample token hashes
- stop reasons
- RNG policy/state hashes
- cache slot/reorder events
- graph capture/replay counts
- effective fast-path flags
