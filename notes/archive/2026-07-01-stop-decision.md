# Stop Decision

## Decision

Stop the current same-calculation RTX 2080/2080 Ti inference optimization loop by the documented stop condition: the first target of `100 tok/s` was not reached, but profiling across the main exact-calculation candidate families no longer supports a plausible remaining `>=10%` improvement in the current architecture.

The retained full-song baseline remains:

- Commit: `3e9033c` behavior retained on branch head
- Job: `49113713`
- Profile: `/work/imt11/Mapperatorinator/runs/full-compile-49113713-3e9033c/beatmapcfb70d0020da473c90f6c1acb32d6bbf.osu.profile.json`
- GPU: RTX 2080 Ti on `dcc-core-ferc-s-z25-21`
- Main generation: `7,639` tokens, `82.615s` synchronized model time, `92.465 tok/s`
- Fixed-seed generated token equivalence against compile-disabled full-song baseline: PASS

## Evidence

Reaching `100 tok/s` from `92.465 tok/s` would require about an `8.1%` full-song main-generation gain. The normal keep threshold remains `>=10%` unless a smaller change is extremely simple.

The retained full-song run reported `82.793s` summed main-generation wall time and `82.615s` synchronized model time across `87` windows. Only `0.178s` total sits outside the timed `model.generate` call, so prompt setup, static-cache construction, device transfer outside the model call, CPU result transfer, and profile bookkeeping cannot close the remaining gap.

The old torch-profiler trace is diagnostic only because the traced first main-generation window took `211s` under profiler overhead, but its event summary still identifies the main cost classes. The visible small-kernel targets are real but not target-sized:

- `aten::cat`: `20,396` calls, `58.3ms` self CUDA in the traced window.
- `aten::index_copy_`: `12,504` calls, `37.3ms` self CUDA.
- `Memcpy DtoD`: `46,908` calls, `68.9ms` self CUDA.
- Top-p `aten::sort` / `_unique`: visible but tens of milliseconds in the traced summary.

The direct attempt to attack the `cat`/mutable-buffer family with a preallocated `_sample` loop was not equivalent and only `+0.5%` token-normalized directionally, with worse model time. That result is stronger evidence than the profiler summary alone.

SDPA remains the correct baseline for this branch. The attention microprofile at `/work/imt11/Mapperatorinator/runs/attn-kernels-49097689/attention_kernel_profile.json` showed SDPA already using fused PyTorch flash-style kernels and beating FA2 in repo-like decode/cross cases by roughly `3x+` on A5000. Static-cache SDPA prefix trimming directly tested reducing attention work and regressed `-3.9%`.

## Remaining Work Class

A fully custom decode loop is still a possible separate research project, especially for a future autoregressive encoder-decoder core, but the current evidence does not support it as a quick `>=10%` exact-calculation win. It would need to preserve:

- HF `generate` sampling semantics, including RNG consumption and EOS behavior.
- Static cache behavior and generated-token accounting.
- The accepted compiled one-token forward path.
- CFG, beams, parallel generation, and server batching either disabled or reimplemented with exact semantics.

The closest prototype, the preallocated `_sample` loop, failed token identity and did not show meaningful speed. Treat future custom decode work as a new strict-equivalence prototype, not a continuation of this quick-win loop.

## Subagent Check

An independent read-only subagent audit on 2026-07-01 reached the same decision: keep `inference_generation_compile=true` as the retained same-calculation baseline, carry the compiled autoregressive decode lesson into future architecture work, and stop scouting current-architecture quick wins unless new profiler evidence changes the bottleneck mix.
