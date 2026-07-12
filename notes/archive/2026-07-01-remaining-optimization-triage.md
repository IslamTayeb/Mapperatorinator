# Remaining Optimization Triage

## Current Retained Baseline

The retained exact-calculation baseline is the accepted generation-compile full-song run:

- Commit: `3e9033c` with `inference_generation_compile=true`
- Job: `49113713`
- Profile: `/work/imt11/Mapperatorinator/runs/full-compile-49113713-3e9033c/beatmapcfb70d0020da473c90f6c1acb32d6bbf.osu.profile.json`
- Main generation: `7,639` tokens, `82.615s` model time, `92.465 tok/s`
- Token equivalence against compile-disabled full-song baseline: PASS

The first target, `100 tok/s`, would need about an `8.1%` full-song main-generation gain from this retained baseline.

## Why Small Wrapper Cleanup Is Not Enough

For the retained full-song run, summed main-generation wall time was `82.793s` and synchronized model time was `82.615s`. Only `0.178s` across `87` main-generation windows sits outside the timed model call.

That rules out prompt setup, static cache allocation, device transfer outside `model.generate`, result CPU transfer, and profiling bookkeeping as target-sized wins for the current throughput metric. Those are still worth keeping tidy, but they cannot plausibly produce a `>=10%` full-song main-generation improvement.

## Remaining Candidate Classes

Rejected by measurement:

- Generation compile disabled -> retained compile path was `+47.0%`.
- Stateful monotonic time masking -> `-4.1%` smoke.
- Last-position logits projection -> `-21.4%` smoke.
- Static-cache SDPA prefix trim -> `-3.9%` smoke.
- Dynamic/default cache -> non-equivalent token output.
- `torch.inference_mode` wrapper -> `+2.3%` full-song, below keep threshold.
- `CompileConfig(dynamic=False)` -> `-23.5%` smoke.
- `CompileConfig(dynamic=True)` -> `-10.0%` smoke.

Likely too small from existing torch-profiler summary:

- Top-p sorting and sampling kernels are visible but not large enough to close the gap.
- `aten::cat`, `index_copy_`, RoPE sin/cos, and small elementwise kernels are real overheads, but individual measured costs are not target-sized unless a future architecture or trace changes their proportion.
- Attention backend switching is not currently promising: SDPA already dispatches fused flash-style kernels, and FA2 lost in decode/cross microprofiles.

## Custom Decode Loop

A hand-written decode loop is the only remaining same-calculation idea with a theoretical path to a large enough win, but it is not a simple optimization:

- It must preserve HF `generate` token semantics exactly: same processors, top-p/top-k filtering, temperature, EOS stopping, fixed-seed CUDA RNG consumption, left padding, static cache behavior, and generated-token accounting.
- It must preserve the accepted compiled one-token forward path. A naive Python loop around `model.forward` could easily regress by losing HF's auto-compile behavior.
- CFG, beams, parallel generation, and server batching should be hard-disabled in the first prototype unless implemented separately.

Given the profile evidence, this should be treated as a separate prototype with strict smoke equivalence rather than a quick cleanup patch. The idea should carry to a future autoregressive encoder-decoder core if implemented, but the exact prompt/context processors may change during the planned architecture migration.

## DCC Smoke Cleanup

Per user request, old smoke artifacts under the user's DCC work area were deleted after their metrics were copied into notes/docs.

- Deleted run dirs pattern: `/work/imt11/Mapperatorinator/runs/smoke-*`
- Deleted log files pattern: `/work/imt11/Mapperatorinator/logs/smoke-*`
- Ownership guard: `-user imt11`
- Manifest: `/work/imt11/Mapperatorinator/logs/deleted-smoke-artifacts-20260630-231929.txt`
- Remaining smoke run dirs/log files after cleanup: `0` / `0`

Full-song accepted runs were left intact.
