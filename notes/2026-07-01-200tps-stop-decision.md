# 200 tok/s Stop Decision

Superseded update: this was the quick-tweak stop decision before the renewed active-prefix runtime pass. Full-song job `49150185` later accepted bucketed active-prefix decode with bucket `256`, raising the retained baseline to `121.926 tok/s` with token equivalence PASS for `7,639/7,639` main tokens. Keep this note as historical closure for the quick-tweak phase.

## Decision

Stop current-architecture same-calculation scouting by the documented 200 tok/s stop condition. The target was not reached, but the measured and audited candidate families no longer show a plausible remaining major exact-calculation path on RTX 2080/2080 Ti.

The retained full-song baseline remains SDPA plus `inference_generation_compile=true`:

- Full-song job: `49113713`
- Profile: `/work/imt11/Mapperatorinator/runs/full-compile-49113713-3e9033c/beatmapcfb70d0020da473c90f6c1acb32d6bbf.osu.profile.json`
- GPU: RTX 2080 Ti on `dcc-core-ferc-s-z25-21`
- Main generation: `7,639` tokens, `82.615s` synchronized model time, `92.465 tok/s`
- Token equivalence: PASS against the compile-disabled full-song baseline, all `7,639 / 7,639` generated main-token IDs matched

Reaching `200 tok/s` from this baseline would require cutting full-song synchronized main-generation model time to about `38.195s`, a `53.8%` reduction. The retained full-song run has only `0.178s` outside synchronized model generation across `87` windows, so wrapper cleanup cannot close the gap.

## Why The Remaining Paths Do Not Justify More Current-Architecture Scouting

| path | evidence | conclusion |
| --- | --- | --- |
| Exact generation compile | Accepted full-song `+47.0%`, token-equivalent PASS | Keep as retained baseline |
| Copy-compatible custom `_sample` hook | Jobs `49135145`/`49135146` and `49135380`/`49135381`; token-equivalent PASS but `-10.9%` compile-disabled and `-14.7%` compile-enabled | Do not retry copy-only custom loop |
| Preallocated `_sample` loop | Token equivalence FAIL at token `1,350`; only `+0.5%` token-normalized directionally with worse model time | Non-equivalent and too small |
| Persistent static mask | Jobs `49137140`/`49137141`; token-equivalent PASS but `-17.6%` overall and slower post-warmup | Rejected |
| Compile config scouting | `dynamic=False`, `dynamic=True`, `max-autotune`, and `fullgraph=True` all regressed or failed keep thresholds | Rejected unless future PyTorch/Transformers changes behavior |
| Fused sampling/logits processors | Post-warmup trace job `49133341`: sampling/logits kernels are milliseconds inside a `2.562s` model record | Not target-sized under current trace |
| Attention backend work | FA2 lost prior A5000 single-song and kernel microprofiles; SDPA already dispatches fused flash-style kernels; prefix trim regressed | Keep SDPA baseline |
| TensorRT-RTX | Isolated env import passed, but toy FP32 lowering in job `49138769` failed TensorRT engine creation and returned GraphModule fallback | Do not proceed to Mapperatorinator export until a toy graph creates a real engine |
| Parallel/server/window batching | Changes prompt dependency, batching, server overhead, or window behavior unless separately proven token-identical | Not an equivalent speed claim |

## Final Subagent Check

Final read-only explorer `019f1c3b-a844-73d1-8e67-858ad3b51732` independently recommended stopping. It found no remaining exact-calculation path both plausible for a `>=10%` full-song win and worth running before the stop decision. It specifically called out wrapper cleanup, fused sampling/logits, copy-style custom decode, TensorRT-RTX fallback, attention/backend work, and batching/parallel/server paths as exhausted, non-equivalent, or not target-sized.

## Carry Forward

Carry forward:

- `inference_generation_compile=true` as the retained same-calculation baseline.
- The 15s SALVALAI smoke gate with fixed-seed generated-token comparison.
- SDPA as the current baseline attention implementation.
- The rule that full-song token equivalence is required for accepted speed claims.

Future work toward `200 tok/s` should be treated as a separate runtime/kernel project, not another quick-tweak loop. Plausible future work would need new evidence from a changed architecture, a changed PyTorch/Transformers/TensorRT runtime, or a true custom decode/CUDA-graph implementation that first proves compile-disabled 15s equivalence, then compile-enabled 15s equivalence, then full-song equivalence.
