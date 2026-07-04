# Cold-Start Considerations For 500 tok/s

## Summary

Cold-start optimization is part of the single-song goal. The current fastest
exact opt-in stack is `270.475 tok/s` on full-song SALVALAI, but warmed replay,
same-process repeat, and verifier CUDA graph timings do not prove cold
single-song speed by themselves.

Current-stack cold first-window setup was measured on 2026-07-04. The result is
specific enough that cold-start work should not become the main optimization
loop unless user-visible total cold wall time becomes the active target.

## Current Baselines

- Conservative cold default: SDPA + `inference_generation_compile=true`,
  active-prefix disabled. Job `49113713` measured full-song SALVALAI at
  `92.465 tok/s`, `7,639` main tokens, `82.615s` synchronized model time, with
  fixed-seed token equivalence PASS.
- Current fastest exact opt-in stack: active-prefix bucket64 CUDA graph,
  warmup0, stateful monotonic, q1 BMM cross-attention, persistent DecodeSession
  graph/cache reuse, native q1 self-attention, and fused RoPE/cache
  self-attention. Job `49230082` measured `270.475 tok/s`, `28.243s`, token
  equivalence PASS for main/timing, and byte-identical `.osu` output.

## Existing Cold-Tax Evidence

- Early active-prefix graph work had a large first-window tax. Cold active512
  `seq0` was around `25-26s`, while warmed active512 `seq0` was around
  `3.8s`; cold active512 after `seq0` was already about `131 tok/s`.
- Active-prefix primer work improved the first long window but moved cost into
  earlier records and worsened aggregate throughput, so it was reverted.
- Active-prefix mask/prepare fast paths and fast-prepare production wiring had
  exactness or local hints, but cold/full-song aggregate results regressed and
  were rejected.
- DecodeSession graph/cache reuse made duplicate graph capture too small to be
  the next lever. Post-fused diagnostics showed only about `0.117s` duplicate
  capture ceiling.
- Native q1 self-attention had a cold setup caveat: main model time improved,
  but first-run outer wall could pay extension/setup overhead.

## Current-Stack Audit

Jobs:

- Isolated extension-cache audit: DCC job `49246259`, run dir
  `/work/imt11/Mapperatorinator/runs/current-cold-optin-audit-49246259-52d6f19`.
- Persistent extension-cache audit: DCC job `49246261`, run dir
  `/work/imt11/Mapperatorinator/runs/current-cold-warmext-audit-49246261-52d6f19`.
- Both ran on RTX 2080 Ti, commit `52d6f19`, full-song SALVALAI, seed `12345`,
  fp32, SDPA, `use_server=false`, `parallel=false`, and the accepted opt-in
  stack:
  active-prefix bucket64 CUDA graph, warmup0, stateful monotonic, q1 BMM
  cross-attention, DecodeSession graph/cache reuse, native q1 self-attention,
  and fused RoPE/cache self-attention.
- All reported output hash
  `483483a1c29ef8a44c4a8d3a82fe0778ae306470ec3157e98969eeabd92c2631`, size
  `31709`, so generated `.osu` output stayed byte-identical across the audit.

| Case | Extension cache policy | Main model time | Main tok/s | Main stage wall | Main seq0 wall | Timing tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `49246259` clean cold | isolated empty `TORCH_EXTENSIONS_DIR` | `28.141s` | `271.456` | `90.247s` | `63.372s` | `99.756` |
| `49246261` clean cold | persistent dir, initially cold | `28.397s` | `269.006` | `96.096s` | `68.961s` | `102.929` |
| `49246261` diagnostic after cache build | persistent dir warm | `31.157s` | `245.177` | `33.130s` | `3.351s` | `99.689` |
| `49246261` warm-repeat run 0 | persistent dir warm | `28.605s` | `267.049` | `31.633s` | `4.258s` | `97.847` |
| `49246261` warm-repeat run 1 | same process warm | `28.599s` | `267.109` | `29.273s` | `1.891s` | `132.520` |

The clean compiler-cold runs preserved the accepted model-time throughput band:
`269-271 tok/s` versus retained `270.475 tok/s`. The problem is not decoder
TPS; it is first-use native extension build/load showing up as first map-window
outer wall and stage wall. Once the extension is built, first map-window wall
falls back to a few seconds while synchronized model time remains similar.

Strict comparison against job `49230082` failed because total stage wall and
per-window no-regression checks catch the first-use setup tax. Token equivalence
passed for main and timing in the compare output. The baseline profile used for
that comparison did not have output-hash fields, but every audit output file
hash matched the same value above.

## Decision

Cold-start native extension handling is worth a small packaging/setup pass, not
a deep decoder-runtime campaign:

1. Prefer a persistent `TORCH_EXTENSIONS_DIR` for normal DCC runs and record its
   policy in cold-start reports.
2. Consider an explicit prebuild/preload step for user-facing deployments so
   first inference does not pay a one-minute extension compile inside
   `main_generation`.
3. Do not claim this as a `tok/s` speedup unless synchronized
   `model_elapsed_seconds` improves. It is a total cold wall-time/setup fix.
4. Do not launch more cold-start profiling unless total cold stage wall becomes
   the active bottleneck again.

## Next Cold-Start Experiments

1. If first-window setup remains target-sized for users, test explicit native extension
   preload and graph/cache priming only when the cost is included in total cold
   stage wall. Do not move setup cost outside the measured claim. The native
   q1 extension preload is a plausible low-risk wall-time cleanup, but likely
   saves little or no synchronized model time.
2. Use the multistep tail graph verifier before any production tail graph
   candidate, but require cold 15s and cold full-song evidence before promotion.
3. Treat broad decoder-layer/native runtime work as the main path to 500 tok/s.
   Narrow cold-start cleanup is useful only if it produces a measured full-song
   non-regressing gain.
4. Refresh the current bottleneck split and theoretical ceilings before the next
   implementation project; do not spend more time on cold setup unless the
   refreshed profile shows it is still a dominant user-facing cost.

## Exactness Risks

- Do not naively reuse DecodeSession graph caches across timing and main
  contexts. Timing contexts intentionally disable some native paths, and graph
  signatures must stay tied to the exact cache/context/backend state.
- Be careful with startup/model-load reordering. The seed is set before model
  and diffusion setup, so delaying or reordering model construction can consume
  RNG differently unless CPU/CUDA RNG states are explicitly preserved and
  restored.
- Static cache preallocation is low-risk only if the same cache object is reset
  before use and memory pressure is checked. It is unlikely to be a major win
  without new profiling evidence.
- Server cold behavior is separate from this single-song path because the
  current accepted DecodeSession/native path is batch-1 and `use_server=false`.

## Reporting Rule

Always separate:

- cold single-song throughput;
- first-window setup and compile/capture cost;
- same-process warm-repeat throughput;
- verifier or CUDA graph replay ceiling;
- multi-song or batch-adjacent throughput.

Do not replace cold single-song evidence with warmed or diagnostic numbers.
