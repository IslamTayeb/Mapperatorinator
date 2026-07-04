# Cold-Start Considerations For 500 tok/s

## Summary

Cold-start optimization is part of the single-song goal. The current fastest
exact opt-in stack is `270.475 tok/s` on full-song SALVALAI, but warmed replay,
same-process repeat, and verifier CUDA graph timings do not prove cold
single-song speed by themselves.

Future work should measure cold first-window setup costs on the current fused
stack before trying preload, graph priming, or compiler tweaks.

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

## Next Cold-Start Experiments

1. Rerun a current-stack cold first-record audit with the fused
   `270.475 tok/s` opt-in stack. Report `main_first_record`,
   `main_remaining_records`, timing `seq0`, output hashes, model-generate CUDA
   ledger, graph capture counts, and total timing+map stage wall.
2. If first-window setup remains target-sized, test explicit native extension
   preload and graph/cache priming only when the cost is included in total cold
   stage wall. Do not move setup cost outside the measured claim. The native
   q1 extension preload is a plausible low-risk wall-time cleanup, but likely
   saves little or no synchronized model time.
3. Use the multistep tail graph verifier before any production tail graph
   candidate, but require cold 15s and cold full-song evidence before promotion.
4. Treat broad decoder-layer/native runtime work as the main path to 500 tok/s.
   Narrow cold-start cleanup is useful only if it produces a measured full-song
   non-regressing gain.

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
