# Inference Experiment Ledger

Last consolidated: 2026-07-10. This is the canonical decision index. Dated notes remain the detailed evidence archive and must not be deleted until a separate audit confirms every needed job, commit, artifact, conclusion, and revisit condition is represented here.

The 2026-07-10 deletion audit found only two fully superseded notes and removed
them: `2026-07-04-current-exact-optimization-frontier.md` and
`2026-07-07-batched-fast-branch-abandoned.md`. The remaining dated notes retain
unique evidence and stay until a self-contained historical evidence index
captures it; a link back to a note does not count as preservation.

Legend:

- **accepted**: retained default-off runtime component or durable verifier/measurement infrastructure;
- **rejected**: runtime/optimization code removed or not promoted;
- **throughput-only**: operational evidence without exact per-request output proof;
- **diagnostic**: target sizing or correctness infrastructure, not a TPS claim.

## Accepted Single-Song Runtime Chain

| Result | Job / commit | Evidence and artifacts | Decision / revisit condition |
| --- | --- | --- | --- |
| Generation compile | `49113713` / `3e9033c` | `7,639` tokens, `121.410 -> 82.615s`, `62.92 -> 92.465 tok/s`, token identity. [note](2026-07-01-generation-compile.md) | Accepted exact compile-only reference; keep default-off for short one-offs. |
| Active-prefix CUDA graph | `49167356` / `8e8757b` | `92.465 -> 106.125 tok/s`, exact tokens; run root `/work/imt11/Mapperatorinator/runs/active-graph-immediate-full-49167356-8e8757b`. [note](2026-07-01-active-prefix-cuda-graph-loop.md) | Accepted default-off B1 path; broaden only with fresh mode-specific gates. |
| Stateful monotonic processor | `49168188` / `a980c8d` | `106.125 -> 134.873 tok/s`, exact tokens, strict 87/87 main windows; run root `/work/imt11/Mapperatorinator/runs/stateful-monotonic-full-49168188-a980c8d`. [note](2026-07-02-stateful-monotonic-graph.md) | Accepted only with simple active graph; do not generalize to server/CFG/beam/batch without exact tests. |
| Active graph warmup zero | `49204568` / `f56f2f5` | reached `146.602 tok/s`, main/timing token identity; run root `/work/imt11/Mapperatorinator/runs/active-warmup0-full-isolated-49204568-f56f2f5`. [note](2026-07-02-active-graph-warmup0.md) | Accepted component; scoped sub-ms window jitter. |
| Active-prefix bucket 64 | `49206207` / `39e85e4` | bucket512 `147.223`, bucket192 `154.733`, bucket64 `155.578 tok/s`; run root `/work/imt11/Mapperatorinator/runs/active-bucket-full-49206207-39e85e4`. [note](2026-07-02-active-bucket-size-sweep.md) | Accepted max-main setting; bucket192 remains timing-stability fallback. |
| q_len=1 BMM cross-attention | `49213490` / `3af8d69` | `155.014 -> 201.125 tok/s`, exact main/timing, stage wall improved; run root `/work/imt11/Mapperatorinator/runs/q1bmm-full-49213490-3af8d69`. [note](2026-07-03-q1-bmm-cross-attention.md) | Accepted only for unmasked FP32 B1 q_len=1 cross-attention. |
| Five-song exact serial validation | `49218365`-`49218368` / `8a2de72` | separate cold aggregate `64.802 -> 195.545 tok/s`, exact per-song main tokens; run root `/work/imt11/Mapperatorinator/runs/five-song-profile-20260702-232516-8a2de72`. [note](2026-07-03-five-song-before-after-profile.md) | Accepted realistic serial evidence for the then-current stack; rerun on the `270.475` stack before using as current denominator. |
| Persistent DecodeSession | `49223294` / `768b50f` | `203.000 -> 216.173 tok/s`, main/timing token identity; run root `/work/imt11/Mapperatorinator/runs/decode-session-runtime-full-49223294-768b50f`. [note](2026-07-03-persistent-decode-session-runtime.md) | Accepted default-off B1 request-local graph/cache reuse; graph lookup cleanup is below threshold unless new traces disagree. |
| Native q1 self-attention | `49225493` / `c563af0` | `207.226 -> 237.111 tok/s`, exact main/timing and byte-identical output; run root `/work/imt11/Mapperatorinator/runs/native-q1-self-full-49225493-c563af0`. [note](2026-07-03-native-q1-self-attention.md) | Accepted for map/main context only; superseded by fused RoPE/cache subflag. |
| Fused RoPE/cache/native self-attention | `49230082` / `d7b8684` | `248.015 -> 270.475 tok/s`, exact main/timing and byte-identical output; run root `/work/imt11/Mapperatorinator/runs/fused-rope-cache-full-49230082-d7b8684`. [note](2026-07-03-fused-self-attention-cache-probe.md) | Current exact-output frontier. Broaden only through one-token, loop-RNG, smoke, full-song, and output-byte gates. |

## Accepted Batch And Verification Infrastructure

| Result | Job / commit | Evidence and artifacts | Decision / revisit condition |
| --- | --- | --- | --- |
| Static IPC batch harness and metadata | `49267768` / `1475062` | real B5 batches, `120.568` scheduler-wall tok/s; manifest under `/work/imt11/Mapperatorinator/runs/static-server-batch-smoke-20260704-142507-1475062`. [note](2026-07-04-batching-server-throughput-track.md) | Accepted infrastructure; throughput-only under shared RNG. |
| Explicit static request/group state | `49268272` / `41de52c` | real batches and no scheduler-wall regression against `49268405`; token totals differed under shared RNG. [note](2026-07-04-batching-server-throughput-track.md) | Accepted state-boundary infrastructure, not an exact speed result. |
| Static manifest self-validation | local tests / `68b63d3` | recomputes aggregates and validates unique batch ledger. [note](2026-07-04-static-server-manifest-self-validation.md) | Accepted promotion guardrail. |
| Max batch 10 operational point | `49268989` / `195dfd7` | max5 `134.781` -> max10 `151.011 tok/s`; run described in [batch track](2026-07-04-batching-server-throughput-track.md) | Accepted workload tuning evidence only; not a default exact optimization. |
| Arrival/state-hash scheduler ledger | `49269123` / `0e6346d`; local strict dry run / `fea2864` | arrivals, queue/state hashes, slot generations; DCC run root `/work/imt11/Mapperatorinator/runs/static-server-ledger-20260704-0e6346d`. [arrival note](2026-07-04-batching-arrival-ledger.md), [state note](2026-07-04-continuous-scheduler-state-validation.md) | Accepted model-free verifier infrastructure; do not call it continuous GPU batching. |
| Capacity-20 characterization | `49269905` / `b4039b0` | max10 `158.123`, max20 `149.374 tok/s`; run root `/work/imt11/Mapperatorinator/runs/static-server-capacity20-20260704-194727-b4039b0`. [note](2026-07-04-static-server-capacity20-profile.md) | Keep evidence and max10 capacity point; reject max20. |

## Durable Diagnostic Infrastructure

These tools remain useful even though their associated experiment did not become a production optimization.

| Tool / boundary | Representative evidence | Keep because |
| --- | --- | --- |
| One-token/direct-loop ABI and exact RNG gates | `49139917`; `49222209` / `c440416` | prevents a fast decoder path from silently changing raw logits, sampled tokens, or final RNG. |
| Persistent-session verifier | `49223121`, `49223151` / `40c411f` | proves request-local cache, encoder buffer, and graph reuse across windows. |
| Attention component profiler | `49222759`, `49222781`, `49222883` / `989359f` | measures real q_len=1 self/cross tensors and rejected PyTorch self-BMM correctly. [note](2026-07-03-decode-attention-component-probe.md) |
| Linear/MLP/decoder-stack profilers and rooflines | `49222623`, `49233687`, `49250117` / `407edbf` | sizes broad compute and bandwidth ceilings before kernel work. [linear roofline](2026-07-04-current-linear-roofline.md) |
| Replay-gap probe | `49250100` / `3257f57` | proved graph shell `0.082s` and static input copy `0.220s` are not targets. [note](2026-07-04-decode-replay-gap-probe.md) |
| Cache-write/decoder ABI verifier | `49253972` / `7044278` | checks that candidates rewrite the expected K/V slot with matching SHA and output behavior. [note](2026-07-04-decoder-layer-candidate-cache-write-checks.md) |
| Weighted decoder-layer summarizer | `49262108` / `fc66274` | prevents a promising single-prefix result from bypassing full-song bucket weighting. [note](2026-07-04-weighted-manual-decoder-layer-confirmation.md) |

## Rejected Or Cut Single-Song Families

| Family | Representative evidence | Why rejected | Revisit only if |
| --- | --- | --- | --- |
| Last-position logits projection | `49110976` / `786a5fd`; [note](2026-07-01-final-logit-projection.md) | no meaningful end-to-end win after correct wrapper routing | Transformers/model output ABI or profiler split materially changes. |
| Static-cache SDPA prefix trim | `49112400` / `aac2c4b`; [note](2026-07-01-static-cache-prefix-trim.md) | non-equivalent or not promotable on the real static-cache path | a new exact cache/mask layout proves current-stack headroom above `5%`. |
| Persistent static-mask mutation | `49137141` / `2d807c1`; [note](2026-07-01-persistent-static-mask.md) | state mutation/capture behavior did not produce a safe win | cache/graph ownership is redesigned and exact replay is proven. |
| Dynamic/default cache | `49114897` / `52b8871`; [note](2026-07-01-dynamic-cache.md) | slower/wrong target versus accepted static-cache generation | backend/cache implementation changes enough to invalidate the profile. |
| Forced compile configs (`dynamic`, `dynamic=false`, `fullgraph`, `max-autotune`) | `49116934` / `d6ce772`; `49118947` / `96aecec`; `49137638` / `149fe88`; `49136380` / `0cccf36` | regression, failure, or no target-sized improvement | a new PyTorch release/current trace supplies a specific reason, then rerun bounded smoke. |
| Copy-compatible/preallocated sample-loop replacements | `49135146`/`49135181` / `07b36a5`; `49121864` / `f7b5222`; [custom note](2026-07-01-custom-decode-loop-hook.md), [prealloc note](2026-07-01-preallocated-sample-loop.md) | did not improve real generation enough and added semantic risk | a broad DecodeSession plan removes measured control cost while preserving HF sampling/RNG. |
| `torch.inference_mode` wrapping | `49115936` / `02b2437`; [note](2026-07-01-inference-mode.md) | exact but only `+2.3%` full-song | current-stack projection exceeds `5%` for a simple isolated change. |
| Old pre-graph monotonic shortcut | `49109743` / `9d7e5b7`; [note](2026-07-01-monotonic-time-mask.md) | old implementation was not a retained win | already superseded by the accepted scoped stateful processor; do not restore the old path. |
| Forced math SDPA / backend quick swaps | `49139420` / `9d92c34`; [note](2026-07-01-sdpa-backend-audit.md) | token-exact paired math result still slower than retained baseline | a current broad trace identifies backend dispatch as target-sized; keep SDPA as present baseline meanwhile. |
| Active-prefix cold quick fixes (mask cap, primer, direct cache dispatch, input preallocation, cudagraph-tree disabling, bucket-local compile) | `49158365`, `49159121`, `49160690`, `49162138`, `49162714`/`49162877`, `49163585`; dated notes | aggregate/timing regressions despite isolated post-warm signals | a refreshed trace shows the same overhead remains above `5%` and the new design accounts for cold setup. |
| Simple stopping-criteria specialization / tiny buckets | `49204960`; `49208036` | only `+3.76%` or smaller/negative; did not attack synchronized control cost | folded into a broader exact device/runtime loop with target-sized combined ceiling. |
| PyTorch q1 BMM self-attention | `49222759`, `49222781`, `49222883` / `989359f`; [note](2026-07-03-decode-attention-component-probe.md) | helps only long-prefix tail; projected `0.463s` (`~1.2%`) | common production buckets `128..640` become faster enough for `>5%`. |
| More duplicate graph-cache cleanup | `49223379` / `80adf86` | remaining duplicate-capture ceiling about `2%` | a current full-song diagnostic shows exclusive capture cost above `5%`. |
| Per-linear call-form/native substitutions | `49222623` / `41a1af0`; `49233393` / `4206d28` | flat/regressing; memory-bandwidth floor leaves limited isolated headroom | broader adjacent operations amortize weight traffic/launches and verifier projection clears threshold. |
| Narrow `fc1+GELU` or MLP residual fusion | `49232032` / `0a89cf7`; `49233007` / `14a077c` | projected `0.379s` and `0.756-0.822s`, below `1.412s` | a broader MLP/layer candidate stably projects above `5%`. |
| Fast prepared-input path | gate `49232069`, full `49232174` / `7213058`; [note](2026-07-03-fast-prepare-fixed-probe.md) | exact but main `273.150 -> 267.683 tok/s`, stage wall regressed | new evidence shows prepare-input work became exclusive and target-sized. |
| Graph replay shell/input copy cleanup | `49250100` / `3257f57`; [note](2026-07-04-decode-replay-gap-probe.md) | only `0.082s` shell and `0.220s` copy | refreshed current-stack gap exceeds `1.412s`. |
| Regional decoder-layer `torch.compile` | `49231529` / `25d18e2`; [note](2026-07-03-compiled-decoder-layer-probe.md) | native pybind op was not traceable/capturable; incompatible with accepted graph runtime | native operations gain a proven graph-safe compiler registration and ceiling. |
| Manual Python/module decoder-layer recomposition | `49250163` / `d051721`; weighted `49262108` / `fc66274` | prefix640 regressed; weighted exact saving only `0.616916s` | a source-of-gap audit plus weighted all-bucket repeat stably projects `>=1.412s`. |
| Native MLP-tail production flag | gates `49258638`/`49258622`, smoke `49258644` / `7e89f17`; [note](2026-07-04-native-mlp-tail-production-rejection.md) | exact but only `+2.8%`; stage/per-window regressions; production wiring reverted | combined multi-segment candidate clears the full-song bar. |
| Native self+cross prefix as same-calculation | sizing `49258712`; classifier `49266140` / `b71c275`; [note](2026-07-04-native-prefix-exactness-classifier.md) | real numeric cache bit drift (`456` K, `595` V mismatches), output drift `7.63e-06` | user explicitly approves `documented-drift`, or a different operation order passes cache SHA/logit/token/RNG/output gates. |
| Fixed-K / multi-step graph | `49235335`, `49235341`; production tail `49250043` / `7891398` reverted by `aac5b7f`; [ceiling note](2026-07-03-kstep-graph-ceiling-probe.md), [rejection](2026-07-04-tail-graph-runtime-rejection.md) | modest replay ceiling, production `+0.375%`, and early EOS can over-advance RNG | an exact device-controlled early-exit/rollback design has a fresh `>5%` end-to-end ceiling. |
| Native extension cold-build work as model TPS | `49246259`, `49246261` / `52d6f19`; [note](2026-07-04-cold-start-considerations.md) | build adds `60-65s` outer wall but synchronized decode stays near `270` | total user-visible cold stage wall becomes the explicit objective. |
| TensorRT/Torch-TensorRT direct adoption | `49134026` / `a2cf83a`; [note](2026-07-01-tensorrt-feasibility.md) | unavailable in current env; SM75/FP32 support and fallback behavior require separate proof | isolated environment proves a real engine (not fallback), FP32 logits/tokens/output, and target-sized speed. |

## Rejected Or Cut Batch Families

| Family | Representative evidence | Why rejected | Revisit only if |
| --- | --- | --- | --- |
| Server generation compile | `49267683` / `2d0d6d7` | TorchInductor cudagraph TLS assertion in background batch thread | dedicated CUDA owner thread compiles/captures/replays and passes exact B1/server gates. |
| Lower static coalescing timeout | `49268950` / `68b63d3` | `0.2s=114.035`, `0.05s=102.046`, `0.02s=78.394 tok/s` | workload or scheduler structure changes and a new knob sweep passes operational gates. |
| Static `max_batch_size=20` | `49269905` / `b4039b0` | `158.123 -> 149.374 tok/s`, token shrink, near VRAM capacity | new kernel/scheduler changes B-scaling or memory floor; do not continue larger static sweeps now. |
| Static-window `parallel=true` as exact optimization | `49267817` / `1475062` | `58.749 tok/s`, sequence-count and all-song token/output mismatch | a dedicated exact window-batch engine preserves window/output semantics. |
| Compiled fast DecodeSession server branch | `codex/batched-fast-decode-session@a74537a` | exact B5 `263.544` below optimized serial; B10 `199.721`, non-equivalent and memory-limited | lower-level graph-step/batched decoder beats optimized serial with private per-request state. Never merge the branch wholesale. |
| Decoder-layer runtime island branch | `experiment/decoder-layer-runtime-island-do-not-merge@f9306d2` | audit-only speculative work; no accepted end-to-end win | cherry-pick only separately reviewed verifier/docs/tooling; never merge wholesale. |

## Current Open Families

| Family | Required pre-production evidence |
| --- | --- |
| Exact speculative verification | n-gram and v32-mini drafts at `K=2/4/8`; sequential-vs-q_len=K numerics; exact target RNG consumption; EOS/cache rollback; acceptance and target-call savings; weighted projection `>5%`. |
| Broad FP32 decoder-layer/stack verifier | ABI and cache-slot exactness; weighted prefixes `128..768`; projected `>=1.412s`, preferably `>=2.824s`; then full correctness ladder. |
| Batch physics: merged B vs independent B1 lanes | distinct-song and permutation exactness, staggered arrival/slot reuse, model-only and complete-step TPS, VRAM, timeline, and `>5%` aggregate gain over optimized serial. |
| Optimized offline scheduler | B1 parity through five-song/three-seed exact queue, explicit request state, scheduler-wall target, cold/latency/memory reporting. |

When an open family is accepted or cut, add the job, commit, artifact paths, exactness class, measured delta, explanation, and concrete revisit condition here before deleting or superseding its dated note.
