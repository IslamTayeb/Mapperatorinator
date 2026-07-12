# Inference Single-Song Frontier

Last consolidated: 2026-07-10. This document is the current single-song source of truth; dated notes remain the detailed evidence archive.

## Objective And Exactness

The target is normal FP32 single-song inference on one RTX 2080/2080 Ti. The accepted frontier is exact-output: same fixed-seed main/timing token IDs and counts, same stop behavior, same final output bytes, and no unreported material regression. Batching, multiple processes, reduced precision, sampling/RNG changes, or output-policy changes cannot count toward this number.

`500 tok/s` for the accepted SALVALAI transcript means `7,639` tokens in `15.278s`. From the current `28.243s` baseline, that requires `12.965s` saved (`45.9%`). The normal `5%` and `10%` keep bars are `1.412s` and `2.824s`.

## Accepted Baseline

Full-song SALVALAI job `49230082`, commit `d7b8684`, RTX 2080 Ti:

| Metric | Result |
| --- | ---: |
| Main tokens | `7,639` |
| Synchronized main model time | `28.243s` |
| Main throughput | `270.475 tok/s` |
| Timing throughput | `101.988 tok/s` |
| Main/timing token identity | PASS (`7,639` / `821`) |
| Generated `.osu` | byte-identical, `31,709` bytes |

Artifacts:

```text
/work/imt11/Mapperatorinator/runs/fused-rope-cache-full-49230082-d7b8684/control.profile.json
/work/imt11/Mapperatorinator/runs/fused-rope-cache-full-49230082-d7b8684/candidate.profile.json
/work/imt11/Mapperatorinator/runs/fused-rope-cache-full-49230082-d7b8684/compare_strict_full.json
```

The accepted immutable effective configuration is
`accepted-fp32-270.475-v1`. It was historically requested through the complete
legacy micro-flag bundle below; the canonical public selector is now
`inference_engine=optimized optimized_inference_mode=single`, and the complete
legacy bundle delegates to that same runtime:

```text
inference_generation_compile=true
inference_active_prefix_decode_loop=true
inference_active_prefix_decode_bucket_size=64
inference_active_prefix_decode_cuda_graph=true
inference_active_prefix_decode_cuda_graph_warmup=0
inference_active_prefix_decode_cuda_graph_min_decode_steps=1
inference_stateful_monotonic_logits_processor=true
inference_q1_bmm_cross_attention=true
inference_decode_session_runtime=true
inference_decode_session_cuda_graph=true
inference_native_decode_kernels=true
inference_native_q1_self_attention=true
inference_native_q1_rope_cache_self_attention=true
```

It is validated only for the simple FP32 batch-1, non-server, non-parallel path. Timing contexts intentionally stay on the normal self-attention path. The fused RoPE/cache candidate improved its same-job control `248.015 -> 270.475 tok/s` (`+9.1%`) and the previous accepted native-self-attention checkpoint `237.111 -> 270.475 tok/s`. Three main windows totaled `4.738ms` of scoped positive jitter versus `2.558s` aggregate model-time saving; timing aggregate improved.

Rollback validation after abandoning the batched-fast branch used job `49325496`, commit `f6add76`, and reproduced the 15-second accepted output hash at `280.3 tok/s`. This is a smoke health check, not a replacement full-song baseline:

```text
/work/imt11/Mapperatorinator/runs/rollback-fastpath-smoke-49325496-f6add76/profile/beatmap13bc54a39d704a799e211e79b1f60d88.osu.profile.json
```

### Current-main reproduction

Jobs `49542937` and `49542938`, commit `a42c250`, reran the full accepted
stack on one RTX 2080 Ti with explicit persistent compiler/cache paths. The
second run measured `7,639 / 28.197s = 270.916 tok/s` main and
`821 / 8.109s = 101.242 tok/s` timing. Both runs reproduced the historical
output exactly: SHA-256
`483483a1c29ef8a44c4a8d3a82fe0778ae306470ec3157e98969eeabd92c2631`,
`31,709` bytes. The `+0.16%` main difference from `270.475` is run noise, not a
promoted optimization.

The paired strict comparison passed calculation metadata, token identity, and
output bytes. Its aggregate main/timing/stage metrics improved on the second
run, but exact zero-tolerance per-window comparison flagged sub-percent jitter;
this is baseline variability, not candidate evidence. An earlier reproduction,
job `49542402`, had one `10.659s` timing compile window while steady windows
matched. That exposed an unset node-local compiler-cache ambiguity; commit
`bb1d441` now pins and records Inductor, Triton, and CUDA cache paths.

```text
/work/imt11/Mapperatorinator/runs/inference-denominator-single_full-49542937-a42c250
/work/imt11/Mapperatorinator/runs/inference-denominator-single_full-49542938-a42c250
/work/imt11/Mapperatorinator/runs/inference-denominator-single_full-49542938-a42c250/compare-warm-vs-cold.json
```

### Optimized-single architecture reproduction

The ownership migration does not replace the `270.475 tok/s` accepted numerical
frontier and makes no speed claim. At final runtime commit `f68cf2b`, optimized
attention dispatch, active-prefix attention state, single-song orchestration,
session state, logits, and kernels are owned under `osuT5/osuT5/inference/optimized/`;
later commits through `7326a62` add only final regression wrappers. Jobs
`49562049`, `49562055`, and `49562091` passed the real one-token, 8-step, and
256-step logit/top-k/token/RNG/cache gates. Reciprocal 15-second job `49562101`
matched all `1,084` main and `164` timing tokens and `.osu` bytes.

Reciprocal full-song job `49562130` compared the public optimized selector with
the delegated complete legacy bundle on an RTX 2080 Ti. Both orders matched all
`7,639` main and `821` timing tokens and output SHA-256
`483483a1c29ef8a44c4a8d3a82fe0778ae306470ec3157e98969eeabd92c2631`,
size `31,709`; aggregate main differences were `+0.2%` and `-0.1%` by launch
order. Job `49562226` then compared the pre-migration accepted legacy path with
the final delegated legacy path and preserved the same tokens and output, with
main differences of `+0.0%` and `-0.1%`. Both jobs exited nonzero only because
the strict comparator rejects ordinary zero-tolerance per-window jitter. Their
absolute throughput drifted below `270.475 tok/s` equally in controls and
candidates, so this is architecture-neutral reproduction evidence, not a new
frontier. Detailed paths, hashes, stage/timing deltas, and ownership boundaries
are in the [migration map](2026-07-10-optimized-single-architecture-migration-map.md).

Final default and compile-only V32 reciprocal smoke jobs `49562311` and
`49562444` also preserved all `1,084` main tokens, `164` timing tokens, and
output bytes. Default main deltas were `-0.8%/+0.4%` by order; compile-only
main deltas were `+0.4%/+0.5%`. Strict nonzero statuses came only from
zero-tolerance per-window jitter.

Reciprocal mixed-song static-server jobs `49562635` and `49562710` preserved
the V32/shared-global-RNG runtime contract, real batching, null/false optimized
state, candidate import coldness, and empty extension caches. Scheduler TPS
flipped from `-20.8%` control-first to `+7.8%` candidate-first because request
ordering changed shared-RNG stop lengths; strict comparisons remained red for
different generated work. The explicitly diagnostic B5 active-step rates were
`+1.3%` and `+23.6%` for the candidate. Neither diagnostic showed a negative
normalized B5 rate, but active-step/prefix work differed and the primary
scheduler comparisons were red. Server performance therefore remains
non-comparable under the shared-RNG contract; this is not exactness evidence or
a throughput win. The ownership migration is accepted across the tested
interfaces, and `270.475 tok/s` remains the single-song frontier.

## Accepted Improvement Chain

| Component | Full-song evidence | Effect | Scope |
| --- | --- | --- | --- |
| Generation compile | `49113713`, `3e9033c` | `62.92 -> 92.465 tok/s`, exact | conservative compile-only reference |
| Active-prefix CUDA graph | `49167356`, `8e8757b` | `92.465 -> 106.125 tok/s`, exact | default-off batch-1 |
| Stateful monotonic processor | `49168188`, `a980c8d` | `106.125 -> 134.873 tok/s`, exact | active graph only |
| Graph warmup zero | `49204568`, `f56f2f5` | reached `146.602 tok/s`, exact | active graph only |
| Bucket size 64 | `49206207`, `39e85e4` | reached `155.578 tok/s`, exact | bucket192 is timing-stability fallback |
| q_len=1 BMM cross-attention | `49213490`, `3af8d69` | `155.014 -> 201.125 tok/s`, exact | unmasked FP32 B1 cross-attention |
| Persistent DecodeSession | `49223294`, `768b50f` | `203.000 -> 216.173 tok/s`, exact | shared cache/graph within a request |
| Native q1 self-attention | `49225493`, `c563af0` | `207.226 -> 237.111 tok/s`, exact output | map/main context only |
| Fused RoPE/cache/native self-attention | `49230082`, `d7b8684` | `248.015 -> 270.475 tok/s`, exact output | current frontier |

Detailed artifacts and scoped regression notes are indexed in [the experiment ledger](inference-experiment-ledger.md).

## Realistic Multi-Song Serial Evidence

Jobs `49218365`-`49218368`, commit `8a2de72`, evaluated Lambada, PEGASUS, Ela ke Leitada, SALVALAI, and Nube Negra before the later DecodeSession/native additions. The opt-in path preserved main token IDs for every song and strict suite scope:

| Scope | Baseline | Optimized |
| --- | ---: | ---: |
| Separate cold aggregate | `64.802` | `195.545 tok/s` |
| Together first run | `64.614` | `201.749 tok/s` |
| Together all | `60.541` | `194.791 tok/s` |
| Together warmed | `56.834` | `194.116 tok/s` |

Run root:

```text
/work/imt11/Mapperatorinator/runs/five-song-profile-20260702-232516-8a2de72
```

This proves realistic cold/serial behavior and exactness for that earlier stack. It is not concurrent batching. The current-stack replacement denominator is recorded in [the batch frontier](inference-batch-frontier.md). Separate timing aggregate improved for all songs, but `timing/sequential/seq0` had a scoped first-record regression.

## Speculative Target-Span Decision

The zero-cost n-gram speculative runtime is rejected on current-stack cost despite clean exact-output target-span numerics:

| Gate | Job / commit | Exactness | Cost result |
| --- | --- | --- | --- |
| K2 eager target span | `49546980` / `8972bb7` | logits/top-k, tokens `[12, 1648]`, final RNG, forced EOS, and cache allclose pass | `11.365ms` span; diagnostic only |
| K4 eager target span | `49547062` / `cdfb577` | logits/top-k, tokens `[12, 1648, 2242, 2717]`, final RNG, forced EOS, and cache allclose pass | `10.770ms` span |
| K4 fixed-shape graph ceiling | `49547134` / `f4c2156` | graph pointers/output stable; replay logits bitwise to eager, tokens/RNG exact, cache allclose | `8.464ms`, above strict `<5.480ms` keep bar |

The K4 graph projects the five-full target path from `163.088s` / `~261.4 tok/s` to `197.472s` / `215.9 tok/s` (`-21.1%`). Slurm job `49547134` is `FAILED` only because the strict cost gate intentionally exited `1`; capture, graph safety, and numerical evidence passed. Reports:

```text
/work/imt11/Mapperatorinator/runs/spec-target-span-k2-49546980-8972bb7/target-span-k2.json
/work/imt11/Mapperatorinator/runs/spec-target-span-k4-49547062-cdfb577/target-span-k4.json
/work/imt11/Mapperatorinator/runs/spec-target-span-k4-49547134-f4c2156/target-span-k4.json
```

Keep the numeric/graph verifier, but do not build the n-gram runtime, run K8, or spend on rollback integration. Revisit only if the accepted q1 denominator materially slows, proposal/acceptance structure changes, or a new fixed-shape target kernel first demonstrates K4 below `5.480ms`. The physical `StaticCache` rollback API remains absent. The mock-only logical stale-suffix oracle in historical commit `cdfb577` never received a GPU gate and was removed by `b21109c` after the cost ceiling killed the family; do not treat it as retained runtime or verifier evidence.

### v32-mini bounded feasibility gate

The greedy v32-mini K4 draft is rejected by the one authorized real-window
scout. This is a fresh proposal family, not a revival of the n-gram call
structure, but its measured draft compute is far above the current-stack budget.

Pinned artifact validation used target revision `74f22583400d259bf424819e11027c17933efe54`
and mini revision `7807f0dc70cab671be012e1f5ddf945b0b8b7278`.
CPU preflight job `49548759` loaded both gamemode-0 FP32 models and tokenizers;
corresponding tokenizer and generation configs are byte-identical, and every
non-capacity config field matches. The gamemode-0 tokenizer SHA-256 is
`6b98be0fc04a95a9e9d4feb8e8b67cc48728a6667e3091dcd5cc528baeca18bd`.
The target is small (`216,304,896` parameters) while mini is base
(`55,646,720` parameters); they share token IDs and prompt construction but not
encoder outputs, weights, caches, graph buffers, or workspaces.

The target denominator is `q=3.825303748ms` per committed q1 output token and
the measured safe K4 target replay is `T=8.4637517ms`. Let `h_j` count K4 calls
that accepted exactly `j` draft tokens before the first mismatch, with `h_4`
meaning full acceptance. The actual closed-loop committed length is:

```text
L = (sum(j=0..3, h_j * (j + 1)) + 4 * h_4) / sum(j=0..4, h_j)
```

This is deliberately not marginal draft-token acceptance. A mismatch emits one
target token and immediately ends the span. For steady draft cost `D` per K4
proposal call:

```text
break even: D < 1.00 * q * L - T
5% keep bar: D < 0.95 * q * L - T
```

| Effective committed `L` | Break-even `D` | Strict 5% `D` |
| ---: | ---: | ---: |
| `1` | impossible (`-4.638ms`) | impossible (`-4.830ms`) |
| `2` | impossible (`-0.813ms`) | impossible (`-1.196ms`) |
| `3` | `<3.012ms` | `<2.438ms` |
| `4` | `<6.837ms` | `<6.072ms` |

Even a zero-cost draft therefore needs `L>2.329`, or more than `58.23%`
committed-span efficiency. Job `49548890`, commit `3ddb585`, ran the exact
seed-`12345`, FP32/SDPA, SALVALAI smoke15 main `sequence_index=9`,
`max_new_tokens=256` scout on an RTX 2080 Ti:

| Field | Result |
| --- | ---: |
| Target transcript / stop / final RNG | exact PASS; `256` tokens, `max_new_tokens` |
| Target/replayed token SHA-256 | `1adc1dbd1d15f6cd96888242beaa271b674125efc7cb6eef5e231d02c153e000` |
| Accepted-prefix histogram `h0..h4` | `[26, 23, 15, 9, 25]` over `98` full calls |
| Draft token acceptance / full accepts | `45.92%` / `25.51%` |
| Effective committed `L` | `2.5816` tokens per target call |
| Strict 5% draft budget at measured `L` | `<0.918ms` per full proposal |
| Warm mini encoder | `7.366ms` CUDA |
| Optimistic ready-cache proposal | `22.596ms` for the required `K-1=3` q1 calls |
| Actual rebuild-inclusive full proposal | `31.970ms` CUDA / `32.153ms` wall mean |
| Runnable finite-window projection | `0.979s -> 4.016s` CUDA (`-310.1%`) |
| Conservative wall projection | `0.979s -> 4.034s` (`-312.0%`) |
| Optimistic steady-cache projection | `0.979s -> 3.078s` (`-214.3%`) |
| Combined allocated VRAM peak | `1.284 GiB` (`1,556 MiB` telemetry peak) |

The accepted-prefix structure clears the zero-cost structural floor, but even
the optimistic draft cost is `24.6x` the `<0.918ms` keep budget. Both strict
CUDA and conservative wall gates fail. Slurm status `FAILED` is the intentional
exit `1` from that valid rejection, not a numerical or loader failure. The
authoritative report is:

```text
/work/imt11/Mapperatorinator/runs/spec-mini-onewindow-49548890-3ddb585/v32-mini-feasibility.json
```

Its SHA-256 is `322be4e4cd6d56d639e81f332ac4333ea555e9b3bb23242713d3c0a003bdc55f`.
Setup-only job `49548295` caught a missing active-prefix companion flag before
model load; `49548337` caught Transformers rejecting `subfolder=None` for an
already-resolved local gamemode path. Neither is performance evidence. CPU job
`49548759` then proved both pinned resolved-path loaders and tokenizers before
the final GPU run. Keep the model-free projection, transcript-replay, and loader
verifier infrastructure, but do not build mini rollback/runtime, run K8, or
expand to a second song/window. Revisit only if a different draft path first
shows sub-millisecond K4 proposal cost at comparable closed-loop acceptance.

## Broad Whole-Layer Gate Decision

The one authorized broad FP32 scout is rejected before weighted buckets or
runtime work. Job `49550902`, branch commit `ce82dda`, measured a verifier-only
prefix-640 whole decoder-layer candidate on an RTX 2080 Ti. All 12 layer outputs
and complete self caches passed the predeclared `atol=rtol=1e-4`; bytes outside
the written self-cache slot and the complete cross cache remained bitwise
unchanged. Baseline logits replay was bitwise exact.

Reciprocal output-discard CUDA-graph timing measured:

| Order | Baseline layer | Candidate layer | Extrapolated full-song saving |
| --- | ---: | ---: | ---: |
| baseline then candidate | `0.177353ms` | `0.155409ms` | `1.988652s` |
| candidate then baseline | `0.178277ms` | `0.155119ms` | `2.098751s` |

The required worst-order strong bar was fixed at `2.8243s`. The independent
validator therefore failed exactly one condition, and Slurm exit `1` is the
intentional performance rejection. The compiler-cold candidate stack took
`136.466s` from an enforced empty `TORCH_EXTENSIONS_DIR`; this is setup evidence,
not part of synchronized layer timing. Peak verifier state included `180 MiB` of
full-cache snapshots.

```text
/work/imt11/Mapperatorinator/runs/native-full-decoder-layer-gate1-49550902-ce82dda/native-full-decoder-layer-gate1.json
report SHA-256 71d8a367d9ef4260b3a201c47b7921c01cecffda2d80441955df8f1d8001c490
branch codex/optimized-native-full-layer-gate @ ce82dda
```

Do not run the weighted all-bucket verifier, a 256-step loop, smoke/full-song,
or production integration for this candidate. Its isolated layer improvement
is real but below the predeclared strong promotion signal, and the prefix-640
projection was already optimistic because it applied one measured delta to all
accepted buckets. Revisit only if a materially different whole-layer candidate
first clears `2.8243s` in the same reciprocal gate.

## Current Bottleneck And Ceiling

Post-frontier diagnostics show that another narrow wrapper or launch tweak cannot reach `500`:

- Weighted full-forward graph replay: about `17.120s`; even making all work outside it free while keeping it unchanged does not reach `500`.
- Weighted decoder stack replay: `16.666s`; optimistic bandwidth roofline `6.348s`, projecting only `426.179 tok/s` if the stack alone reached that floor.
- Captured one-token linears: `7.156s`; nominal bandwidth floor `5.027s`, only `2.129s` above-floor headroom and a `292.5 tok/s` projected total if linears reach that floor.
- Production graph shell versus isolated full forward: only `0.082s`; static input copy: `0.220s`.
- Production-like sampling/logits/EOS/append tail: about `1.6s` total, distributed across several operations and below a comfortable standalone keep case.
- Duplicate graph capture, final projection, individual linears, self-attention alone, cross-attention alone, and MLP alone are not `500`-capable boundaries.

The broad decoder-layer scout consumed the one roofline-authorized verifier and
missed its strong bar. No current single-song performance experiment is now
authorized: a new family must begin with a refreshed current-stack profile and
show more than `5%` avoidable end-to-end headroom before code is written.

## Current Stop/Go Rule

The next production-facing experiment must first be verifier-only and must:

1. replace multiple adjacent decoder operation classes, not another single wrapper or kernel;
2. preserve the decoder-layer ABI and exact cache-write slot checks where required;
3. show weighted CUDA-graph projected saving of at least `1.412s`, preferably `2.824s`;
4. then pass one-token logits/top-k/cache, 256-step token/logit/RNG, 15-second smoke, reciprocal-order full-song main/timing, and `.osu` byte gates.

There is no ranked single-song scout left on current evidence. A whole-step
device-controlled graph remains conditional only if a fresh profile first
shows more than `5%` exclusive headroom and exact early EOS/RNG rollback can be
proven; the existing graph-shell, input-copy, and tail measurements do not meet
that prerequisite.

Speculation must consume target RNG exactly one output position at a time, commit only matching draft tokens, discard uncommitted cache suffixes on mismatch, and preserve final token/RNG/output identity. Stop before production if draft cost plus verified target calls saved project below `5%`.

## Explicit Cuts

Do not restart these without new current-stack evidence:

- per-linear call-form rewrites, standalone final projection, narrow MLP or attention islands;
- graph-cache dictionary cleanup, static input-copy cleanup, fast prepare-input rewrites;
- naive fixed-K graphing that over-advances RNG at early EOS;
- zero-cost n-gram speculative runtime on the current q1 denominator;
- greedy `OliBomby/Mapperatorinator-v32-mini` K4 drafting on the current target denominator;
- manual Python/module decoder-layer recomposition;
- native self+cross prefix as `bitwise-calculation-exact` (cache writes have real FP32 bit drift);
- cold native-extension compilation as a synchronized model-TPS target;
- documented-drift/native reordering without explicit approval.

See [the experiment ledger](inference-experiment-ledger.md) for the representative job, commit, artifact, and revisit condition for each family.
