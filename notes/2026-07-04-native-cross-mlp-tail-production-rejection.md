# Native Cross+MLP Tail Production Rejection

## Summary

The verifier-only native cross+MLP tail looked target-sized in weighted CUDA
graph replay, but the production opt-in spike did not clear the promotion bar
cleanly enough to keep. It passed exactness gates and improved full-song
main-generation model time by about `4.6-5.4%`, but one reciprocal full-song
pair landed below the `5%` keep threshold, timing-context strict checks failed,
and the branch touched the production decoder forward path. Leave it unmerged.

The accepted single-song baseline remains DCC job `49230082`: `270.475 tok/s`,
`7,639` main tokens, `28.243s` synchronized model time, main/timing token
equivalence PASS, and byte-identical `.osu` output.

## Branch

```text
codex/native-cross-mlp-production
```

Commits left unmerged:

```text
f9bf235 add native cross mlp production flag
4e7c73f extend decode verifiers for native cross mlp
```

The branch added a default-off flag:

```text
inference_native_cross_mlp_tail=true
```

It routed the flag through `inference.py`, `server.py:model_generate()`, runtime
profile metadata, and `VarWhisperDecoderLayer.forward()`. The candidate kept
the accepted native self-attention/RoPE/cache path and replaced decoder
cross-attention plus MLP tail with native one-token helpers for map/main
generation only. Timing contexts stayed on the normal path.

## Exactness Gates

| Gate | Job | Result |
| --- | ---: | --- |
| one-token logits/top-k | `49266703` | PASS |
| direct-loop 256-step token/logit/RNG | `49266798` | PASS |
| 15s reciprocal smoke token/output equivalence | `49266821` | PASS in both orders |
| full-song reciprocal token/output equivalence | `49266934` | PASS in both orders |

One-token report:

```text
/work/imt11/Mapperatorinator/runs/cross-mlp-one-token-49266703-4e7c73f/one_token_cross_mlp.json
```

Direct-loop report:

```text
/work/imt11/Mapperatorinator/runs/cross-mlp-direct-49266798-4e7c73f/direct_cross_mlp.json
```

15s smoke run:

```text
/work/imt11/Mapperatorinator/runs/cross-mlp-smoke15-49266821-4e7c73f
```

Full-song run:

```text
/work/imt11/Mapperatorinator/runs/cross-mlp-full-49266934-4e7c73f
```

Environment for the accepted gates was DCC `gpu-common`, RTX 2080 Ti on
`dcc-core-ferc-s-z25-20`, torch `2.10.0+cu128`, transformers `4.57.3`, fp32,
SDPA, `use_server=false`, `parallel=false`, seed `12345`.

## 15s Smoke

The 15s smoke used reciprocal order to expose cache/order effects:

| Pair | Control tok/s | Candidate tok/s | Model-time delta | Strict result |
| --- | ---: | ---: | ---: | --- |
| control then candidate | `287.424` | `308.875` | `3.771s -> 3.510s` (`+7.5% tok/s`) | PASS |
| candidate then control | `290.992` | `306.676` | `3.725s -> 3.535s` (`+5.4% tok/s`) | FAIL |

Both smoke orders matched all `1,084 / 1,084` main generated token IDs and the
output artifact hash. The candidate-first strict failure came from early
setup-heavy map windows: `seq0`, `seq1`, and `seq2`. Token-heavy windows
`seq3+` improved by roughly `4-7%`.

## Full-Song Result

Full-song validation also used reciprocal order:

| Pair | Main tokens | Control model time | Candidate model time | Control tok/s | Candidate tok/s | Stage wall | Strict result |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| control then candidate | `7,639` | `28.126s` | `26.824s` | `271.597` | `284.780` | `43.654s -> 42.204s` | FAIL |
| candidate then control | `7,639` | `28.426s` | `26.888s` | `268.733` | `284.103` | `43.716s -> 42.534s` | FAIL |

Both pairs matched all `7,639 / 7,639` main tokens, all `821 / 821` timing
tokens, and the final output bytes:

```text
sha256=483483a1c29ef8a44c4a8d3a82fe0778ae306470ec3157e98969eeabd92c2631
size_bytes=31709
```

The main-generation model-time gain was real but marginal:

- pair A: `28.126s -> 26.824s`, saving `1.302s` (`+4.9% tok/s`);
- pair B: `28.426s -> 26.888s`, saving `1.538s` (`+5.7% tok/s`);
- average saving: about `1.42s`, roughly the `5%` bar for the accepted
  `28.243s` baseline.

Strict full-song failed because per-window no-regression failed. Main-generation
failures were tiny in aggregate, but timing-context checks also failed:

- pair A main-generation bad windows totaled about `0.0043s`, but
  timing-context had `78` model-time regressions totaling about `0.080s`;
- pair B main-generation had no model-time regressions in the direct parser,
  but timing-context regressed `8.144s -> 8.233s` and had `25` regressing
  timing windows totaling about `0.121s`.

## Decision

Reject the production path and leave branch
`codex/native-cross-mlp-production` unmerged.

Rationale:

- the exactness gates passed, so this is a performance decision, not a
  correctness failure;
- full-song main-generation improvement is only borderline `5%`;
- timing-context strict checks failed despite the candidate being disabled
  there, showing the run is still order/noise sensitive;
- the code is not simple: it adds a public flag, control-plane routing, profile
  metadata, and a broad decoder-forward branch;
- the projected verifier signal did transfer to production only partially.

Keep the verifier-only infrastructure from `codex/native-cross-mlp-verifier`,
but do not reintroduce the production flag without new bottleneck evidence that
projects comfortably above the keep bar, preferably above `10%`, and explains
why timing/stage regressions will not recur.

## Next Target

Return to broad decoder-layer or decoder-stack math/memory verifier work. The
next candidate should attack multiple adjacent operation classes and should
remain verifier-only until CUDA-graph replay projects a comfortably target-sized
full-song saving. Do not spend more time tuning this narrow cross+MLP
production path.
