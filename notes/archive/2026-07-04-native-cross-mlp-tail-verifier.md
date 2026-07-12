# Native Cross+MLP Tail Verifier

## Purpose

Test one multi-segment decoder-layer candidate without touching production
inference routing. The candidate keeps the reference self-attention segment and
self-cache write, then replaces the adjacent cross-attention residual and MLP
residual tail with existing native one-token helpers.

This is diagnostic target sizing only. It is not a `profile_inference`
throughput claim and it does not change the accepted `270.475 tok/s` baseline.

## Why This Was The Right Target

The current frontier requires target-sized work, not another narrow loop. The
single-segment and per-linear paths were already below threshold, while the
native self+cross prefix failed strict cache-slot SHA because it changed the
self-cache write numerically.

This candidate is a bounded middle ground:

- it attacks two measured above-floor segments, cross-attention and MLP;
- it preserves the reference self-attention cache write;
- it stays inside `utils/profile_decode_decoder_layer_island.py`;
- it adds no production flags and does not touch `inference.py`, `server.py`,
  `config.py`, or Hydra defaults.

## Code

Branch:

```text
codex/native-cross-mlp-verifier
```

Commits:

```text
b022e86 add native cross mlp decoder verifier
02813ee fix weighted decoder variant requirements
```

Files:

- `utils/profile_decode_decoder_layer_island.py`
- `utils/validate_decoder_layer_abi.py`
- `utils/summarize_weighted_decoder_layer_island.py`

The new flag is:

```text
--candidate-native-cross-mlp-tail
```

It emits variants:

```text
native_cross_mlp_tail_warp2
native_cross_mlp_tail_warp4
native_cross_mlp_tail_warp8
```

## Single-Prefix Verifier

DCC job:

```text
49266324
```

Run directory:

```text
/work/imt11/Mapperatorinator/runs/native-cross-mlp-49266324-b022e86
```

Artifacts:

```text
/work/imt11/Mapperatorinator/runs/native-cross-mlp-49266324-b022e86/native_cross_mlp_tail.json
/work/imt11/Mapperatorinator/runs/native-cross-mlp-49266324-b022e86/native_cross_mlp_tail_validation.json
```

Environment:

| Field | Value |
| --- | --- |
| node | `dcc-core-ferc-s-z25-20` |
| GPU | `NVIDIA GeForce RTX 2080 Ti`, UUID `GPU-de50b870-b708-690d-74a4-6e855163a133` |
| driver/CUDA | `595.71.05` / `13.2` |
| torch | `2.10.0+cu128` |
| transformers | `4.57.3` |
| config | `profile_salvalai_smoke15`, sequence `9`, active prefix `128`, fp32, SDPA |

Slurm state was `FAILED` only because a final helper heredoc tried to read a
literal `$RUN_DIR/native_cross_mlp_tail.json` after the verifier artifacts were
already written. The report and validation JSONs are valid.

Result:

| Check | Result |
| --- | --- |
| report pass | `True` |
| logits replay | PASS, max abs `0.0` |
| candidate cache-write checks | PASS |
| ABI validation | PASS |
| best single-prefix projected saving | warp4: `2.232917s`, projected `293.694 tok/s` |

All native cross+MLP variants preserved the expected self-cache K/V slot exactly:
key/value SHA matched, bitwise mismatch count `0`, numeric mismatch count `0`.
Candidate output allclose passed with max abs `7.62939453125e-06`.

## Weighted Full-Bucket Verifier

DCC job:

```text
49266355
```

Run directory:

```text
/work/imt11/Mapperatorinator/runs/native-cross-mlp-weighted-49266355-b022e86
```

Fixed weighted summary:

```text
/work/imt11/Mapperatorinator/runs/native-cross-mlp-weighted-49266355-b022e86/weighted_native_cross_mlp_tail_summary.fixed.json
```

Environment:

| Field | Value |
| --- | --- |
| node | `dcc-core-ferc-s-z25-20` |
| GPU | `NVIDIA GeForce RTX 2080 Ti`, UUID `GPU-ca8caefc-47d0-69fb-2718-9f51f0cb67c1` |
| driver/CUDA | `595.71.05` / `13.2` |
| torch | `2.10.0+cu128` |
| transformers | `4.57.3` |
| weighted source | `/work/imt11/Mapperatorinator/runs/decoder-stack-weighted-49233687/weighted_decoder_stack_summary.json` |

Slurm state was `FAILED` only because the original weighted summarizer CLI
incorrectly appended explicit `--require-variant` values to its default
`manual_decoder_runtime_island` requirement. Commit `02813ee` fixed the harness
and the fixed summary over the same bucket reports passes.

Weighted result:

| Variant | Weighted seconds | Saved vs repo | Projected tok/s | Clears 5% | Clears 10% |
| --- | ---: | ---: | ---: | --- | --- |
| `repo_decoder_layer` | `16.766751s` | `0.000000s` | `270.474` | no | no |
| `native_cross_mlp_tail_warp2` | `15.120354s` | `1.646397s` | `287.217` | yes | no |
| `native_cross_mlp_tail_warp4` | `15.116964s` | `1.649787s` | `287.254` | yes | no |
| `native_cross_mlp_tail_warp8` | `15.045932s` | `1.720819s` | `288.023` | yes | no |

All `11` bucket reports passed logits replay, candidate cache-write checks, and
ABI validation. The weighted decode replay count was `7,552`.

## Decision

Keep the verifier infrastructure and treat the candidate as eligible for a
bounded production spike, not as a graduated speedup.

Rationale:

- It is exact under the current cache-write/logits verifier.
- It clears the `1.412s` 5% projected full-song model-time bar when weighted.
- It does not clear the `2.824s` 10% strong bar.
- It projects only `~288 tok/s`, so it is not a path to `500 tok/s` by itself.
- Production transfer is unproven; the prior MLP-tail path also looked useful
  in replay and then landed below the promotion threshold in `profile_inference`.

Next production spike, if attempted, must be default-off and routed through
`inference.py` / `server.py:model_generate()` only after the normal gates:

1. one-token logits/top-k gate;
2. direct-loop token/logit/RNG gate;
3. 15s smoke generated-token equivalence;
4. full-song SALVALAI `--strict-full-song` token/output/no-regression gate.

Stop or revert production wiring if the 15s smoke or full-song untraced
`profile_inference` result is below the 5% keep bar, regresses stage wall, or
fails any equivalence/output check.
