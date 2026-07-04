# Native Self+Cross Prefix Verifier

## Hypothesis

The July 4 bottleneck refresh said narrow MLP, self-attention-only, cross-attention-only,
tail, graph-shell, and input-copy work are the wrong targets. The remaining
target-sized boundary is broad one-token decoder-layer runtime/compute. A
bounded verifier should therefore measure a multi-segment decoder-layer prefix,
not another single-operation production flag.

## Implementation

Branch:
`codex/native-self-cross-prefix-verifier`

Commits:

- `23bebe6`: added verifier-only native one-token `RMSNorm+Linear` and
  `Linear+Residual` helpers plus
  `utils/profile_decode_decoder_layer_island.py --candidate-native-self-cross-prefix`.
- `e370bd5`: added cache-slot allclose/max-abs diagnostics for cache-writing
  candidates while keeping the existing strict SHA pass gate.

The new candidate replaces:

`self_attn_norm -> Wqkv -> RoPE/cache write/native q1 self-attn -> Wo/residual -> cross_attn_norm -> Wq -> q1 BMM cross-attn -> Wo/residual`

then runs the existing MLP residual segment so the verifier still compares at
the full decoder-layer output boundary.

No `inference.py`, `server.py`, `config.py`, or Hydra production flag was added.
This is verifier/profiler infrastructure only.

## DCC Results

### Combined manual + native prefix diagnostic

- Job: `49258712`
- Commit: `e370bd5`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti
  `GPU-de50b870-b708-690d-74a4-6e855163a133`
- Report:
  `/work/imt11/Mapperatorinator/runs/native-self-cross-prefix-49258712-e370bd5/native_self_cross_prefix.json`
- Slurm state: `FAILED` only because the strict report pass was false for the
  native candidate cache SHA check.
- Logits replay: PASS, `max_abs=0.0`

Projected full-song CUDA graph replay results at `profile_salvalai_smoke15`
seq9, forced active prefix `640`:

| candidate | strict cache SHA | cache allclose | output allclose | projected saved | projected tok/s |
| --- | --- | --- | --- | ---: | ---: |
| manual decoder runtime island | PASS | PASS | PASS | `3.537833s` | `309.207` |
| native self+cross prefix warp2 | FAIL | PASS | PASS | `4.936389s` | `327.761` |
| native self+cross prefix warp4 | FAIL | PASS | PASS | `4.961293s` | `328.112` |
| native self+cross prefix warp8 | FAIL | PASS | PASS | `4.928855s` | `327.655` |

The native prefix cache drift was numerically tiny but not byte-identical:

- warp2 K/V max abs: `7.15e-07` / `4.77e-07`
- warp4 K/V max abs: `0.0` / `0.0`, but SHA still differed
- warp8 K/V max abs: `0.0` / `0.0`, but SHA still differed
- output max abs for native prefix variants: `7.63e-06`

This is not exact same-calculation evidence. It only says the approximate native
prefix remains target-sized and may be worth revisiting if a later direct-loop
token/logit/RNG gate proves generated-token identity.

### Manual-only confirmation

- Job: `49258725`
- Commit: `e370bd5`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti
  `GPU-de50b870-b708-690d-74a4-6e855163a133`
- Report:
  `/work/imt11/Mapperatorinator/runs/manual-island-verify-49258725-e370bd5/manual_island.json`
- ABI validation:
  `/work/imt11/Mapperatorinator/runs/manual-island-verify-49258725-e370bd5/manual_island_abi_validation.json`
- Slurm state: `COMPLETED`
- Logits replay: PASS, `max_abs=0.0`
- Candidate cache-write checks: PASS for repo layer, self-attention residual
  segment, and manual decoder runtime island.
- ABI validation: PASS, zero failures/warnings.

Manual-only projected full-song CUDA graph replay:

| metric | value |
| --- | ---: |
| repo decoder layer | `17.847813s` |
| manual decoder runtime island | `16.022411s` |
| projected saved time | `1.825402s` |
| projected speedup over layer boundary | `1.113928x` |
| projected main-generation throughput | `289.163 tok/s` |
| residual segment sum | `14.527303s` |
| unexplained layer-vs-segments gap | `3.320510s` |

## Decision

Keep the verifier branch as an active diagnostic candidate for now, but do not
merge the native self+cross prefix as an accepted exact path and do not wire any
production flag yet.

The manual-only run is exact at the verifier boundary and clears the 5% projected
full-song bar (`1.825s` versus the `~1.41s` keep threshold), but it is not a
throughput claim. It conflicts with earlier manual-island measurements that were
flat or negative, so the next step must be a weighted/full-bucket confirmation
and source-of-gap audit before production integration.

The native self+cross prefix is larger (`~4.9-5.0s` projected), but strict cache
SHA fails. Cache and output are allclose, so it is useful as target sizing, not
same-calculation evidence. It must not advance without direct-loop token/logit
RNG gates and generated-output equivalence.

## Next

1. Rerun the manual decoder runtime island across the weighted full-song
   active-prefix bucket distribution, not just prefix640 seq9.
2. If the exact manual boundary repeatedly saves `>=1.4s`, inspect the actual
   repo-layer-vs-manual gap and design a production candidate through the normal
   `inference.py`/`server.py` control plane.
3. Treat the native self+cross prefix as a later approximate/native candidate
   only after the exact manual gap is understood. Its SHA mismatch means it needs
   stronger end-to-end token/logit/RNG proof before any speed claim.
