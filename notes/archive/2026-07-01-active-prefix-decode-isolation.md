# Active-Prefix Decode Isolation

## Summary

This note records the first strong positive runtime signal after the retained generation-compile win. At the time it was a diagnostic ceiling probe, not an accepted inference speedup. Follow-up full-song jobs kept the generated-token loop as an opt-in strategic candidate rather than the retained cold single-song baseline; see `notes/2026-07-01-active-prefix-decode-loop-full-song.md`.

The active-prefix idea should only be applied to the one-token decode step. Applying it during prefill is not equivalent.

## Inputs

- Commit: `51f189f`
- Job: `49140082`
- Node/GPU: `dcc-core-ferc-s-z25-20`, NVIDIA GeForce RTX 2080 Ti, driver `595.71.05`, capability `7.5`
- Config: `configs/inference/profile_salvalai_smoke15.yaml`
- Key flags: `precision=fp32`, `attn_implementation=sdpa`, `use_server=false`, `parallel=false`, `cfg_scale=1.0`, `num_beams=1`, `sequence_index=9`
- Run dir: `/work/imt11/Mapperatorinator/runs/active-prefix-isolate-49140082-51f189f`
- Slurm logs:
  - `/work/imt11/Mapperatorinator/logs/active-prefix-isolate-49140082.out`
  - `/work/imt11/Mapperatorinator/logs/active-prefix-isolate-49140082.err`

## Results

| Variant | Compile | Report | Gate result | Timing result |
| --- | --- | --- | --- | --- |
| baseline | false | `gate_baseline_compile_false.json` | PASS, `max_abs=0.0`, top-k match | not timed in this job |
| active-prefix prefill only | false | `gate_prefill_compile_false.json` | FAIL, `max_abs=15.413055`, top-k mismatch | skipped |
| active-prefix decode only | false | `gate_decode_compile_false.json` | PASS, `max_abs=0.0`, top-k match | graph report PASS |
| active-prefix prefill + decode | false | `gate_both_compile_false.json` | FAIL, `max_abs=15.413055`, top-k mismatch | skipped |
| baseline | true | `gate_baseline_compile_true.json` | PASS, `max_abs=2.2888e-05`, top-k match | not timed in this job |
| active-prefix prefill only | true | `gate_prefill_compile_true.json` | FAIL, `max_abs=15.413059`, top-k mismatch | skipped |
| active-prefix decode only | true | `gate_decode_compile_true.json` | PASS, `max_abs=2.2888e-05`, top-k match | graph report PASS |
| active-prefix prefill + decode | true | `gate_both_compile_true.json` | FAIL, `max_abs=15.413059`, top-k mismatch | skipped |

Graph timing for the decode-only passing variant:

| Compile | Report | Eager ms/step | Graph ms/step | Speedup | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| false | `graph_decode_compile_false.json` | `11.4996ms` | `3.7891ms` | `3.035x` | exact vs HF raw logits |
| true | `graph_decode_compile_true.json` | `11.7837ms` | `3.7899ms` | `3.109x` | exact within compile-enabled tolerance |

Previous fixed-step graph baseline job `49139948` measured about `8.09ms` per graph replay without active-prefix decode. The decode-only active-prefix graph ceiling is therefore about `2.13x` faster than the previous fixed-step graph ceiling.

## Interpretation

- Active-prefix during prefill changes the calculation enough to fail the one-token logits gate. Do not apply active-prefix self-attention to prompt prefill.
- Active-prefix during the one-token decode step is logits-equivalent for the tested `seq9` gate with compile disabled and enabled.
- The fixed-step graph timing moves from a `~124 tok/s` ceiling (`8.09ms`) to a `~264 tok/s` fixed-step ceiling (`3.79ms`) before real loop overhead. This is the first measured path with plausible arithmetic for `200 tok/s`.
- This is still not a full inference speedup. The current timing captures one prepared candidate step with fixed shape, not a real loop with changing tokens, changing active prefix length, logits processors, sampling, EOS behavior, RNG consumption, or generated-token accounting.

## Decision

Keep the diagnostic infrastructure. Promote only the decode-only concept to a real candidate path. The next implementation should leave prefill unchanged, then try a direct decode loop that uses active-prefix only after the normal static-cache prefill.

Graduation gates remain:

1. One-token logits gate PASS across representative sequence positions and active prefix lengths.
2. 15s middle-song smoke generated-token equivalence PASS with compile disabled.
3. 15s middle-song smoke generated-token equivalence PASS with compile enabled or an explicitly documented replacement runtime.
4. Full-song SALVALAI generated-token equivalence PASS and untraced throughput before any accepted speed claim.

## Next

- Build an opt-in batch-1 direct decode loop using `osuT5.osuT5.inference.direct_decode`.
- Preserve HF generation semantics by reusing existing logits processors and stopping criteria before changing any sampling code.
- Use normal prefill, then decode-only active-prefix inside the one-token model forward.
- Expect a shape/capture challenge: the active prefix length changes per generated token. A real implementation may need bucketed CUDA graphs, per-length graph caches, or an active-prefix kernel/cache layout that keeps graph shapes stable.
- Do not revive prefill active-prefix unless a new isolated test explains and fixes the large logits mismatch.

## Bucketed Decode Follow-Up

Job `49140217` tested graph-reusable active-prefix decode bucket lengths after commit `fb2b2ae`.

- Node/GPU: `dcc-core-ferc-s-z25-21`, NVIDIA GeForce RTX 2080 Ti, driver `595.71.05`, capability `7.5`
- Run dir: `/work/imt11/Mapperatorinator/runs/active-prefix-buckets-49140217-fb2b2ae`
- Logs:
  - `/work/imt11/Mapperatorinator/logs/active-prefix-buckets-49140217.out`
  - `/work/imt11/Mapperatorinator/logs/active-prefix-buckets-49140217.err`

All tested buckets preserved the one-token logits gate:

| Bucket | Compile | Gate max_abs | Gate top-k | Graph ms/step |
| ---: | --- | ---: | --- | ---: |
| 128 | false | `0.0` | PASS | `3.8920ms` |
| 256 | false | `0.0` | PASS | `4.1022ms` |
| 512 | false | `0.0` | PASS | `4.4178ms` |
| 128 | true | `2.2888e-05` | PASS | `3.7597ms` |
| 256 | true | `2.2888e-05` | PASS | `3.9840ms` |
| 512 | true | `2.2888e-05` | PASS | `4.4166ms` |

Interpretation:

- Bucketed decode is exact for the tested `seq9` probe as long as prefill remains unchanged.
- Bucket lengths preserve the causal mask for future positions while avoiding the full `kv2560` static-cache SDPA shape.
- Bucketed shapes are a better candidate for real runtime work than exact-prefix shapes, because they create reusable graph/kernel shape families such as `128`, `256`, and `512`.
- Even bucket512 remains much faster than the prior full-static fixed-step graph ceiling (`8.09ms`), so a graph-backed bucketed loop is still plausibly `200 tok/s`-class if loop overhead and cache-buffer discipline can be controlled.

Next direct-loop work should compute `bucket = ceil(prefix_length / bucket_size) * bucket_size`, cap it at the static cache length, and apply active-prefix only during one-token decode. Generated-token equivalence is still required before any throughput claim.
