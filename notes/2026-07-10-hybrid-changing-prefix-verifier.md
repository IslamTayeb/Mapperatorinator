# Hybrid B2 changing-prefix verifier

This branch now contains the next bounded gate authorized by fixed-shape job
`49552768`. It is still verifier infrastructure, not an offline engine or
scheduler.

The verifier runs eight real closed-loop steps first. It advances private B1
model graphs/caches, a shared neutral-padded `[2,128]` monotonic/temperature
processor graph, and private TopP/sample/EOS graphs. It compares every step to
independent unpadded B1 generation, including raw logits/top-20, processed
scores, tokens, RNG, stopping, normal prepared static inputs, prefix state, and
self/cross caches. It times both reciprocal orders only after exactness passes.
Phase B (16 steps, five reset-separated timing trials) is unreachable unless
Phase A is exact and its slower complete-wall order exceeds `500 tok/s`.

The bridge reads `lane.graph_outputs.logits[:, -1, :]` through a persistent
zero-copy view; `CapturedB1Lane.logits` is intentionally forbidden because it
returns a clone. Cache reset restores post-prefill snapshots in place, and each
captured sampling generator starts from the independently verified post-anchor
RNG state. The timed loop uses precomputed normal-preparation tensors and
precomputed write views; it contains no per-step tensor assembly or sync.
The report keeps prefill/setup, graph warmup/capture, phase-total wall, current
and peak VRAM, and `TORCH_EXTENSIONS_DIR` before/after cache counts visible;
these costs are explicitly excluded from complete decode-loop TPS.

## Accepted bounded result

DCC job `49555060`, commit `2b6e690`, completed in `00:05:52` with exit
`0:0` on one RTX 2080 Ti. Phase A passed its eight-step exactness and `500
tok/s` gates in both reciprocal orders. Phase B then passed 16 changing-prefix
steps and five reset-separated timing trials per reciprocal order.

| Evidence | Result |
| --- | ---: |
| Phase A complete wall, order `0,1` / `1,0` | `622.671` / `619.208 tok/s` |
| Phase B complete wall, order `0,1` / `1,0` | `628.868` / `621.893 tok/s` |
| Phase B five-trial ranges, order `0,1` / `1,0` | `626.726-630.113` / `620.430-622.791 tok/s` |
| same-job private-B1 Phase B control | `416.683` / `416.643 tok/s` |
| worst Phase B gain versus same-job B1 | `+49.26%` |
| worst Phase B change versus fixed-prefix `670.026` | `-7.18%` |
| current / peak allocated VRAM | `1.971 / 2.512 GB` |
| current / peak reserved VRAM | `3.326 / 3.326 GB` |
| timed allocated / reserved delta | `0 / 0 bytes` |

All per-step raw logits, active self-cache prefixes, and current cache slots had
`max_abs=0` and matching SHA-256 values. Cross-cache hashes were unchanged and
matched. Processed scores were bitwise identical, neutral padding was proven
inert, static inputs and sampled tokens matched, and both private RNG streams
matched after every step and at the end. Cache restore, pointer stability,
reciprocal order, stop flags, and no-dummy-sampling gates also passed. Neither
row reached EOS, so this run did not exercise an actual EOS transition.

The five Phase B hybrid trial TPS values were:

- order `0,1`: `626.726`, `627.761`, `630.037`, `629.719`, `630.113`;
- order `1,0`: `621.706`, `621.949`, `622.791`, `622.594`, `620.430`.

Setup stayed outside complete decode timing but remains visible. The two serial
prefills cost `0.053646s` and `0.053109s` wall; model-graph warmup/capture cost
about `0.0113/0.0118s` and `0.0113/0.0110s` per lane. Native-context entry cost
`4.299s`. `TORCH_EXTENSIONS_DIR` was warm before the run and unchanged after it
(`10` entries, `112` files). Total gate wall was `306.126s`, including exact
references, reset checks, capture, both phases, and report construction; it is
not a decode-throughput denominator.

Artifacts:

```text
/work/imt11/Mapperatorinator/runs/hybrid-changing-prefix-49555060-2b6e690/hybrid-changing-prefix.json
/work/imt11/Mapperatorinator/runs/hybrid-changing-prefix-49555060-2b6e690/nvidia-smi.csv
/work/imt11/Mapperatorinator/logs/hybrid-prefix-49555060.out
/work/imt11/Mapperatorinator/logs/hybrid-prefix-49555060.err
```

The report SHA-256 is
`365c9ffa99878ee68d45d8152634398b254ab731f2d75bb01d3d86dfdc8f6a6b`.

This remains one identical-input, two-seed SALVALAI smoke15 `seq9`, bucket-128
shape. Accepted five-song traces contain only `305 / 5,905` decode tokens in
that bucket. Applying `621.893 tok/s` beyond that observed bucket would project
`542.663` decode-only tok/s and `443.896` tok/s with current full setup charges,
requiring `61.65%` setup removal to reach `500`. That is an intentionally
invalid all-bucket extrapolation, not queue evidence.

The result authorizes production-weighted active-prefix bucket/queue physics
only. It does not authorize prefill/setup optimization, scheduler/runtime
wiring, server work, 256-step promotion, or an offline-engine claim. The exact
five-song B2-only ceiling analysis is pending and must precede the next
experiment.

For any later reproduction, use
`scripts/dcc/verify_hybrid_changing_prefix.sbatch` only from the isolated DCC
worktree after setting `MAPPERATORINATOR_REPO` and `EXPECTED_COMMIT` to a pushed,
clean checkpoint. The wrapper pins fixed report SHA
`d7f47e5c174a8f3504401ec0d428b0070f4724e95971efc150c3d13007f343ef`.
