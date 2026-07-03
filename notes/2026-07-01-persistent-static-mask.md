# Persistent Static Causal-Mask Buffer

## Idea

Reuse a full-size 4D static-cache causal mask buffer during one-token SDPA decode. This was meant to differ from the rejected static-cache prefix trim:

- Keep the full static-cache target length.
- Avoid changing K/V attention length.
- Try to stabilize the mask data pointer for compiled decode.
- Guard to batch size 1, one-token decode, CUDA, SDPA, and static cache.

## Result

Rejected and reverted.

- Candidate commit: `2d807c1`
- Baseline job: `49137140`
- Candidate job: `49137141`
- Node: `dcc-core-ferc-s-z25-21`
- GPU: RTX 2080 Ti
- Baseline profile: `/work/imt11/Mapperatorinator/runs/smoke15-mask-base-2d807c1/beatmap4438946a6a864548b8d968d4a3a24682.osu.profile.json`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/smoke15-mask-cand-2d807c1/beatmap63be4f635c7147cd9b127038054db1d2.osu.profile.json`

Comparator result:

```text
tokens_per_second: baseline=51.392, candidate=42.361, delta=-9.031 (-17.6%, worse)
model_elapsed_seconds: baseline=21.093, candidate=25.590, delta=+4.497 (+21.3%, worse)
generated_tokens: baseline=1084, candidate=1084
Token equivalence: PASS (1084 generated token IDs match).
```

The candidate profile recorded `persistent_static_mask_enabled=true`, so the experimental path actually ran. Post-warmup windows regressed too:

- Baseline seq9: `104.6 tok/s`
- Candidate seq9: `91.5 tok/s`

## Interpretation

Stable mask storage alone did not help. The replacement added per-token `fill_`, boolean visible-mask construction, and `masked_fill_` work inside the timed generation path. That extra work outweighed any benefit from reusing the 4D mask allocation or stabilizing the mask data pointer.

Future mask work should not mutate a persistent full mask per token unless a trace proves mask construction itself has become a target-sized cost and the replacement avoids extra kernels. This result also reinforces that the 200 tok/s path probably needs to reduce compiled forward/attention/model-kernel cost, not just rearrange mask construction.
