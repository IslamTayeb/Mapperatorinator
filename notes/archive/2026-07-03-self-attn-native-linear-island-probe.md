# Native-Linear Self-Attention Island Probe

## Summary

Rejected. The probe was exact/allclose, but native one-token linear wrappers for `Wqkv` and `Wo` around the existing fused RoPE/cache/native q1 self-attention island did not save target-sized time. Explicit active-prefix repeats regressed the projected full-song model time.

## Evidence

- Job: `49233393`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, UUID `GPU-ca8caefc-47d0-69fb-2718-9f51f0cb67c1`
- Branch/commit: `experiment/self-attn-native-linear-island`, `4206d28`
- Run dir: `/work/imt11/Mapperatorinator/runs/self-attn-native-linear-island-49233393`
- Reports:
  - `seq9_warm50_iter500.json`
  - `seq9_prefix128_warm100_iter2000.json`
  - `seq9_prefix640_warm50_iter500.json`
- Environment: torch `2.10.0+cu128`, CUDA `12.8`, transformers `4.57.3`

All three reports passed the self-attention island verifier:

- `pass=True`
- `logits_replay_allclose=True`
- `logits_replay_max_abs=1.9073486328125e-05`

## Results

| Probe | Repo self-attn island | Best native-linear result | Projected full-song effect |
| --- | ---: | ---: | ---: |
| default seq9/prefix128 first run | `4.921s` | warp4 `4.886s` | `+0.035s`, `270.810 tok/s` |
| explicit prefix128 repeat | `4.006s` | warp4 `4.115s` | `-0.109s`, `269.433 tok/s` |
| explicit prefix640 repeat | `6.735s` | warp4 `7.013s` | `-0.278s`, `267.837 tok/s` |

The only positive number was the first default run and was far below noise/threshold. The more controlled explicit-prefix repeats were negative.

## Decision

Do not merge the experiment branch. The narrow composition still pays too much overhead around the native linear calls and does not improve the actual production-like CUDA graph replay path. It is also far below the roughly `1.4s` full-song saving needed to clear the `5%` keep threshold from the current `270.475 tok/s` baseline.

Future self-attention/native work should move to a broader residual/layer island or a better native linear strategy that reduces multiple decoder-layer components together. Do not retry this `Wqkv/Wo` wrapper composition unless new profiling shows a materially different target.
