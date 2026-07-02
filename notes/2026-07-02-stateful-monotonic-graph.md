# Stateful Monotonic Processor on Active Graph

## Hypothesis

After active-prefix CUDA graph replay moved the one-token forward into a tighter runtime path, CPU-side logits processors became target-sized. Diagnostic job `49167587` showed `MonotonicTimeShiftLogitsProcessor` took `5.373s / 5.602s` of active-graph logits-processor time on the 15s smoke. A batch-1 stateful monotonic path should preserve exact HF sampling semantics while avoiding the full-prefix scan on every generated token.

## Implementation

Commit `a980c8d` added default-off `inference_stateful_monotonic_logits_processor=true`. It wires through inference metadata, non-server generation, server generation, generation stats, and profile records. The fast path is intentionally narrow: batch size 1, sequential input growth by one token, and fallback to the original full scan when the input shape/growth does not match the expected decode pattern.

Use it with the accepted active-prefix CUDA graph path:

```bash
python inference.py --config-name profile_salvalai \
  inference_generation_compile=true \
  inference_active_prefix_decode_loop=true \
  inference_active_prefix_decode_bucket_size=512 \
  inference_active_prefix_decode_cuda_graph=true \
  inference_active_prefix_decode_cuda_graph_min_decode_steps=1 \
  inference_stateful_monotonic_logits_processor=true
```

## 15s Smoke

- Job: `49168158`
- Node/GPU: `dcc-dhvimdcore-gpu-ferc-s-p15-3`, RTX 2080 Ti
- Commit: `a980c8d`
- Run dir: `/work/imt11/Mapperatorinator/runs/stateful-monotonic-smoke-49168158-a980c8d`
- Comparison: active512 graph baseline vs active512 graph + stateful monotonic
- Same-calculation metadata: PASS
- Token equivalence: PASS, `1,084 / 1,084`
- Main generation: `132.593 -> 149.321 tok/s`, `+12.6%`
- Model time: `8.175s -> 7.260s`, `-11.2%`
- Total stage: `17.320s -> 15.930s`, `-8.0%`
- Per-window no-regression: failed only `seq0`, a one-token window, by `0.299ms`

## Full Song

- Job: `49168188`
- Node/GPU: `dcc-dhvimdcore-gpu-ferc-s-p15-3`, RTX 2080 Ti
- Stack: torch `2.10.0+cu128`, Transformers `4.57.3`
- Commit: `a980c8d`
- Run dir: `/work/imt11/Mapperatorinator/runs/stateful-monotonic-full-49168188-a980c8d`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/stateful-monotonic-full-49168188-a980c8d/active512_graph_stateful/beatmap07c1643a974c4e1ca6bbbca34a282478.osu.profile.json`

Against the retained compile-only baseline:

| metric | retained compile-only | active512 graph + stateful | delta |
| --- | ---: | ---: | ---: |
| main tokens | `7,639` | `7,639` | unchanged |
| main model time | `82.615s` | `56.639s` | `-25.976s`, `-31.4%` |
| main throughput | `92.465 tok/s` | `134.873 tok/s` | `+45.9%` |
| total timing+map stage | `113.928s` | `78.473s` | `-35.455s`, `-31.1%` |
| token equivalence | baseline | PASS, `7,639 / 7,639` | exact |
| per-window no-regression | baseline | PASS, `87 / 87` | strict pass |

Against the prior active512 graph full-song run:

| metric | active512 graph | active512 graph + stateful | delta |
| --- | ---: | ---: | ---: |
| main model time | `71.981s` | `56.639s` | `-15.342s`, `-21.3%` |
| main throughput | `106.125 tok/s` | `134.873 tok/s` | `+27.1%` |
| total timing+map stage | `101.481s` | `78.473s` | `-23.008s`, `-22.7%` |
| token equivalence | baseline | PASS, `7,639 / 7,639` | exact |
| per-window no-regression | baseline | PASS, `87 / 87` | strict pass |

The retained-baseline metadata comparison failed only because the old retained profile lacks newer metadata keys. The active512 graph comparison had same-calculation metadata PASS.

## Decision

Keep `inference_stateful_monotonic_logits_processor=true` as a default-off opt-in performance flag with active-prefix CUDA graph/simple batch-1 generation. This is an accepted exact full-song win and the current fastest measured path for 200 tok/s work: `134.873 tok/s`.

Do not make it the cold default yet, because the active-prefix graph stack itself remains opt-in and scoped. Do not generalize it to server, CFG, beam, batch, or non-active-prefix generation without new equivalence and non-regression runs.

## Next

From `134.873 tok/s`, reaching `200 tok/s` still requires reducing full-song main model time from `56.639s` to about `38.195s`, another `18.444s` or `32.6%`. The next profiling pass should measure the stateful active-graph path again, because monotonic masking is no longer expected to dominate. Likely next targets are graph-forward replay time, bucket size/scheduling, remaining logits processors, sampling/stopping overhead, and q_len=1 attention/cache layout.
