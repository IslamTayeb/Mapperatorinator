# Native q_len=1 Self-Attention

## Hypothesis

Post-DecodeSession profiling showed duplicate CUDA graph capture was no longer target-sized, while decoder CUDA compute was still dominated by one-token linear/GEMV work and FMHA attention. PyTorch q1 BMM self-attention had already been rejected because it only helped the long-prefix tail. A narrow native CUDA kernel for the actual active-prefix q_len=1 self-attention shape could improve the common `128..640` decode buckets without changing precision, sampling, RNG, output policy, or generated-token behavior.

This is intended to carry forward to future autoregressive encoder-decoder work because q_len=1 decoder self-attention over a KV cache remains a common inference primitive.

## Implementation

Commits:

- `b45d5f3` added a diagnostic native q1 attention probe to `utils/profile_decode_attention_components.py`.
- `2ee1d68` added the default-off production native q1 self-attention path.
- `c563af0` scoped native q1 self-attention to non-timing/map-generation contexts after the first smoke showed timing regression.

Flags for the accepted path:

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
```

Scope:

- fp32 only.
- batch size 1 only.
- non-server, non-parallel, CFG 1, beams 1 only.
- active-prefix decode loop plus DecodeSession CUDA graph only.
- decoder self-attention only.
- q_len=1 only.
- timing contexts keep the normal path; native q1 self-attention is enabled for map generation.

## Diagnostic Probe

DCC job `49224045`, run dir:

```text
/work/imt11/Mapperatorinator/runs/native-q1-attn-probe-49224045-b45d5f3
```

The Slurm job ended `FAILED` only because the summary script globbed `*.stdout.json`. The actual measurement JSON files completed and a fixed summary was written to:

```text
/work/imt11/Mapperatorinator/runs/native-q1-attn-probe-49224045-b45d5f3/summary_fixed.json
```

Native self-attention output matched SDPA within allclose tolerance across all tested lengths; max abs was `3.58e-07`.

| active-prefix length | SDPA ms | native ms | speedup |
| ---: | ---: | ---: | ---: |
| `128` | `0.031150` | `0.023623` | `1.32x` |
| `192` | `0.042239` | `0.023327` | `1.81x` |
| `256` | `0.054434` | `0.030738` | `1.77x` |
| `320` | `0.066458` | `0.027974` | `2.38x` |
| `384` | `0.055176` | `0.022856` | `2.41x` |
| `448` | `0.090571` | `0.037387` | `2.42x` |
| `512` | `0.072169` | `0.029348` | `2.46x` |
| `576` | `0.114628` | `0.046633` | `2.46x` |
| `640` | `0.088720` | `0.036065` | `2.46x` |
| `704` | `0.097069` | `0.039530` | `2.46x` |
| `768` | `0.105513` | `0.042638` | `2.47x` |
| `1024` | `0.208855` | `0.091423` | `2.28x` |

Using full-song active64 replay counts from job `49207288`, the self-attention-only projection was `4.943s` saved from the current `35.337s` model-time baseline, or about `216.173 -> 251.33 tok/s` if integration held perfectly.

The same probe did not justify replacing the accepted q1 BMM cross-attention branch. Cross-attention remains on the existing q1 BMM path.

## Correctness Gates

First production smoke, DCC job `49224965`, commit `2ee1d68`:

```text
/work/imt11/Mapperatorinator/runs/native-q1-self-gates-49224965-2ee1d68
```

- One-token logits gate: PASS, `max_abs=2.6702880859375e-05`.
- Direct-loop 64-token gate: PASS, token match true, RNG match true, logits pass.
- 15s smoke main: `231.928 -> 263.558 tok/s`, `+13.6%`, token equivalence PASS (`1,084 / 1,084`).
- 15s smoke timing: `54.332 -> 51.601 tok/s`, regression.

The timing regression caused the native self-attention path to be scoped off for `ContextType.TIMING`.

Scoped smoke, DCC job `49225227`, commit `c563af0`:

```text
/work/imt11/Mapperatorinator/runs/native-q1-self-gates-49225227-c563af0
```

- One-token logits gate: PASS, `max_abs=2.6702880859375e-05`.
- Direct-loop 64-token gate: PASS, token match true, RNG match true.
- 15s smoke main: `234.594 -> 262.900 tok/s`, `+12.1%`, token equivalence PASS (`1,084 / 1,084`).
- 15s smoke timing: `55.229 -> 55.238 tok/s`, no aggregate regression, token equivalence PASS (`164 / 164`).

The scoped smoke showed worse candidate main outer wall than model time because the process loaded the native extension inside the first main profile. That setup cost was treated as a caveat and checked in the full-song total-stage comparison.

## Full-Song Validation

DCC job `49225493`, node `dcc-core-gpu-ferc-s-h36-5`, RTX 2080 Ti `GPU-cbcd7909-6bd2-1b73-d336-a63ead10aa5d`, driver `595.71.05`, torch `2.10.0+cu128`, Transformers `4.57.3`, commit `c563af0`.

Run dir:

```text
/work/imt11/Mapperatorinator/runs/native-q1-self-full-49225493-c563af0
```

| run | main tokens | main model time | main tok/s | timing tokens | timing model time | timing tok/s | total timing+map stage | token/map equivalence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| native-off DecodeSession control | `7,639` | `36.863s` | `207.226` | `821` | `8.754s` | `93.785` | `50.736s` | baseline |
| native q1 self-attn candidate | `7,639` | `32.217s` | `237.111` | `821` | `8.143s` | `100.822` | `48.527s` | PASS main/timing, byte-identical `.osu` |

Compared with the previous accepted DecodeSession checkpoint (`216.173 tok/s`, job `49223294`), the new retained opt-in result is about `+9.7%` faster on SALVALAI main-generation throughput. The same-job native-off control shows `+14.4%`, likely because the paired control was slower than the prior accepted run.

Strict compare:

- Same-calculation metadata: PASS.
- Main generated-token equivalence: PASS (`7,639 / 7,639`).
- Timing generated-token equivalence: PASS (`821 / 821`).
- Generated `.osu`: byte-identical.
- Main aggregate: `207.226 -> 237.111 tok/s`, `+14.4%`; `36.863s -> 32.217s`, `-12.6%`.
- Timing aggregate: `93.785 -> 100.822 tok/s`, `+7.5%`; `8.754s -> 8.143s`, `-7.0%`.
- Total timing+map stage: `50.736s -> 48.527s`, `-4.4%`.

Strict per-window no-regression failed in scoped places:

- Main `seq0`: model time improved by `244ms`, but outer wall regressed by `2.411s`, consistent with native extension setup/load.
- Main `seq44`, `seq45`, `seq84`: only `1.511ms` total failed-window model overhead.
- Timing `seq8`: `0.347ms` failed-window model overhead.

## Decision

Keep as the current fastest exact single-song opt-in path.

Why:

- It is default-off and hard-gated to the validated simple fp32 batch-1 path.
- It preserves fixed-seed generated token IDs for main and timing.
- It produces byte-identical `.osu` output in the full-song validation.
- It improves full-song main-generation model time by `4.646s` against same-job native-off control.
- It improves timing aggregate and total profiled stage wall even though native self-attention is disabled for timing contexts.
- It is measured native-kernel work on a real hotspot and should carry to future autoregressive decoder or encoder-decoder inference.

Risks and caveats:

- This native kernel changes floating-point reduction order. Exact generated-token and output identity passed for this run, but any expansion must repeat the logits, direct-loop, smoke, and full-song gates.
- First-use native extension setup shows up as `seq0` outer-wall cost. Do not hide that cost when claiming cold single-song total-stage wins.
- The native kernel is narrow and not a general attention backend. Do not broaden it to timing, server, batch, fp16, cross-attention, prefill, masked cases, or non-active-prefix paths without fresh gates.

## Next Work

Run a fresh post-native profiling pass before choosing the next implementation target. The prior profile was before this kernel and likely overstates remaining FMHA attention share. Good next questions:

1. How much of full-song or seq9 self-CUDA is now one-token linear/GEMV/MLP versus residual attention?
2. Did sampling/sort become large enough to matter after attention shrank?
3. Is native extension setup large enough to justify explicit preload in the measured setup path, without hiding cold cost?
4. Is there a single narrow native/CUTLASS/cuBLASLt MLP or projection island with a credible `>5%` full-song ceiling?

Do not start another kernel before the post-native profile identifies a target-sized hotspot.
