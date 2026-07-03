# Decode Attention Component Probe

## Purpose

Measure the actual one-token decoder attention tensors after the accepted q1 BMM cross-attention stack, before adding any more attention branches or native kernels.

Campaign baseline at the time was DCC job `49213490`: active-prefix bucket64, CUDA graph warmup0, stateful monotonic logits processor, q_len=1 BMM cross-attention, `7,639` SALVALAI main tokens, `37.981s` model time, `201.125 tok/s`, token equivalence PASS. This was later superseded by persistent DecodeSession runtime job `49223294`: `35.337s`, `216.173 tok/s`, main/timing token equivalence PASS.

## Utility

Added `utils/profile_decode_attention_components.py`.

The utility:

- loads the real Mapperatorinator model and SALVALAI 15s smoke prompt;
- builds a direct one-token cached decode state through the existing verifier helpers;
- monkeypatches the VarWhisper SDPA attention wrapper only during the probe step;
- captures real post-projection q/k/v/mask/output tensors for decoder self-attention and cross-attention;
- benchmarks equivalent SDPA, output-transform-only, and `q_len=1` `bmm -> softmax -> bmm` variants with CUDA events;
- records logits replay equality so the captured step is tied to the real decode path.

This is diagnostic only. It does not claim an inference speedup.

## Jobs

| job | commit | state | note |
| --- | --- | --- | --- |
| `49222759` | `989359f` | `COMPLETED` | Single attention component probe at active-prefix length `128`. |
| `49222781` | `989359f` | `COMPLETED` | Length sweep over `128`, `256`, `512`, `704`, `1024`, `2048`. |
| `49222868` | `989359f` | `FAILED` | Launcher bug: attempted to source missing `bin/activate`; no model work. |
| `49222883` | `989359f` | `FAILED` after valid JSONs | Production-prefix sweep completed all probe JSONs, then failed only in an ad-hoc summary snippet. Parsed manually into `summary_fixed.json`. |

Key run roots:

- `/work/imt11/Mapperatorinator/runs/decode-attention-components-20260703-024837-989359f`
- `/work/imt11/Mapperatorinator/runs/decode-attention-length-sweep-20260703-025029-989359f`
- `/work/imt11/Mapperatorinator/runs/decode-attention-production-prefix-sweep-20260703-030118-989359f`

Production-prefix sweep environment:

- Node: `dcc-core-gpu-ferc-s-h36-5`
- GPU: RTX 2080 Ti `GPU-5db878e3-a34e-853b-e5be-357ca7d5862a`
- Driver/CUDA from `nvidia-smi`: `595.71.05` / `13.2`
- Torch: `2.10.0+cu128`
- Transformers: `4.57.3`
- Precision: `fp32`
- Attention: `sdpa`
- Config: `profile_salvalai_smoke15`, `sequence_index=9`, `seed=12345`, `q1_bmm_cross_attention=true`

## Results

All valid probe JSONs passed logits replay with `max_abs=0.0`. The captured shape groups match the production q1 path:

- Cross-attention: `q=[1,12,1,64]`, `k/v=[1,12,1024,64]`, no mask.
- Self-attention: `q=[1,12,1,64]`, `k/v=[1,12,L,64]`, mask `[1,1,1,L]`.

Production-prefix sweep:

| active prefix | self SDPA | self BMM | self delta | self BMM speedup | cross SDPA | cross BMM | cross BMM speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `128` | `0.021617ms` | `0.084218ms` | `-0.062602ms` | `0.257x` | `0.150382ms` | `0.064850ms` | `2.319x` |
| `192` | `0.030156ms` | `0.082767ms` | `-0.052611ms` | `0.364x` | `0.150390ms` | `0.060774ms` | `2.475x` |
| `256` | `0.038779ms` | `0.082058ms` | `-0.043279ms` | `0.473x` | `0.150533ms` | `0.058952ms` | `2.554x` |
| `320` | `0.047526ms` | `0.084756ms` | `-0.037230ms` | `0.561x` | `0.150608ms` | `0.060129ms` | `2.505x` |
| `384` | `0.056174ms` | `0.085565ms` | `-0.029392ms` | `0.657x` | `0.150451ms` | `0.060731ms` | `2.477x` |
| `448` | `0.064821ms` | `0.086356ms` | `-0.021536ms` | `0.751x` | `0.150383ms` | `0.061583ms` | `2.442x` |
| `512` | `0.073753ms` | `0.083478ms` | `-0.009725ms` | `0.884x` | `0.151204ms` | `0.060723ms` | `2.490x` |
| `576` | `0.082452ms` | `0.086648ms` | `-0.004196ms` | `0.952x` | `0.151161ms` | `0.061935ms` | `2.441x` |
| `640` | `0.091103ms` | `0.084544ms` | `0.006559ms` | `1.078x` | `0.151056ms` | `0.060975ms` | `2.477x` |
| `704` | `0.100721ms` | `0.086740ms` | `0.013981ms` | `1.161x` | `0.151173ms` | `0.062866ms` | `2.405x` |
| `768` | `0.110317ms` | `0.087932ms` | `0.022385ms` | `1.255x` | `0.151267ms` | `0.063206ms` | `2.393x` |

The cross-attention result confirms the accepted production q1 BMM branch remains well-justified. The self-attention result is different: BMM is bad for short active-prefix buckets, near break-even at `576`, and only mildly positive at `640+`.

Using full-song active64 diagnostics from job `49207288` (`7,552` graph replays) as the replay distribution:

| thresholded self-BMM policy | eligible positive replays | projected saved model time | projected SALVALAI tok/s from `37.981s` |
| --- | ---: | ---: | ---: |
| `L >= 640` | `4,236` | `0.463s` | `203.607 tok/s` |
| `L >= 704` | `1,329` | `0.234s` | `202.373 tok/s` |
| `L >= 768` | `108` | `0.029s` | `201.281 tok/s` |

This projection assumes all 12 decoder self-attention layers benefit exactly like the isolated captured op, so it is already optimistic for an integrated branch.

## Decision

Do not implement a PyTorch thresholded q1 BMM self-attention branch. It is exact in the microprobe but the full-song ceiling is only about `0.46s`, or `1.2%` of accepted q1 model time. That is below the keep threshold and not worth adding another attention-path flag.

Keep the accepted q1 BMM cross-attention branch. It remains a real win because cross-attention uses a long unmasked `L=1024` shape every one-token decode step.

Future attention work should move beyond simple PyTorch `bmm` substitutions:

1. Persistent/bufferized `DecodeSession` graph/cache/encoder reuse, because it can remove repeated capture/runtime costs and creates the stable buffers needed for larger native kernels.
2. A narrow native/CUTLASS/CUDA q_len=1 active-prefix self-attention kernel only if it improves the `128..640` production buckets, not just the long tail.
3. Fused decoder-block or scheduler work that reduces launches across attention, linears, layernorm, and elementwise operations together.
