# SDPA Backend Audit on RTX 2080 Ti

## Summary

This audit checks whether PyTorch SDPA backend dispatch is hiding a simple exact-calculation backend win on SM75. It is not a model-quality, precision, sampling, output, or token-behavior change. Any backend replacement still needs full-song SALVALAI token equivalence before it can replace the retained baseline.

## Inputs

- Retained full-song baseline: SDPA + `inference_generation_compile=true`, job `49113713`, `7,639` main tokens, `82.615s` synchronized model time, `92.465 tok/s`, token equivalence PASS.
- Current 15s smoke reference: job `49139323`, commit `9681150`, profile `/work/imt11/Mapperatorinator/runs/smoke15-current-49139323-9681150/beatmap2c6f46b9eeb04e4698f7fe12d43dd7ee.osu.profile.json`.
- Backend audit job: `49139404`, node `dcc-core-ferc-s-z25-21`, RTX 2080 Ti, torch `2.10.0+cu128`, Transformers `4.57.3`, commit `9d92c34`.
- Full-song paired audit job: `49139420`, node `dcc-core-ferc-s-z25-21`, RTX 2080 Ti, torch `2.10.0+cu128`, Transformers `4.57.3`, commit `9d92c34`.

## Results

| Backend | Status | 15s main tokens | Main model time | Main tok/s | Token equivalence | Decision |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `flash` | Failed | n/a | n/a | n/a | n/a | Not viable on SM75 in this PyTorch build |
| `efficient` | Completed | 1,084 | `21.253s` | `51.004` | PASS, `1,084 / 1,084` | Not a win; +0.3% vs smoke reference |
| `math` | Completed | 1,084 | `13.746s` | `78.862` | PASS, `1,084 / 1,084` | Promising smoke signal; full-song validation required |

Flash failure evidence from stderr:

```text
Flash attention only supports gpu architectures in the range [sm80, sm121]. Attempting to run on a sm 7.5 gpu.
RuntimeError: No available kernel. Aborting execution.
```

The `efficient` result confirms the unforced SDPA path was already effectively using the memory-efficient CUTLASS-style path. It is within smoke noise and should not be promoted.

The `math` smoke result is surprisingly positive, but the per-window shape matters:

- `seq=3`: `499` tokens, `7.588s`, `65.8 tok/s`.
- `seq=9`: `234` tokens, `2.255s`, `103.9 tok/s`.

The post-warmup `seq=9` speed is basically the same as the efficient/default smoke (`105.3 tok/s`), so the smoke improvement may mostly be first-window compilation or warmup behavior rather than steady-state decode speed.

Full-song validation confirmed that the smoke win should not graduate:

| Full-song run | Profile | Main tokens | Main model time | Main tok/s | Token equivalence | Decision |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Current default SDPA paired run | `/work/imt11/Mapperatorinator/runs/full-default-sdpa-49139420-9d92c34/beatmap71d2382c13934c40970d2443eb8a5315.osu.profile.json` | 7,639 | `90.286s` | `84.609` | PASS vs retained baseline | Same-job comparator only; slower than retained baseline |
| Forced `profile_sdpa_backend=math` | `/work/imt11/Mapperatorinator/runs/full-sdpa-math-49139420-9d92c34/beatmap8241a2b3f0be495592a5cf2b9bb9d6a2.osu.profile.json` | 7,639 | `85.177s` | `89.684` | PASS vs retained baseline | Reject for main generation; `-3.0%` vs retained `92.465 tok/s` |

The forced math backend did improve the same-job current default run by about `6.0%`, but the paired default was itself slower than the retained full-song baseline. Since accepted results must beat the retained baseline, not just a slower paired rerun, `profile_sdpa_backend=math` is a rejected main-generation optimization for now.

One useful side observation: timing-context generation improved substantially in the paired full-song job (`821` tokens, `40.358s` default to `24.030s` math). That may be worth revisiting only if timing-context generation becomes an explicit objective; it does not justify changing the retained main-generation baseline.

## Trace Context

The deeper seq9 torch profile job `49139349` showed:

- `fmha_cutlassF_f32_aligned_64x64_rf_sm75`: `2031.747ms` self CUDA, `5,628` calls.
- `Torch-Compiled Region`: `2047.904ms` self CUDA across retained compiled-region events.
- cuBLAS GEMV/GEMM-like kernels: `227.063ms` self CUDA.
- `aten::addmm`/linear events: `282.210ms` self CUDA.
- sampling/sort/softmax: `12.029ms` self CUDA.

This keeps attention/backend dispatch target-sized, but not enough to claim a backend change without full-song evidence.

## Profiler Note

The first event-limit trace sorted bounded key averages only by self CUDA time, which can drop nested semantic ranges whose total CUDA time is large. The profiler was patched to sort bounded event summaries by the largest self/total CUDA or self CPU signal before truncation. This is a profiling-quality change only, not an inference speed change.

## Decision

Do not promote forced `math` SDPA based on 15s smoke. Keep SDPA plus `inference_generation_compile=true` as the retained baseline. Continue with direct `q_len=1` runtime/cache profiling, semantic range attribution, and exact direct-step/CUDA-graph feasibility rather than backend toggling.
