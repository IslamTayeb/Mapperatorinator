# Fused Self-Attention Cache Probe

## Purpose

Test the next implementation-class target after the accepted native q_len=1 self-attention path and the weighted decoder-layer accounting pass.

Starting accepted full-song single-song baseline was DCC job `49225493`: `7,639` SALVALAI main tokens, `32.217s` synchronized model time, `237.111 tok/s`, fixed-seed token equivalence PASS for main/timing, and byte-identical `.osu` output.

The weighted split says the best next target is not standalone MLP or sampling:

| candidate | weighted full-song seconds | reason |
| --- | ---: | --- |
| self-attention module | `8.806s` | largest measured sub-target |
| self-attention pre-attention setup | `4.378s` | qkv/RoPE/cache/layout before native q1 attention |
| native q1 attention only | `3.075s` | already improved but still target-sized |
| MLP island | `3.724s` | would need roughly `1.76x` whole-MLP speedup to clear 5% full-song |
| cross-attention module | `3.124s` | useful but smaller than self-attention and already has q1 BMM |

The 5% full-song bar is about `1.61s`; the 10% bar is about `3.22s`. A fused self-attention setup/cache probe only graduates if the weighted projection is large enough before production integration.

## Diagnostic Implementation

Commit `8ca1241` adds a verifier-only CUDA extension entrypoint:

```text
native_q1_rope_cache_attention(qkv, cache_keys, cache_values, cos, sin, cache_position, attention_mask, active_prefix_length)
```

It is not wired into production inference. It is only exposed through `utils/profile_decode_self_attention_island.py`.

The diagnostic kernel targets the batch-1, q_len=1, fp32, active-prefix StaticCache path and fuses:

- RoPE application for the current q/k vectors;
- direct StaticCache K/V write at the one-token `cache_position`;
- active-prefix q_len=1 attention over the static cache.

The first probe deliberately leaves `Wqkv`, cos/sin generation, and `Wo` in PyTorch for the production-shaped island. This isolates whether the post-`Wqkv` cache/setup work is worth deeper production work before adding cuBLASLt/CUTLASS or changing the real model path.

## Gates

Do not claim an inference speedup from this diagnostic alone.

Required before production integration:

- captured self-attention island logits/allclose must pass across weighted active-prefix buckets;
- weighted full-song projection should exceed `1.61s` saved, preferably `3.22s`;
- one-token logits gate must pass;
- direct-loop generated-token/logit/RNG gate must pass;
- 15s SALVALAI smoke token equivalence must pass;
- full-song SALVALAI token equivalence and total-stage non-regression must pass.

If the fused post-`Wqkv` variant is small or unstable, do not keep adding isolated self-attention kernel complexity. The next realistic path would be a broader decoder-layer or multi-layer runtime prototype.

## Runs

Initial single-bucket smoke:

- DCC job: `49229655`
- Commit: `8ca1241`
- Active-prefix length: `640`
- Purpose: catch CUDA compile/indexing/graph-capture failures before the full weighted sweep.
- Slurm state: `FAILED` only because the post-run shell summary had a Python quoting typo.
- Valid JSON: `/work/imt11/Mapperatorinator/runs/fused-self-attn-smoke-49229655-8ca1241/self_attn_len640.json`
- Result: profile PASS, logits replay `max_abs=0.0`.
- Fused island correctness: allclose PASS, graph replay allclose PASS, `max_abs=5.96e-07`.

Prefix `640` CUDA graph replay timings:

| variant | ms/layer |
| --- | ---: |
| repo module forward | `0.109146` |
| manual native attention island | `0.108990` |
| fused RoPE/cache attention island | `0.079599` |
| pre-attention setup only | `0.051831` |
| native attention only | `0.039751` |
| fused post-`Wqkv` attention only | `0.040046` |
| output projection only | `0.007253` |

This single bucket is not a full-song projection, but it is promising enough for the weighted active-prefix sweep. The key signal is that the production-shaped fused island was faster, while the post-`Wqkv` attention-only kernel was essentially tied with the old native attention-only kernel. That suggests any real win comes from replacing setup/cache/layout work around the existing attention, not from the attention math alone.

Weighted active-prefix sweep:

- DCC job: `49229681`
- Commit: `ddd75f9`
- Slurm state: `FAILED` only because the post-run reducer had a shell quoting bug.
- Valid run dir: `/work/imt11/Mapperatorinator/runs/fused-self-attn-buckets-49229681-ddd75f9`
- Summary: `/work/imt11/Mapperatorinator/runs/fused-self-attn-buckets-49229681-ddd75f9/summary.json`
- Active-prefix lengths: `128,192,256,320,384,448,512,576,640,704,768`
- Result: all captured variants PASS, all graph-replay variants PASS, logits replay exact.

Weighted CUDA graph replay totals over the full-song active64 replay distribution:

| variant | weighted seconds |
| --- | ---: |
| repo self-attention module | `9.104s` |
| manual native attention island | `9.118s` |
| fused RoPE/cache attention island | `6.599s` |
| pre-attention setup only | `4.485s` |
| native attention only | `3.167s` |
| fused post-`Wqkv` attention only | `3.185s` |
| output projection only | `0.637s` |

Projected savings:

| comparison | projected seconds | model-time share |
| --- | ---: | ---: |
| repo module -> fused island | `2.505s` | `7.77%` |
| manual native island -> fused island | `2.518s` | `7.82%` |
| native attention only -> fused post-`Wqkv` attention only | `-0.018s` | `-0.06%` |

The weighted probe clears the `>5%` strategic threshold but not the `>=10%` automatic keep threshold. It is worth one production-candidate attempt because the diagnostic kernel is narrow, fp32, batch1, default-off, and directly targets the largest remaining self-attention setup/cache/layout bucket. If production verification does not preserve token identity or loses most of the projected `2.5s`, revert or leave it diagnostic-only.

Production correctness gates:

- DCC job: `49230011`
- Commit: `d7b8684`
- Run dir: `/work/imt11/Mapperatorinator/runs/fused-rope-cache-gates3-49230011-d7b8684`
- One-token logits gate: PASS at sequence index 9, active-prefix decode length `640`, max abs `1.9073486328125e-05`, allclose PASS, top-k order PASS.
- Direct-loop gate: PASS over `64` sampled decode steps, generated-token identity PASS, raw-logit allclose/top-k PASS, final RNG-state PASS.

15s SALVALAI smoke:

- DCC job: `49230035`
- Commit: `d7b8684`
- Run dir: `/work/imt11/Mapperatorinator/runs/fused-rope-cache-smoke15-49230035-d7b8684`
- Control flags: accepted native q1 self-attention stack, fused rope/cache flag disabled.
- Candidate flags: same stack with `inference_native_q1_rope_cache_self_attention=true`.

| label | tokens | model time | tok/s | token equivalence |
| --- | ---: | ---: | ---: | --- |
| control main | `1,084` | `4.183s` | `259.168` | baseline |
| candidate main | `1,084` | `3.774s` | `287.257` | PASS |
| control timing | `164` | `3.020s` | `54.313` | baseline |
| candidate timing | `164` | `3.160s` | `51.906` | PASS |

The smoke main-generation gain was `+10.8%`, enough to promote. Timing regressed in the smoke, but the fused flag was requested and not enabled for timing records because timing contexts disable `native_q1_self_attention`; treat that timing smoke result as noise/order sensitivity until full-song validation.

Full-song SALVALAI validation:

- DCC job: `49230082`
- Commit: `d7b8684`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, UUID `GPU-de50b870-b708-690d-74a4-6e855163a133`
- Run dir: `/work/imt11/Mapperatorinator/runs/fused-rope-cache-full-49230082-d7b8684`
- Control profile: `/work/imt11/Mapperatorinator/runs/fused-rope-cache-full-49230082-d7b8684/control.profile.json`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/fused-rope-cache-full-49230082-d7b8684/candidate.profile.json`
- Strict compare: `/work/imt11/Mapperatorinator/runs/fused-rope-cache-full-49230082-d7b8684/compare_strict_full.json`

| label | tokens | model time | tok/s | timing tok/s | token/map equivalence |
| --- | ---: | ---: | ---: | ---: | --- |
| same-job control | `7,639` | `30.801s` | `248.015` | `101.706` | baseline |
| fused rope/cache candidate | `7,639` | `28.243s` | `270.475` | `101.988` | PASS main/timing, byte-identical `.osu` |

Against the same-job control, main generation improved `+9.1%` and saved `2.558s` synchronized model time. Against the prior retained native q1 self-attention result from job `49225493`, the new candidate improves `237.111 -> 270.475 tok/s` and saves `3.974s` model time.

Generated output sanity passed: the control and candidate `.osu` files were byte-identical (`cmp_exit=0`, both `31,709` bytes). Main generated-token IDs matched `7,639 / 7,639`; timing generated-token IDs matched `821 / 821`.

Strict zero-tolerance comparison still failed on scoped micro-regressions. Main generation failed `3 / 87` windows (`seq45`, `seq84`, `seq85`) totaling `4.738ms` positive model-time overhead versus `2.558s` aggregate main model-time savings. Timing failed many tiny windows by jitter, but timing aggregate still improved by `22.3ms`. This is not a strict zero-regression result, but it is an accepted default-off opt-in component because it is exact, byte-identical at output, improves main/timing aggregates, and clears the strategic 5-10% keep band with a narrow kernel change.

Why it worked:

- The attention math itself did not get faster; weighted `native attention only -> fused post-Wqkv attention only` was slightly negative.
- The win came from fusing the post-`Wqkv` setup around q_len=1 self-attention: current-token RoPE application, direct StaticCache K/V write, active-prefix cache load, and attention launch shape handling.
- This removes PyTorch-level transpose/unbind/RoPE/cache-update/layout plumbing inside the hottest self-attention setup bucket while preserving fp32 math, generated-token behavior, and output bytes.

Decision:

- Keep `inference_native_q1_rope_cache_self_attention=true` as a default-off opt-in only with the validated simple fp32 batch-1 DecodeSession active-prefix native path.
- Do not enable it for timing contexts unless a separate timing-context run proves non-regression; current production gating leaves timing on the normal path.
- Keep using one-token logits, direct-loop token/logit/RNG, 15s smoke, full-song token equivalence, and generated output checks before extending the fused kernel to broader modes.
