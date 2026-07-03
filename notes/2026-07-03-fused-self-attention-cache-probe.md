# Fused Self-Attention Cache Probe

## Purpose

Test the next implementation-class target after the accepted native q_len=1 self-attention path and the weighted decoder-layer accounting pass.

Current accepted full-song single-song baseline remains DCC job `49225493`: `7,639` SALVALAI main tokens, `32.217s` synchronized model time, `237.111 tok/s`, fixed-seed token equivalence PASS for main/timing, and byte-identical `.osu` output.

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
