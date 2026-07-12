# Fixed-K Tail Graph Audit

## Purpose

Test whether fixed-size multi-token CUDA graph tail work can preserve exact `torch.multinomial` RNG behavior when generation stops early on EOS. This is verifier infrastructure for future `DecodeSession`/tail-runtime work, not an inference speedup.

## DCC Run

- Job: `49232592`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti
- Commit: `11fd14c`
- Run dir: `/work/imt11/Mapperatorinator/runs/fixed-k-tail-audit-20260703-134115-11fd14c`
- Report: `/work/imt11/Mapperatorinator/runs/fixed-k-tail-audit-20260703-134115-11fd14c/fixed_k_tail_graph_audit.json`
- Config: `profile_salvalai_smoke15`, seq9, seed `12345`, fp32, SDPA, accepted opt-in stack with `DecodeSession`, native q1 self-attention, fused RoPE/cache self-attention, and `profile_record_token_ids=true`

Command shape:

```bash
python utils/verify_direct_decode_loop.py \
  --config-name profile_salvalai_smoke15 \
  --sequence-index 9 \
  --max-new-tokens 16 \
  --candidate-active-prefix-decode \
  --candidate-active-prefix-decode-bucket-size 64 \
  --candidate-cuda-graph-forward \
  --candidate-cuda-graph-warmup 0 \
  --candidate-q1-bmm-cross-attention \
  --candidate-native-q1-self-attention \
  --candidate-native-q1-rope-cache-self-attention \
  --candidate-decode-session \
  --force-eos-at-generated-token 2 \
  --fixed-k-tail-graph-audit \
  --tail-graph-k 4
```

## Result

Overall gate: PASS.

| Check | Result | Meaning |
| --- | --- | --- |
| Forced-EOS direct loop | PASS | Candidate direct loop still matched HF-style reference token behavior, raw logits, stop reason, and final RNG state when EOS was forced. |
| No-EOS fixed-K graph block | PASS | Four eager multinomial draws matched four graph-replayed draws, final CUDA RNG state matched, and the next eager samples matched. |
| Naive forced-EOS fixed-K block | Expected divergence detected | The graph output prefix through EOS matched eager output, but final CUDA RNG diverged because the graph consumed two extra samples after the eager stop point. |

Key report fields:

```text
force_eos_direct_loop.pass=true
force_eos_direct_loop.token_match=true
force_eos_direct_loop.rng_match=true
force_eos_direct_loop.logits_pass=true
fixed_k_tail_graph_audit.pass=true
no_eos_block.sequence_match=true
no_eos_block.final_rng_match=true
no_eos_block.next_eager_sample_match=true
forced_eos_naive_block.output_prefix_through_eos_match=true
forced_eos_naive_block.final_rng_match=false
forced_eos_naive_block.expected_rng_divergence_detected=true
```

## Decision

Keep the verifier changes. Do not claim a speedup.

This confirms the design constraint for future multi-token/tail graph work: fixed-K graph replay is exact only when it does not sample past the point where eager HF generation would stop, or when the runtime implements exact rollback or true device-side early exit. Matching generated tokens through EOS is insufficient if final RNG state diverges.

Future candidates that graph multiple tail steps must run:

- `utils/verify_direct_decode_loop.py --force-eos-at-generated-token N --fixed-k-tail-graph-audit`
- 15s fixed-seed generated-token equivalence
- full-song generated-token and output/map equivalence
- untraced full-song performance non-regression
