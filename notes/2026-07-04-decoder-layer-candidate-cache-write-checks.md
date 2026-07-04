# Decoder Layer Candidate Cache-Write Checks

## Purpose

Tighten the whole decoder-layer verifier before any native/CUDA replacement
work. The earlier ABI fingerprint proved the reference q_len=1 self-cache slot,
but benchmark candidates were still only judged by hidden-output allclose. A
candidate could return the right one-token hidden state while failing to write
the `StaticCache` K/V slot needed by the next token.

This is verifier infrastructure only. It is not an inference throughput claim.

## Code Change

`utils/profile_decode_decoder_layer_island.py` now accepts:

```text
--verify-cache-write-candidates
```

When enabled, the profiler:

1. captures the reference q_len=1 self-cache K/V slot fingerprint,
2. resets the candidate cache slot to `NaN`,
3. runs each cache-writing candidate once,
4. fingerprints the actual written K/V slot,
5. requires key SHA, value SHA, and candidate output allclose to match the
   reference, and
6. restores the reference slot before timing.

`utils/validate_decoder_layer_abi.py` now accepts:

```text
--require-candidate-cache-write-checks
```

That strict mode requires `repo_decoder_layer`, `self_attn_residual_segment`,
`manual_decoder_runtime_island`, and any present `compiled_decoder_layer`
result to include passing cache-write checks. Old reports fail this stricter
mode, which is intentional.

## Validation

DCC job:

```text
49253972
```

Run details:

| Field | Value |
| --- | --- |
| node/GPU | `dcc-core-ferc-s-z25-20`, RTX 2080 Ti `GPU-de50b870-b708-690d-74a4-6e855163a133` |
| branch/commit | `codex/candidate-cache-write-checks`, `7044278` |
| run dir | `/work/imt11/Mapperatorinator/runs/decoder-layer-candidate-cache-write-49253972-7044278` |
| report | `/work/imt11/Mapperatorinator/runs/decoder-layer-candidate-cache-write-49253972-7044278/decoder_layer_candidate_cache_check_report.json` |
| validation | `/work/imt11/Mapperatorinator/runs/decoder-layer-candidate-cache-write-49253972-7044278/decoder_layer_candidate_cache_check_validation.json` |
| Slurm state | `COMPLETED`, elapsed `00:02:17`, `gres/gpu:2080=1` |
| environment | Python `3.10.12`, torch `2.10.0+cu128`, transformers `4.57.3` |
| config | `profile_salvalai_smoke15`, sequence `9`, bucket `64`, fp32, SDPA, fused RoPE/cache self-attention |

Command shape:

```text
python utils/profile_decode_decoder_layer_island.py \
  --config-name profile_salvalai_smoke15 \
  --sequence-index 9 \
  --active-prefix-bucket-size 64 \
  --native-q1-rope-cache-self-attention \
  --candidate-decoder-runtime-island \
  --cuda-graph-replay \
  --warmup 10 \
  --iters 50 \
  --include-cache-write-fingerprint \
  --verify-cache-write-candidates \
  --report-path /work/imt11/Mapperatorinator/runs/decoder-layer-candidate-cache-write-49253972-7044278/decoder_layer_candidate_cache_check_report.json \
  audio_path=/work/imt11/Mapperatorinator/data/salvalai.mp3

python utils/validate_decoder_layer_abi.py \
  /work/imt11/Mapperatorinator/runs/decoder-layer-candidate-cache-write-49253972-7044278/decoder_layer_candidate_cache_check_report.json \
  --require-cache-write-fingerprint \
  --require-candidate-cache-write-checks \
  --json-output /work/imt11/Mapperatorinator/runs/decoder-layer-candidate-cache-write-49253972-7044278/decoder_layer_candidate_cache_check_validation.json
```

Result:

```text
report_pass=True
candidate_cache_write_checks_pass=True
validation_pass=True
validation_failures=[]
logits_replay_allclose=True
logits_replay_max_abs=0.0
```

Candidate checks:

| Variant | Output allclose | Key slot match | Value slot match | Pass |
| --- | --- | --- | --- | --- |
| `repo_decoder_layer` | true | true | true | true |
| `self_attn_residual_segment` | true | true | true | true |
| `manual_decoder_runtime_island` | true | true | true | true |

## Bottleneck Interpretation

The result keeps the whole-layer native verifier path alive as a safe
experiment boundary, but it does not promote any runtime speed path. The same
diagnostic still shows the manual decoder runtime island is below the production
keep threshold:

| Replay boundary | Projected full-song seconds |
| --- | ---: |
| repo decoder layer, CUDA graph replay | `15.739054s` |
| manual decoder runtime island, CUDA graph replay | `15.058896s` |
| projected saving | `0.680159s` |
| projected TPS | `277.148 tok/s` |

That is about `4.5%` on the isolated replay boundary and below the `5%`
minimum for production/runtime complexity. The next accepted optimization still
needs to attack real decoder-layer math or memory behavior, prove current-stack
headroom above the bottleneck bar, and then pass full-song token/output
equivalence before any speed claim.
