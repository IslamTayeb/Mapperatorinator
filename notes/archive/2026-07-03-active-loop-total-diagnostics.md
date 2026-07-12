# Active Loop Total Diagnostics

## Purpose

Add a diagnostic-only runtime gap ledger to the accepted active-prefix
DecodeSession/native stack. The goal was to explain the remaining gap between
isolated one-token graph replay and production synchronized model time before
starting broader runtime or native-kernel work.

This is not a throughput optimization and no speedup graduated.

Accepted baseline entering this pass:

- Full-song job: `49230082`
- Main tokens: `7,639`
- Synchronized main model time: `28.243s`
- Throughput: `270.475 tok/s`
- Equivalence: main/timing token identity PASS and byte-identical `.osu`

## Code Change

`osuT5/osuT5/inference/decode_loop.py` now emits extra fields only when
`profile_active_prefix_decode_diagnostics=true`:

- total active-prefix loop wall/CUDA event time;
- derived child wall total and unattributed loop wall;
- graph signature lookup wall/CUDA event time;
- graph capture wall/CUDA event time;
- graph static-input copy wall/CUDA event time;
- graph replay wall/CUDA event time;
- static input copy bytes/calls by key;
- graph replay fraction of loop CUDA event time.

The diagnostics-off path does not record the loop timer.

Local checks:

```text
python -m py_compile osuT5/osuT5/inference/decode_loop.py
git diff --check
```

## DCC Jobs

Failed preflight:

- Job: `49232537`
- Commit: `b43e45a`
- Failure: diagnostics-off control called `_record_diagnostic_cuda_start(None)`
  before the null-safe helper was added.
- Result: failed before producing profile evidence.

Completed diagnostic:

- Job: `49232548`
- Commit: `5d08f8a`
- Branch: `experiment/active-loop-total-diagnostics`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti
  `GPU-de50b870-b708-690d-74a4-6e855163a133`
- Driver: `595.71.05`
- Torch: `2.10.0+cu128`
- Transformers: `4.57.3`
- Run dir:

```text
/work/imt11/Mapperatorinator/runs/active-loop-total-diag-20260703-133158-5d08f8a
```

Artifacts:

- `control.profile.json`
- `diag.profile.json`
- `control_vs_diag.main.json`
- `control_vs_diag.timing.json`

Config:

- `profile_salvalai_smoke15`
- `seed=12345`
- `precision=fp32`
- `attn_implementation=sdpa`
- `use_server=false`
- `parallel=false`
- active-prefix bucket64 CUDA graph, warmup0, min decode steps 1
- stateful monotonic logits processor
- q1 BMM cross-attention
- DecodeSession runtime + CUDA graph
- native q1 self-attention + fused RoPE/cache self-attention
- `profile_record_token_ids=true`

## Equivalence And Overhead

The diagnostic profile preserved generated token IDs:

| label | token equivalence |
| --- | --- |
| main generation | PASS, `1,084 / 1,084` |
| timing context | PASS, `164 / 164` |

The diagnostic run is not a speed claim. The added diagnostic ranges changed
timing:

| label | control tok/s | diagnostic tok/s | model-time delta |
| --- | ---: | ---: | ---: |
| main generation | `267.946` | `242.199` | `+0.430s`, worse |
| timing context | `43.902` | `54.070` | `-0.703s`, better/noisy |

Use this only for attribution.

## Main-Generation Ledger

Aggregated over the 15s smoke main-generation records:

| field | value |
| --- | ---: |
| generated tokens | `1,084` |
| diagnostic model time | `4.476s` |
| loop total wall | `4.254s` |
| loop total CUDA event | `4.147s` |
| decode-forward CUDA graph event | `2.631s` |
| graph replay event | `2.221s` |
| graph capture event | `0.172s` |
| graph static-input copy event | `0.084s` |
| graph signature lookup event | `0.053s` |
| prepare inputs event | `0.569s` |
| update model kwargs event | `0.009s` |
| logits processors event | `0.109s` |
| sampling softmax event | `0.007s` |
| sampling multinomial event | `0.047s` |
| stopping criteria event | `0.302s` |

Static input copy volume:

| key | bytes copied |
| --- | ---: |
| `decoder_attention_mask` | `10,895,360` |
| `decoder_input_ids` | `8,512` |
| `decoder_position_ids` | `8,512` |
| `cache_position` | `8,512` |

Interpretation:

- Static graph input copy is visible but not target-sized by itself. Scaling the
  smoke `84ms` event by full-song token count gives roughly `0.6s`, below the
  full-song `5%` keep threshold.
- Graph replay remains the largest measured bucket in the production loop.
- `prepare_inputs` and `stopping_criteria` are visible in this diagnostic, but
  these ranges are not exclusive and add synchronization overhead. Fast prepare
  already failed full-song promotion, so this ledger is not enough to restart
  that path.
- Child wall ranges are intentionally nested and non-exclusive; the child-wall
  sum exceeded total loop wall for main generation. Use CUDA-event totals and
  relative bucket sizing cautiously.

## Timing-Context Ledger

Aggregated over the timing-context smoke records:

| field | value |
| --- | ---: |
| generated tokens | `164` |
| diagnostic model time | `3.033s` |
| loop total wall | `2.620s` |
| loop total CUDA event | `2.508s` |
| decode-forward CUDA graph event | `0.451s` |
| graph replay event | `0.363s` |
| graph capture event | `0.053s` |
| graph static-input copy event | `0.012s` |
| prepare inputs event | `0.142s` |
| logits processors event | `0.078s` |
| stopping criteria event | `0.078s` |

Timing diagnostics remain noisy and should not drive a production optimization
without a separate timing-specific gate.

## Decision

Keep the runtime gap ledger as diagnostic infrastructure.

Do not implement a production optimization from this pass. The ledger rejects
static graph input copying as a standalone target, and it does not overturn the
previous fast-prepare or tail-fusion rejections.

Next plausible implementation work still needs a broad exact runtime/native
decoder-layer island, or a more precise low-overhead ledger that can prove a
single production control bucket exceeds the `~1.4s` full-song keep threshold.
