# Fast Prepare Fixed Probe

## Hypothesis

The accepted fused RoPE/cache stack still pays host-side per-token preparation
before each one-token decode. A direct active-prefix prepared-input builder might
avoid some Hugging Face `prepare_inputs_for_generation` overhead without changing
tokens, logits, RNG, or output behavior.

This was a production-flag retry of verifier infrastructure, not a new accepted
runtime path.

## Probe

Branch: `experiment/fast-prepare-fixed`

Temporary commit `7213058` added default-off
`inference_active_prefix_fast_prepare` and routed it through the existing control
plane:

- config/Hydra default: default `false`
- `inference.py`: validated as a simple sequential active-prefix CUDA graph mode
- `osuT5/osuT5/inference/server.py:model_generate()`: passed the mode into
  `active_prefix_decode_generate()`
- `processor.py` and profile metadata: recorded whether the flag was active
- `direct_decode.py`: excluded wrapper-only kwargs such as `negative_prompt` and
  `negative_prompt_attention_mask`

The branch was not merged to `main` because the full-song promotion gate failed.

## Gates

### Direct-loop verifier

- Job: `49232069`
- Commit: `7213058`
- Node/GPU: `dcc-core-ferc-s-z25-21`, RTX 2080 Ti
  `GPU-8de8e67b-289d-a1ae-3745-25f40db81f55`
- Run dir:
  `/work/imt11/Mapperatorinator/runs/fastprep-fixed-gate-49232069-7213058`
- Report:
  `/work/imt11/Mapperatorinator/runs/fastprep-fixed-gate-49232069-7213058/verify_fast_prepare_fixed.json`
- Result: PASS
- Generated tokens: PASS
- Raw logits: PASS
- Final CPU/CUDA RNG state: PASS
- Sampled steps: `256`

### 15s smoke

- Job: `49232092`
- Commit: `7213058`
- Run dir:
  `/work/imt11/Mapperatorinator/runs/fastprep-fixed-smoke15-49232092-7213058`
- Config: `configs/inference/profile_salvalai_smoke15.yaml`
- Stack: current accepted opt-in stack plus `inference_active_prefix_fast_prepare`
  for candidate runs
- Structure: four fresh processes with isolated compiler/cache dirs:
  `control_a`, `fast_a`, `fast_b`, `control_b`
- Token equivalence: PASS for main (`1,084 / 1,084`) and timing
  (`164 / 164`) in paired strict compares

Smoke aggregate:

| group | main tokens | main model s | main tok/s | main outer wall s |
| --- | ---: | ---: | ---: | ---: |
| control | `2,168` | `7.476751` | `289.966` | `11.904840` |
| fast prepare | `2,168` | `7.501902` | `288.993` | `9.093114` |

The smoke result did not show a model-time win (`-0.335%` main tok/s), but it
hinted at lower outer wall and lower total stage wall (`-17.352%`), so it was
promoted once to full-song.

## Full-Song Result

- Job: `49232174`
- Commit: `7213058`
- State: `COMPLETED`
- Node/GPU: `dcc-core-ferc-s-z25-21`, RTX 2080 Ti
  `GPU-8de8e67b-289d-a1ae-3745-25f40db81f55`
- Run dir:
  `/work/imt11/Mapperatorinator/runs/fastprep-fixed-fullsong-49232174-7213058`
- Profiles:
  - `control_a.profile.json`
  - `fast_a.profile.json`
  - `fast_b.profile.json`
  - `control_b.profile.json`
- Token equivalence: PASS for main (`7,639 / 7,639`) and timing
  (`821 / 821`) in paired strict compares

Per-run profiles:

| run | main model s | main tok/s | main wall s | timing model s | timing tok/s | stage wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control_a | `27.778665` | `274.995` | `30.127744` | `7.961541` | `103.121` | `43.058100` |
| fast_a | `29.238527` | `261.265` | `31.611988` | `8.307137` | `98.831` | `44.867497` |
| fast_b | `27.836399` | `274.425` | `30.182607` | `7.986340` | `102.801` | `42.937524` |
| control_b | `28.153878` | `271.330` | `30.752334` | `8.313126` | `98.759` | `44.019047` |

Averaged by group:

| group | main model s | main tok/s | main wall s | timing model s | timing tok/s | stage wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | `55.932543` | `273.150` | `60.880079` | `16.274667` | `100.893` | `87.077147` |
| fast prepare | `57.074925` | `267.683` | `61.794595` | `16.293477` | `100.777` | `87.805022` |

Aggregate deltas:

| metric | delta |
| --- | ---: |
| main model tok/s | `-2.002%` |
| main model time | `+2.042%` |
| main outer wall | `+1.502%` |
| timing model tok/s | `-0.115%` |
| total stage wall | `+0.836%` |

Strict paired compares:

| compare | main result | timing result |
| --- | --- | --- |
| `control_a` vs `fast_a` | `274.995 -> 261.265 tok/s`, FAIL | `103.121 -> 98.831 tok/s`, FAIL |
| `control_b` vs `fast_b` | `271.330 -> 274.425 tok/s`, small noisy PASS on aggregate but per-window FAIL | `98.759 -> 102.801 tok/s`, per-window FAIL |

## Decision

Reject and do not merge `inference_active_prefix_fast_prepare`.

The fixed flag is exact, but it does not improve full-song synchronized model
time and it slightly regresses total stage wall on average. The smoke outer-wall
signal did not survive the full-song gate. The correct conclusion is that
`prepare_inputs_for_generation` is not target-sized enough under the current
accepted DecodeSession/native stack to justify a production side path.

Leave `prepare_one_token_decode_inputs_fast` as verifier/diagnostic
infrastructure. Only revisit production fast prepare if a fresh post-runtime
profile shows input preparation has become a meaningful full-song bottleneck and
the candidate again goes through direct-loop, 15s smoke, full-song
token-equivalence, total-stage, and per-window non-regression gates.
