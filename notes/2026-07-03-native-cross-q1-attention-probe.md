# Native Cross q_len=1 Attention Probe

## Purpose

Close a cheap uncertainty before broad decoder-layer work: test whether the
existing native q_len=1 attention kernel can replace the accepted q1 BMM
cross-attention reduction for the decoder's encoder-cache attention path.

This was diagnostic-only. It did not touch production inference and does not
claim a throughput win.

## Context

Current accepted single-song opt-in baseline:

- DCC job `49230082`
- full-song SALVALAI, `7,639` main tokens
- `28.243s` synchronized main-generation model time
- `270.475 tok/s`
- main/timing generated-token equivalence PASS
- byte-identical generated `.osu`

The weighted accounting pass sizes cross-attention at about `3.124s` full-song,
with q1 BMM attention around `1.713s` and q/out projections around `1.286s`.
That is borderline target-sized, so this probe tested an already-existing
native attention kernel before considering any new cross-attention kernel work.

## Probe

Branch:

`experiment/native-cross-q1-attn-probe`

Experiment commit:

`ec47b34` (`profile: add native cross attention probe`)

DCC job:

`49236047`

Run root:

`/work/imt11/Mapperatorinator/runs/native-cross-q1-probe-49236047-ec47b34`

Logs:

- `/work/imt11/Mapperatorinator/logs/native-cross-q1-probe-49236047.out`
- `/work/imt11/Mapperatorinator/logs/native-cross-q1-probe-49236047.err`

Reports:

- `cross_prefix128.json`
- `cross_prefix640.json`

Environment:

- Node: `dcc-chsi-gpu-ferc-s-i11-1`
- GPU: RTX 2080 Ti, UUID `GPU-5b9229da-8433-e112-04c8-086f173136fb`
- Driver: `595.71.05`
- Torch: `2.10.0+cu128`
- Transformers: `4.57.3`
- CUDA runtime reported by PyTorch: `12.8`
- Precision: `fp32`
- Attention: `sdpa`
- Config: `profile_salvalai_smoke15`, `sequence_index=9`, seed `12345`

The temporary diagnostic added `--native-q1-attention-variant` to
`utils/profile_decode_cross_attention_island.py`. It compared the accepted
`torch.bmm -> softmax -> torch.bmm` q1 cross-attention reduction against the
existing `native_q1_attention()` kernel and a manual native-q1 cross-attention
island.

The code was not merged to `main`.

## Results

Both reports passed the existing verifier gates:

| prefix | pass | logits replay | max abs | captured cross-attn modules |
| --- | --- | --- | ---: | ---: |
| `128` | PASS | PASS | `0.0` | `12` |
| `640` | PASS | PASS | `0.0` | `12` |

CUDA graph replay timings per decoder layer:

| prefix | repo cross module | manual q1 BMM island | q1 BMM attention only | native q1 attention only | manual native-q1 island |
| --- | ---: | ---: | ---: | ---: | ---: |
| `128` | `0.032527ms` | `0.032469ms` | `0.017988ms` | `0.066503ms` | `0.081090ms` |
| `640` | `0.032476ms` | `0.032538ms` | `0.017918ms` | `0.066462ms` | `0.081196ms` |

The native attention outputs were close enough for this diagnostic:

- native q1 attention max abs: `4.41e-06`
- manual native-q1 island max abs: `3.81e-06`

But it was much slower than the accepted q1 BMM path. The full manual
native-q1 cross-attention island was about `2.5x` slower than the repo q1 BMM
cross module under graph replay.

## Decision

Reject reusing the existing native q_len=1 attention kernel for cross-attention.

The accepted q1 BMM cross-attention path remains the right implementation for
the current fp32 batch-1 encoder-cache shape. Cross-attention is still a useful
future encoder-decoder concept, but a production candidate would need a new
fused cross-attention residual island that saves at least `~1.6s` full-song.
Do not broaden the existing native self-attention kernel to cross-attention.
