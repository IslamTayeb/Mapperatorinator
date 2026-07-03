# Native Self-Attention Residual Probe

## Hypothesis

After the accepted fused RoPE/cache q1 self-attention path, a broader native
self-attention residual island might still help by fusing `RMSNorm -> Wqkv`
and `Wo -> residual` around the existing native q1 attention kernel. This was
intended as a verifier-first diagnostic, not a production inference path.

## Implementation

Experiment branch:
`experiment/native-self-attn-residual-island`

Commit:
`6d9fc4d03427a3d39097d851e76f519aeebde484`

Temporary files/changes:

- Added `osuT5/osuT5/inference/native_self_attention_residual.py`.
- Extended `utils/profile_decode_decoder_layer_island.py` with
  `--candidate-native-self-attn-residual`.
- The native helper tested three fp32 CUDA warp-group variants:
  `native_self_attn_residual_warp2`, `warp4`, and `warp8`.
- The probe kept the existing accepted native q1 RoPE/cache attention kernel in
  the middle, and only replaced the boundary around it.

This branch was not merged to `main`.

## DCC Result

DCC job:
`49235105`

Hardware:
RTX 2080 Ti, `GPU-70edd336-9ffb-9a85-c0ee-c04d77b76094`,
node `dcc-gehmlab-gpu-ferc-s-z25-19`

Stack:

- Torch `2.10.0+cu128`
- Transformers `4.57.3`
- Precision `fp32`
- `attn_implementation=sdpa`
- Config `profile_salvalai_smoke15`
- Sequence index `9`
- Active-prefix decode length forced to `640`
- CUDA graph replay enabled

Artifacts:

- JSON:
  `/work/imt11/Mapperatorinator/runs/native-selfattn-residual-49235105/self_attn_residual_L640.json`
- stdout:
  `/work/imt11/Mapperatorinator/logs/native-selfres-smoke-49235105.out`
- stderr:
  `/work/imt11/Mapperatorinator/logs/native-selfres-smoke-49235105.err`

Correctness:

- Decoder logits replay pass: `true`
- Logits replay max abs: `0.0`
- Native self-attention residual max abs versus the captured boundary:
  `8.344650268554688e-07`
- CUDA graph replay allclose: `true`

## Timing

Prefix640 self-attention residual CUDA graph replay projection:

| Variant | Graph replay ms/layer | Projected full-song seconds | Saved vs repo self-attn residual | Projected TPS |
| --- | ---: | ---: | ---: | ---: |
| Repo self-attn residual segment | `0.077526` | `7.025723s` | baseline | baseline |
| Native residual warp2 | `0.078070` | `7.074982s` | `-0.049259s` | `270.003` |
| Native residual warp4 | `0.077973` | `7.066248s` | `-0.040525s` | `270.087` |
| Native residual warp8 | `0.078325` | `7.098159s` | `-0.072436s` | `269.782` |

The ungraphed event timing looked better for the native wrappers
(`~0.256-0.259ms` versus `0.395ms` for the repo boundary), which is consistent
with launch-count cleanup. That does not help the current fastest path enough,
because production decode is already under CUDA graph replay, where the native
variants were slightly slower.

## Decision

Rejected. Do not merge the experiment branch.

Reason:

- The candidate was exact/allclose, but regressed the graph-replayed boundary
  that matters for the current fastest single-song path.
- The best projected result was still negative: `-0.0405s` saved at prefix640.
- This is far below the `>5%` full-song keep threshold and below the roughly
  `~1.4s` minimum useful projected saving for new native-kernel complexity.

## Lesson

For the current CUDA-graph DecodeSession path, launch-count cleanup around
self-attention is mostly exhausted. Native kernels must improve real math or
memory behavior inside the graph, not just reduce eager-launch overhead.

This does leave a possible future lesson for non-graph or service paths: if a
future architecture/runtime cannot use CUDA graph replay effectively, similar
fused residual islands could matter again. For the present exact single-song
campaign, this branch should remain unmerged.
