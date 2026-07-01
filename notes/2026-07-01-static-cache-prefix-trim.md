# Static-Cache SDPA Prefix-Trim Experiment

## Idea

The torch-profile summary showed SDPA self-attention running with max-target static-cache shapes such as `[1, 12, 1, 2560]` for every decode token. The candidate kept the static cache allocation but built the 4D causal mask to the current decoder mask length and sliced self-attention K/V tensors to that same valid prefix before calling SDPA.

This was intended to preserve exact logits because the removed positions were future or unfilled static-cache slots that the original 4D mask excluded.

## Result

Rejected and reverted.

RTX 2080 Ti smoke slice on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

- Baseline commit: `01c18d6`
- Baseline job: `49109301`
- Baseline profile: `/work/imt11/Mapperatorinator/runs/smoke-base-49109301-01c18d6/beatmap6f980906005d441fb87edde94f269b83.osu.profile.json`
- Candidate commit: `aac2c4b`
- Candidate job: `49112400`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/smoke-prefix-49112400-aac2c4b/beatmapdd5d12b22215462a973be40cd9143954.osu.profile.json`

Comparator result:

```text
tokens_per_second: baseline=69.389, candidate=66.686, delta=-2.703 (-3.9%, worse)
model_elapsed_seconds: baseline=41.707, candidate=43.398, delta=+1.691 (+4.1%, worse)
generated_tokens: baseline=2894, candidate=2894
Token equivalence: PASS (2894 generated token IDs match).
```

## Interpretation

The output tokens matched, so the calculation was equivalent for this smoke run. It still lost speed. The likely explanation is that the shorter attention shape did not pay for the extra per-token slicing and produced less favorable SDPA/kernel behavior than the original full static-cache path.

Do not reintroduce this exact static-cache K/V prefix trim unless a future trace shows a different kernel profile or the implementation can avoid per-token slicing overhead.
