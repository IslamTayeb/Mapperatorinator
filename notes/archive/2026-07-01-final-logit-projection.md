# Final-Logit Projection Experiment

## Idea

During `generate`, Transformers samples from the final decoder position. The candidate passed `logits_to_keep=1` through the Mapperatorinator wrapper to VarWhisper so `VarWhisperForConditionalGeneration.forward` projected only `outputs[0][:, -1:, :]` through `proj_out` when `labels is None`.

## Result

Rejected and reverted.

The first implementation commit (`1011c1d`) failed validation in job `49110663` because the top-level Mapperatorinator wrapper did not declare/pass through `logits_to_keep`, and Transformers rejected it as an unused model kwarg.

The fixed implementation (`786a5fd`) ran successfully but was much slower.

RTX 2080 Ti smoke slice on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

- Baseline commit: `01c18d6`
- Baseline job: `49109301`
- Baseline profile: `/work/imt11/Mapperatorinator/runs/smoke-base-49109301-01c18d6/beatmap6f980906005d441fb87edde94f269b83.osu.profile.json`
- Candidate commit: `786a5fd`
- Candidate job: `49110976`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/smoke-logits2-49110976-786a5fd/beatmapac03a230f58f4de5a23bf66b2074c844.osu.profile.json`

Comparator result:

```text
tokens_per_second: baseline=69.389, candidate=54.510, delta=-14.879 (-21.4%, worse)
model_elapsed_seconds: baseline=41.707, candidate=53.091, delta=+11.384 (+27.3%, worse)
generated_tokens: baseline=2894, candidate=2894
Token equivalence: PASS (2894 generated token IDs match).
```

## Interpretation

The output tokens matched, so the calculation was semantically equivalent for this smoke run. It still lost badly. The likely explanation is that the smaller projection shape falls off the efficient larger-GEMM path or creates worse per-token launch behavior; nominally less arithmetic did not mean faster inference.

Do not reintroduce final-position-only VarWhisper logits projection unless a future torch trace shows a changed projection/kernel profile that justifies retesting.
