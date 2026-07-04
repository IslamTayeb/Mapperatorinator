# Current Exact Optimization Frontier

## Purpose

Refresh the bottleneck and acceptance boundary after two July 4 cuts:

- weighted manual decoder-runtime recomposition is exact but too small;
- native self+cross prefix is target-shaped but numerically non-equivalent under
  the current native cache-write verifier.

This note is a stop/go artifact. It is not a throughput claim and does not
change the accepted `270.475 tok/s` baseline.

## Current Baseline And Target

Accepted exact single-song baseline:

- DCC job: `49230082`
- main tokens: `7,639`
- synchronized main model time: `28.243s`
- main-generation throughput: `270.475 tok/s`
- fixed-seed main/timing token equivalence: PASS
- generated `.osu` byte equivalence: PASS

Target:

- `500 tok/s` needs `15.278s` model time;
- remaining required saving is `12.965s`, or `45.9%` of current model time;
- `5%` keep bar is `1.412s`;
- `10%` strong bar is `2.824s`.

## Acceptance Boundary

Full-song promotion is based on same-calculation metadata, generated token IDs,
output artifact bytes, and no-regression checks:

```bash
python utils/summarize_inference_profile.py \
  --compare "$BASELINE/profile.json" "$CANDIDATE/profile.json" \
  --strict-full-song \
  --json-output "$RUN_DIR/compare-full-song.json"
```

For cache-writing native decoder-layer candidates, the current pre-production
verifier is stricter: `utils/profile_decode_decoder_layer_island.py
--verify-cache-write-candidates` requires the q_len=1 self-cache K/V write to
match the expected slot SHA, plus output allclose. This is a verifier/preflight
gate, not part of `--strict-full-song`, but the current project docs require it
before production wiring for this class of candidate.

The one-token logits gate uses allclose and top-k matching; the direct-loop gate
adds generated-token and final-RNG equality. So tiny fp32 drift is not
automatically fatal for every experiment, but a candidate that fails a required
cache-write verifier cannot be called an exact same-calculation native-layer win
under the current rules.

## Current Candidate Ledger

| Candidate | Evidence | Exactness | Decision |
| --- | --- | --- | --- |
| Broad native decoder-layer or decoder-stack math/memory verifier | Weighted decoder layer is `16.512s` of `28.243s`; weighted stack roofline has `10.319s` above-floor headroom | Unproven; must preserve ABI, cache-write SHA, logits/top-k, direct-loop token/RNG, smoke/full-song token IDs, and output bytes | Only remaining exact target-sized class |
| Manual decoder-runtime recomposition | Job `49262108`: `16.512013s -> 15.895097s`, saving `0.616916s`, projected `276.514 tok/s` | Exact/cache checks passed | Rejected: below `5%` |
| Native self+cross prefix | Job `49258712`: prefix640 projected `~4.93-4.96s` saved, `~328 tok/s`; job `49266140`: `456` key and `595` value numeric bit mismatches, output drift `7.63e-06` | Non-equivalent under current cache-write gate | Rejected for same-calculation; only revisit as explicit documented-drift |
| Fixed-K / multistep tail graph | Prior verifier ceiling around `6.134s`, but production one-step tail graph was only `+0.375%`; forced-EOS audit proves naive fixed-K over-advances RNG | High EOS/RNG risk | Not next; only inside broader DecodeSession runtime if exact early-exit/rollback is proven |
| Cold-start/native extension preload | Compiler-cold first-window outer wall can add `60-65s`; synchronized model-time TPS stays near `270` | Exact packaging/cache issue | Not a main-generation TPS path unless user-visible cold wall becomes the target |
| Graph shell/input copy/final projection/per-linear/MLP-only/self-attention-only/cross-attention-only | Existing probes are below threshold or non-equivalent | Mixed | Cut unless fresh profiling changes target size |

## External Determinism Context

NVIDIA documents that cuBLAS routines are bitwise reproducible for a given
toolkit version on GPUs with the same architecture and SM count, but not across
toolkit versions; multi-stream execution needs separate workspace/handles,
cuBLASLt with owned workspace, or `CUBLAS_WORKSPACE_CONFIG` to avoid internal
workspace nondeterminism. PyTorch documents that complete reproducibility is not
guaranteed across releases, commits, platforms, or CPU/GPU, and that enforcing
deterministic algorithms can reduce performance.

Practical implication for this project:

- Using stable cuBLAS/cuBLASLt/CUTLASS can be repeatable on one DCC stack, but
  repeatability is not the same as matching PyTorch/HF byte-for-byte.
- Custom warp reductions for linears/RMSNorms are expected to change fp32
  operation order and can fail cache/logit exactness even when allclose passes.
- Exact native work must either preserve the reference operation boundary for
  cache-writing state or prove exactness through the established gates before
  production wiring.

References:

- NVIDIA cuBLAS documentation, Results Reproducibility:
  https://docs.nvidia.com/cuda/cublas/#results-reproducibility
- PyTorch Reproducibility note:
  https://docs.pytorch.org/docs/2.9/notes/randomness.html

## Next Stop/Go

Do not start another narrow production flag.

The only exact next implementation class worth considering is a verifier-only
whole-layer or whole-stack native math/memory candidate. It must:

1. replace multiple adjacent operation classes together, not one linear or one
   attention kernel;
2. pass the decoder-layer ABI and cache-write SHA checks;
3. show weighted CUDA-graph projected saving `>=1.4s`, preferably `>=2.8s`,
   before any production integration;
4. then pass one-token logits/top-k, direct-loop token/logit/RNG, 15s smoke
   token equivalence, full-song token/output equivalence, and no-regression.

If no such exact candidate can be designed without changing fp32 operation
order, the honest next fork is a user decision: either stop exact
same-calculation work, or explicitly evaluate a `documented-drift` native path
with separate tables and no same-calculation speed claim.
