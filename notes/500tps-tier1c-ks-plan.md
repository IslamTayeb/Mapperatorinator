# TIER1(c) — 30-seed KS distribution parity (plan)

Depends on **TIER1a PASS** (greedy canary) and TIER1b rejection unit tests.

## Scope

- Engine: `inference_engine=turbo` @ tip `codex/turbo-tiny-draft`
- Baseline: `inference_engine=optimized` bit-exact tip `55949274`
- Precision: FP16; temp 0.9 / top-p 0.9; γ=5; draft ckpt from §37 train `50146289`
- Seeds: ≥30; songs: ≥3 (include SALVALAI + 2 tip gallery maps)
- Metrics (KS, α=0.01, Bonferroni): HO count, note-density curve, HO-type histogram, timeshift, slider-length
- Record pack paths + job IDs in ledger §36; no 500 claim from KS alone

## Jobs (submit after TIER1a)

1. `s36-tier1c-generate-optimized.sbatch` — 30×3 optimized `.osu` set
2. `s36-tier1c-generate-turbo.sbatch` — matching turbo set
3. `s36-tier1c-ks.sbatch` — KS harness over both sets

## Not this work

§39 hybrid TIER3; tip graduation; INT8-as-FP16.
