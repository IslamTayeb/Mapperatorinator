# W3 — Tiger Batched Encoder Prefill (Handoff)

**Status:** **GPU RECIPROCAL SUBMITTED / HARVEST PENDING** — scout path wired; FP16 2080 Ti B/C/C/B vs tip optimized  
**Branch:** `codex/w3-batched-encoder-prefill`  
**Base tip:** `55949274` / FP16 **366.11** — **no merge to main**; **no 500 claim**  
**Package:** `notes/500tps-cross-arch-package.md` · W1: `notes/500tps-cross-arch-w1-handoff.md`

## Why W3 (provisional; Ada still pending ~Jul 19–20)

A5000 W1 lean: Tiger compiled-decode fp16 recon **~350 ≥** optimized **~316**. Package prefers **W3** over **W2**. 2080: opt **267** > tiger **229** (specialist). Tip sanity: W1 cell opt fp16 **267 ≠ 366** — **do not demote tip**; measure W3 vs tip auth carefully (song wall + main TPS) and vs same-job tip optimized baseline.

Prior related STOP: encoder precompute `49903861` **STOP_ENCODER_PRECOMPUTE** (B1 complete-request wall regress). W3 revisits with **B16**, CPU store, and decode-skip credited to complete wall.

## Tiger source (PR #120)

- `server.precompute_encoder_outputs` — chunked encoder over all windows (`batch_size=16`)
- Processor path hoists precompute before sequential window decode
- Store on **CPU**; quality-equivalent / **not bit-identical** (batched kernel drift)
- Opt-in; V32 path unchanged when disabled

## Our port (ownership)

| Piece | Path | Role |
| --- | --- | --- |
| Core scout | `osuT5/.../optimized/scout/batched_encoder_prefill.py` | B16 precompute + opt-in Processor hooks |
| Lazy scout export | `osuT5/.../optimized/scout/__init__.py` | `__getattr__` only — no cold eager import |
| Reciprocal runner | `utils/run_batched_encoder_prefill_candidate.py` | baseline cold / candidate install |
| DCC wrapper | `scripts/dcc/profile_batched_encoder_prefill_reciprocal.sbatch` | FP16 2080 Ti B/C/C/B + gates |
| Unit/smoke | `tests/test_batched_encoder_prefill_scaffold.py` | CPU-only; mock encoder B1≡B16 |
| Ceiling harness | `utils/profile_batched_encoder_precompute_ceiling.py` | Verifier-only B1..B16 timings |

**Not wired into** accepted `OptimizedSingleRuntime` / selectors / server. Install only via:

```python
from osuT5.osuT5.inference.optimized.scout.batched_encoder_prefill import (
    install_batched_encoder_prefill_candidate,
)

with install_batched_encoder_prefill_candidate(Processor, batch_size=16, storage="cpu"):
    ...
```

Defaults: `batch_size=16`, `storage="cpu"`, precisions `{fp16, fp32}`, `result_class=documented-drift`.

## Gates (promotion)

| Gate | Pass rule |
| --- | --- |
| **Ownership** | Changes only under `inference/optimized/` (+ scout runner/sbatch); V32 cold / tip bytes unchanged when candidate off |
| **Exactness** | Documented-drift: default require greedy token streams + `.osu` equal vs same-job tip optimized; do **not** require bit-identical encoder hiddens |
| **Wall** | Candidate complete-request wall improves ≥**5%** vs same-job tip optimized; precompute charged to request |
| **TPS** | Main-gen model TPS must not regress ≥1% vs **same-job baseline**; also report vs tip auth **366.11** (do not demote tip on SANITY_GAP) |
| **Memory** | Peak VRAM with CPU store ≤ tip + small H2D scratch; fail loudly on OOM |
| **Stop** | First exactness / negative-wall / ownership / insufficient-gain failure → no production wiring |

## DCC reciprocal

```bash
# on DCC, after push + worktree sync
export MAPPERATORINATOR_REPO=/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/w3-batched-encoder-prefill
# (actual path set at submit time)
export MAPPERATORINATOR_COMMIT=<pushed HEAD>
export MAPPERATORINATOR_BRANCH=codex/w3-batched-encoder-prefill
export MAPPERATORINATOR_REMOTE_REF=refs/remotes/origin/codex/w3-batched-encoder-prefill
export MAPPERATORINATOR_PRECISION=fp16
export MAPPERATORINATOR_ALLOW_PARALLEL_RECIPROCAL=1   # Ada W1 + §57b may be pending
sbatch scripts/dcc/profile_batched_encoder_prefill_reciprocal.sbatch
```

Unique per-job `TMPDIR` / `TORCH_EXTENSIONS_DIR`. Harvest: `$WORK/runs/w3-b16-encoder-prefill-fp16-<job>/gate.json`.

## Jobs / harvest

| Field | Value |
| --- | --- |
| Job | _(fill on submit)_ |
| Decision | _(PROMOTE_SCOUT / STOP)_ |
| Wall Δ | _(fill)_ |
| Main TPS Δ vs baseline / tip auth | _(fill)_ |

## Next after harvest

1. If **STOP** (wall &lt;5% or drift): keep scout+runner; no `optimized/single/` wiring; record revisit condition.
2. If **PROMOTE_SCOUT**: optional production wiring under `optimized/single/` still needs a separate gate pass — still no V32/server/merge.
3. Ada firm-up can wait (~Jul 19–20); do not block W3 harvest on Ada.
4. Do not stack with turbo until each track seals independently.

## Smoke (CPU)

```bash
cd /work/projects/Mapperatorinator-worktrees/w3-batched-encoder-prefill
python -m pytest tests/test_batched_encoder_prefill_scaffold.py -q
```

## One-line summary

W3 B16 encoder prefill is an opt-in optimized scout (CPU store, documented-drift) with a DCC FP16 reciprocal harness vs tip `55949274`/366.11; no merge, no 500 claim; Ada firm-up deferred.
