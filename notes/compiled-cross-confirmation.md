# Compiled-cross authoritative confirmation pin

- Base tip: `0dbab9e5` (`codex/500tps-arena-compiled-cross-last-mile`)
- Composition: already clean — compiled accepted cross-BMM is on selected-stack lineage
  (shared-arena tip `3cd327df` is an ancestor of `0dbab9e5`).
- Independent win: job `49910316` (h36-9) relaxed PASS — main TPS 477.96→483.61 (+5.65),
  main_model −0.208s, complete_request wall −0.319s; final_map_equal; dispatch declaration pass.
- Caveat: that job's absolute baseline (~478 TPS) is below prior selected reference (~494.27);
  this pin re-runs the same same-commit reciprocal alone to confirm the delta without concurrent GPU jobs.
- Isolation: reciprocal wrapper uses `TMPDIR=/work/imt11/Mapperatorinator/tmp/reciprocal-$SLURM_JOB_ID`
  and job-local torch_extensions / TORCHINDUCTOR cache under the run root.
- Do not compose DP4A (49908852 DROP). Do not claim cold one-off win (AOT/setup ~13s).
- Mode: AOT/persistent compile path; `max-autotune-no-cudagraphs` (scout 49906034).
- Graduation detail: `notes/compiled-cross-graduation.md` (independence + non-additive TPS caveat).
