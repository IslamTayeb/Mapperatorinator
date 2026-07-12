# Persistent Setup Audit

Job `49617869` at `c733941` measured accepted five-song smoke windows on one
RTX 2080 Ti in FP32/SDPA, in both song orders with five reset-separated trials.
Token hashes and final `.osu` hashes matched accepted job `49543717`.

The old synthetic charge was `48.832990 ms/window`; the strict warm persistent
transition was `89.065463 ms/window`. The resulting dependency-aware K3 ceiling
was `447.694 tok/s`. Even making conditioning, encoder, stable copy, and
cross-KV free left `60.773577 ms/window` and only `495.080 tok/s`. Encoder
storage, cross-KV prefetch, and K3 runtime work were therefore rejected.

Artifacts:

- setup: `/work/imt11/Mapperatorinator/runs/persistent-setup-49617869-c733941/persistent-setup.json`, SHA `047e80dc034d3033b30297ff85e17470030ac6b867071262b61d8267ba94be6a`
- K3: `/work/imt11/Mapperatorinator/runs/persistent-setup-49617869-c733941/dependency-k3.json`, SHA `d0302af20966b40d6132fabb3c27ca5cca3206e51952ba3fcf7d58fde97fbe4b`
- precompute ceiling: `/work/imt11/Mapperatorinator/runs/persistent-setup-49617869-c733941/persistent-precompute-ceiling.json`, SHA `456f6f0ee167dcae0814e14dcf35a2b616568dd27bbe709bcb55236340f172c7`

Slurm exit `1:0` was the intentional keep-gate rejection. Revisit only after a
materially different accepted setup/decode result restores target-sized
dependency-aware headroom.
