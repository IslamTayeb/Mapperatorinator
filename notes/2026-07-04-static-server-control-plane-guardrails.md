# Static Server Control-Plane Guardrails

## Purpose

Prevent future Track B static-server profiles from reusing stale IPC servers or
bypassing known unsupported runtime combinations. This is mergeable
control-plane infrastructure, not a throughput claim.

## Change

- Added `get_server_runtime_key()` and runtime-keyed server socket paths for
  `use_server=true` loaders. The socket now distinguishes:
  - `max_batch_size`;
  - `server_batch_timeout`;
  - device;
  - precision;
  - attention implementation;
  - generation compile enablement.
- Added a loader-level fail-loud guard for
  `use_server=true` plus `inference_generation_compile=true`, so direct callers
  cannot bypass `compile_args()` validation.
- Updated `utils/profile_static_server_batch.py` to check the same
  runtime-keyed sockets that `load_model_with_server()` uses.
- Fixed `web-ui.py` server startup to use the current `get_server_address()`
  signature, pass `server_batch_timeout`, and reject server generation compile.
- Aligned `InferenceConfig.use_server` with the Hydra default:
  `use_server=false`.

## Validation

Local only:

```text
tests/test_inference_server_control_plane.py
```

No DCC/GPU profiling was run for this checkpoint. Future static-server
capacity runs should use these keyed sockets so `max_batch_size` and
`server_batch_timeout` changes cannot silently attach to stale owner servers.
