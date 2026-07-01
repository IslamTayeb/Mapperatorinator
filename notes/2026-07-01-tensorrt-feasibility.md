# TensorRT Feasibility Probe

## Result

Torch-TensorRT / TensorRT is not immediately runnable in the current DCC Mapperatorinator environment.

- Job: `49134026`
- Node: `dcc-core-ferc-s-z25-21`
- GPU: `NVIDIA GeForce RTX 2080 Ti`, capability `(7, 5)`
- Env: `/hpc/group/romerolab/imt11/envs/mapperatorinator`
- Repo commit: `a2cf83a`
- Log: `/work/imt11/Mapperatorinator/logs/trt-probe-49134026.out`

The GPU-node probe found:

- `torch 2.10.0+cu128`
- CUDA runtime visible, `torch.cuda.is_available() == True`
- `torch_tensorrt`: missing
- `tensorrt`: missing
- `onnx`: missing
- `onnxruntime`: missing
- `polygraphy`: missing

## Interpretation

The TensorRT spike remains a plausible 200 tok/s project, but it is currently an environment/bootstrap task rather than an immediate inference-code experiment. It should not be allowed to displace the custom decode/CUDA-graph work until an isolated DCC env has the TensorRT stack installed and can compile the repeated one-token decoder forward.

Graduation requirements remain unchanged:

1. Compile or export only the repeated one-token decoder forward first.
2. Prove logits match within the accepted dtype tolerance.
3. Prove fixed-seed generated token IDs match end-to-end on the 15s smoke slice.
4. Promote only after a plausible speed signal, then require full-song token equivalence before any speed claim graduates.

## Next Step

Keep the retained runtime baseline as SDPA plus `inference_generation_compile=true`. Start the exact custom decode-loop feasibility spike behind an opt-in flag, with v1 hard-disabled for CFG, beams, parallel generation, and server batching.
