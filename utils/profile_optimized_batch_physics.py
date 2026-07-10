"""Plan/record scaffold for the optimized GPU batch-physics scout.

No GPU runner is wired yet.  ``--describe-plan`` emits the bounded experiment
matrix.  Attempting execution fails loudly so planning output cannot be mistaken
for throughput evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.optimized.benchmark import (
    BatchPhysicsObservation,
    BatchPhysicsPlan,
    compare_batch_physics_observations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--describe-plan",
        action="store_true",
        help="Emit the required merged-batch/lane-pool matrix without running CUDA.",
    )
    actions.add_argument(
        "--record-observation",
        type=Path,
        help="Validate and normalize one completed GPU observation JSON.",
    )
    actions.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("BASELINE", "CANDIDATE"),
        help="Strictly compare two exact-output GPU observation JSON files.",
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.describe_plan:
        payload = BatchPhysicsPlan().as_dict()
    elif args.record_observation is not None:
        payload = BatchPhysicsObservation.from_dict(
            json.loads(args.record_observation.read_text(encoding="utf-8"))
        ).as_dict()
    elif args.compare is not None:
        baseline_path, candidate_path = args.compare
        baseline = BatchPhysicsObservation.from_dict(
            json.loads(baseline_path.read_text(encoding="utf-8"))
        )
        candidate = BatchPhysicsObservation.from_dict(
            json.loads(candidate_path.read_text(encoding="utf-8"))
        )
        payload = compare_batch_physics_observations(baseline, candidate)
    else:
        raise RuntimeError(
            "GPU batch-physics execution is not implemented. Implement and verify the merged B=1 "
            "one-token runner before merged B=2/5/8, and implement independent streams, graph "
            "instances, buffers, per-request generators, caches, and cuBLAS workspaces before "
            "enabling any B1 lane-pool measurement."
        )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
