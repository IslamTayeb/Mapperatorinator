from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.optimized.scout.conditional_while_multinomial import (  # noqa: E402
    run_conditional_while_multinomial_probe,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()

    if args.output.suffix != ".json":
        raise ValueError("probe output must be a JSON file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run_conditional_while_multinomial_probe(
        seed=args.seed,
        iterations=args.iterations,
    )
    payload = result.to_dict()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    if not result.exact_conditional_while_feasible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
