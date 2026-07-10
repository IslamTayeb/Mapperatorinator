"""CLI surface for a future bounded GPU speculative-decoding scout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.optimized.speculative import (  # noqa: E402
    ProposalSourceKind,
    SpeculationConfig,
    SpeculativeScoutConfig,
    load_scout_adapter_factory,
    run_bounded_speculative_scout,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded exact speculative-decoding scout. This utility does not load a model "
            "unless an explicit module:function GPU adapter factory is provided."
        )
    )
    parser.add_argument("--proposal-source", choices=[kind.value for kind in ProposalSourceKind], required=True)
    parser.add_argument("--speculation-k", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--eos-token-id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--draft-model-path",
        default=None,
        help="Optional v32-mini model path; defaults to OliBomby/Mapperatorinator-v32-mini.",
    )
    parser.add_argument(
        "--adapter-factory",
        help="Explicit GPU adapter factory as module:function. Omission fails before model loading.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = SpeculativeScoutConfig(
        proposal_source=ProposalSourceKind(args.proposal_source),
        speculation=SpeculationConfig(
            speculation_k=args.speculation_k,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=args.eos_token_id,
        ),
        seed=args.seed,
        draft_model_path=args.draft_model_path,
    )
    adapter = None
    if args.adapter_factory:
        adapter = load_scout_adapter_factory(args.adapter_factory)(config)
    comparison = run_bounded_speculative_scout(config, adapter)
    manifest = {
        "result_class": "verifier_only_exact_transcript",
        "proposal_source": config.proposal_source.value,
        "seed": config.seed,
        "draft_model_path": config.draft_model_path,
        **comparison.to_manifest(),
    }
    rendered = json.dumps(manifest, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
