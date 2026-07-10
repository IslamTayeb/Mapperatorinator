"""Strict contracts for the bounded production-weighted B2 scout.

The model-free ceiling is intentionally separate from the GPU verifier.  In
the accepted-profile optimistic cost model, five requests cannot reach 500
tok/s with only two lanes.  A duplicated ten-request queue can still reach the
target only if the weighted complete B2 decode step clears the derived
harmonic-throughput requirement.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


WEIGHTED_CEILING_SCHEMA_VERSION = 1
WEIGHTED_CEILING_GATE = "accepted_five_song_model_free_b2_ceiling"
WEIGHTED_SOURCE_CONTRACT_SCHEMA_VERSION = 1
WEIGHTED_SOURCE_CONTRACT_GATE = "lambada_repeat01_seq9_real_prefix"
WEIGHTED_CONTEXT_REPLAY_SCHEMA_VERSION = 1
WEIGHTED_CONTEXT_REPLAY_GATE = "lambada_repeat01_context_only_replay"
# The canonical context-only report was produced without a model or CUDA by
# Slurm job 49558528.  Authorization uses the canonical JSON digest, not the
# pretty-printed file SHA.  The complete reviewed report is reconstructed below
# from immutable fields so a nonempty digest cannot authorize capture alone.
REVIEWED_CONTEXT_REPLAY_JOB_ID = "49558528"
REVIEWED_CONTEXT_REPLAY_COMMIT = "c6b45fdf919a09c570bdfa7ce51f60568862613f"
REVIEWED_CONTEXT_REPLAY_REPORT_SHA256: str | None = (
    "bf2e787bff309e8ce62382e2e3a00af3e4230b00b210331399c471cc4236e72b"
)
REVIEWED_CONTEXT_REPLAY_FILE_SHA256 = (
    "d47bc29cfb2755133bc48fa722f4475d06a94bfd467bc2b077e9bdfa9e7573de"
)
REVIEWED_CONTEXT_PROMPT_SHA256 = (
    "216fde48e17e2b4b9cebebadb454efe81b80a86bd2c1c81e4c710db35bb27663"
)
REVIEWED_CONTEXT_PROMPT_MASK_SHA256 = (
    "d9ad68f2a11712547d5eb0fa4bb5c0e80e319cfab464516aa02b0e141ee8b5f4"
)
REVIEWED_CONTEXT_FRAMES_SHA256 = (
    "a2c94c30626130130d5b90c6a458a7bf5bed5f752cfdab91073837c9f874d642"
)
MODEL_SNAPSHOT_REVISION = "74f22583400d259bf424819e11027c17933efe54"
MODEL_CONFIG_SHA256 = (
    "ff96b8c059179978c93b6f938e39fac945d5682a3495789c00cc159d571a1a22"
)
REVIEWED_SOURCE_CAPTURE_JOB_ID = "49559386"
REVIEWED_SOURCE_CAPTURE_COMMIT = "1896b1990e118cc9643e45256e7a80128a3b6fc4"
REVIEWED_SOURCE_CONTRACT_SHA256 = (
    "de55e2d16c4b085c5cce62e83be8cd4f83bb23f2a40e87f80be23a88f216b99a"
)
REVIEWED_SOURCE_REPORT_FILE_SHA256 = (
    "8757f97874fd7249420feb1d2e03de43a646da8187cdebb34d3ee306baf7de62"
)
REVIEWED_SOURCE_CAPTURE_PROMPT_SHA256 = (
    "4b4fa1542791fd34a54d9284b4a540735e20f872437bb0010aa275ceae00a75d"
)
REVIEWED_SOURCE_CAPTURE_MASK_SHA256 = (
    "c2f4844a7e6da9306b69984ee75348da246e9775d1a50275cf5f97ebb8aca299"
)
REVIEWED_SOURCE_INITIAL_RNG_SHA256 = (
    "6312c6ffff33419f0b24ac872ab51654ad2ea956cc004fa61f6c95016f7c9389"
)
REVIEWED_SOURCE_POST_TIMING_RNG_SHA256 = (
    "a7b19f81999f0e622e63411e89d195a84ddb4a8919c5f603109f1280e5f3976c"
)
REVIEWED_SOURCE_PRE_TARGET_RNG_SHA256 = (
    "3903731b09c872d94364baa79c33a2794a0d7b129abf900cb5dace10cba1133d"
)
REVIEWED_SOURCE_POST_REPLAY_RNG_SHA256 = (
    "61758aa77b7fe8f0603235093d2b38c9ecc3658344b31c6e2757e15a305f030b"
)
REVIEWED_SOURCE_RAW_LOGITS_SHA256 = (
    "8e0fee1b926d5f9f7f25f02042983e47c7d391e9c439beffe26f134782ed01ae"
)
REVIEWED_SOURCE_PREPARED_MASK_SHA256 = (
    "7411b76017def4ea0c62a52705ba9e3d5c8a665f4769b85358a52cc31c0b2418"
)
REVIEWED_SOURCE_ACTIVE_MASK_SHA256 = (
    "6c2d8cf911b5d720b855b872ac580f64a9af211e7a56c1e0441ce603da9889b5"
)
REVIEWED_SOURCE_SELF_CACHE_SHA256 = (
    "77ec075467a970497a99cd6cbb0b266269fe51db53674480272b8058aa74b466"
)
REVIEWED_SOURCE_CROSS_CACHE_SHA256 = (
    "0b4307698b3556f9881590cb6ce9ff9a9a24330dc33883b1b994006702920ed4"
)

SOURCE_JOB_ID = "49543717"
SOURCE_COMMIT = "a709b86c37484c4ef9754d582e4506939273bf67"
SOURCE_PROFILE_SHA256 = "05e49f8799fb9611c2d5974877b83fc57940edafd45940ef7ed7b504ab12145f"
SOURCE_AUDIO_SHA256 = "1e692d9ffbc2a3051219ac19e96f2c32b428fa0e4fc36278022866e19e6e6183"
SOURCE_AUDIO_SIZE_BYTES = 4382685
SOURCE_AUDIO_SAMPLES = 3310237
SOURCE_PROFILE_SIZE_BYTES = 90254
SOURCE_RESULT_SHA256 = "007a98e050f045669ac0fc5bcaa0099baf32173e2020f9b23dcdccaf4983820e"
SOURCE_PROFILE_LABEL = "main_generation"
SOURCE_SEQUENCE_INDEX = 9
SOURCE_PROMPT_LENGTH = 478
SOURCE_GENERATED_TOKENS = 327
SOURCE_PREFIX_REPLAY_TOKENS = 42
CAPTURE_PROMPT_LENGTH = SOURCE_PROMPT_LENGTH + SOURCE_PREFIX_REPLAY_TOKENS
ACTIVE_PREFIX_LENGTH = 576
PHASE_A_HORIZON = 8
MODEL_MAX_TARGET_POSITIONS = 2560

TIMING_TRANSCRIPT_TOKENS = 132
PRE_TARGET_MAIN_TRANSCRIPT_TOKENS = 1042
PRE_TARGET_RNG_DRAWS = TIMING_TRANSCRIPT_TOKENS + PRE_TARGET_MAIN_TRANSCRIPT_TOKENS
BASE_TIMING_TOKENIZER_SCOPE = "base_v32_auto_select_gamemode_model_false"
MAIN_TOKENIZER_SCOPE = "gamemode_0_auto_selected"
BASE_TIMING_VOCAB_SIZE = 4097
MAIN_VOCAB_SIZE = 4069
BASE_TIMING_BEAT_TOKEN_ID = 4081
BASE_TIMING_MEASURE_TOKEN_ID = 4082
MAIN_BEAT_TOKEN_ID = 4063
MAIN_MEASURE_TOKEN_ID = 4064
TIMING_TRANSCRIPT_SHA256 = "a665eba7d64940434da93e2b4d7d935924109147da29031a6ca4d848619e5ba2"
PRE_TARGET_MAIN_TRANSCRIPT_SHA256 = (
    "55fe58553bc4a32b95c50f6a4277c215bab59430c478c80a0011e486555ca4a3"
)
TARGET_TRANSCRIPT_SHA256 = "a956a0c2b7c955e952bf68c070948f745d62134b22737cab46a5aa0ee4e0dcde"
TARGET_PREFIX_REPLAY_SHA256 = "66960797a93b3f094f36cffadfcc44cbf27ad837eb30cad3d9cdde95a1381f06"
TARGET_PREFIX_REPLAY_TOKEN_IDS = (
    848, 1648, 2236, 2721, 3878, 2977, 3880, 3982, 4056, 2237, 2727,
    4058, 2271, 2635, 4061, 875, 1649, 3879, 3982, 4061, 875, 1649,
    2238, 2786, 3879, 3982, 4062, 887, 1651, 2272, 2693, 3880, 3982,
    4053, 900, 1648, 2273, 2855, 3880, 3982, 4053, 924,
)

PACKED_PREFILL_JOB_ID = "49548273"
PACKED_PREFILL_COMMIT = "8a7517903fedd9deb692aa9b332c663bbb16537d"
PACKED_PREFILL_REPORT_SHA256 = (
    "93397a3cf042669ea507f343c661f1fadf2e88012c290893872b7cfa296569dd"
)
PACKED_PREFILL_BATCH_SIZE = 8
TEN_REQUEST_WINDOW_COUNT = 100
TARGET_TPS = 500.0

# Accepted five-song decode-token histogram.  In the duplicated-ten workload,
# each original token becomes exactly one B2 pair step, so these counts sum to
# TEN_REQUEST_DECODE_PAIR_STEPS.
PRODUCTION_PAIR_STEPS_BY_BUCKET = {
    64: 21,
    128: 305,
    192: 304,
    256: 316,
    320: 314,
    384: 338,
    448: 612,
    512: 884,
    576: 1099,
    640: 833,
    704: 358,
    768: 192,
    832: 164,
    896: 75,
    960: 64,
    1024: 26,
}


@dataclass(frozen=True)
class QueueProfileSource:
    song_id: str
    run_index: int
    sha256: str
    main_records_sha256: str


QUEUE_PROFILE_SOURCES = (
    QueueProfileSource(
        "lambada", 5, SOURCE_PROFILE_SHA256,
        "469efd95dfdcb8598c669cc39a487aa73619a9af3a19d8c767aedf4bac58d50d",
    ),
    QueueProfileSource(
        "pegasus", 6,
        "30fc0fedbed38cca0d59f02a870852c6b567e1803e1d227559a3c49fb6c2746c",
        "04a98402937fc968f1f5c33867cacc076dbc4b98f8adbe5e6d58859a37ef63d0",
    ),
    QueueProfileSource(
        "ela-ke-leitada", 7,
        "704ac699d762b4580c4f8c3f07311a384b4071f9c5e990bfe7f3a64dee2223a7",
        "b31ba9a0ba5c5691cc054540c451b439291d42a9008549545502e9417221fe03",
    ),
    QueueProfileSource(
        "salvalai", 8,
        "387264c12874f1bbee9a7a635e489a809b2ac0360c3fec38b8867292f8c0f912",
        "61ed1a12754821c786b3f12541a3382267a738f9300db26db5e4eb4718ee8208",
    ),
    QueueProfileSource(
        "nube-negra", 9,
        "139385064872a347093d3609adf2ad4c9b24858e9d555a6eaea275e03aaf8059",
        "3c7a5497cd72ba3ce68af42f5e731d8196ebecad68e91a25b2c5d1f9b7f2c4cf",
    ),
)

EXPECTED_FRAME_TIMES_MS = (62227, 63864, 65502, 67140, 68777, 70415, 72052, 73690, 75327, 76965)
EXPECTED_TIMING_PROMPT_TOKENS = (16, 48, 48, 50, 48, 48, 48, 50, 48, 48)
EXPECTED_TIMING_GENERATED_TOKENS = (39, 7, 9, 7, 7, 7, 9, 7, 7, 33)
EXPECTED_MAIN_PROMPT_TOKENS = (48, 400, 441, 446, 450, 471, 472, 475, 480, 478)
EXPECTED_MAIN_GENERATED_TOKENS = (409, 89, 70, 73, 82, 79, 92, 77, 71, 327)


@dataclass(frozen=True)
class AcceptedProfileContract:
    profile_sha256: str = SOURCE_PROFILE_SHA256
    source_commit: str = SOURCE_COMMIT
    audio_sha256: str = SOURCE_AUDIO_SHA256
    result_sha256: str = SOURCE_RESULT_SHA256
    timing_transcript_sha256: str = TIMING_TRANSCRIPT_SHA256
    pre_target_main_transcript_sha256: str = PRE_TARGET_MAIN_TRANSCRIPT_SHA256
    target_transcript_sha256: str = TARGET_TRANSCRIPT_SHA256
    target_prefix_replay_sha256: str = TARGET_PREFIX_REPLAY_SHA256
    frame_times_ms: tuple[int, ...] = EXPECTED_FRAME_TIMES_MS
    timing_prompt_tokens: tuple[int, ...] = EXPECTED_TIMING_PROMPT_TOKENS
    timing_generated_tokens: tuple[int, ...] = EXPECTED_TIMING_GENERATED_TOKENS
    main_prompt_tokens: tuple[int, ...] = EXPECTED_MAIN_PROMPT_TOKENS
    main_generated_tokens: tuple[int, ...] = EXPECTED_MAIN_GENERATED_TOKENS


def canonical_json_sha256(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def token_ids_sha256(values: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _generation_records(
        profile: Mapping[str, Any],
        *,
        label: str,
) -> list[Mapping[str, Any]]:
    generation = profile.get("generation")
    if (
            not isinstance(generation, Sequence)
            or isinstance(generation, (str, bytes))
    ):
        raise ValueError("accepted source profile generation ledger is malformed.")
    records = [
        record for record in generation
        if isinstance(record, Mapping) and record.get("profile_label") == label
    ]
    if len(records) != 10:
        raise ValueError(f"accepted source profile requires ten {label} records.")
    return records


def validate_accepted_profile(
        profile: Mapping[str, Any],
        *,
        contract: AcceptedProfileContract = AcceptedProfileContract(),
) -> dict[str, Any]:
    """Validate the exact accepted transcript used to reconstruct Lambada seq9."""

    if profile.get("schema_version") != 1:
        raise ValueError("accepted source profile schema changed.")
    metadata = profile.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("accepted source profile metadata is missing.")
    expected_metadata = {
        "git_commit": contract.source_commit,
        "result_file_sha256": contract.result_sha256,
        "model_path": "OliBomby/Mapperatorinator-v32",
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "start_time": 71000,
        "end_time": 86000,
        "song_id": "lambada",
        "repeat_index": 1,
        "run_index": 5,
        "seed": 12345,
        "profile_record_token_ids": True,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(f"accepted source profile metadata field {field} changed.")

    timing = _generation_records(profile, label="timing_context")
    main = _generation_records(profile, label="main_generation")
    for label, records, context_type, prompt_counts, generated_counts in (
            (
                "timing_context",
                timing,
                "timing",
                contract.timing_prompt_tokens,
                contract.timing_generated_tokens,
            ),
            (
                "main_generation",
                main,
                "map",
                contract.main_prompt_tokens,
                contract.main_generated_tokens,
            ),
    ):
        for index, record in enumerate(records):
            expected_shape = {
                "profile_label": label,
                "context_type": context_type,
                "mode": "sequential",
                "sequence_index": index,
                "frame_time_ms": contract.frame_times_ms[index],
                "prompt_tokens": prompt_counts[index],
                "generated_tokens": generated_counts[index],
                "trim_lookback": False,
                "trim_lookahead": index != 9,
            }
            for field, expected in expected_shape.items():
                if record.get(field) != expected:
                    raise ValueError(
                        f"accepted {label} sequence {index} field {field} changed."
                    )
            token_ids = record.get("generated_token_ids")
            if (
                    not isinstance(token_ids, Sequence)
                    or isinstance(token_ids, (str, bytes))
                    or len(token_ids) != generated_counts[index]
                    or any(
                        not isinstance(token, int) or isinstance(token, bool) or token < 0
                        for token in token_ids
                    )
            ):
                raise ValueError(
                    f"accepted {label} sequence {index} token transcript is malformed."
                )

    timing_ids = [int(token) for record in timing for token in record["generated_token_ids"]]
    pre_target_main_ids = [
        int(token) for record in main[:SOURCE_SEQUENCE_INDEX]
        for token in record["generated_token_ids"]
    ]
    target_ids = [int(token) for token in main[SOURCE_SEQUENCE_INDEX]["generated_token_ids"]]
    observed_hashes = {
        "timing_transcript_sha256": token_ids_sha256(timing_ids),
        "pre_target_main_transcript_sha256": token_ids_sha256(pre_target_main_ids),
        "target_transcript_sha256": token_ids_sha256(target_ids),
        "target_prefix_replay_sha256": token_ids_sha256(
            target_ids[:SOURCE_PREFIX_REPLAY_TOKENS]
        ),
    }
    expected_hashes = {
        "timing_transcript_sha256": contract.timing_transcript_sha256,
        "pre_target_main_transcript_sha256": contract.pre_target_main_transcript_sha256,
        "target_transcript_sha256": contract.target_transcript_sha256,
        "target_prefix_replay_sha256": contract.target_prefix_replay_sha256,
    }
    if observed_hashes != expected_hashes:
        raise ValueError("accepted source profile token transcript hashes changed.")
    return {
        "timing_records": timing,
        "main_records": main,
        "timing_token_ids": timing_ids,
        "pre_target_main_token_ids": pre_target_main_ids,
        "target_token_ids": target_ids,
        "transcript_hashes": observed_hashes,
    }


def load_accepted_profile(
        path: Path,
        *,
        contract: AcceptedProfileContract = AcceptedProfileContract(),
) -> tuple[dict[str, Any], dict[str, Any]]:
    observed_sha256 = _file_sha256(path)
    if observed_sha256 != contract.profile_sha256:
        raise ValueError("accepted source profile file SHA-256 changed.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("accepted source profile must contain one JSON object.")
    evidence = validate_accepted_profile(payload, contract=contract)
    return payload, evidence


def _finite_positive(value: Any, name: str) -> float:
    if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive.")
    return float(value)


def _validate_queue_profile_inputs(
        profile_inputs: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if len(profile_inputs) != len(QUEUE_PROFILE_SOURCES):
        raise ValueError("weighted ceiling requires all five accepted repeat01 profiles.")
    values: list[Mapping[str, Any]] = []
    for observed, source in zip(profile_inputs, QUEUE_PROFILE_SOURCES, strict=True):
        if not isinstance(observed, Mapping):
            raise ValueError("weighted queue profile evidence is malformed.")
        if {
            "song_id": observed.get("song_id"),
            "run_index": observed.get("run_index"),
            "profile_sha256": observed.get("profile_sha256"),
            "main_records_sha256": observed.get("main_records_sha256"),
        } != {
            "song_id": source.song_id,
            "run_index": source.run_index,
            "profile_sha256": source.sha256,
            "main_records_sha256": source.main_records_sha256,
        }:
            raise ValueError("weighted queue profile source order/SHA changed.")
        records = observed.get("main_records")
        if (
                not isinstance(records, Sequence)
                or isinstance(records, (str, bytes))
                or len(records) != 10
        ):
            raise ValueError(f"weighted queue profile {source.song_id} needs ten main records.")
        if canonical_json_sha256(records) != source.main_records_sha256:
            raise ValueError(f"weighted queue profile records changed for {source.song_id}.")
        for sequence_index, record in enumerate(records):
            if (
                    not isinstance(record, Mapping)
                    or record.get("sequence_index") != sequence_index
                    or not isinstance(record.get("prompt_tokens"), int)
                    or isinstance(record.get("prompt_tokens"), bool)
                    or int(record["prompt_tokens"]) <= 0
                    or not isinstance(record.get("generated_tokens"), int)
                    or isinstance(record.get("generated_tokens"), bool)
                    or int(record["generated_tokens"]) <= 0
            ):
                raise ValueError(
                    f"weighted queue {source.song_id} main record {sequence_index} is malformed."
                )
            _finite_positive(
                record.get("model_elapsed_seconds"),
                f"weighted queue {source.song_id} main record {sequence_index} model time",
            )
        values.append(observed)
    return values


def _validate_setup_input(setup: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = {
        "job_id": PACKED_PREFILL_JOB_ID,
        "commit": PACKED_PREFILL_COMMIT,
        "report_sha256": PACKED_PREFILL_REPORT_SHA256,
        "batch_size": PACKED_PREFILL_BATCH_SIZE,
        "linearized_window_count": TEN_REQUEST_WINDOW_COUNT,
        "serial_b1_prefill_wall_seconds_for_batch8": 0.34391318541020155,
        "packer_wall_seconds_for_batch8": 0.04675073456019163,
        "provenance_scope": "optimistic linearized synthetic setup charge",
    }
    if setup != expected:
        raise ValueError("weighted setup source differs from packed-prefill job 49548273.")
    return setup


def _active_prefix_bucket(length: int) -> int:
    return min(1024, ((length + 63) // 64) * 64)


def _ideal_overlap_wall(
        chains: Sequence[Sequence[float]],
        *,
        lanes: int,
) -> float:
    if lanes <= 0:
        raise ValueError("ideal overlap lane count must be positive.")
    total = 0.0
    max_length = max((len(chain) for chain in chains), default=0)
    for ordinal in range(max_length):
        costs = sorted(
            (float(chain[ordinal]) for chain in chains if ordinal < len(chain)),
            reverse=True,
        )
        total += sum(costs[index] for index in range(0, len(costs), lanes))
    return total


def _derive_model_free_ceiling_report(
        profile_inputs: Sequence[Mapping[str, Any]],
        setup_input: Mapping[str, Any],
) -> dict[str, Any]:
    profiles = _validate_queue_profile_inputs(profile_inputs)
    setup = _validate_setup_input(setup_input)
    chains: list[list[float]] = []
    histogram: dict[int, int] = {}
    five_main_tokens = 0
    for profile in profiles:
        chain: list[float] = []
        for record in profile["main_records"]:
            prompt_tokens = int(record["prompt_tokens"])
            generated_tokens = int(record["generated_tokens"])
            five_main_tokens += generated_tokens
            decode_tokens = generated_tokens - 1
            if decode_tokens > 0:
                per_token_cost = float(record["model_elapsed_seconds"]) / decode_tokens
                chain.extend([per_token_cost] * decode_tokens)
                for offset in range(1, generated_tokens):
                    bucket = _active_prefix_bucket(prompt_tokens + offset)
                    histogram[bucket] = histogram.get(bucket, 0) + 1
        chains.append(chain)

    decode_pair_steps = sum(histogram.values())
    k2_wall = _ideal_overlap_wall(chains, lanes=2)
    k3_wall = _ideal_overlap_wall(chains, lanes=3)
    k2_tps = five_main_tokens / k2_wall
    k3_tps = five_main_tokens / k3_wall
    setup_seconds = (
        float(setup["serial_b1_prefill_wall_seconds_for_batch8"])
        + float(setup["packer_wall_seconds_for_batch8"])
    ) / int(setup["batch_size"]) * int(setup["linearized_window_count"])
    ten_main_tokens = five_main_tokens * 2
    target_total_wall = ten_main_tokens / TARGET_TPS
    decode_budget = target_total_wall - setup_seconds
    if decode_budget <= 0.0:
        raise ValueError("weighted setup consumes the complete ten-request target wall.")
    required_pair_seconds = decode_budget / decode_pair_steps
    required_weighted_tps = 2.0 / required_pair_seconds
    return {
        "schema_version": WEIGHTED_CEILING_SCHEMA_VERSION,
        "gate": WEIGHTED_CEILING_GATE,
        "source": {
            "job_id": SOURCE_JOB_ID,
            "commit": SOURCE_COMMIT,
            "workload": "accepted exact repeat01 five-song smoke15 profiles",
            "profile_inputs": [dict(profile) for profile in profiles],
            "setup_input": dict(setup),
        },
        "five_request_ideal_k2": {
            "main_tokens": five_main_tokens,
            "decode_chain_tokens": decode_pair_steps,
            "ideal_wall_seconds": k2_wall,
            "ideal_scheduler_wall_tokens_per_second": k2_tps,
            "target_tokens_per_second": TARGET_TPS,
            "target_closed_in_accepted_profile_cost_model": k2_tps < TARGET_TPS,
            "algorithm": (
                "charge each window model_elapsed_seconds uniformly over generated_tokens-1; "
                "for each dependency-chain token ordinal sort active costs descending, group "
                "adjacent K=2, and sum each group maximum"
            ),
            "optimism": (
                "coordination, tensor-shape compatibility, setup, and slot-refill costs are free"
            ),
            "interpretation": "K2 is closed only under this accepted-profile cost model",
        },
        "five_request_ideal_k3": {
            "main_tokens": five_main_tokens,
            "decode_chain_tokens": decode_pair_steps,
            "ideal_wall_seconds": k3_wall,
            "ideal_scheduler_wall_tokens_per_second": k3_tps,
            "target_tokens_per_second": TARGET_TPS,
            "target_open_in_accepted_profile_cost_model": k3_tps > TARGET_TPS,
            "algorithm": (
                "same synchronous token-ordinal perfect-overlap model as K2 with K=3"
            ),
            "interpretation": "K3 remains open only under this accepted-profile cost model",
        },
        "duplicated_ten_request_b2": {
            "main_tokens": ten_main_tokens,
            "decode_pair_steps": decode_pair_steps,
            "pair_steps_by_active_prefix_bucket": {
                str(bucket): count
                for bucket, count in sorted(histogram.items())
            },
            "optimistic_linearized_setup_seconds": setup_seconds,
            "target_total_wall_seconds": target_total_wall,
            "decode_wall_budget_seconds": decode_budget,
            "required_seconds_per_pair": required_pair_seconds,
            "required_weighted_complete_tokens_per_second": required_weighted_tps,
            "target_tokens_per_second": TARGET_TPS,
        },
        "decision": {
            "gpu_scout_scope": "B2 weighted active-prefix Phase A only",
            "first_bucket": 576,
            "horizon": PHASE_A_HORIZON,
            "scheduler_or_runtime_wiring_authorized": False,
            "phase_b_authorized": False,
            "v32_reference_behavior_unchanged": True,
            "runtime_code_touched": False,
        },
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object.")
    return value


def build_model_free_ceiling_report(
        profile_paths: Sequence[Path],
        *,
        packed_prefill_report_path: Path,
) -> dict[str, Any]:
    if len(profile_paths) != len(QUEUE_PROFILE_SOURCES):
        raise ValueError("weighted ceiling requires exactly five --profile paths.")
    profile_inputs: list[dict[str, Any]] = []
    for path, source in zip(profile_paths, QUEUE_PROFILE_SOURCES, strict=True):
        if _file_sha256(path) != source.sha256:
            raise ValueError(f"accepted queue profile SHA changed for {source.song_id}.")
        payload = _load_json_object(path)
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"accepted queue profile {source.song_id} is missing metadata.")
        if (
                metadata.get("git_commit") != SOURCE_COMMIT
                or metadata.get("song_id") != source.song_id
                or metadata.get("run_index") != source.run_index
                or metadata.get("repeat_index") != 1
        ):
            raise ValueError(f"accepted queue profile metadata changed for {source.song_id}.")
        records = _generation_records(payload, label="main_generation")
        normalized_records = [
            {
                "sequence_index": int(record["sequence_index"]),
                "prompt_tokens": int(record["prompt_tokens"]),
                "generated_tokens": int(record["generated_tokens"]),
                "model_elapsed_seconds": float(record["model_elapsed_seconds"]),
            }
            for record in records
        ]
        if canonical_json_sha256(normalized_records) != source.main_records_sha256:
            raise ValueError(f"accepted queue main-record hash changed for {source.song_id}.")
        profile_inputs.append({
            "song_id": source.song_id,
            "run_index": source.run_index,
            "profile_sha256": source.sha256,
            "main_records_sha256": source.main_records_sha256,
            "main_records": normalized_records,
        })

    if _file_sha256(packed_prefill_report_path) != PACKED_PREFILL_REPORT_SHA256:
        raise ValueError("packed-prefill source report SHA-256 changed.")
    packed = _load_json_object(packed_prefill_report_path)
    runtime = packed.get("runtime")
    prefill = packed.get("prefill")
    if (
            packed.get("schema_version") != 1
            or packed.get("batch_size") != PACKED_PREFILL_BATCH_SIZE
            or packed.get("exactness_pass") is not True
            or packed.get("merged_prefill_mode") != "packed_b1"
            or not isinstance(runtime, Mapping)
            or runtime.get("git_commit") != PACKED_PREFILL_COMMIT
            or runtime.get("slurm_job_id") != PACKED_PREFILL_JOB_ID
            or not isinstance(prefill, Mapping)
            or prefill.get("pass") is not True
    ):
        raise ValueError("packed-prefill source report contract changed.")
    setup_input = {
        "job_id": PACKED_PREFILL_JOB_ID,
        "commit": PACKED_PREFILL_COMMIT,
        "report_sha256": PACKED_PREFILL_REPORT_SHA256,
        "batch_size": PACKED_PREFILL_BATCH_SIZE,
        "linearized_window_count": TEN_REQUEST_WINDOW_COUNT,
        "serial_b1_prefill_wall_seconds_for_batch8": float(
            prefill["serial_B1"]["wall_seconds"]
        ),
        "packer_wall_seconds_for_batch8": float(
            prefill["packed_prefill_gate"]["packer"]["wall_seconds"]
        ),
        "provenance_scope": "optimistic linearized synthetic setup charge",
    }
    return _derive_model_free_ceiling_report(profile_inputs, setup_input)


def validate_model_free_ceiling_report(report: Mapping[str, Any]) -> None:
    source = report.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("weighted model-free report is missing source evidence.")
    profile_inputs = source.get("profile_inputs")
    setup_input = source.get("setup_input")
    if (
            not isinstance(profile_inputs, Sequence)
            or isinstance(profile_inputs, (str, bytes))
            or not isinstance(setup_input, Mapping)
    ):
        raise ValueError("weighted model-free report source inputs are malformed.")
    expected = _derive_model_free_ceiling_report(profile_inputs, setup_input)
    if report != expected:
        raise ValueError("weighted model-free ceiling report differs from source-derived arithmetic.")
    five = report["five_request_ideal_k2"]
    if five["target_closed_in_accepted_profile_cost_model"] is not True or not (
            495.98 < float(five["ideal_scheduler_wall_tokens_per_second"]) < 496.0
    ):
        raise ValueError("five-request ideal-K2 closure did not self-validate.")
    k3 = report["five_request_ideal_k3"]
    if k3["target_open_in_accepted_profile_cost_model"] is not True or not (
            640.76 < float(k3["ideal_scheduler_wall_tokens_per_second"]) < 640.78
    ):
        raise ValueError("five-request ideal-K3 branch did not self-validate.")
    ten = report["duplicated_ten_request_b2"]
    if not (
            0.00320689 < float(ten["required_seconds_per_pair"]) < 0.00320690
            and 623.65
            < float(ten["required_weighted_complete_tokens_per_second"])
            < 623.66
    ):
        raise ValueError("ten-request weighted B2 requirement did not self-validate.")


def required_weighted_complete_tps() -> float:
    raise RuntimeError(
        "required weighted TPS is source-derived; read it from a validated ceiling report."
    )


def _validate_sha256(value: Any, *, label: str) -> str:
    if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase hexadecimal SHA-256.")
    return value


def _validate_tensor_descriptor(
        value: Any,
        *,
        label: str,
        expected_dtype: str | None = None,
        expected_shape: Sequence[int] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"sha256", "shape", "stride", "dtype"}:
        raise ValueError(f"{label} tensor descriptor fields are malformed.")
    _validate_sha256(value["sha256"], label=f"{label} hash")
    shape = value["shape"]
    stride = value["stride"]
    if (
            not isinstance(shape, Sequence)
            or isinstance(shape, (str, bytes))
            or not shape
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item <= 0
                for item in shape
            )
            or not isinstance(stride, Sequence)
            or isinstance(stride, (str, bytes))
            or len(stride) != len(shape)
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item < 0
                for item in stride
            )
    ):
        raise ValueError(f"{label} tensor shape/stride is malformed.")
    if expected_shape is not None and list(shape) != list(expected_shape):
        raise ValueError(f"{label} tensor shape changed.")
    dtype = value["dtype"]
    if not isinstance(dtype, str) or not dtype.startswith("torch."):
        raise ValueError(f"{label} tensor dtype is malformed.")
    if expected_dtype is not None and dtype != expected_dtype:
        raise ValueError(f"{label} tensor dtype changed.")
    return value


def validate_context_replay_report(report: Mapping[str, Any]) -> None:
    """Validate the model-free tokenizer/context reconstruction prerequisite."""

    if set(report) != {
        "schema_version",
        "gate",
        "source",
        "tokenizers",
        "timing",
        "main",
        "maintainer_boundary",
    }:
        raise ValueError("weighted context-only report fields changed.")
    if report.get("schema_version") != WEIGHTED_CONTEXT_REPLAY_SCHEMA_VERSION:
        raise ValueError("unsupported weighted context-only schema.")
    if report.get("gate") != WEIGHTED_CONTEXT_REPLAY_GATE:
        raise ValueError("report is not the weighted context-only gate.")
    expected_source = {
        "job_id": SOURCE_JOB_ID,
        "commit": SOURCE_COMMIT,
        "profile_sha256": SOURCE_PROFILE_SHA256,
        "profile_size_bytes": SOURCE_PROFILE_SIZE_BYTES,
        "audio_sha256": SOURCE_AUDIO_SHA256,
        "audio_size_bytes": SOURCE_AUDIO_SIZE_BYTES,
        "audio_samples": SOURCE_AUDIO_SAMPLES,
        "timing_transcript_sha256": TIMING_TRANSCRIPT_SHA256,
        "pre_target_main_transcript_sha256": PRE_TARGET_MAIN_TRANSCRIPT_SHA256,
        "target_transcript_sha256": TARGET_TRANSCRIPT_SHA256,
    }
    if report.get("source") != expected_source:
        raise ValueError("weighted context-only source evidence changed.")
    expected_tokenizers = {
        "timing_scope": BASE_TIMING_TOKENIZER_SCOPE,
        "timing_vocab_size": BASE_TIMING_VOCAB_SIZE,
        "timing_beat_token_id": BASE_TIMING_BEAT_TOKEN_ID,
        "timing_measure_token_id": BASE_TIMING_MEASURE_TOKEN_ID,
        "main_scope": MAIN_TOKENIZER_SCOPE,
        "main_vocab_size": MAIN_VOCAB_SIZE,
        "main_beat_token_id": MAIN_BEAT_TOKEN_ID,
        "main_measure_token_id": MAIN_MEASURE_TOKEN_ID,
    }
    if report.get("tokenizers") != expected_tokenizers:
        raise ValueError("weighted context-only tokenizer scopes changed.")
    timing = report.get("timing")
    expected_timing = {
        "pretrim_event_count": 122,
        "event_count": 60,
        "time_shift_count": 30,
        "beat_count": 22,
        "measure_count": 8,
        "marker_count": 30,
        "marker_min_time_ms": 71237,
        "marker_max_time_ms": 85875,
        "timing_point_count": 1,
    }
    if not isinstance(timing, Mapping):
        raise ValueError("weighted context-only timing evidence is missing.")
    for field, expected in expected_timing.items():
        if timing.get(field) != expected:
            raise ValueError(f"weighted context-only timing field {field} changed.")
    if set(timing) != set(expected_timing):
        raise ValueError("weighted context-only timing fields changed.")
    main = report.get("main")
    if not isinstance(main, Mapping) or set(main) != {
        "prompt_token_counts",
        "target_sequence_index",
        "target_frame_time_ms",
        "target_context_type",
        "target_prompt",
        "target_prompt_attention_mask",
        "target_frames",
        "condition_tensor_keys",
    }:
        raise ValueError("weighted context-only main evidence fields changed.")
    if main["prompt_token_counts"] != list(EXPECTED_MAIN_PROMPT_TOKENS):
        raise ValueError("weighted context-only main prompt counts changed.")
    if (
            main["target_sequence_index"] != SOURCE_SEQUENCE_INDEX
            or main["target_frame_time_ms"] != 76965.0
            or main["target_context_type"] != "map"
            or main["condition_tensor_keys"] != []
    ):
        raise ValueError("weighted context-only target contract changed.")
    prompt_descriptor = _validate_tensor_descriptor(
        main["target_prompt"],
        label="weighted context-only target prompt",
        expected_dtype="torch.int64",
        expected_shape=[1, SOURCE_PROMPT_LENGTH],
    )
    if prompt_descriptor != {
        "sha256": REVIEWED_CONTEXT_PROMPT_SHA256,
        "shape": [1, SOURCE_PROMPT_LENGTH],
        "stride": [SOURCE_PROMPT_LENGTH, 1],
        "dtype": "torch.int64",
    }:
        raise ValueError("weighted context-only target prompt descriptor changed.")
    prompt_mask_descriptor = _validate_tensor_descriptor(
        main["target_prompt_attention_mask"],
        label="weighted context-only target prompt mask",
        expected_dtype="torch.bool",
        expected_shape=[1, SOURCE_PROMPT_LENGTH],
    )
    if prompt_mask_descriptor != {
        "sha256": REVIEWED_CONTEXT_PROMPT_MASK_SHA256,
        "shape": [1, SOURCE_PROMPT_LENGTH],
        "stride": [SOURCE_PROMPT_LENGTH, 1],
        "dtype": "torch.bool",
    }:
        raise ValueError("weighted context-only target prompt mask descriptor changed.")
    frames_descriptor = _validate_tensor_descriptor(
        main["target_frames"],
        label="weighted context-only target frames",
        expected_dtype="torch.float32",
        expected_shape=[1, 262016],
    )
    if frames_descriptor != {
        "sha256": REVIEWED_CONTEXT_FRAMES_SHA256,
        "shape": [1, 262016],
        "stride": [262016, 1],
        "dtype": "torch.float32",
    }:
        raise ValueError("weighted context-only target frames descriptor changed.")
    expected_boundary = {
        "engine": "v32",
        "model_loaded": False,
        "cuda_used": False,
        "gpu_capture_authorized": False,
        "authorization_requires_reviewed_context_sha_commit": True,
    }
    if report.get("maintainer_boundary") != expected_boundary:
        raise ValueError("weighted context-only boundary changed.")


def reviewed_context_replay_report() -> dict[str, Any]:
    """Return the immutable CPU context report reviewed after job 49558528."""

    return {
        "schema_version": WEIGHTED_CONTEXT_REPLAY_SCHEMA_VERSION,
        "gate": WEIGHTED_CONTEXT_REPLAY_GATE,
        "source": {
            "job_id": SOURCE_JOB_ID,
            "commit": SOURCE_COMMIT,
            "profile_sha256": SOURCE_PROFILE_SHA256,
            "profile_size_bytes": SOURCE_PROFILE_SIZE_BYTES,
            "audio_sha256": SOURCE_AUDIO_SHA256,
            "audio_size_bytes": SOURCE_AUDIO_SIZE_BYTES,
            "audio_samples": SOURCE_AUDIO_SAMPLES,
            "timing_transcript_sha256": TIMING_TRANSCRIPT_SHA256,
            "pre_target_main_transcript_sha256": PRE_TARGET_MAIN_TRANSCRIPT_SHA256,
            "target_transcript_sha256": TARGET_TRANSCRIPT_SHA256,
        },
        "tokenizers": {
            "timing_scope": BASE_TIMING_TOKENIZER_SCOPE,
            "timing_vocab_size": BASE_TIMING_VOCAB_SIZE,
            "timing_beat_token_id": BASE_TIMING_BEAT_TOKEN_ID,
            "timing_measure_token_id": BASE_TIMING_MEASURE_TOKEN_ID,
            "main_scope": MAIN_TOKENIZER_SCOPE,
            "main_vocab_size": MAIN_VOCAB_SIZE,
            "main_beat_token_id": MAIN_BEAT_TOKEN_ID,
            "main_measure_token_id": MAIN_MEASURE_TOKEN_ID,
        },
        "timing": {
            "pretrim_event_count": 122,
            "event_count": 60,
            "time_shift_count": 30,
            "beat_count": 22,
            "measure_count": 8,
            "marker_count": 30,
            "marker_min_time_ms": 71237,
            "marker_max_time_ms": 85875,
            "timing_point_count": 1,
        },
        "main": {
            "prompt_token_counts": list(EXPECTED_MAIN_PROMPT_TOKENS),
            "target_sequence_index": SOURCE_SEQUENCE_INDEX,
            "target_frame_time_ms": 76965.0,
            "target_context_type": "map",
            "target_prompt": {
                "sha256": REVIEWED_CONTEXT_PROMPT_SHA256,
                "shape": [1, SOURCE_PROMPT_LENGTH],
                "stride": [SOURCE_PROMPT_LENGTH, 1],
                "dtype": "torch.int64",
            },
            "target_prompt_attention_mask": {
                "sha256": REVIEWED_CONTEXT_PROMPT_MASK_SHA256,
                "shape": [1, SOURCE_PROMPT_LENGTH],
                "stride": [SOURCE_PROMPT_LENGTH, 1],
                "dtype": "torch.bool",
            },
            "target_frames": {
                "sha256": REVIEWED_CONTEXT_FRAMES_SHA256,
                "shape": [1, 262016],
                "stride": [262016, 1],
                "dtype": "torch.float32",
            },
            "condition_tensor_keys": [],
        },
        "maintainer_boundary": {
            "engine": "v32",
            "model_loaded": False,
            "cuda_used": False,
            "gpu_capture_authorized": False,
            "authorization_requires_reviewed_context_sha_commit": True,
        },
    }


def validate_reviewed_context_replay_pin() -> str:
    """Validate the immutable in-code CPU report and its canonical digest."""

    digest = REVIEWED_CONTEXT_REPLAY_REPORT_SHA256
    if digest is None:
        raise RuntimeError(
            "weighted full GPU source capture is blocked until a separate "
            "post-context-report commit pins REVIEWED_CONTEXT_REPLAY_REPORT_SHA256."
        )
    digest = _validate_sha256(digest, label="reviewed weighted context-only report")
    expected_report = reviewed_context_replay_report()
    validate_context_replay_report(expected_report)
    expected_digest = canonical_json_sha256(expected_report)
    if expected_digest != digest:
        raise RuntimeError(
            "in-code reviewed weighted context report does not match its pinned "
            f"canonical digest: expected={digest}, observed={expected_digest}."
        )
    return digest


def require_reviewed_context_replay(report: Mapping[str, Any]) -> str:
    """Require a fresh local replay to equal the reviewed CPU contract."""

    digest = validate_reviewed_context_replay_pin()
    expected_report = reviewed_context_replay_report()
    validate_context_replay_report(report)
    observed_digest = canonical_json_sha256(report)
    if report != expected_report or observed_digest != digest:
        raise RuntimeError(
            "fresh weighted context replay does not match the reviewed CPU "
            f"contract: expected={digest}, observed={observed_digest}."
        )
    return digest


def validate_reviewed_context_evidence(evidence: Mapping[str, Any]) -> None:
    """Match live model-side context inputs to the reviewed CPU evidence."""

    fields = ("source", "tokenizers", "timing", "main")
    if set(evidence) != set(fields):
        raise ValueError("weighted live context evidence fields changed.")
    expected = reviewed_context_replay_report()
    for field in fields:
        if evidence.get(field) != expected[field]:
            raise ValueError(
                f"weighted live context evidence field {field} differs from the "
                "reviewed CPU report."
            )


def validate_source_contract(contract: Mapping[str, Any]) -> None:
    """Validate a separately reviewed real-prefix capture artifact.

    The capture job first created the tensor evidence.  The separate
    post-capture review now pins the critical tensor hashes here and the full
    canonical contract plus raw report file in the strict loader.
    """

    if set(contract) != {
        "schema_version",
        "gate",
        "source",
        "transcript",
        "reconstruction",
        "accepted_prefix_replay",
        "maintainer_boundary",
    }:
        raise ValueError("weighted source contract top-level fields changed.")
    if contract.get("schema_version") != WEIGHTED_SOURCE_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported weighted source-contract schema.")
    if contract.get("gate") != WEIGHTED_SOURCE_CONTRACT_GATE:
        raise ValueError("source contract is not Lambada repeat01 seq9.")
    expected_source = {
        "job_id": SOURCE_JOB_ID,
        "commit": SOURCE_COMMIT,
        "profile_sha256": SOURCE_PROFILE_SHA256,
        "profile_size_bytes": SOURCE_PROFILE_SIZE_BYTES,
        "audio_sha256": SOURCE_AUDIO_SHA256,
        "audio_size_bytes": SOURCE_AUDIO_SIZE_BYTES,
        "audio_samples": SOURCE_AUDIO_SAMPLES,
        "result_sha256": SOURCE_RESULT_SHA256,
        "profile_label": SOURCE_PROFILE_LABEL,
        "sequence_index": SOURCE_SEQUENCE_INDEX,
        "source_prompt_length": SOURCE_PROMPT_LENGTH,
        "source_generated_tokens": SOURCE_GENERATED_TOKENS,
        "model_snapshot_revision": MODEL_SNAPSHOT_REVISION,
        "model_config_sha256": MODEL_CONFIG_SHA256,
    }
    if contract.get("source") != expected_source:
        raise ValueError("weighted source evidence differs from the accepted profile.")
    transcript = contract.get("transcript")
    expected_transcript = {
        "timing_tokens": TIMING_TRANSCRIPT_TOKENS,
        "timing_sha256": TIMING_TRANSCRIPT_SHA256,
        "pre_target_main_tokens": PRE_TARGET_MAIN_TRANSCRIPT_TOKENS,
        "pre_target_main_sha256": PRE_TARGET_MAIN_TRANSCRIPT_SHA256,
        "pre_target_rng_draws": PRE_TARGET_RNG_DRAWS,
        "target_tokens": SOURCE_GENERATED_TOKENS,
        "target_sha256": TARGET_TRANSCRIPT_SHA256,
        "target_prefix_replay_tokens": SOURCE_PREFIX_REPLAY_TOKENS,
        "target_prefix_replay_sha256": TARGET_PREFIX_REPLAY_SHA256,
    }
    if transcript != expected_transcript:
        raise ValueError("weighted source transcript differs from accepted token IDs.")
    reconstruction = contract.get("reconstruction")
    if not isinstance(reconstruction, Mapping):
        raise ValueError("weighted source contract is missing reconstruction hashes.")
    if set(reconstruction) != {
        "source_prompt_length",
        "capture_prompt_length",
        "active_prefix_length",
        "phase_a_horizon",
        "prefix_build",
        "prior_context_tokens_source",
        "target_prefix_forced_tokens",
        "dummy_rng_advance_only_before_target",
        "timing_tokenizer_scope",
        "main_tokenizer_scope",
        "timing_vocab_size",
        "main_vocab_size",
        "timing_beat_token_id",
        "timing_measure_token_id",
        "main_beat_token_id",
        "main_measure_token_id",
        "timing_pretrim_event_count",
        "timing_event_count",
        "timing_time_shift_count",
        "timing_beat_count",
        "timing_measure_count",
        "timing_marker_count",
        "timing_marker_min_time_ms",
        "timing_marker_max_time_ms",
        "timing_point_count",
        "model_max_target_positions",
        "self_cache_max_length",
        "rng_seed",
        "rng_advance_method",
        "timing_rng_advance_draw_calls",
        "main_rng_advance_draw_calls",
        "cache_populated_through_position",
        "prepared_cache_position",
        "prepared_decoder_token_id",
        "next_output_position",
        "processor_calls",
        "model_decode_calls_after_prefill",
        "self_cache_position518_populated",
        "self_cache_position519_zero",
        "prepared_attention_mask_allowed_positions",
        "prepared_attention_mask_masked_positions",
        "active_attention_mask_allowed_positions",
        "active_attention_mask_masked_positions",
        "prepared_attention_mask_values_exact",
        "active_attention_mask_values_exact",
        "source_prompt",
        "source_prompt_attention_mask",
        "frames",
        "condition_tensors",
        "capture_prompt",
        "capture_prompt_attention_mask",
        "initial_rng_state",
        "post_timing_rng_state",
        "pre_target_rng_state",
        "post_replay_rng_state",
        "timing_rng_advance_probability",
        "main_rng_advance_probability",
        "pre_last_sample_raw_logits",
        "prepared_static_inputs",
        "prospective_active_attention_mask",
        "pre_next_forward_cache",
    }:
        raise ValueError("weighted reconstruction fields changed.")
    expected_scalars = {
        "source_prompt_length": SOURCE_PROMPT_LENGTH,
        "capture_prompt_length": CAPTURE_PROMPT_LENGTH,
        "active_prefix_length": ACTIVE_PREFIX_LENGTH,
        "phase_a_horizon": PHASE_A_HORIZON,
        "prefix_build": "DecodeSession prompt478 plus exact sampled replay42",
        "prior_context_tokens_source": "accepted_profile",
        "target_prefix_forced_tokens": False,
        "dummy_rng_advance_only_before_target": True,
        "timing_tokenizer_scope": BASE_TIMING_TOKENIZER_SCOPE,
        "main_tokenizer_scope": MAIN_TOKENIZER_SCOPE,
        "timing_vocab_size": BASE_TIMING_VOCAB_SIZE,
        "main_vocab_size": MAIN_VOCAB_SIZE,
        "timing_beat_token_id": BASE_TIMING_BEAT_TOKEN_ID,
        "timing_measure_token_id": BASE_TIMING_MEASURE_TOKEN_ID,
        "main_beat_token_id": MAIN_BEAT_TOKEN_ID,
        "main_measure_token_id": MAIN_MEASURE_TOKEN_ID,
        "timing_pretrim_event_count": 122,
        "timing_event_count": 60,
        "timing_time_shift_count": 30,
        "timing_beat_count": 22,
        "timing_measure_count": 8,
        "timing_marker_count": 30,
        "timing_marker_min_time_ms": 71237,
        "timing_marker_max_time_ms": 85875,
        "timing_point_count": 1,
        "model_max_target_positions": MODEL_MAX_TARGET_POSITIONS,
        "self_cache_max_length": MODEL_MAX_TARGET_POSITIONS,
        "rng_seed": 12345,
        "rng_advance_method": (
            "one CUDA multinomial(num_samples=1) per accepted prior generated token"
        ),
        "timing_rng_advance_draw_calls": TIMING_TRANSCRIPT_TOKENS,
        "main_rng_advance_draw_calls": PRE_TARGET_MAIN_TRANSCRIPT_TOKENS,
        "cache_populated_through_position": 518,
        "prepared_cache_position": 519,
        "prepared_decoder_token_id": TARGET_PREFIX_REPLAY_TOKEN_IDS[-1],
        "next_output_position": SOURCE_PREFIX_REPLAY_TOKENS,
        "processor_calls": SOURCE_PREFIX_REPLAY_TOKENS,
        "model_decode_calls_after_prefill": SOURCE_PREFIX_REPLAY_TOKENS - 1,
        "self_cache_position518_populated": True,
        "self_cache_position519_zero": True,
        "prepared_attention_mask_allowed_positions": CAPTURE_PROMPT_LENGTH,
        "prepared_attention_mask_masked_positions": (
            MODEL_MAX_TARGET_POSITIONS - CAPTURE_PROMPT_LENGTH
        ),
        "active_attention_mask_allowed_positions": CAPTURE_PROMPT_LENGTH,
        "active_attention_mask_masked_positions": (
            ACTIVE_PREFIX_LENGTH - CAPTURE_PROMPT_LENGTH
        ),
        "prepared_attention_mask_values_exact": True,
        "active_attention_mask_values_exact": True,
    }
    for field, expected in expected_scalars.items():
        if reconstruction.get(field) != expected:
            raise ValueError(f"weighted reconstruction field {field} changed.")
    _validate_tensor_descriptor(
        reconstruction.get("source_prompt"), label="weighted source prompt",
        expected_dtype="torch.int64", expected_shape=[1, SOURCE_PROMPT_LENGTH],
    )
    _validate_tensor_descriptor(
        reconstruction.get("source_prompt_attention_mask"),
        label="weighted source prompt attention mask",
        expected_dtype="torch.bool", expected_shape=[1, SOURCE_PROMPT_LENGTH],
    )
    _validate_tensor_descriptor(
        reconstruction.get("frames"), label="weighted source frames",
        expected_dtype="torch.float32",
    )
    _validate_tensor_descriptor(
        reconstruction.get("capture_prompt"), label="weighted capture prompt",
        expected_dtype="torch.int64", expected_shape=[1, CAPTURE_PROMPT_LENGTH],
    )
    _validate_tensor_descriptor(
        reconstruction.get("capture_prompt_attention_mask"),
        label="weighted capture prompt attention mask",
        expected_dtype="torch.bool", expected_shape=[1, CAPTURE_PROMPT_LENGTH],
    )
    _validate_tensor_descriptor(
        reconstruction.get("initial_rng_state"), label="weighted initial RNG state",
        expected_dtype="torch.uint8",
    )
    _validate_tensor_descriptor(
        reconstruction.get("post_timing_rng_state"),
        label="weighted post-timing RNG state",
        expected_dtype="torch.uint8",
    )
    _validate_tensor_descriptor(
        reconstruction.get("pre_target_rng_state"), label="weighted pre-target RNG state",
        expected_dtype="torch.uint8",
    )
    _validate_tensor_descriptor(
        reconstruction.get("post_replay_rng_state"), label="weighted post-replay RNG state",
        expected_dtype="torch.uint8",
    )
    _validate_tensor_descriptor(
        reconstruction.get("pre_last_sample_raw_logits"),
        label="weighted pre-last-sample raw logits",
        expected_dtype="torch.float32",
        expected_shape=[1, 4069],
    )
    _validate_tensor_descriptor(
        reconstruction.get("timing_rng_advance_probability"),
        label="weighted timing RNG advance probability",
        expected_dtype="torch.float32",
        expected_shape=[1, BASE_TIMING_VOCAB_SIZE],
    )
    _validate_tensor_descriptor(
        reconstruction.get("main_rng_advance_probability"),
        label="weighted main RNG advance probability",
        expected_dtype="torch.float32",
        expected_shape=[1, MAIN_VOCAB_SIZE],
    )
    reviewed_descriptor_hashes = {
        "source_prompt": REVIEWED_CONTEXT_PROMPT_SHA256,
        "source_prompt_attention_mask": REVIEWED_CONTEXT_PROMPT_MASK_SHA256,
        "frames": REVIEWED_CONTEXT_FRAMES_SHA256,
        "capture_prompt": REVIEWED_SOURCE_CAPTURE_PROMPT_SHA256,
        "capture_prompt_attention_mask": REVIEWED_SOURCE_CAPTURE_MASK_SHA256,
        "initial_rng_state": REVIEWED_SOURCE_INITIAL_RNG_SHA256,
        "post_timing_rng_state": REVIEWED_SOURCE_POST_TIMING_RNG_SHA256,
        "pre_target_rng_state": REVIEWED_SOURCE_PRE_TARGET_RNG_SHA256,
        "post_replay_rng_state": REVIEWED_SOURCE_POST_REPLAY_RNG_SHA256,
        "pre_last_sample_raw_logits": REVIEWED_SOURCE_RAW_LOGITS_SHA256,
    }
    for field, expected_sha256 in reviewed_descriptor_hashes.items():
        descriptor = reconstruction.get(field)
        if not isinstance(descriptor, Mapping) or descriptor.get("sha256") != expected_sha256:
            raise ValueError(f"weighted reviewed descriptor {field} hash changed.")
    if reconstruction.get("condition_tensors") != {}:
        raise ValueError("weighted Lambada source must have no condition tensor keys.")
    prepared = reconstruction.get("prepared_static_inputs")
    if not isinstance(prepared, Mapping) or set(prepared) != {
        "cache_position", "decoder_attention_mask", "decoder_input_ids", "decoder_position_ids"
    }:
        raise ValueError("weighted prepared static input keys changed.")
    prepared_contracts = {
        "cache_position": ("torch.int64", [1]),
        "decoder_attention_mask": (
            "torch.float32",
            [1, 1, 1, MODEL_MAX_TARGET_POSITIONS],
        ),
        "decoder_input_ids": ("torch.int64", [1, 1]),
        "decoder_position_ids": ("torch.int64", [1, 1]),
    }
    for key, (dtype, shape) in prepared_contracts.items():
        _validate_tensor_descriptor(
            prepared[key],
            label=f"weighted prepared static input {key}",
            expected_dtype=dtype,
            expected_shape=shape,
        )
    if prepared["decoder_attention_mask"]["sha256"] != REVIEWED_SOURCE_PREPARED_MASK_SHA256:
        raise ValueError("weighted reviewed prepared attention mask hash changed.")
    _validate_tensor_descriptor(
        reconstruction.get("prospective_active_attention_mask"),
        label="weighted prospective active attention mask",
        expected_dtype="torch.float32",
        expected_shape=[1, 1, 1, ACTIVE_PREFIX_LENGTH],
    )
    if (
            reconstruction["prospective_active_attention_mask"]["sha256"]
            != REVIEWED_SOURCE_ACTIVE_MASK_SHA256
    ):
        raise ValueError("weighted reviewed active attention mask hash changed.")
    cache = reconstruction.get("pre_next_forward_cache")
    if not isinstance(cache, Mapping) or set(cache) != {
        "self_sha256", "cross_sha256", "self_tensors", "cross_tensors"
    }:
        raise ValueError("weighted pre-next-forward cache descriptor is malformed.")
    _validate_sha256(cache["self_sha256"], label="weighted self-cache aggregate")
    _validate_sha256(cache["cross_sha256"], label="weighted cross-cache aggregate")
    if (
            cache["self_sha256"] != REVIEWED_SOURCE_SELF_CACHE_SHA256
            or cache["cross_sha256"] != REVIEWED_SOURCE_CROSS_CACHE_SHA256
    ):
        raise ValueError("weighted reviewed cache aggregate hash changed.")
    for cache_name in ("self_tensors", "cross_tensors"):
        descriptors = cache[cache_name]
        if (
                not isinstance(descriptors, Sequence)
                or isinstance(descriptors, (str, bytes))
                or not descriptors
        ):
            raise ValueError(f"weighted {cache_name} descriptors are malformed.")
        for index, descriptor in enumerate(descriptors):
            _validate_tensor_descriptor(
                descriptor,
                label=f"weighted {cache_name} tensor {index}",
                expected_dtype="torch.float32",
            )
    replay = contract.get("accepted_prefix_replay")
    if not isinstance(replay, Mapping):
        raise ValueError("weighted source contract is missing accepted-prefix replay evidence.")
    if set(replay) != {
        "committed_tokens",
        "all_tokens_match",
        "forced_tokens",
        "stopped",
        "accepted_token_ids_sha256",
        "sampled_token_ids_sha256",
        "steps",
    }:
        raise ValueError("weighted accepted-prefix replay fields changed.")
    if (
            replay.get("committed_tokens") != SOURCE_PREFIX_REPLAY_TOKENS
            or replay.get("all_tokens_match") is not True
            or replay.get("forced_tokens") is not False
            or replay.get("stopped") is not False
    ):
        raise ValueError("weighted source prefix was not sampled exactly from the accepted transcript.")
    steps = replay.get("steps")
    if (
            not isinstance(steps, Sequence)
            or isinstance(steps, (str, bytes))
            or len(steps) != SOURCE_PREFIX_REPLAY_TOKENS
    ):
        raise ValueError("weighted accepted-prefix replay must retain every step.")
    accepted_tokens: list[int] = []
    sampled_tokens: list[int] = []
    previous_rng_after: str | None = None
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping) or set(step) != {
            "step",
            "accepted_token",
            "sampled_token",
            "match",
            "rng_before_sha256",
            "rng_after_sha256",
        }:
            raise ValueError(f"weighted accepted-prefix replay step {index} fields changed.")
        for token_field in ("accepted_token", "sampled_token"):
            token = step.get(token_field)
            if (
                    not isinstance(token, int)
                    or isinstance(token, bool)
                    or token < 0
            ):
                raise ValueError(
                    f"weighted accepted-prefix replay step {index} {token_field} "
                    "must be a non-negative integer."
                )
        if (
                step.get("step") != index
                or step.get("match") is not True
                or step.get("sampled_token") != step.get("accepted_token")
        ):
            raise ValueError(f"weighted accepted-prefix replay step {index} differs.")
        for field in ("rng_before_sha256", "rng_after_sha256"):
            _validate_sha256(
                step.get(field), label=f"weighted replay step {index} {field}"
            )
        if previous_rng_after is not None and step["rng_before_sha256"] != previous_rng_after:
            raise ValueError(f"weighted replay RNG chain breaks before step {index}.")
        previous_rng_after = step["rng_after_sha256"]
        accepted_tokens.append(int(step["accepted_token"]))
        sampled_tokens.append(int(step["sampled_token"]))
    if tuple(accepted_tokens) != TARGET_PREFIX_REPLAY_TOKEN_IDS:
        raise ValueError("weighted accepted-prefix replay token list changed.")
    if token_ids_sha256(accepted_tokens) != TARGET_PREFIX_REPLAY_SHA256:
        raise ValueError("weighted accepted-prefix replay token hash changed.")
    if token_ids_sha256(sampled_tokens) != TARGET_PREFIX_REPLAY_SHA256:
        raise ValueError("weighted sampled-prefix replay token hash changed.")
    if replay.get("accepted_token_ids_sha256") != TARGET_PREFIX_REPLAY_SHA256:
        raise ValueError("weighted accepted replay aggregate hash changed.")
    if replay.get("sampled_token_ids_sha256") != TARGET_PREFIX_REPLAY_SHA256:
        raise ValueError("weighted sampled replay aggregate hash changed.")
    if steps[0]["rng_before_sha256"] != reconstruction["pre_target_rng_state"]["sha256"]:
        raise ValueError("weighted replay does not begin at the pre-target RNG state.")
    if steps[-1]["rng_after_sha256"] != reconstruction["post_replay_rng_state"]["sha256"]:
        raise ValueError("weighted replay does not end at the post-replay RNG state.")
    expected_boundary = {
        "engine": "v32",
        "default_behavior_changed": False,
        "runtime_wiring_added": False,
        "phase_a_authorized": False,
        "authorization_requires_reviewed_contract_sha_commit": True,
    }
    if contract.get("maintainer_boundary") != expected_boundary:
        raise ValueError("weighted source contract changed the V32 maintainer boundary.")


def validate_reviewed_source_contract(
        contract: Mapping[str, Any],
        *,
        expected_contract_sha256: str,
) -> None:
    validate_source_contract(contract)
    if len(expected_contract_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in expected_contract_sha256
    ):
        raise ValueError("reviewed source-contract SHA-256 must contain 64 hex characters.")
    if canonical_json_sha256(contract) != expected_contract_sha256:
        raise ValueError("weighted source contract differs from the separately reviewed SHA-256.")


def validate_reviewed_source_capture_report(report: Mapping[str, Any]) -> None:
    """Validate the complete separately reviewed capture-only artifact."""

    contract_fields = {
        "schema_version",
        "gate",
        "source",
        "transcript",
        "reconstruction",
        "accepted_prefix_replay",
        "maintainer_boundary",
    }
    if set(report) != {
        *contract_fields,
        "rng_advance",
        "runtime_metadata",
        "source_contract_sha256",
    }:
        raise ValueError("weighted reviewed source-capture report fields changed.")
    contract = {field: report[field] for field in contract_fields}
    reported_digest = report.get("source_contract_sha256")
    if reported_digest != REVIEWED_SOURCE_CONTRACT_SHA256:
        raise ValueError("weighted source-capture reported contract digest changed.")
    validate_reviewed_source_contract(
        contract,
        expected_contract_sha256=REVIEWED_SOURCE_CONTRACT_SHA256,
    )

    runtime = report.get("runtime_metadata")
    if not isinstance(runtime, Mapping):
        raise ValueError("weighted source-capture runtime metadata is missing.")
    if set(runtime) != {
        "git_commit",
        "config_name",
        "torch_version",
        "transformers_version",
        "cuda_version",
        "gpu_name",
        "gpu_capability",
        "probe",
        "torch_extensions_dir",
        "torch_extensions_cache_before",
        "torch_extensions_cache_after",
        "total_wall_seconds",
        "capture_only",
        "h8_executed",
        "scheduler_or_runtime_wiring_authorized",
        "reviewed_context_report_sha256",
        "context_preflight_wall_seconds",
        "fresh_context_replay_match",
        "live_context_evidence_match",
    }:
        raise ValueError("weighted source-capture runtime metadata fields changed.")
    expected_runtime = {
        "git_commit": REVIEWED_SOURCE_CAPTURE_COMMIT,
        "config_name": "profile_salvalai_smoke15",
        "torch_version": "2.10.0+cu128",
        "transformers_version": "4.57.3",
        "cuda_version": "12.8",
        "gpu_name": "NVIDIA GeForce RTX 2080 Ti",
        "gpu_capability": [7, 5],
        "capture_only": True,
        "h8_executed": False,
        "scheduler_or_runtime_wiring_authorized": False,
        "reviewed_context_report_sha256": REVIEWED_CONTEXT_REPLAY_REPORT_SHA256,
        "fresh_context_replay_match": True,
        "live_context_evidence_match": True,
        "torch_extensions_dir": "/work/imt11/Mapperatorinator/torch_extensions",
    }
    for field, expected in expected_runtime.items():
        if runtime.get(field) != expected:
            raise ValueError(
                f"weighted source-capture runtime field {field} changed."
            )
    for field in ("context_preflight_wall_seconds", "total_wall_seconds"):
        value = runtime.get(field)
        if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
        ):
            raise ValueError(f"weighted source-capture runtime field {field} is invalid.")
    if runtime["context_preflight_wall_seconds"] >= runtime["total_wall_seconds"]:
        raise ValueError("weighted source-capture preflight cannot exceed total wall.")
    if runtime.get("torch_extensions_cache_before") != runtime.get(
            "torch_extensions_cache_after"
    ):
        raise ValueError("weighted source-capture extension cache changed during capture.")
    expected_extension_cache = {
        "path": "/work/imt11/Mapperatorinator/torch_extensions",
        "exists": True,
        "entry_count": 10,
        "file_count": 112,
    }
    if runtime["torch_extensions_cache_before"] != expected_extension_cache:
        raise ValueError("weighted source-capture extension-cache provenance changed.")
    expected_probe = {
        "context_type": "map",
        "sequence_index": SOURCE_SEQUENCE_INDEX,
        "frame_time_ms": 76965.0,
        "lookback_time": 0.0,
        "lookahead_time": 0.0,
        "timing_pretrim_event_count": 122,
        "timing_event_count": 60,
        "timing_time_shift_count": 30,
        "timing_beat_count": 22,
        "timing_measure_count": 8,
        "timing_marker_count": 30,
        "timing_marker_min_time_ms": 71237,
        "timing_marker_max_time_ms": 85875,
        "timing_point_count": 1,
        "main_prompt_token_counts": list(EXPECTED_MAIN_PROMPT_TOKENS),
    }
    if runtime["probe"] != expected_probe:
        raise ValueError("weighted source-capture runtime probe changed.")

    reconstruction = contract["reconstruction"]
    rng_advance = report.get("rng_advance")
    if not isinstance(rng_advance, Mapping) or set(rng_advance) != {
        "segments",
        "initial_rng_state",
        "post_timing_rng_state",
        "pre_target_rng_state",
    }:
        raise ValueError("weighted source-capture RNG advance fields changed.")
    for field in (
            "initial_rng_state",
            "post_timing_rng_state",
            "pre_target_rng_state",
    ):
        if rng_advance[field] != reconstruction[field]:
            raise ValueError(
                f"weighted source-capture RNG advance field {field} diverged."
            )
    segments = rng_advance["segments"]
    if not isinstance(segments, Sequence) or len(segments) != 2:
        raise ValueError("weighted source-capture must retain two RNG advance segments.")
    expected_segments = (
        {
            "scope": BASE_TIMING_TOKENIZER_SCOPE,
            "vocab_size": BASE_TIMING_VOCAB_SIZE,
            "draw_calls": TIMING_TRANSCRIPT_TOKENS,
            "probability": reconstruction["timing_rng_advance_probability"],
            "rng_before": reconstruction["initial_rng_state"],
            "rng_after": reconstruction["post_timing_rng_state"],
        },
        {
            "scope": MAIN_TOKENIZER_SCOPE,
            "vocab_size": MAIN_VOCAB_SIZE,
            "draw_calls": PRE_TARGET_MAIN_TRANSCRIPT_TOKENS,
            "probability": reconstruction["main_rng_advance_probability"],
            "rng_before": reconstruction["post_timing_rng_state"],
            "rng_after": reconstruction["pre_target_rng_state"],
        },
    )
    for index, (segment, expected) in enumerate(
            zip(segments, expected_segments, strict=True)
    ):
        if not isinstance(segment, Mapping) or set(segment) != {
            "scope",
            "method",
            "vocab_size",
            "draw_calls",
            "probability",
            "rng_before",
            "rng_after",
        }:
            raise ValueError(f"weighted source-capture RNG segment {index} fields changed.")
        for field, value in expected.items():
            if segment.get(field) != value:
                raise ValueError(
                    f"weighted source-capture RNG segment {index} field {field} changed."
                )
        if segment.get("method") != reconstruction["rng_advance_method"]:
            raise ValueError(
                f"weighted source-capture RNG segment {index} method changed."
            )


def load_reviewed_source_capture_report(path: Path) -> dict[str, Any]:
    """Load the pinned capture artifact without accepting digest overrides."""

    if _file_sha256(path) != REVIEWED_SOURCE_REPORT_FILE_SHA256:
        raise ValueError("weighted reviewed source-capture report file SHA-256 changed.")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("weighted reviewed source-capture report must be one object.")
    validate_reviewed_source_capture_report(report)
    return report
