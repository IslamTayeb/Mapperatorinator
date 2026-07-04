from argparse import Namespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from osuT5.osuT5.inference.continuous_batching import (
    ContinuousBatchRequest,
    ContinuousBatchScheduler,
    ContinuousBatchSchedulerConfig,
)
from utils.profile_continuous_scheduler import run_scheduler_dry_run


def _assert_raises(exc_type, fn, message_fragment):
    try:
        fn()
    except exc_type as exc:
        assert message_fragment in str(exc)
    else:
        raise AssertionError(f"Expected {exc_type.__name__}")


def _request(
        request_id,
        script_tokens,
        *,
        max_new_tokens=8,
        eos_token_ids=(99,),
        key=("model", "same"),
        initial_rng_state_hash=None,
        final_rng_state_hash=None,
        logits_processor_state_hash=None,
        cache_state_hash=None,
        planned_arrival_step=None,
):
    return ContinuousBatchRequest(
        request_id=request_id,
        compatibility_key=key,
        prompt_tokens=4,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        script_tokens=list(script_tokens),
        metadata={"song": request_id},
        planned_arrival_step=planned_arrival_step,
        initial_rng_state_hash=initial_rng_state_hash,
        final_rng_state_hash=final_rng_state_hash,
        logits_processor_state_hash=logits_processor_state_hash,
        cache_state_hash=cache_state_hash,
    )


def test_fifo_lifecycle_slot_replacement_and_report_metadata():
    scheduler = ContinuousBatchScheduler(ContinuousBatchSchedulerConfig(max_active_sequences=2))
    scheduler.enqueue(_request("a", [10, 99]))
    scheduler.enqueue(_request("b", [20, 21, 22], max_new_tokens=2))
    scheduler.enqueue(_request("c", [30]))

    step0 = scheduler.step()
    assert step0.step_index == 0
    assert [item["request_id"] for item in step0.activated] == ["a", "b"]
    assert [item["request_id"] for item in step0.decoded] == ["a", "b"]
    assert step0.finished == []

    step1 = scheduler.step()
    assert [item["request_id"] for item in step1.finished] == ["a", "b"]
    assert [item["stop_reason"] for item in step1.finished] == ["eos", "max_new_tokens"]
    assert [item["request_id"] for item in step1.activated] == ["c"]

    step2 = scheduler.step()
    assert step2.activated == []
    assert [item["request_id"] for item in step2.finished] == ["c"]
    assert step2.finished[0]["stop_reason"] == "script_exhausted"
    assert not scheduler.has_work()

    report = scheduler.report().to_dict()
    assert report["active_batch_size_histogram"] == {"1": 1, "2": 2}
    assert [
        (
            request["request_id"],
            request["generated_tokens"],
            request["stop_reason"],
            request["enqueue_step"],
            request["activation_step"],
            request["finish_step"],
            request["queue_wait_steps"],
            request["decode_steps"],
            request["latency_steps"],
            request["cache_slot_id"],
            request["slot_generation"],
        )
        for request in report["requests"]
    ] == [
        ("a", [10, 99], "eos", 0, 0, 1, 0, 2, 2, 0, 1),
        ("b", [20, 21], "max_new_tokens", 0, 0, 1, 0, 2, 2, 1, 1),
        ("c", [30], "script_exhausted", 0, 1, 2, 1, 2, 3, 0, 2),
    ]
    assert step1.activated == [
        {"request_id": "c", "cache_slot_id": 0, "slot_generation": 2, "activation_step": 1, "queue_wait_steps": 1}
    ]
    assert report["cache_slot_events"] == [
        {"event": "acquire", "step_index": 0, "request_id": "a", "cache_slot_id": 0, "slot_generation": 1},
        {"event": "acquire", "step_index": 0, "request_id": "b", "cache_slot_id": 1, "slot_generation": 1},
        {
            "event": "release",
            "step_index": 1,
            "request_id": "a",
            "cache_slot_id": 0,
            "slot_generation": 1,
            "stop_reason": "eos",
        },
        {
            "event": "release",
            "step_index": 1,
            "request_id": "b",
            "cache_slot_id": 1,
            "slot_generation": 1,
            "stop_reason": "max_new_tokens",
        },
        {"event": "acquire", "step_index": 1, "request_id": "c", "cache_slot_id": 0, "slot_generation": 2},
        {
            "event": "release",
            "step_index": 2,
            "request_id": "c",
            "cache_slot_id": 0,
            "slot_generation": 2,
            "stop_reason": "script_exhausted",
        },
    ]


def test_empty_scheduler_step_returns_noop_metadata():
    scheduler = ContinuousBatchScheduler(ContinuousBatchSchedulerConfig(max_active_sequences=2))
    step = scheduler.step()

    assert step.is_noop
    assert step.step_index == 0
    assert scheduler.current_step == 1
    assert scheduler.report().to_dict()["steps"] == [
        {"step_index": 0, "activated": [], "decoded": [], "finished": [], "active_batch_size": 0}
    ]


def test_compatibility_key_guard_and_config_validation():
    _assert_raises(
        ValueError,
        lambda: ContinuousBatchScheduler(ContinuousBatchSchedulerConfig(max_active_sequences=0)),
        "max_active_sequences",
    )
    _assert_raises(
        ValueError,
        lambda: ContinuousBatchScheduler(
            ContinuousBatchSchedulerConfig(max_active_sequences=1, max_wait_ms=-1)
        ),
        "max_wait_ms",
    )
    _assert_raises(
        ValueError,
        lambda: ContinuousBatchScheduler(
            ContinuousBatchSchedulerConfig(max_active_sequences=1, prefill_policy="surprise")
        ),
        "prefill_policy",
    )
    _assert_raises(
        ValueError,
        lambda: ContinuousBatchScheduler(
            ContinuousBatchSchedulerConfig(max_active_sequences=1, decode_order_policy="random")
        ),
        "decode_order_policy",
    )
    _assert_raises(
        ValueError,
        lambda: ContinuousBatchScheduler(
            ContinuousBatchSchedulerConfig(max_active_sequences=1, rng_policy="surprise")
        ),
        "rng_policy",
    )

    scheduler = ContinuousBatchScheduler(ContinuousBatchSchedulerConfig(max_active_sequences=1))
    scheduler.enqueue(_request("a", [1], key=("model", "a")))
    _assert_raises(
        ValueError,
        lambda: scheduler.enqueue(_request("b", [2], key=("model", "b"))),
        "compatibility_key",
    )
    _assert_raises(
        ValueError,
        lambda: scheduler.enqueue(_request("a", [3], key=("model", "a"))),
        "Duplicate request_id",
    )
    _assert_raises(
        ValueError,
        lambda: scheduler.enqueue(_request("bad", [3], max_new_tokens=0, key=("model", "a"))),
        "max_new_tokens",
    )
    _assert_raises(
        ValueError,
        lambda: scheduler.enqueue(_request("bad-arrival", [3], key=("model", "a"), planned_arrival_step=-1)),
        "planned_arrival_step",
    )
    _assert_raises(
        ValueError,
        lambda: scheduler.enqueue(_request("future-arrival", [3], key=("model", "a"), planned_arrival_step=2)),
        "before its planned arrival step",
    )


def test_round_robin_changes_decode_order_without_changing_slots():
    scheduler = ContinuousBatchScheduler(
        ContinuousBatchSchedulerConfig(max_active_sequences=3, decode_order_policy="round_robin")
    )
    for request_id in ["a", "b", "c"]:
        scheduler.enqueue(_request(request_id, [1, 2, 3], max_new_tokens=3))

    orders = []
    for _ in range(2):
        step = scheduler.step()
        orders.append([item["request_id"] for item in step.decoded])

    assert orders == [["a", "b", "c"], ["b", "c", "a"]]
    assert {
        slot.request_id: slot.slot_id
        for slot in scheduler.active_slots()
    } == {
        "a": 0,
        "b": 1,
        "c": 2,
    }


def test_run_until_idle_is_deterministic():
    def run_once():
        scheduler = ContinuousBatchScheduler(ContinuousBatchSchedulerConfig(max_active_sequences=2))
        scheduler.enqueue(_request("a", [1, 2], max_new_tokens=2))
        scheduler.enqueue(_request("b", [3, 4, 99], max_new_tokens=5))
        return scheduler.run_until_idle(max_steps=10).to_dict()

    assert run_once() == run_once()


def test_no_post_stop_decode_for_finished_requests():
    scheduler = ContinuousBatchScheduler(ContinuousBatchSchedulerConfig(max_active_sequences=1))
    scheduler.enqueue(_request("a", [99, 100, 101], max_new_tokens=5))
    scheduler.step()
    report_after_finish = scheduler.report().to_dict()

    assert not scheduler.has_work()
    assert report_after_finish["requests"][0]["generated_tokens"] == [99]
    assert report_after_finish["requests"][0]["stop_reason"] == "eos"
    assert scheduler.step().is_noop
    assert scheduler.report().to_dict()["requests"][0]["generated_tokens"] == [99]


def test_report_preserves_exactness_ledger_hashes_and_config():
    scheduler = ContinuousBatchScheduler(
        ContinuousBatchSchedulerConfig(
            max_active_sequences=1,
            max_wait_ms=25,
            prefill_policy="batch_prefill",
            decode_order_policy="arrival_order",
            rng_policy="per_request_generator",
        )
    )
    scheduler.enqueue(_request(
        "a",
        [1],
        initial_rng_state_hash="rng-before",
        final_rng_state_hash="rng-after",
        logits_processor_state_hash="logits-state",
        cache_state_hash="cache-state",
    ))

    report = scheduler.run_until_idle(max_steps=4).to_dict()

    assert report["config"] == {
        "max_active_sequences": 1,
        "max_wait_ms": 25,
        "prefill_policy": "batch_prefill",
        "decode_order_policy": "arrival_order",
        "rng_policy": "per_request_generator",
    }
    assert report["requests"][0]["initial_rng_state_hash"] == "rng-before"
    assert report["requests"][0]["final_rng_state_hash"] == "rng-after"
    assert report["requests"][0]["logits_processor_state_hash"] == "logits-state"
    assert report["requests"][0]["cache_state_hash"] == "cache-state"


def test_profile_continuous_scheduler_honors_staggered_arrivals():
    requests = [
        {
            "request_id": "a",
            "arrival_step": 0,
            "prompt_tokens": 4,
            "max_new_tokens": 2,
            "script_tokens": [1, 2],
            "generate_kwargs": {"do_sample": True, "top_p": 0.9, "top_k": 20, "temperature": 1.0},
            "initial_rng_state_hash": "rng-before-a",
            "final_rng_state_hash": "rng-after-a",
            "logits_processor_state_hash": "logits-a",
            "cache_state_hash": "cache-a",
        },
        {
            "request_id": "b",
            "arrival_step": 0,
            "prompt_tokens": 4,
            "max_new_tokens": 1,
            "script_tokens": [3],
            "generate_kwargs": {"do_sample": True, "top_p": 0.9, "top_k": 20, "temperature": 1.0},
            "initial_rng_state_hash": "rng-before-b",
            "final_rng_state_hash": "rng-after-b",
            "logits_processor_state_hash": "logits-b",
            "cache_state_hash": "cache-b",
        },
        {
            "request_id": "c",
            "arrival_step": 4,
            "prompt_tokens": 4,
            "max_new_tokens": 1,
            "script_tokens": [4],
            "generate_kwargs": {"do_sample": True, "top_p": 0.9, "top_k": 20, "temperature": 1.0},
            "initial_rng_state_hash": "rng-before-c",
            "final_rng_state_hash": "rng-after-c",
            "logits_processor_state_hash": "logits-c",
            "cache_state_hash": "cache-c",
        },
    ]
    with TemporaryDirectory() as tmpdir:
        requests_path = Path(tmpdir) / "requests.json"
        requests_path.write_text(json.dumps(requests), encoding="utf-8")
        manifest = run_scheduler_dry_run(
            Namespace(
                requests_json=requests_path,
                output_root=None,
                suite_id="arrival-test",
                max_active_sequences=1,
                max_wait_ms=0,
                prefill_policy="serial",
                decode_order_policy="arrival_order",
                rng_policy="serial_global",
                allow_missing_state_hashes=False,
                max_steps=16,
            )
        )

    assert manifest["aggregate"]["scheduler_step_count"] == 5
    assert manifest["aggregate"]["idle_step_count"] == 1
    assert manifest["aggregate"]["planned_arrival_step_histogram"] == {"0": 2, "4": 1}
    assert manifest["active_batch_size_histogram"] == {"1": 4}
    rows = {request["request_id"]: request for request in manifest["requests"]}
    assert rows["a"]["planned_arrival_step"] == 0
    assert rows["a"]["queue_wait_steps"] == 0
    assert rows["b"]["planned_arrival_step"] == 0
    assert rows["b"]["queue_wait_steps"] == 1
    assert rows["c"]["planned_arrival_step"] == 4
    assert rows["c"]["enqueue_step"] == 4
    assert rows["c"]["queue_wait_steps"] == 0
    assert rows["c"]["missing_state_hash_fields"] == []


def test_profile_continuous_scheduler_requires_state_hashes_by_default():
    requests = [
        {
            "request_id": "missing-state",
            "arrival_step": 0,
            "prompt_tokens": 4,
            "max_new_tokens": 1,
            "script_tokens": [1],
            "generate_kwargs": {"do_sample": True},
        },
    ]
    with TemporaryDirectory() as tmpdir:
        requests_path = Path(tmpdir) / "requests.json"
        requests_path.write_text(json.dumps(requests), encoding="utf-8")
        args = Namespace(
            requests_json=requests_path,
            output_root=None,
            suite_id="missing-state-test",
            max_active_sequences=1,
            max_wait_ms=0,
            prefill_policy="serial",
            decode_order_policy="arrival_order",
            rng_policy="serial_global",
            allow_missing_state_hashes=False,
            max_steps=4,
        )

        _assert_raises(
            ValueError,
            lambda: run_scheduler_dry_run(args),
            "missing state hash fields",
        )

        args.allow_missing_state_hashes = True
        manifest = run_scheduler_dry_run(args)

    assert manifest["state_hash_policy"] == "allow_missing_planning_only"
    assert manifest["aggregate"]["missing_state_hash_request_count"] == 1
    assert manifest["requests"][0]["missing_state_hash_fields"] == [
        "initial_rng_state_hash",
        "final_rng_state_hash",
        "logits_processor_state_hash",
        "cache_state_hash",
    ]
