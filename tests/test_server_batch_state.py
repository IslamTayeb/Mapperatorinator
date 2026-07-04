import threading

import torch

from osuT5.osuT5.inference.server import (
    InferenceServer,
    StaticServerRequest,
    generation_compatibility_key,
)


def test_generation_compatibility_key_handles_nested_hashables_independent_of_dict_order():
    left = generation_compatibility_key({
        "temperature": 1.0,
        "stops": [1, 2, 3],
        "policies": {"rng": "shared_global", "prefill": "serial"},
    })
    right = generation_compatibility_key({
        "policies": {"prefill": "serial", "rng": "shared_global"},
        "stops": [1, 2, 3],
        "temperature": 1.0,
    })

    assert left == right


def test_generation_compatibility_key_distinguishes_ordered_values():
    assert generation_compatibility_key({"stops": [1, 2, 3]}) != generation_compatibility_key({"stops": [3, 2, 1]})


def test_generation_compatibility_key_rejects_tensors():
    try:
        generation_compatibility_key({"bad_state": torch.tensor([1])})
    except TypeError as exc:
        assert "server batching compatibility key" in str(exc)
    else:
        raise AssertionError("Expected tensor-valued generate_kwargs to be rejected")


def test_static_server_request_starts_with_isolated_batch_metadata_lists():
    first = StaticServerRequest(model_kwargs={}, total_work=3, conn=None, event=threading.Event())
    second = StaticServerRequest(model_kwargs={}, total_work=5, conn=None, event=threading.Event())

    first.server_batch_ids.append(11)
    first.work_done = 1

    assert first.remaining_work == 2
    assert second.remaining_work == 5
    assert second.server_batch_ids == []


def test_cut_model_kwargs_slices_tensors_and_keeps_metadata():
    server = InferenceServer(model=None, tokenizer=None)
    model_kwargs = {
        "inputs": torch.arange(12).reshape(4, 3),
        "decoder_input_ids": torch.arange(20).reshape(4, 5),
        "context_type": "map",
        "optional": None,
    }

    cut = server._cut_model_kwargs(model_kwargs, start=1, length=2)

    assert torch.equal(cut["inputs"], torch.tensor([[3, 4, 5], [6, 7, 8]]))
    assert torch.equal(
        cut["decoder_input_ids"],
        torch.tensor([[5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]),
    )
    assert cut["context_type"] == "map"
    assert cut["optional"] is None
