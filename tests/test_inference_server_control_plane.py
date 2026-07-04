from inference import get_server_address, get_server_runtime_key, load_model_with_server


def test_server_runtime_key_changes_with_batching_knobs():
    base = get_server_runtime_key(
        max_batch_size=10,
        server_batch_timeout=0.2,
        device="cuda",
        precision="fp32",
        attn_implementation="sdpa",
        generation_compile=False,
    )
    changed = get_server_runtime_key(
        max_batch_size=20,
        server_batch_timeout=0.2,
        device="cuda",
        precision="fp32",
        attn_implementation="sdpa",
        generation_compile=False,
    )

    assert base != changed
    assert "mb-10" in base
    assert "bt-0_2" in base
    assert "dev-cuda" in base


def test_server_address_includes_runtime_key():
    first = get_server_address(None, server_runtime_key="mb-10__bt-0_2")
    second = get_server_address(None, server_runtime_key="mb-20__bt-0_2")

    assert first != second
    assert "mb-10__bt-0_2" in first


def test_load_model_with_server_rejects_generation_compile_server_path():
    try:
        load_model_with_server(
            ckpt_path=None,
            t5_args=None,
            device="cpu",
            use_server=True,
            generation_compile=True,
        )
    except ValueError as exc:
        assert "not supported with use_server=true" in str(exc)
    else:
        raise AssertionError("Expected server + generation compile to fail before model loading")
