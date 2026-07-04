import _pickle
import os
import time
import threading
import traceback
import torch
from dataclasses import dataclass, field
from functools import partial
from multiprocessing.connection import Listener, Client
from typing import Any

from transformers import LogitsProcessorList, ClassifierFreeGuidanceLogitsProcessor, TemperatureLogitsWarper

from ..event import EventType, ContextType
from .logit_processors import ConditionalTemperatureLogitsWarper, get_beat_type_tokens, \
    get_mania_type_tokens, get_scroll_speed_tokens, TimeshiftBias, LookbackBiasLogitsWarper, \
    MonotonicTimeShiftLogitsProcessor
from .cache_utils import MapperatorinatorCache, get_cache
from .decode_loop import active_prefix_decode_generate
from .generation_compatibility import generation_compatibility_key
from ..runtime_profiling import generation_profile_context
from ..model import Mapperatorinator
from ..tokenizer import Tokenizer

# The default address used for IPC
SOCKET_PATH = r'\\.\pipe\Mapperatorinator'

MILISECONDS_PER_SECOND = 1000
MILISECONDS_PER_STEP = 10

RETRY_SIGNAL = "RETRY_SIGNAL"


@dataclass
class StaticServerRequest:
    model_kwargs: dict[str, Any]
    total_work: int
    conn: Any
    event: threading.Event
    work_done: int = 0
    result: Any = None
    generated_tokens: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    generated_tokens_per_sample: list[int] = field(default_factory=list)
    prompt_tokens_per_sample: list[int] = field(default_factory=list)
    output_tokens_per_sample: list[int] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    enqueued_at: float = field(default_factory=time.perf_counter)
    queued_at: float = field(default_factory=time.perf_counter)
    server_batch_ids: list[int] = field(default_factory=list)
    server_batch_sizes: list[int] = field(default_factory=list)
    server_batch_request_counts: list[int] = field(default_factory=list)
    server_batch_work_items: list[int] = field(default_factory=list)
    server_batch_elapsed_seconds: list[float] = field(default_factory=list)
    server_queue_wait_seconds: list[float] = field(default_factory=list)
    server_first_queue_wait_seconds: float | None = None

    @property
    def remaining_work(self) -> int:
        return self.total_work - self.work_done


@dataclass
class StaticServerRequestGroup:
    generate_kwargs: dict[str, Any]
    requests: list[StaticServerRequest] = field(default_factory=list)


def _prompt_token_counts(model_kwargs, pad_token_id: int | None) -> torch.Tensor | None:
    decoder_attention_mask = model_kwargs.get("decoder_attention_mask")
    if isinstance(decoder_attention_mask, torch.Tensor):
        return decoder_attention_mask.to(torch.long).sum(dim=-1).cpu()

    decoder_input_ids = model_kwargs.get("decoder_input_ids")
    if not isinstance(decoder_input_ids, torch.Tensor):
        return None

    if pad_token_id is None:
        return torch.full((decoder_input_ids.shape[0],), decoder_input_ids.shape[1], dtype=torch.long)

    return decoder_input_ids.ne(pad_token_id).to(torch.long).sum(dim=-1).cpu()


def _output_token_counts(result: torch.Tensor, pad_token_id: int | None) -> torch.Tensor:
    if pad_token_id is None:
        return torch.full((result.shape[0],), result.shape[1], dtype=torch.long)

    return result.ne(pad_token_id).to(torch.long).sum(dim=-1)


def _build_generation_stats(
        result: torch.Tensor,
        model_kwargs: dict,
        pad_token_id: int | None,
        elapsed_seconds: float,
) -> dict:
    prompt_token_counts = _prompt_token_counts(model_kwargs, pad_token_id)
    output_token_counts = _output_token_counts(result, pad_token_id).cpu()
    generated_token_counts = output_token_counts.clone()
    if prompt_token_counts is not None:
        generated_token_counts = torch.clamp(generated_token_counts - prompt_token_counts, min=0)

    generated_tokens = int(generated_token_counts.sum().item())
    tokens_per_second = generated_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
    prompt_tokens = int(prompt_token_counts.sum().item()) if prompt_token_counts is not None else None

    return {
        "batch_size": int(result.shape[0]),
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_per_sample": prompt_token_counts.tolist() if prompt_token_counts is not None else None,
        "output_tokens": int(output_token_counts.sum().item()),
        "output_tokens_per_sample": output_token_counts.tolist(),
        "generated_tokens": generated_tokens,
        "generated_tokens_per_sample": generated_token_counts.tolist(),
        "elapsed_seconds": float(elapsed_seconds),
        "tokens_per_second": tokens_per_second,
    }


def get_eos_token_id(tokenizer, lookback_time: float = 0, lookahead_time: float = 0, context_type: ContextType = None):
    eos_token_id = [tokenizer.eos_id]
    if context_type is not None and context_type in tokenizer.context_eos:
        eos_token_id.append(tokenizer.context_eos[context_type])
    if lookback_time > 0:
        eos_token_id.extend(range(tokenizer.event_start[EventType.TIME_SHIFT], tokenizer.event_start[EventType.TIME_SHIFT] + int(lookback_time / MILISECONDS_PER_STEP)))
    if lookahead_time > 0:
        eos_token_id.extend(range(tokenizer.event_end[EventType.TIME_SHIFT] - int(lookahead_time / MILISECONDS_PER_STEP), tokenizer.event_end[EventType.TIME_SHIFT]))
    return eos_token_id


def build_logits_processor_list(
        tokenizer,
        *,
        cfg_scale: float = 1.0,
        timeshift_bias: float = 0.0,
        types_first: bool = False,
        temperature: float = 1.0,
        timing_temperature: float | None = None,
        mania_column_temperature: float | None = None,
        taiko_hit_temperature: float | None = None,
        lookback_time: float = 0.0,
        device=None,
        stateful_monotonic: bool = False,
) -> LogitsProcessorList:
    timing_temperature = temperature if timing_temperature is None else timing_temperature
    mania_column_temperature = temperature if mania_column_temperature is None else mania_column_temperature
    taiko_hit_temperature = temperature if taiko_hit_temperature is None else taiko_hit_temperature
    device = device if device is not None else getattr(tokenizer, "device", None)

    logits_processor_list = LogitsProcessorList()
    if cfg_scale > 1.0:
        logits_processor_list.append(ClassifierFreeGuidanceLogitsProcessor(cfg_scale))

    logits_processor_list.append(
        MonotonicTimeShiftLogitsProcessor(tokenizer, stateful_batch1=stateful_monotonic)
    )

    if timeshift_bias != 0:
        logits_processor_list.append(
            TimeshiftBias(
                timeshift_bias,
                tokenizer.event_start[EventType.TIME_SHIFT],
                tokenizer.event_end[EventType.TIME_SHIFT]
            )
        )
    if types_first:
        logits_processor_list.append(ConditionalTemperatureLogitsWarper(
            temperature,
            timing_temperature,
            mania_column_temperature,
            taiko_hit_temperature,
            types_first,
            get_beat_type_tokens(tokenizer),
            get_mania_type_tokens(tokenizer),
            get_scroll_speed_tokens(tokenizer),
        ))
    else:
        logits_processor_list.append(TemperatureLogitsWarper(temperature))
    if lookback_time > 0:
        logits_processor_list.append(LookbackBiasLogitsWarper(lookback_time, tokenizer, types_first, device))

    return logits_processor_list


def _sync_cuda_for_model(model) -> None:
    if torch.cuda.is_available() and getattr(getattr(model, "device", None), "type", None) == "cuda":
        torch.cuda.synchronize(model.device)


def _reset_mapperatorinator_cache(cache: MapperatorinatorCache) -> None:
    cache.self_attention_cache.reset()
    cache.cross_attention_cache.reset()
    for layer_idx in list(cache.is_updated):
        cache.is_updated[layer_idx] = False


def _session_cache(
        decode_session_state: dict | None,
        model,
        *,
        batch_size: int,
        num_beams: int,
        cfg_scale: float,
) -> MapperatorinatorCache:
    if decode_session_state is None:
        return get_cache(model, batch_size, num_beams, cfg_scale)

    cache = decode_session_state.get("cache")
    if cache is None:
        cache = get_cache(model, batch_size, num_beams, cfg_scale)
        decode_session_state["cache"] = cache
    else:
        _reset_mapperatorinator_cache(cache)
    return cache


@torch.no_grad()
def model_generate(model, tokenizer, model_kwargs, generate_kwargs):
    # To device
    model_kwargs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in model_kwargs.items()}
    model_kwargs = {k: v.to(model.dtype) if k != "inputs" and isinstance(v, torch.Tensor) and v.dtype == torch.float32 else v for k, v in model_kwargs.items()}
    batch_size = model_kwargs['inputs'].shape[0]
    # print(f"[Model Generate] Batch size: {batch_size}, Model device: {model.device}")

    precision = generate_kwargs.pop('precision', 'fp32')
    cfg_scale = generate_kwargs.pop('cfg_scale', 1.0)
    timeshift_bias = generate_kwargs.pop('timeshift_bias', 0)
    types_first = generate_kwargs.pop('types_first', False)
    temperature = generate_kwargs.pop('temperature', 1.0)
    timing_temperature = generate_kwargs.pop('timing_temperature', temperature)
    mania_column_temperature = generate_kwargs.pop('mania_column_temperature', temperature)
    taiko_hit_temperature = generate_kwargs.pop('taiko_hit_temperature', temperature)
    lookback_time = generate_kwargs.pop('lookback_time', 0.0)
    lookahead_time = generate_kwargs.pop('lookahead_time', 0.0)
    context_type = generate_kwargs.pop('context_type', None)
    sync_model_timing = bool(generate_kwargs.pop('sync_model_timing', False))
    profile_model_generate_cuda_ledger = bool(generate_kwargs.pop('profile_model_generate_cuda_ledger', False))
    profile_generation_detail_ranges = bool(generate_kwargs.pop('profile_generation_detail_ranges', False))
    profile_active_prefix_decode_diagnostics = bool(generate_kwargs.pop('profile_active_prefix_decode_diagnostics', False))
    profile_sdpa_backend = generate_kwargs.pop('profile_sdpa_backend', None)
    active_prefix_decode_loop = bool(generate_kwargs.pop('active_prefix_decode_loop', False))
    active_prefix_decode_bucket_size = int(generate_kwargs.pop('active_prefix_decode_bucket_size', 128))
    active_prefix_decode_cuda_graph = bool(generate_kwargs.pop('active_prefix_decode_cuda_graph', False))
    active_prefix_decode_cuda_graph_warmup = int(generate_kwargs.pop('active_prefix_decode_cuda_graph_warmup', 0))
    active_prefix_decode_cuda_graph_min_decode_steps = int(
        generate_kwargs.pop('active_prefix_decode_cuda_graph_min_decode_steps', 1)
    )
    stateful_monotonic_logits_processor = bool(
        generate_kwargs.pop('stateful_monotonic_logits_processor', False)
    )
    q1_bmm_cross_attention = bool(generate_kwargs.pop('q1_bmm_cross_attention', False))
    native_q1_self_attention_requested = bool(generate_kwargs.pop('native_q1_self_attention', False))
    native_q1_rope_cache_self_attention_requested = bool(
        generate_kwargs.pop('native_q1_rope_cache_self_attention', False)
    )
    decode_session_state = generate_kwargs.pop('decode_session_state', None)
    decode_session_cuda_graph = bool(generate_kwargs.pop('decode_session_cuda_graph', False))
    if context_type is not None:
        context_type = ContextType(context_type)  # Convert to ContextType enum
    if native_q1_rope_cache_self_attention_requested and not native_q1_self_attention_requested:
        raise ValueError("native_q1_rope_cache_self_attention requires native_q1_self_attention.")
    if native_q1_rope_cache_self_attention_requested and not active_prefix_decode_loop:
        raise ValueError("native_q1_rope_cache_self_attention requires active_prefix_decode_loop.")
    native_q1_self_attention = (
        native_q1_self_attention_requested
        and context_type != ContextType.TIMING
    )
    native_q1_rope_cache_self_attention = (
        native_q1_rope_cache_self_attention_requested
        and native_q1_self_attention
    )

    # Create the logits processors
    logits_processor_list = build_logits_processor_list(
        tokenizer,
        cfg_scale=cfg_scale,
        timeshift_bias=timeshift_bias,
        types_first=types_first,
        temperature=temperature,
        timing_temperature=timing_temperature,
        mania_column_temperature=mania_column_temperature,
        taiko_hit_temperature=taiko_hit_temperature,
        lookback_time=lookback_time,
        device=model.device,
        stateful_monotonic=stateful_monotonic_logits_processor,
    )

    # Prepare cache
    cache = _session_cache(
        decode_session_state,
        model,
        batch_size=batch_size,
        num_beams=generate_kwargs.get('num_beams', 1),
        cfg_scale=cfg_scale,
    )
    pad_token_id = generate_kwargs.get('pad_token_id', getattr(tokenizer, 'pad_id', None))
    if active_prefix_decode_cuda_graph and not active_prefix_decode_loop:
        raise ValueError("active_prefix_decode_cuda_graph requires active_prefix_decode_loop.")
    if active_prefix_decode_loop:
        if batch_size != 1:
            raise ValueError("active_prefix_decode_loop currently supports batch_size=1 only.")
        if cfg_scale != 1.0:
            raise ValueError("active_prefix_decode_loop currently does not support classifier-free guidance.")
        if int(generate_kwargs.get('num_beams', 1)) != 1:
            raise ValueError("active_prefix_decode_loop currently supports num_beams=1 only.")
        if active_prefix_decode_bucket_size <= 0:
            raise ValueError("active_prefix_decode_bucket_size must be positive.")
        if active_prefix_decode_cuda_graph_min_decode_steps <= 0:
            raise ValueError("active_prefix_decode_cuda_graph_min_decode_steps must be positive.")
    if decode_session_state is not None:
        if not active_prefix_decode_loop:
            raise ValueError("decode_session_state requires active_prefix_decode_loop.")
        if not active_prefix_decode_cuda_graph or not decode_session_cuda_graph:
            raise ValueError("decode_session_state currently requires active-prefix CUDA graph replay.")
        decode_session_state.setdefault("graph_cache", {})
        decode_session_state.setdefault("stable_encoder_holder", {})
    active_prefix_decode_diagnostics = (
        {
            "enabled": True,
            "decode_steps": 0,
            "bucket_lengths_seen": [],
            "bucket_transition_count": 0,
        }
        if active_prefix_decode_loop and profile_active_prefix_decode_diagnostics
        else None
    )

    # Perform batched generation
    generate_start_event = generate_end_event = None
    generate_cuda_event_seconds = None
    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=precision == 'amp'), \
            generation_profile_context(
                detail_ranges=profile_generation_detail_ranges,
                sdpa_backend=profile_sdpa_backend,
                q1_bmm_cross_attention=q1_bmm_cross_attention,
                native_q1_self_attention=native_q1_self_attention,
                native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
            ):
        if sync_model_timing:
            _sync_cuda_for_model(model)
            if (
                profile_model_generate_cuda_ledger
                and torch.cuda.is_available()
                and getattr(getattr(model, "device", None), "type", None) == "cuda"
            ):
                generate_start_event = torch.cuda.Event(enable_timing=True)
                generate_end_event = torch.cuda.Event(enable_timing=True)
        start_time = time.perf_counter()
        if generate_start_event is not None:
            generate_start_event.record()
        result = model.generate(
            **model_kwargs,
            **generate_kwargs,
            use_cache=True,
            past_key_values=cache,
            logits_processor=logits_processor_list,
            eos_token_id=get_eos_token_id(tokenizer, lookback_time=lookback_time, lookahead_time=lookahead_time, context_type=context_type),
            custom_generate=partial(
                active_prefix_decode_generate,
                active_prefix_bucket_size=active_prefix_decode_bucket_size,
                cuda_graph_forward=active_prefix_decode_cuda_graph,
                cuda_graph_warmup=active_prefix_decode_cuda_graph_warmup,
                cuda_graph_min_decode_steps=active_prefix_decode_cuda_graph_min_decode_steps,
                active_prefix_decode_diagnostics=active_prefix_decode_diagnostics,
                shared_graph_cache=(
                    decode_session_state.get("graph_cache")
                    if decode_session_state is not None and decode_session_cuda_graph
                    else None
                ),
                stable_encoder_holder=(
                    decode_session_state.get("stable_encoder_holder")
                    if decode_session_state is not None and decode_session_cuda_graph
                    else None
                ),
            ) if active_prefix_decode_loop else None,
        )
        if generate_end_event is not None:
            generate_end_event.record()
        if sync_model_timing:
            _sync_cuda_for_model(model)
        elapsed_seconds = time.perf_counter() - start_time
        if generate_start_event is not None and generate_end_event is not None:
            generate_cuda_event_seconds = float(generate_start_event.elapsed_time(generate_end_event)) / 1000.0

    result = result.cpu()
    stats = _build_generation_stats(result, model_kwargs, pad_token_id, elapsed_seconds)
    stats.update({
        "precision": precision,
        "context_type": context_type.value if context_type is not None else None,
        "num_beams": int(generate_kwargs.get("num_beams", 1)),
        "cfg_scale": float(cfg_scale),
        "do_sample": bool(generate_kwargs.get("do_sample", False)),
        "sync_model_timing": sync_model_timing,
        "profile_model_generate_cuda_ledger": profile_model_generate_cuda_ledger,
        "model_generate_cpu_elapsed_seconds": elapsed_seconds if profile_model_generate_cuda_ledger else None,
        "model_generate_cuda_event_seconds": generate_cuda_event_seconds,
        "model_generate_host_gap_seconds": (
            elapsed_seconds - generate_cuda_event_seconds
            if generate_cuda_event_seconds is not None
            else None
        ),
        "generation_compile_enabled": not bool(getattr(getattr(model, "generation_config", None), "disable_compile", True)),
        "profile_generation_detail_ranges": profile_generation_detail_ranges,
        "profile_active_prefix_decode_diagnostics": profile_active_prefix_decode_diagnostics,
        "profile_sdpa_backend": profile_sdpa_backend,
        "stateful_monotonic_logits_processor": stateful_monotonic_logits_processor,
        "q1_bmm_cross_attention_enabled": q1_bmm_cross_attention,
        "native_q1_self_attention_requested": native_q1_self_attention_requested,
        "native_q1_self_attention_enabled": native_q1_self_attention,
        "native_q1_self_attention_disabled_reason": (
            "timing_context"
            if native_q1_self_attention_requested and not native_q1_self_attention
            else None
        ),
        "native_q1_rope_cache_self_attention_requested": native_q1_rope_cache_self_attention_requested,
        "native_q1_rope_cache_self_attention_enabled": native_q1_rope_cache_self_attention,
        "native_q1_rope_cache_self_attention_disabled_reason": (
            "timing_context"
            if native_q1_rope_cache_self_attention_requested and not native_q1_rope_cache_self_attention
            else None
        ),
        "decode_session_runtime_enabled": decode_session_state is not None,
        "decode_session_cuda_graph_enabled": bool(decode_session_cuda_graph),
        "decode_session_graph_count": (
            len(decode_session_state.get("graph_cache", {}))
            if decode_session_state is not None
            else None
        ),
        "active_prefix_decode_loop_enabled": active_prefix_decode_loop,
        "active_prefix_decode_bucket_size": active_prefix_decode_bucket_size if active_prefix_decode_loop else None,
        "active_prefix_decode_cuda_graph_enabled": active_prefix_decode_cuda_graph if active_prefix_decode_loop else False,
        "active_prefix_decode_cuda_graph_warmup": (
            active_prefix_decode_cuda_graph_warmup if active_prefix_decode_cuda_graph else None
        ),
        "active_prefix_decode_cuda_graph_min_decode_steps": (
            active_prefix_decode_cuda_graph_min_decode_steps if active_prefix_decode_cuda_graph else None
        ),
    })
    if active_prefix_decode_diagnostics is not None:
        stats["active_prefix_decode_diagnostics"] = active_prefix_decode_diagnostics

    return result, stats


@torch.no_grad()
def model_forward(model, model_kwargs, generate_kwargs):
    # To device
    model_kwargs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in model_kwargs.items()}
    model_kwargs = {k: v.to(model.dtype) if k != "inputs" and isinstance(v, torch.Tensor) and v.dtype == torch.float32 else v for k, v in model_kwargs.items()}
    model_kwargs["frames"] = model_kwargs.pop('inputs', None)  # Rename for compatibility
    precision = generate_kwargs.pop('precision', 'fp32')
    cfg_scale = generate_kwargs.pop('cfg_scale', 1.0)

    # Prepare inputs for the model
    model_kwargs = model.prepare_inputs_for_generation(**model_kwargs)

    # Create the logits processors
    logits_processor_list = LogitsProcessorList()
    if cfg_scale > 1.0:
        logits_processor_list.append(ClassifierFreeGuidanceLogitsProcessor(cfg_scale))

    # Perform forward pass
    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=precision == 'amp'):
        logits = model.forward(**model_kwargs).logits.to(torch.float32)

    logits = logits_processor_list(model_kwargs["decoder_input_ids"], logits).cpu()
    return logits


if os.name == "nt":
    import msvcrt

    def portable_lock(fp):
        fp.seek(0)
        msvcrt.locking(fp.fileno(), msvcrt.LK_LOCK, 1)

    def portable_unlock(fp):
        fp.seek(0)
        msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def portable_lock(fp):
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)

    def portable_unlock(fp):
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


class Locker:
    def __enter__(self):
        self.fp = open("./lockfile.lck", 'w+')
        portable_lock(self.fp)

    def __exit__(self, _type, value, tb):
        portable_unlock(self.fp)
        self.fp.close()



class InferenceServer:
    def __init__(
            self,
            model,
            tokenizer,
            max_batch_size=8,
            batch_timeout=0.2,
            idle_timeout=20,
            socket_path=SOCKET_PATH
    ):
        """
        Initializes the inference server.
        :param model: The model to use for inference.
        :param tokenizer: The tokenizer to use for processing inputs.
        :param max_batch_size: Maximum batch size for processing requests.
        :param batch_timeout: Time in seconds to wait for more requests before processing a batch.
        :param idle_timeout: Time in seconds to wait before shutting down due to no clients.
        :param socket_path: The address used for IPC.
        """
        self.model: Mapperatorinator = model
        self.tokenizer: Tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.idle_timeout = idle_timeout
        self.socket_path = socket_path
        self.grouped_requests = {}  # holds pending requests
        self.lock = threading.Lock()
        self.shutdown_flag = threading.Event()
        self.listener = None
        self.connections = 0
        self.batch_counter = 0

    def start(self):
        # Remove stale socket
        try:
            os.unlink(self.socket_path)
        except (FileNotFoundError, OSError):
            pass

        # Start IPC listener
        self.listener = Listener(self.socket_path)
        threading.Thread(target=self._listener_thread, daemon=True).start()
        # Start batcher thread
        threading.Thread(target=self._batch_thread, daemon=True).start()
        # Start idle monitor
        threading.Thread(target=self._idle_monitor, daemon=True).start()

    def stop(self):
        self.shutdown_flag.set()
        try:
            if self.listener is not None:
                self.listener.close()
        except Exception:
            pass
        try:
            os.unlink(self.socket_path)
        except Exception:
            pass

    def _listener_thread(self):
        while not self.shutdown_flag.is_set():
            try:
                conn = self.listener.accept()
                # Handle each client in its own thread
                threading.Thread(target=self._client_handler, args=(conn,), daemon=True).start()
            except (OSError, EOFError) as e:
                print(f"[Listener] Error in accept: {e}")
                time.sleep(1)  # Wait before retrying

    def _client_handler(self, conn):
        with self.lock:
            self.connections += 1
        try:
            with conn:
                while True:
                    try:
                        model_kwargs, generate_kwargs = conn.recv()
                    except _pickle.UnpicklingError:
                        print("UnpicklingError detected! Requesting a retry from the client.")
                        # Tell the client to try again
                        conn.send(RETRY_SIGNAL)
                        # Loop back to conn.recv() to wait for the resent data
                        continue
                    except (EOFError, OSError):
                        break

                    try:
                        generate_key = generation_compatibility_key(generate_kwargs)
                    except TypeError as exc:
                        conn.send({'error': str(exc)})
                        continue

                    # Prepare a response event
                    response_event = threading.Event()
                    batch_size = model_kwargs['inputs'].shape[0]
                    record = StaticServerRequest(
                        model_kwargs=model_kwargs,
                        total_work=batch_size,
                        conn=conn,
                        event=response_event,
                    )

                    # Enqueue request
                    with self.lock:
                        group = self.grouped_requests.get(generate_key)
                        if group is None:
                            group = StaticServerRequestGroup(generate_kwargs=dict(generate_kwargs))
                            self.grouped_requests[generate_key] = group
                        group.requests.append(record)

                    # Wait until batch thread processes it
                    response_event.wait()

                    # Send back result
                    try:
                        conn.send(record.result)
                    except BrokenPipeError:
                        # Client disconnected
                        break
        finally:  # Ensure we always close the connection
            with self.lock:
                self.connections -= 1

    def _batch_thread(self):
        while not self.shutdown_flag.is_set():
            time.sleep(self.batch_timeout)
            with self.lock:
                if not self.grouped_requests:
                    continue
                generate_key = list(self.grouped_requests.keys())[0]
                group = self.grouped_requests[generate_key]
                requests = group.requests

                generate_kwargs: dict = dict(group.generate_kwargs)
                cfg_scale = generate_kwargs.get('cfg_scale', 1.0)
                num_beams = generate_kwargs.get('num_beams', 1)
                batch_multiplier = 2 * num_beams if cfg_scale > 1 else num_beams

                # Grab full or partial requests until BATCH_SIZE is reached or requests is empty
                batch_requests = []
                remaining_batch_size = self.max_batch_size // batch_multiplier
                if remaining_batch_size <= 0:
                    error_message = (
                        "max_batch_size is too small for server batching: "
                        f"max_batch_size={self.max_batch_size}, num_beams={num_beams}, cfg_scale={cfg_scale}."
                    )
                    for request in requests:
                        request.result = {'error': error_message}
                        request.event.set()
                    del self.grouped_requests[generate_key]
                    continue
                while remaining_batch_size > 0 and len(requests) > 0:
                    request = requests.pop(0)
                    req_kwargs = request.model_kwargs
                    req_work_done = request.work_done
                    req_remaining_work = request.remaining_work
                    work = min(req_remaining_work, remaining_batch_size)
                    batch_requests.append((self._cut_model_kwargs(req_kwargs, req_work_done, work), request, work))
                    remaining_batch_size -= work
                    if req_remaining_work > work:
                        # If there is still work left, re-add the record to the queue
                        request.queued_at = time.perf_counter()
                        requests.insert(0, request)

                if not group.requests:
                    del self.grouped_requests[generate_key]
                self.batch_counter += 1
                batch_id = self.batch_counter
                batch_started_at = time.perf_counter()

            try:
                # Collate inputs
                keys = [k for k in batch_requests[0][0].keys() if batch_requests[0][0][k] is not None]
                model_kwargs = {}
                paddings = [0 for _ in range(len(batch_requests))]  # For padding left
                for k in keys:
                    kwargses = [b[0][k] for b in batch_requests]
                    # Pad left if necessary
                    if kwargses[0].dim() > 1:
                        max_len = max(tensor.size(-1) for tensor in kwargses)
                        if k == 'decoder_input_ids':
                            paddings = [max_len - tensor.size(-1) for tensor in kwargses]
                        kwargses = [torch.nn.functional.pad(tensor, (max_len - tensor.size(-1), 0)) for tensor in kwargses]
                    model_kwargs[k] = torch.cat(kwargses, dim=0)

                outputs, stats = model_generate(self.model, self.tokenizer, model_kwargs, generate_kwargs)
                server_batch_size = int(stats.get('batch_size') or model_kwargs['inputs'].shape[0])
                server_batch_elapsed_seconds = float(stats.get('elapsed_seconds', 0.0) or 0.0)
                generated_tokens_per_sample = stats.get('generated_tokens_per_sample', [])
                prompt_tokens_per_sample = stats.get('prompt_tokens_per_sample') or []
                output_tokens_per_sample = stats.get('output_tokens_per_sample', [])

                # Split and dispatch results
                batch_i = 0
                for i, (_, request, work_done) in enumerate(batch_requests):
                    padding = paddings[i]
                    out = outputs[batch_i:batch_i + work_done, padding:]  # Remove padding from the left
                    request_generated_counts = generated_tokens_per_sample[batch_i:batch_i + work_done]
                    request_prompt_counts = prompt_tokens_per_sample[batch_i:batch_i + work_done]
                    request_output_counts = output_tokens_per_sample[batch_i:batch_i + work_done]
                    request_generated_tokens = sum(request_generated_counts)
                    batch_i += work_done
                    request.result = out if request.result is None else torch.cat((request.result, out), dim=0)
                    request.work_done += work_done
                    request.generated_tokens += request_generated_tokens
                    request.prompt_tokens += sum(request_prompt_counts)
                    request.output_tokens += sum(request_output_counts)
                    request.generated_tokens_per_sample.extend(request_generated_counts)
                    request.prompt_tokens_per_sample.extend(request_prompt_counts)
                    request.output_tokens_per_sample.extend(request_output_counts)
                    request.elapsed_seconds += stats.get('elapsed_seconds', 0.0)
                    request.server_batch_ids.append(batch_id)
                    request.server_batch_sizes.append(server_batch_size)
                    request.server_batch_request_counts.append(len(batch_requests))
                    request.server_batch_work_items.append(work_done)
                    request.server_batch_elapsed_seconds.append(server_batch_elapsed_seconds)
                    slice_queue_wait = max(0.0, batch_started_at - request.queued_at)
                    if request.server_first_queue_wait_seconds is None:
                        request.server_first_queue_wait_seconds = max(0.0, batch_started_at - request.enqueued_at)
                    request.server_queue_wait_seconds.append(slice_queue_wait)
                    if request.work_done >= request.total_work:
                        # All work done for this record, signal completion
                        elapsed_seconds = request.elapsed_seconds
                        generated_tokens = request.generated_tokens
                        server_queue_wait_seconds = request.server_queue_wait_seconds
                        detail_stats = dict(stats)
                        for key in (
                            'batch_size',
                            'prompt_tokens',
                            'prompt_tokens_per_sample',
                            'output_tokens',
                            'output_tokens_per_sample',
                            'generated_tokens',
                            'generated_tokens_per_sample',
                            'elapsed_seconds',
                            'tokens_per_second',
                            'precision',
                            'context_type',
                            'num_beams',
                            'cfg_scale',
                            'do_sample',
                        ):
                            detail_stats.pop(key, None)
                        request.result = {
                            'output': request.result,
                            'stats': {
                                'batch_size': request.total_work,
                                'prompt_tokens': request.prompt_tokens,
                                'prompt_tokens_per_sample': request.prompt_tokens_per_sample,
                                'output_tokens': request.output_tokens,
                                'output_tokens_per_sample': request.output_tokens_per_sample,
                                'generated_tokens': generated_tokens,
                                'generated_tokens_per_sample': request.generated_tokens_per_sample,
                                'elapsed_seconds': elapsed_seconds,
                                'tokens_per_second': generated_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0,
                                'precision': stats.get('precision'),
                                'context_type': stats.get('context_type'),
                                'num_beams': stats.get('num_beams'),
                                'cfg_scale': stats.get('cfg_scale'),
                                'do_sample': stats.get('do_sample'),
                                **detail_stats,
                                'server_batching_mode': 'static_ipc',
                                'server_elapsed_seconds_attribution': 'merged_batch_elapsed_replicated_per_request',
                                'server_max_batch_size': self.max_batch_size,
                                'server_batch_timeout_seconds': self.batch_timeout,
                                'server_batch_count': len(request.server_batch_ids),
                                'server_batch_ids': request.server_batch_ids,
                                'server_batch_sizes': request.server_batch_sizes,
                                'server_batch_request_counts': request.server_batch_request_counts,
                                'server_batch_work_items': request.server_batch_work_items,
                                'server_batch_elapsed_seconds': request.server_batch_elapsed_seconds,
                                'server_queue_wait_seconds': server_queue_wait_seconds,
                                'server_first_queue_wait_seconds': request.server_first_queue_wait_seconds,
                                'server_total_queue_wait_seconds': sum(server_queue_wait_seconds),
                                'server_max_queue_wait_seconds': (
                                    max(server_queue_wait_seconds) if server_queue_wait_seconds else 0.0
                                ),
                                'server_rng_policy': 'shared_global',
                                'token_equivalence_status': 'not_checked_shared_server_rng',
                            },
                        }
                        request.event.set()
            except Exception as e:
                print(f"[Batch Thread] Error processing batch: {e}")
                traceback.print_exc()
                # Signal all requests in this batch to retry
                for _, request, _ in batch_requests:
                    request.result = RETRY_SIGNAL
                    request.event.set()  # Signal completion
            finally:
                torch.cuda.empty_cache()  # Clear any cached memory, otherwise will definitely run out of memory if multiple batch sizes are used

    def _cut_model_kwargs(self, model_kwargs, start, length):
        """Cuts the model_kwargs tensors to the specified range."""
        return {k: v[start:start + length] if isinstance(v, torch.Tensor) else v for k, v in model_kwargs.items()}

    def _idle_monitor(self):
        last_activity = time.time()
        while not self.shutdown_flag.is_set():
            time.sleep(self.idle_timeout / 2)
            with self.lock:
                if self.connections > 0:
                    last_activity = time.time()
            if time.time() - last_activity > self.idle_timeout:
                # No requests for a while: shutdown
                self.shutdown_flag.set()
                try:
                    self.listener.close()
                    os.unlink(self.socket_path)
                except Exception:
                    pass


class InferenceClient:
    def __init__(
            self,
            model_loader,
            tokenizer_loader,
            max_batch_size=8,
            batch_timeout=0.2,
            idle_timeout=20,
            server_thread_daemon=False,
            socket_path=SOCKET_PATH,
            allow_auto_start=True,
            connect_timeout=60.0,
            request_timeout=None,
    ):
        """
        Initializes the inference client. Automatically starts the inference server if it is not running.
        :param model_loader: Function to load the model.
        :param tokenizer_loader: Function to load the tokenizer.
        :param max_batch_size: Maximum batch size for processing requests.
        :param batch_timeout: Time in seconds to wait for more requests before processing a batch.
        :param idle_timeout: Time in seconds to wait before shutting down due to no clients.
        :param server_thread_daemon: Whether the auto-started background server thread should be daemonized.
        :param socket_path: The address used for IPC.
        """
        self.model_loader = model_loader
        self.tokenizer_loader = tokenizer_loader
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.idle_timeout = idle_timeout
        self.server_thread_daemon = server_thread_daemon
        self.socket_path = socket_path
        self.allow_auto_start = allow_auto_start
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self.conn = None
        self.last_generation_stats = None
        self._server_start_error = None
        self._server = None
        self._server_thread = None

    def __enter__(self):
        self._reconnect()
        return self

    def _reconnect(self):
        with Locker():
            try:
                self.conn = Client(self.socket_path)
            except FileNotFoundError:
                if not self.allow_auto_start:
                    raise ConnectionError(
                        f"Inference server socket does not exist and auto-start is disabled: {self.socket_path}"
                    )
                # No server: start one
                self._start_server_thread()
                # Wait for server socket to appear
                started_at = time.perf_counter()
                while not os.path.exists(self.socket_path):
                    if self._server_start_error is not None:
                        raise RuntimeError(
                            f"Inference server failed to start for socket {self.socket_path}"
                        ) from self._server_start_error
                    if self._server_thread is not None and not self._server_thread.is_alive():
                        raise RuntimeError(
                            f"Inference server thread exited before creating socket {self.socket_path}"
                        )
                    if self.connect_timeout is not None and time.perf_counter() - started_at > self.connect_timeout:
                        raise TimeoutError(
                            f"Timed out waiting for inference server socket after "
                            f"{self.connect_timeout:.1f}s: {self.socket_path}"
                        )
                    time.sleep(0.1)
                self.conn = Client(self.socket_path)

    def _start_server_thread(self):
        if self._server_thread is not None and self._server_thread.is_alive():
            return

        self._server_thread = threading.Thread(
            target=self._start_server,
            args=(self.model_loader, self.tokenizer_loader),
            daemon=self.server_thread_daemon,
        )
        self._server_thread.start()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if self.conn:
            self.conn.close()

    def _start_server(self, model_loader, tokenizer_loader):
        try:
            # Load model inside server process
            model = model_loader()
            tokenizer = tokenizer_loader()
            server = InferenceServer(
                model,
                tokenizer,
                max_batch_size=self.max_batch_size,
                batch_timeout=self.batch_timeout,
                idle_timeout=self.idle_timeout,
                socket_path=self.socket_path
            )
            self._server = server
            server.start()
            # Block until shutdown
            while not server.shutdown_flag.is_set():
                time.sleep(1)
        except BaseException as exc:
            self._server_start_error = exc
            raise
        finally:
            self._server = None

    def generate(self, model_kwargs, generate_kwargs, max_retries=3):
        attempts = 0
        while attempts < max_retries:
            # Send request and wait for response
            try:
                if self.conn is None:
                    self._reconnect()
                self.conn.send((model_kwargs, generate_kwargs))
                if self.request_timeout is not None and not self.conn.poll(self.request_timeout):
                    raise TimeoutError(
                        f"Timed out waiting for inference server response after "
                        f"{self.request_timeout:.1f}s: {self.socket_path}"
                    )
                result = self.conn.recv()
            except (EOFError, OSError):
                print("Connection error, attempting to reconnect...")
                self._reconnect()
                attempts += 1
                continue

            if result == RETRY_SIGNAL:
                print("Retrying request due to Error.")
                attempts += 1
                continue
            if isinstance(result, dict) and 'error' in result:
                raise RuntimeError(str(result['error']))
            else:
                if isinstance(result, dict) and 'output' in result:
                    self.last_generation_stats = result.get('stats')
                    return result['output']
                self.last_generation_stats = None
                return result

        raise RuntimeError(f"Failed to get a valid response after {max_retries} attempts.")

    def ensure_server(self, timeout=None):
        """Ensure the background inference server is running.

        This is useful when an external owner (e.g., a web UI) wants to start the
        server once and keep it alive independently of any per-job client
        connections.
        """
        with Locker():
            if not os.path.exists(self.socket_path):
                self._start_server_thread()

        # Wait for server socket to appear.
        started_at = time.perf_counter()
        while not os.path.exists(self.socket_path):
            if self._server_start_error is not None:
                raise RuntimeError(
                    f"Inference server failed to start for socket {self.socket_path}"
                ) from self._server_start_error
            if self._server_thread is not None and not self._server_thread.is_alive():
                raise RuntimeError(
                    f"Inference server thread exited before creating socket {self.socket_path}"
                )
            wait_timeout = self.connect_timeout if timeout is None else timeout
            if wait_timeout is not None and time.perf_counter() - started_at > wait_timeout:
                raise TimeoutError(
                    f"Timed out waiting for inference server socket after "
                    f"{wait_timeout:.1f}s: {self.socket_path}"
                )
            time.sleep(0.1)

    def shutdown_server(self, join_timeout=5.0):
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            finally:
                self.conn = None

        if self._server is not None:
            self._server.stop()

        server_thread = self._server_thread
        if server_thread is not None and server_thread.is_alive():
            server_thread.join(timeout=join_timeout)


if __name__ == "__main__":
    ckpt_path_str = "OliBomby/Mapperatorinator-v32"

    # Example usage
    def model_loader():
        model = Mapperatorinator.from_pretrained(ckpt_path_str)
        model.generation_config.disable_compile = True
        model.eval()
        model.to('cuda')
        return model

    def tokenizer_loader():
        return Tokenizer.from_pretrained(ckpt_path_str)

    client = InferenceClient(model_loader, tokenizer_loader)
    tokenizer = Tokenizer.from_pretrained(ckpt_path_str)

    # Example model_kwargs and generate_kwargs
    model_kwargs = {
        'inputs': torch.rand((1, 524160)),  # Example input
        'difficulty': torch.tensor([7.]),
        'mapper_idx': torch.tensor([-1]),
        'song_position': torch.tensor([[0., .112]]),
    }
    generate_kwargs = {
        'num_beams': 1,
        'max_length': 2048,
        'do_sample': True,
        'cfg_scale': 1.0,
        'top_p': 0.9,
        'top_k': 0,
        'pad_token_id': tokenizer.pad_id,
        'timeshift_bias': 0,
        'types_first': False,
        'temperature': 0.9,
        'timing_temperature': 0.0,
        'mania_column_temperature': 0.7,
        'taiko_hit_temperature': 0.7,
        'lookback_time': 0,
        'lookahead_time': 3000,
    }

    result = client.generate(model_kwargs, generate_kwargs)
    events = [tokenizer.decode(t) if t > 10 else t for t in result[0].numpy()]
    print(events)  # Process the result as needed
    print(client.last_generation_stats)
