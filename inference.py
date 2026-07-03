import logging
import multiprocessing
import os
import socket
import subprocess
import sys

import utils.excepthook  # noqa
import uuid
from functools import reduce
from pathlib import Path
import random

import hydra
import torch
from accelerate.utils import set_seed
from omegaconf import OmegaConf, DictConfig
from slider import Beatmap
from transformers.utils import cached_file, is_flash_attn_2_available
from importlib import metadata
from packaging.version import Version

import osu_diffusion
from utils import routed_pickle
from config import InferenceConfig
from diffusion_pipeline import DiffisionPipeline
from osuT5.osuT5.config import TrainConfig
from osuT5.osuT5.dataset.data_utils import events_of_type, TIMING_TYPES, merge_events
from osuT5.osuT5.inference import Preprocessor, Processor, Postprocessor, BeatmapConfig, GenerationConfig, \
    generation_config_from_beatmap, beatmap_config_from_beatmap, background_line
from osuT5.osuT5.inference.profiler import InferenceProfiler
from osuT5.osuT5.inference.server import InferenceClient
from osuT5.osuT5.inference.super_timing_generator import SuperTimingGenerator
from osuT5.osuT5.model import Mapperatorinator
from osuT5.osuT5.tokenizer import ContextType
from osuT5.osuT5.utils import load_model_loaders, resolve_compatible_lora_path, resolve_model_checkpoint_path, get_model_checkpoint_subfolder
from osu_diffusion import DiT_models
from osu_diffusion.config import DiffusionTrainConfig


def get_default_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_profile_runtime_metadata() -> dict:
    metadata = {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "hf_home": os.environ.get("HF_HOME"),
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE"),
        "xdg_cache_home": os.environ.get("XDG_CACHE_HOME"),
        "tmpdir": os.environ.get("TMPDIR"),
    }
    if torch.cuda.is_available():
        metadata["cuda_device_name"] = torch.cuda.get_device_name()
        metadata["cuda_device_capability"] = list(torch.cuda.get_device_capability())

    repo_root = Path(__file__).resolve().parent
    for key, command in {
        "git_commit": ["git", "rev-parse", "HEAD"],
        "git_branch": ["git", "branch", "--show-current"],
    }.items():
        try:
            result = subprocess.run(
                command,
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            continue
        metadata[key] = result.stdout.strip()

    return metadata


def assert_package_version(package_name: str, required_version: str):
    required_version = Version(required_version)
    try:
        installed_version = Version(metadata.version(package_name))
    except metadata.PackageNotFoundError as e:
        raise RuntimeError(
            f"Missing dependency: '{package_name}' is not installed. "
            "Please install requirements.txt."
        ) from e

    assert installed_version >= required_version, (
        f"{package_name}>={required_version} is required, but {installed_version} is installed. "
        f"Please install requirements.txt."
    )


def assert_package_versions():
    assert_package_version("transformers", "4.57.3")


def setup_inference_environment(seed: int):
    assert_package_versions()
    multiprocessing.set_start_method('spawn', force=True)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision('high')
    set_seed(seed)


def compile_device_and_seed(args: InferenceConfig, verbose=True):
    message = None
    if args.device == "auto":
        if torch.cuda.is_available():
            message = "Using CUDA for inference (auto-selected)."
            args.device = "cuda"
        elif torch.mps.is_available():
            message = "Using MPS for inference (auto-selected)."
            args.device = "mps"
        else:
            message = "Using CPU for inference (auto-selected fallback)."
            args.device = "cpu"
    elif args.device != "cpu":
        if args.device == "cuda":
            if not torch.cuda.is_available():
                message = "CUDA is not available. Falling back to CPU."
                args.device = "cpu"
        elif args.device == "mps":
            if not torch.mps.is_available():
                message = "MPS is not available. Falling back to CPU."
                args.device = "cpu"
        else:
            message = f"Requested device '{args.device}' not available. Falling back to CPU."
            args.device = "cpu"

    if verbose and message is not None:
        print(message)

    message = None
    if args.attn_implementation == "auto":
        if args.precision in ("bf16", "fp16") and args.device == "cuda" and is_flash_attn_2_available():
            message = "Using Flash Attention for attention (auto-selected)."
            args.attn_implementation = "flash_attention_2"
        else:
            message = "Using SDPA for attention (auto-selected)."
            args.attn_implementation = "sdpa"
    elif args.attn_implementation == "flash_attention_2":
        if not is_flash_attn_2_available():
            message = "Flash Attention is not available. Falling back to SDPA."
            args.attn_implementation = "sdpa"
        elif args.precision not in ("bf16", "fp16") or args.device != "cuda":
            message = "Flash Attention requires bf16/fp16 precision and CUDA device. Falling back to SDPA."
            args.attn_implementation = "sdpa"

    if verbose and message is not None:
        print(message)

    if args.seed is None:
        args.seed = random.randint(0, 2 ** 16)
        if verbose:
            print(f"Random seed: {args.seed}")


def compile_paths(args: InferenceConfig):
    # Convert paths to Path objects for easier manipulation
    beatmap_path = Path(args.beatmap_path) if args.beatmap_path else None
    output_path = Path(args.output_path) if args.output_path else None
    audio_path = Path(args.audio_path) if args.audio_path else None

    # Case 1: Beatmap path is provided - autofill audio and output
    if beatmap_path:
        if not beatmap_path.exists():
            raise ValueError(f"Beatmap file not found: {beatmap_path}")
        elif not beatmap_path.suffix.lower() == '.osu':
            raise ValueError(f"Beatmap file must have .osu extension: {beatmap_path}")

        try:
            beatmap = Beatmap.from_path(beatmap_path)

            # Autofill audio path if empty
            if not audio_path and beatmap.audio_filename:
                audio_path = beatmap_path.parent / beatmap.audio_filename

            # Autofill output path if empty
            if not output_path:
                output_path = beatmap_path.parent

        except Exception as e:
            raise ValueError(f"Error reading beatmap file: {e}")

    # Case 2: Audio path is provided but no output path - autofill output
    elif audio_path and audio_path.exists() and not output_path:
        output_path = audio_path.parent

    # Validate all paths
    valid_audio_extensions = {'.mp3', '.wav', '.ogg', '.m4a', '.flac'}
    if not audio_path:
        raise ValueError("Audio file path is required.")
    elif not audio_path.exists():
        raise ValueError(f"Audio file not found: {audio_path}")
    elif audio_path.suffix.lower() not in valid_audio_extensions:
        raise ValueError(
            f"Audio file must have one of the following extensions: {', '.join(valid_audio_extensions)}: {audio_path}")

    # Update args
    args.audio_path = str(audio_path) if audio_path else ""
    args.output_path = str(output_path) if output_path else ""
    args.beatmap_path = str(beatmap_path) if beatmap_path else ""


def compile_args_from_beatmap(args: InferenceConfig, verbose=True):
    beatmap_path = Path(args.beatmap_path)
    beatmap = Beatmap.from_path(beatmap_path)

    if beatmap.mode not in args.train.data.gamemodes and (any(
            c in [ContextType.MAP, ContextType.GD, ContextType.NO_HS] for c in args.in_context) or args.add_to_beatmap):
        raise ValueError(
            f"Reference beatmap mode {beatmap.mode} is not supported by the model. Supported modes: {args.train.data.gamemodes}")

    if verbose:
        print(f"Using metadata from beatmap: {beatmap.display_name}")
    generation_config = generation_config_from_beatmap(beatmap, beatmap_path)
    beatmap_config = beatmap_config_from_beatmap(beatmap)

    beatmap_args = {
        "gamemode": generation_config.gamemode,
        "beatmap_id": generation_config.beatmap_id,
        "difficulty": generation_config.difficulty,
        "mapper_id": generation_config.mapper_id,
        "descriptors": generation_config.descriptors,
        "hp_drain_rate": generation_config.hp_drain_rate,
        "circle_size": generation_config.circle_size,
        "overall_difficulty": generation_config.overall_difficulty,
        "approach_rate": generation_config.approach_rate,
        "slider_multiplier": generation_config.slider_multiplier,
        "slider_tick_rate": generation_config.slider_tick_rate,
        "hitsounded": generation_config.hitsounded,
        "keycount": generation_config.keycount,
        "hold_note_ratio": generation_config.hold_note_ratio,
        "scroll_speed_ratio": generation_config.scroll_speed_ratio,
        "bpm": beatmap_config.bpm,
        "offset": beatmap_config.offset,
        "title": beatmap_config.title,
        "title_unicode": beatmap_config.title_unicode,
        "artist": beatmap_config.artist,
        "artist_unicode": beatmap_config.artist_unicode,
        "creator": beatmap_config.creator,
        "version": beatmap_config.version,
        "source": beatmap_config.source,
        "background": str(beatmap_path.parent / beatmap.background) if beatmap.background else None,
        "preview_time": beatmap_config.preview_time,
    }

    for key, value in beatmap_args.items():
        if getattr(args, key) is None and value is not None:
            setattr(args, key, value)
            if verbose:
                print(f"Using beatmap {key} {value}")


def compile_default_args(args: InferenceConfig, verbose=True):
    # Populate fair defaults for any inherited args that need to be filled
    default_args = {
        "gamemode": 0,
        "hitsounded": True,
        "keycount": 4,
        "hp_drain_rate": 5,
        "circle_size": 4,
        "overall_difficulty": 8,
        "approach_rate": 9,
        "slider_multiplier": 1.4,
        "slider_tick_rate": 1,
        "bpm": 120,
        "offset": 0,
        "title": "Unknown Title",
        "artist": "Unknown Artist",
        "creator": "Mapperatorinator",
        "version": "Mapperatorinator",
        "source": "",
        "preview_time": -1,
    }

    for key, value in default_args.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
            if verbose:
                print(f"Using default {key} {value}")


def get_tags_dict(args: DictConfig | InferenceConfig):
    return dict(
        model=args.model_path,
        lora=args.lora_path,
        lookback=args.lookback,
        lookahead=args.lookahead,
        beatmap_id=args.beatmap_id,
        difficulty=args.difficulty,
        mapper_id=args.mapper_id,
        year=args.year,
        hitsounded=args.hitsounded,
        hold_note_ratio=args.hold_note_ratio,
        scroll_speed_ratio=args.scroll_speed_ratio,
        descriptors=f"\"[{','.join(args.descriptors)}]\"" if args.descriptors else None,
        negative_descriptors=f"\"[{','.join(args.negative_descriptors)}]\"" if args.negative_descriptors else None,
        timing_leniency=args.timing_leniency,
        seed=args.seed,
        add_to_beatmap=args.add_to_beatmap,
        start_time=args.start_time,
        end_time=args.end_time,
        in_context=f"[{','.join(ctx.value.upper() if isinstance(ctx, ContextType) else ctx for ctx in args.in_context)}]",
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        timing_temperature=args.timing_temperature,
        mania_column_temperature=args.mania_column_temperature,
        taiko_hit_temperature=args.taiko_hit_temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        parallel=args.parallel,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        super_timing=args.super_timing,
        timer_num_beams=args.timer_num_beams,
        timer_bpm_threshold=args.timer_bpm_threshold,
        timer_cfg_scale=args.timer_cfg_scale,
        timer_iterations=args.timer_iterations,
        generate_positions=args.generate_positions,
        diff_cfg_scale=args.diff_cfg_scale,
        max_seq_len=args.max_seq_len,
        overlap_buffer=args.overlap_buffer,
    )


def compile_derived_args(args: InferenceConfig):
    # Any args that can be derived from other args
    derived_args = {
        "title_unicode": args.title,
        "artist_unicode": args.artist,
    }

    for key, value in derived_args.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if args.tags is None:
        # Create tags that describes args
        tags = get_tags_dict(args)
        # Filter to all non-default values
        defaults = get_tags_dict(OmegaConf.load("configs/inference/default.yaml"))
        tags = {k: v for k, v in tags.items() if v != defaults[k]}
        # To string separated by spaces
        args.tags = " ".join(f"{k}={v}" for k, v in tags.items())


def validate_reserved_runtime_flags(args: InferenceConfig):
    if args.inference_decode_session_runtime:
        if args.use_server:
            raise ValueError("inference_decode_session_runtime requires use_server=false.")
        if args.parallel:
            raise ValueError("inference_decode_session_runtime currently supports sequential inference only.")
        if args.cfg_scale != 1.0:
            raise ValueError("inference_decode_session_runtime currently requires cfg_scale=1.0.")
        if args.num_beams != 1:
            raise ValueError("inference_decode_session_runtime currently requires num_beams=1.")
        if not args.inference_active_prefix_decode_loop:
            raise ValueError("inference_decode_session_runtime requires inference_active_prefix_decode_loop=true.")
        if not args.inference_active_prefix_decode_cuda_graph:
            raise ValueError("inference_decode_session_runtime requires inference_active_prefix_decode_cuda_graph=true.")
        if not args.inference_decode_session_cuda_graph:
            raise ValueError("inference_decode_session_runtime requires inference_decode_session_cuda_graph=true.")
    if args.inference_decode_session_cuda_graph:
        if not args.inference_decode_session_runtime:
            raise ValueError("inference_decode_session_cuda_graph requires inference_decode_session_runtime=true.")
    if args.inference_decode_session_chunk_size != 1:
        raise NotImplementedError(
            "inference_decode_session_chunk_size is reserved and must remain 1 until chunked DecodeSession "
            "generation preserves exact RNG/token behavior."
        )
    if args.inference_native_q1_rope_cache_self_attention and not args.inference_native_q1_self_attention:
        raise ValueError(
            "inference_native_q1_rope_cache_self_attention requires inference_native_q1_self_attention=true."
        )
    if args.inference_native_q1_self_attention:
        if not args.inference_native_decode_kernels:
            raise ValueError("inference_native_q1_self_attention requires inference_native_decode_kernels=true.")
        if args.precision != "fp32":
            raise ValueError("inference_native_q1_self_attention currently requires precision=fp32.")
        if args.attn_implementation != "sdpa":
            raise ValueError("inference_native_q1_self_attention currently requires attn_implementation=sdpa.")
        if args.use_server:
            raise ValueError("inference_native_q1_self_attention requires use_server=false.")
        if args.parallel:
            raise ValueError("inference_native_q1_self_attention currently supports sequential inference only.")
        if args.cfg_scale != 1.0:
            raise ValueError("inference_native_q1_self_attention currently requires cfg_scale=1.0.")
        if args.num_beams != 1:
            raise ValueError("inference_native_q1_self_attention currently requires num_beams=1.")
        if not args.inference_active_prefix_decode_loop:
            raise ValueError("inference_native_q1_self_attention requires inference_active_prefix_decode_loop=true.")
        if not args.inference_active_prefix_decode_cuda_graph:
            raise ValueError("inference_native_q1_self_attention requires inference_active_prefix_decode_cuda_graph=true.")
        if not args.inference_decode_session_runtime:
            raise ValueError("inference_native_q1_self_attention requires inference_decode_session_runtime=true.")
        if not args.inference_decode_session_cuda_graph:
            raise ValueError("inference_native_q1_self_attention requires inference_decode_session_cuda_graph=true.")
    if args.inference_native_one_token_linear:
        if not args.inference_native_decode_kernels:
            raise ValueError("inference_native_one_token_linear requires inference_native_decode_kernels=true.")
        if args.precision != "fp32":
            raise ValueError("inference_native_one_token_linear currently requires precision=fp32.")
        if args.attn_implementation != "sdpa":
            raise ValueError("inference_native_one_token_linear currently requires attn_implementation=sdpa.")
        if args.use_server:
            raise ValueError("inference_native_one_token_linear requires use_server=false.")
        if args.parallel:
            raise ValueError("inference_native_one_token_linear currently supports sequential inference only.")
        if args.cfg_scale != 1.0:
            raise ValueError("inference_native_one_token_linear currently requires cfg_scale=1.0.")
        if args.num_beams != 1:
            raise ValueError("inference_native_one_token_linear currently requires num_beams=1.")
        if not args.inference_active_prefix_decode_loop:
            raise ValueError("inference_native_one_token_linear requires inference_active_prefix_decode_loop=true.")
        if not args.inference_active_prefix_decode_cuda_graph:
            raise ValueError("inference_native_one_token_linear requires inference_active_prefix_decode_cuda_graph=true.")
        if not args.inference_decode_session_runtime:
            raise ValueError("inference_native_one_token_linear requires inference_decode_session_runtime=true.")
        if not args.inference_decode_session_cuda_graph:
            raise ValueError("inference_native_one_token_linear requires inference_decode_session_cuda_graph=true.")
    if (
            args.inference_native_decode_kernels
            and not args.inference_native_q1_self_attention
            and not args.inference_native_one_token_linear
    ):
        raise NotImplementedError(
            "inference_native_decode_kernels currently requires an explicitly selected native subflag."
        )


def compile_args(args: InferenceConfig, verbose=True):
    """Validates and populates missing args."""
    validate_reserved_runtime_flags(args)
    compile_device_and_seed(args, verbose=verbose)
    compile_paths(args)

    if args.beatmap_path:
        compile_args_from_beatmap(args, verbose=verbose)
    else:
        compile_default_args(args, verbose=verbose)

    compile_derived_args(args)


def get_config(args: InferenceConfig):
    # Set defaults for generation config that does not allow an unknown value
    return GenerationConfig(
        gamemode=args.gamemode,
        beatmap_id=args.beatmap_id,
        difficulty=args.difficulty,
        mapper_id=args.mapper_id,
        year=args.year,
        hitsounded=args.hitsounded,
        hp_drain_rate=args.hp_drain_rate,
        circle_size=args.circle_size,
        overall_difficulty=args.overall_difficulty,
        approach_rate=args.approach_rate,
        slider_multiplier=args.slider_multiplier,
        slider_tick_rate=args.slider_tick_rate,
        keycount=args.keycount,
        hold_note_ratio=args.hold_note_ratio,
        scroll_speed_ratio=args.scroll_speed_ratio,
        descriptors=args.descriptors,
        negative_descriptors=args.negative_descriptors,
    ), BeatmapConfig(
        title=args.title,
        title_unicode=args.title_unicode,
        artist=args.artist,
        artist_unicode=args.artist_unicode,
        audio_filename=Path(args.audio_path).name,
        hp_drain_rate=args.hp_drain_rate,
        circle_size=(args.keycount if args.gamemode == 3 else args.circle_size) or 4,
        overall_difficulty=args.overall_difficulty,
        approach_rate=args.approach_rate,
        slider_multiplier=args.slider_multiplier,
        slider_tick_rate=args.slider_tick_rate,
        creator=args.creator,
        version=args.version,
        source=args.source,
        tags=args.tags,
        background_line=background_line(args.background),
        preview_time=args.preview_time,
        bpm=args.bpm,
        offset=args.offset,
        mode=args.gamemode,
    )


def supports_explicit_timing_output(args: InferenceConfig) -> bool:
    return any(ContextType.TIMING in context_type["out"] for context_type in args.train.data.context_types)


def should_generate_timing_context(args: InferenceConfig, output_type: list[ContextType]) -> bool:
    has_empty_or_none_context = len(args.in_context) == 0 or ContextType.NONE in args.in_context
    return has_empty_or_none_context and supports_explicit_timing_output(args) and any(
        context_type in output_type for context_type in [ContextType.TIMING, ContextType.MAP]
    )


def should_load_separate_timing_model(args: InferenceConfig, output_type: list[ContextType] | None = None) -> bool:
    output_type = args.output_type if output_type is None else output_type
    needs_generated_timing = (
        args.super_timing and (len(args.in_context) == 0 or ContextType.NONE in args.in_context)
    ) or should_generate_timing_context(args, output_type)

    if not needs_generated_timing:
        return False

    current_ckpt_path, current_subfolder = resolve_model_checkpoint_path(
        args.model_path,
        gamemode=args.gamemode,
        auto_select_gamemode_model=args.auto_select_gamemode_model,
    )
    base_ckpt_path, base_subfolder = resolve_model_checkpoint_path(
        args.model_path,
        gamemode=args.gamemode,
        auto_select_gamemode_model=False,
    )

    return current_ckpt_path != base_ckpt_path or current_subfolder != base_subfolder


def generate(
        args: InferenceConfig,
        *,
        audio_path: str = None,
        beatmap_path: str = None,
        output_path: str = None,
        generation_config: GenerationConfig,
        beatmap_config: BeatmapConfig,
        model: Mapperatorinator | InferenceClient,
        tokenizer,
        timing_model: Mapperatorinator | InferenceClient | None = None,
        timing_tokenizer=None,
        diff_model=None,
        diff_tokenizer=None,
        refine_model=None,
        verbose=True,
        logger=None,
        profiler: InferenceProfiler | None = None,
):
    audio_path = args.audio_path if audio_path is None else audio_path
    beatmap_path = args.beatmap_path if beatmap_path is None else beatmap_path
    output_path = args.output_path if output_path is None else output_path
    logger = get_default_logger() if logger is None else logger.getChild(__name__)
    profiler = profiler or InferenceProfiler.from_args(args)
    profile_metadata = {
        "audio_path": audio_path,
        "beatmap_path": beatmap_path,
        "output_path": output_path,
        "model_path": args.model_path,
        "device": args.device,
        "precision": args.precision,
        "attn_implementation": args.attn_implementation,
        "use_server": args.use_server,
        "parallel": args.parallel,
        "max_batch_size": args.max_batch_size,
        "inference_generation_compile": args.inference_generation_compile,
        "inference_active_prefix_decode_loop": args.inference_active_prefix_decode_loop,
        "inference_active_prefix_decode_bucket_size": args.inference_active_prefix_decode_bucket_size,
        "inference_active_prefix_decode_cuda_graph": args.inference_active_prefix_decode_cuda_graph,
        "inference_active_prefix_decode_cuda_graph_warmup": args.inference_active_prefix_decode_cuda_graph_warmup,
        "inference_active_prefix_decode_cuda_graph_min_decode_steps": (
            args.inference_active_prefix_decode_cuda_graph_min_decode_steps
        ),
        "inference_stateful_monotonic_logits_processor": args.inference_stateful_monotonic_logits_processor,
        "inference_q1_bmm_cross_attention": args.inference_q1_bmm_cross_attention,
        "inference_decode_session_runtime": args.inference_decode_session_runtime,
        "inference_decode_session_cuda_graph": args.inference_decode_session_cuda_graph,
        "inference_decode_session_chunk_size": args.inference_decode_session_chunk_size,
        "inference_native_decode_kernels": args.inference_native_decode_kernels,
        "inference_native_q1_self_attention": args.inference_native_q1_self_attention,
        "inference_native_q1_rope_cache_self_attention": args.inference_native_q1_rope_cache_self_attention,
        "inference_native_one_token_linear": args.inference_native_one_token_linear,
        "profile_record_token_ids": args.profile_record_token_ids,
        "profile_sync_cuda": args.profile_sync_cuda,
        "profile_torch_generation": args.profile_torch_generation,
        "profile_nvtx_generation_ranges": args.profile_nvtx_generation_ranges,
        "profile_generation_detail_ranges": args.profile_generation_detail_ranges,
        "profile_sdpa_backend": args.profile_sdpa_backend,
        "seed": args.seed,
        "temperature": args.temperature,
        "timing_temperature": args.timing_temperature,
        "mania_column_temperature": args.mania_column_temperature,
        "taiko_hit_temperature": args.taiko_hit_temperature,
        "timeshift_bias": args.timeshift_bias,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": args.do_sample,
        "num_beams": args.num_beams,
        "cfg_scale": args.cfg_scale,
        "lookback": args.lookback,
        "lookahead": args.lookahead,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "in_context": [context.value for context in args.in_context],
        "output_type": [context.value for context in args.output_type],
    }
    if profiler.enabled:
        profile_metadata.update(get_profile_runtime_metadata())
    profiler.set_metadata(**profile_metadata)

    # Do some validation
    with profiler.stage("validate_inputs"):
        if not Path(audio_path).exists() or not Path(audio_path).is_file():
            raise FileNotFoundError(f"Provided audio file path does not exist: {audio_path}")
        if beatmap_path:
            beatmap_path_obj = Path(beatmap_path)
            if not beatmap_path_obj.exists() or not beatmap_path_obj.is_file():
                raise FileNotFoundError(f"Provided beatmap file path does not exist: {beatmap_path}")
            # Validate beatmap file type
            if beatmap_path_obj.suffix.lower() != '.osu':
                raise ValueError(f"Beatmap file must have .osu extension: {beatmap_path}")
        if (output_path is None or output_path == "") and (not args.add_to_beatmap or not args.overwrite_reference_beatmap or args.export_osz):
            raise ValueError("Output path is required.")

    with profiler.stage("setup_processors"):
        preprocessor = Preprocessor(args, parallel=args.parallel)
        processor = Processor(args, model, tokenizer, profiler=profiler)
        postprocessor = Postprocessor(args, logger=logger)

    with profiler.stage("audio_load"):
        audio = preprocessor.load(audio_path)
    with profiler.stage("audio_segment"):
        sequences = preprocessor.segment(audio)
    profiler.set_metadata(
        audio_samples=len(audio),
        song_length_ms=sequences[2],
        sequence_count=len(sequences[0]),
        milliseconds_per_sequence=preprocessor.miliseconds_per_sequence,
        milliseconds_per_stride=preprocessor.miliseconds_per_stride,
    )
    extra_in_context = {}
    output_type = args.output_type.copy()
    timing_model = model if timing_model is None else timing_model
    timing_tokenizer = tokenizer if timing_tokenizer is None else timing_tokenizer

    # Auto generate timing if not provided in in_context and required for the model and this output_type
    timing_events, timing_times, timing = None, None, None
    if args.super_timing and (len(args.in_context) == 0 or ContextType.NONE in args.in_context):
        with profiler.stage("super_timing_generation"):
            super_timing_generator = SuperTimingGenerator(args, timing_model, timing_tokenizer, profiler=profiler)
            timing_events, timing_times = super_timing_generator.generate(audio, generation_config, verbose=verbose)
        with profiler.stage("super_timing_postprocess"):
            timing = postprocessor.generate_timing(timing_events)
        extra_in_context[ContextType.TIMING] = timing
        if ContextType.TIMING in output_type:
            output_type.remove(ContextType.TIMING)
    elif should_generate_timing_context(args, output_type):
        # Generate timing context with the base model and reuse it for the main generation pass.
        with profiler.stage("timing_context_generation"):
            timing_processor = Processor(args, timing_model, timing_tokenizer, profiler=profiler)
            timing_events, timing_times = timing_processor.generate(
                sequences=sequences,
                generation_config=generation_config,
                in_context=[ContextType.NONE],
                out_context=[ContextType.TIMING],
                beatmap_path=beatmap_path,
                verbose=verbose,
                profile_label="timing_context",
            )[0]
        with profiler.stage("timing_context_postprocess"):
            timing_events, timing_times = events_of_type(timing_events, timing_times, TIMING_TYPES)
            timing = postprocessor.generate_timing(timing_events)
        extra_in_context[ContextType.TIMING] = timing
        if ContextType.TIMING in output_type:
            output_type.remove(ContextType.TIMING)
    elif ContextType.TIMING in args.in_context or (
            args.train.data.add_timing and any(t in args.in_context for t in [ContextType.GD, ContextType.NO_HS])):
        # Exact timing is provided in the other beatmap, so we don't need to generate it
        with profiler.stage("load_reference_timing"):
            timing = [tp for tp in Beatmap.from_path(Path(beatmap_path)).timing_points if tp.parent is None]

    # Generate beatmap
    if len(output_type) > 0:
        with profiler.stage("main_generation", output_type=[context.value for context in output_type]):
            result = processor.generate(
                sequences=sequences,
                generation_config=generation_config,
                in_context=args.in_context,
                out_context=output_type,
                beatmap_path=beatmap_path,
                extra_in_context=extra_in_context,
                verbose=verbose,
                profile_label="main_generation",
            )

        with profiler.stage("merge_generated_events"):
            events, _ = reduce(merge_events, result)

        if timing is None and (ContextType.TIMING in args.output_type or args.train.data.add_timing):
            with profiler.stage("derive_timing_from_generated_events"):
                timing = postprocessor.generate_timing(events)

        # Resnap timing events
        if args.resnap_events and timing is not None:
            with profiler.stage("resnap_events"):
                events = postprocessor.resnap_events(events, timing)
    else:
        events = timing_events

    # Generate positions with diffusion
    if args.generate_positions and args.gamemode in [0, 2] and ContextType.MAP in output_type:
        with profiler.stage("diffusion_position_generation"):
            diffusion_pipeline = DiffisionPipeline(args, diff_model, diff_tokenizer, refine_model)
            events = diffusion_pipeline.generate(
                events=events,
                generation_config=generation_config,
                timing=timing,
                verbose=verbose,
            )

    with profiler.stage("postprocess_generate_osu"):
        result = postprocessor.generate(
            events=events,
            beatmap_config=beatmap_config,
            timing=timing,
        )

    if args.add_to_beatmap:
        with profiler.stage("merge_with_reference_beatmap"):
            result = postprocessor.add_to_beatmap(result, beatmap_path)
        if verbose:
            logger.info(f"Merged generated content with reference beatmap")

    if args.add_to_beatmap and args.overwrite_reference_beatmap:
        output_osu_path = Path(beatmap_path)
    else:
        # noinspection PyTypeChecker
        output_osu_path = Path(output_path) / f"beatmap{str(uuid.uuid4().hex)}.osu"

    if args.export_osz:
        # noinspection PyTypeChecker
        result_path = Path(output_path) / f"beatmap{str(uuid.uuid4().hex)}.osz"
        with profiler.stage("write_osz"):
            postprocessor.export_osz(result_path, result, output_osu_path.name, audio_path, args.background)
        if verbose:
            logger.info(f"Generated .osz saved to {result_path}")
    else:
        result_path = output_osu_path
        with profiler.stage("write_osu"):
            postprocessor.write_result(result_path, result)
        if verbose:
            logger.info(f"Generated beatmap saved to {result_path}")

    profiler.set_metadata(result_path=result_path)
    profile_path = profiler.write(args.profile_output_path or profiler.default_output_path(result_path))
    if verbose and profile_path is not None:
        logger.info(f"Inference profile saved to {profile_path}")

    return result, result_path


def load_model_with_server(ckpt_path: str | Path | None, t5_args: TrainConfig, device, max_batch_size: int = 8,
                           use_server: bool = False, precision: str = "fp32", attn_implementation: str = "sdpa",
                           eval_mode: bool = True, lora_path=None, gamemode: int | None = None,
                           auto_select_gamemode_model: bool = True, generation_compile: bool = False):
    model_loader, tokenizer_loader = load_model_loaders(
        ckpt_path=ckpt_path,
        t5_args=t5_args,
        device=device,
        precision=precision,
        attn_implementation=attn_implementation,
        eval_mode=eval_mode,
        pickle_module=routed_pickle,
        lora_path=lora_path,
        gamemode=gamemode,
        auto_select_gamemode_model=auto_select_gamemode_model,
        generation_compile=generation_compile,
    )

    return InferenceClient(
        model_loader,
        tokenizer_loader,
        max_batch_size=max_batch_size,
        socket_path=get_server_address(
            ckpt_path,
            lora_path=lora_path,
            gamemode=gamemode,
            auto_select_gamemode_model=auto_select_gamemode_model,
        ),
    ) if use_server else model_loader(), tokenizer_loader()


def get_server_address(
        ckpt_path_str: str | Path | None,
        lora_path: str | Path | None = None,
        gamemode: int | None = None,
        auto_select_gamemode_model: bool = True,
):
    """
    Get a valid socket address for the OS and model version.
    """
    resolved_ckpt_path, subfolder = resolve_model_checkpoint_path(
        ckpt_path_str,
        gamemode=gamemode,
        auto_select_gamemode_model=auto_select_gamemode_model,
    )
    ckpt_path_str = "" if not resolved_ckpt_path else (resolved_ckpt_path.as_posix() if isinstance(resolved_ckpt_path, Path) else str(resolved_ckpt_path))
    if subfolder:
        ckpt_path_str = f"{ckpt_path_str}/{subfolder}"
    ckpt_subfolder = get_model_checkpoint_subfolder(resolved_ckpt_path, subfolder)
    effective_lora_path, _ = resolve_compatible_lora_path(
        lora_path,
        ckpt_subfolder=ckpt_subfolder,
        verbose=False,
    )
    if effective_lora_path:
        effective_lora_str = effective_lora_path.as_posix() if isinstance(effective_lora_path, Path) else str(effective_lora_path)
        ckpt_path_str = f"{ckpt_path_str}__lora__{effective_lora_str}"
    ckpt_path_str = ckpt_path_str.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(".", "_")
    # Check if the OS supports Unix sockets
    if os.name == 'posix':
        # Use a Unix socket for Linux and macOS
        return f"/tmp/{ckpt_path_str}.sock"
    else:
        # Use a Windows named pipe
        return fr"\\.\pipe\{ckpt_path_str}"


def load_diff_model(
        ckpt_path,
        diff_args: DiffusionTrainConfig,
        device,
):
    if not os.path.exists(ckpt_path) and ckpt_path != "":
        tokenizer_file = cached_file(ckpt_path, "tokenizer.pkl")
        model_file = cached_file(ckpt_path, "model_ema.pkl")
    else:
        ckpt_path = Path(ckpt_path)
        tokenizer_file = ckpt_path / "tokenizer.pkl"
        model_file = ckpt_path / "model_ema.pkl"

    tokenizer_state = torch.load(tokenizer_file, pickle_module=routed_pickle, weights_only=False)
    tokenizer = osu_diffusion.utils.tokenizer.Tokenizer()
    tokenizer.load_state_dict(tokenizer_state)

    ema_state = torch.load(model_file, pickle_module=routed_pickle, weights_only=False, map_location=device)
    model = DiT_models[diff_args.model.model](
        context_size=diff_args.model.context_size,
        class_size=tokenizer.num_tokens,
    ).to(device)
    model.load_state_dict(ema_state)
    model.eval()  # important!
    return model, tokenizer


@hydra.main(config_path="configs/inference", config_name="v32", version_base="1.1")
def main(args: InferenceConfig):
    args = OmegaConf.to_object(args) if isinstance(args, DictConfig) else args
    profiler = InferenceProfiler.from_args(args)
    with profiler.stage("compile_args"):
        compile_args(args)
    with profiler.stage("setup_inference_environment"):
        setup_inference_environment(args.seed)

    with profiler.stage("load_main_model"):
        model, tokenizer = load_model_with_server(args.model_path, args.train, args.device,
                                                  max_batch_size=args.max_batch_size, use_server=args.use_server,
                                                  precision=args.precision, attn_implementation=args.attn_implementation,
                                                  lora_path=args.lora_path, gamemode=args.gamemode,
                                                  auto_select_gamemode_model=args.auto_select_gamemode_model,
                                                  generation_compile=args.inference_generation_compile)

    timing_model, timing_tokenizer = None, None
    if should_load_separate_timing_model(args):
        print("Using base model for timing generation.")
        with profiler.stage("load_timing_model"):
            timing_model, timing_tokenizer = load_model_with_server(
                args.model_path,
                args.train,
                args.device,
                max_batch_size=args.max_batch_size,
                use_server=args.use_server,
                precision=args.precision,
                attn_implementation=args.attn_implementation,
                gamemode=args.gamemode,
                auto_select_gamemode_model=False,
                generation_compile=args.inference_generation_compile,
            )

    diff_model, diff_tokenizer, refine_model = None, None, None
    if args.generate_positions:
        with profiler.stage("load_diffusion_model"):
            diff_model, diff_tokenizer = load_diff_model(args.diff_ckpt, args.diffusion, args.device)

            if os.path.exists(args.diff_refine_ckpt):
                refine_model = load_diff_model(args.diff_refine_ckpt, args.diffusion, args.device)[0]

            if args.compile:
                diff_model.forward = torch.compile(diff_model.forward, mode="reduce-overhead", fullgraph=True)

    with profiler.stage("build_generation_config"):
        generation_config, beatmap_config = get_config(args)

    return generate(
        args,
        generation_config=generation_config,
        beatmap_path=args.beatmap_path,
        beatmap_config=beatmap_config,
        model=model,
        tokenizer=tokenizer,
        timing_model=timing_model,
        timing_tokenizer=timing_tokenizer,
        diff_model=diff_model,
        diff_tokenizer=diff_tokenizer,
        refine_model=refine_model,
        profiler=profiler,
    )


if __name__ == "__main__":
    main()
