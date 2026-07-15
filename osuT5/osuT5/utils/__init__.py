"""Lazy public exports for training and inference utilities.

Importing the inference entry point must not initialize the training-only
datasets, tracking, and accelerator stack.  Attribute access preserves the
existing public names while importing only the module that owns the requested
symbol.
"""

from __future__ import annotations

from importlib import import_module
from typing import Final


_INIT_EXPORTS: Final = (
    "check_args_and_env",
    "opti_flags",
    "update_args_with_env_info",
    "setup_args",
)
_MODEL_EXPORTS: Final = (
    "get_shared_training_state",
    "get_lora_checkpoint_metadata",
    "save_lora_checkpoint_metadata",
    "load_lora_checkpoint_metadata",
    "get_model_checkpoint_subfolder",
    "resolve_compatible_lora_path",
    "resolve_model_checkpoint_path",
    "load_model",
    "load_model_loaders",
    "get_tokenizer",
    "get_optimizer",
    "get_scheduler",
    "get_dataset",
    "get_dataloaders",
    "worker_init_fn",
    "TokenBalancedBatcher",
)
_TRAIN_EXPORTS: Final = (
    "maybe_cleanup_wandb_cache",
    "forward",
    "forward_eval",
    "add_prefix",
    "maybe_save_checkpoint",
    "maybe_eval",
    "maybe_logging",
    "maybe_grad_clip_and_grad_calc",
    "eval_model",
    "calc_loss",
    "get_stats",
    "acc_range",
    "fuzzy_acc_range",
    "train",
    "train_profiling",
)

_EXPORT_MODULE: Final = {
    **{name: ".init_utils" for name in _INIT_EXPORTS},
    **{name: ".model_utils" for name in _MODEL_EXPORTS},
    **{name: ".train_utils" for name in _TRAIN_EXPORTS},
}
__all__ = tuple(_EXPORT_MODULE)


def __getattr__(name: str):
    module_name = _EXPORT_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
