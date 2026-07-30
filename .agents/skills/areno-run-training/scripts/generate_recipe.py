#!/usr/bin/env python3
"""Training-recipe generator for the areno-run-training skill.

Accepts SFT/DPO/online-RL mode, GPU count, context length, and target batch,
then writes a complete editable config and launch command with per-value
provenance.  When a checkpoint path is available, the generator also estimates
weight / optimizer / KV-cache memory and compares against probed free VRAM to
derive safe training parameters.

The script works standalone (without the ``areno`` package installed) using
baked-in field definitions synced from ``areno/api/trainer_config.py``.  When
``areno`` is importable, the generator additionally validates the recipe
against the real dataclass ``__post_init__`` constraints.

Usage::

    python .agents/skills/areno-run-training/scripts/generate_recipe.py \\
        --algo gspo --gpus 8 --tp-size 4 \\
        --context-len 4096 --batch-size 32 \\
        --ckpt Qwen/Qwen3-0.6B \\
        --reward-fn-path examples/math/math_verify_reward.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Protocol

# == constants ==============================================================

RL_ALGOS = {"gspo", "grpo", "ppo"}
OFFLINE_ALGOS = {"sft", "dpo"}

# CLI option name -> config field name, for fields where they differ.
# Synced from areno.cli.train._trainer_config_from_args inline mappings.
_CLI_TO_FIELD: dict[str, str] = {
    "lr": "optimizer_lr",
    "min_lr": "optimizer_min_lr",
    "adam_beta1": "optimizer_beta1",
    "adam_beta2": "optimizer_beta2",
    "disable_thinking": "chat_template_enable_thinking",
}
_FIELD_TO_CLI: dict[str, str] = {v: k for k, v in _CLI_TO_FIELD.items()}
_INVERTED_BOOL: dict[str, str] = {"drop_rollout_state": "keep_rollout_state"}

# CLI-only utility flags with no TrainerConfig field.
# Synced from areno.cli.train Click option declarations.
_CLI_ONLY_OPTIONS: frozenset[str] = frozenset({
    "tune_params", "mem_frac", "tune_max_samples", "smoke_infer", "smoke_train",
})

# Built-in algorithm names and their narrowest config class.
# Algo->config-class mapping synced from areno.cli.train._trainer_config_from_args.
VALID_ALGOS: dict[str, str] = {
    "sft": "TrainerConfig",
    "dpo": "DPOTrainerConfig",
    "gspo": "PolicyTrainerConfig",
    "grpo": "PolicyTrainerConfig",
    "ppo": "PPOTrainerConfig",
}

# CLI option groups (synced from areno.cli.train.TRAIN_OPTION_GROUPS).
_TRAIN_OPTION_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Basic", (
        "algo", "ckpt", "dataset_path", "model_hub", "dataset_loader_fn",
        "tune_params", "mem_frac", "tune_max_samples", "smoke_infer", "smoke_train",
        "epochs", "max_steps", "world_size", "tp_size",
    )),
    ("Rollout", (
        "batch_size", "n_samples", "max_running_prompts", "max_prompt_tokens",
        "max_new_tokens", "max_context_len", "temperature", "top_k", "top_p", "greedy",
        "eager_decode", "drop_rollout_state", "attn_backend", "disable_thinking",
        "agent_fn", "agent_timeout_s", "train_tool_results", "reward_fn_path", "reward_ckpt",
    )),
    ("Train", (
        "mini_bs", "score_micro_bs", "gradient_accumulation_steps",
        "activation_checkpointing", "lr", "min_lr", "lr_decay_steps", "lr_decay_style",
        "adam_beta1", "adam_beta2", "adam_8bit", "weight_decay", "grad_clip_norm",
        "ref_ckpt", "critic_ckpt", "critic_lr", "critic_warmup_steps",
        "gspo_clip_eps", "grpo_clip_eps", "dpo_beta",
        "use_kl_loss", "kl_loss_coef", "kl_loss_type",
        "clip_eps", "clip_ratio_c", "value_clip_eps", "value_loss_coef", "gamma", "lam",
    )),
    ("Checkpoint", ("save_path", "save_interval")),
    ("Observability", ("metrics_log_dir",)),
)


def _valid_option_names() -> set[str]:
    """Return real areno-train CLI option names, excluding CLI-only utility flags."""
    names: set[str] = set()
    for _section, opts in _TRAIN_OPTION_GROUPS:
        names.update(opts)
    names -= _CLI_ONLY_OPTIONS
    return names


# == baked-in config field defaults (synced from trainer_config.py) =========
# Each entry: field_name -> default_value.  Split per config class so we can
# build provenance for the narrowest type without importing areno.

# TrainerConfig (base, used by sft)
_BASE_DEFAULTS: dict[str, Any] = {
    "algo": "gspo", "ckpt": None, "dataset_path": None, "model_hub": "modelscope",
    "dataset_loader_fn": None, "save_path": None, "save_interval": 100,
    "epochs": 10, "max_steps": None, "tp_size": 4, "world_size": 8,
    "batch_size": 32, "mini_bs": 16, "score_micro_bs": 8,
    "gradient_accumulation_steps": None, "max_prompt_tokens": 1024,
    "max_new_tokens": 3071, "max_context_len": None,
    "optimizer_lr": 1e-6, "optimizer_min_lr": 1e-7, "lr_decay_steps": 1000,
    "lr_decay_style": "cosine", "optimizer_beta1": 0.9, "optimizer_beta2": 0.999,
    "weight_decay": 1e-2, "grad_clip_norm": 1.0, "adam_8bit": False,
    "activation_checkpointing": True, "keep_rollout_state": True,
    "eager_decode": False, "attn_backend": "flash",
    "metrics_log_dir": "/tmp/areno/tfevent",
    "agent_fn": None, "agent_timeout_s": 300.0, "train_tool_results": False,
    "chat_template_enable_thinking": None,
}

# RolloutTrainerConfig adds rollout fields (gspo, grpo, ppo)
_ROLLOUT_DEFAULTS: dict[str, Any] = {
    "n_samples": 8, "greedy": False, "temperature": 1.0, "top_k": -1,
    "top_p": 1.0, "max_running_prompts": None,
}

# PolicyTrainerConfig adds reward/clipping fields (gspo, grpo, ppo)
_POLICY_DEFAULTS: dict[str, Any] = {
    "reward_fn_path": None, "gspo_clip_eps": 3e-4, "grpo_clip_eps": 0.2,
}

# DPOTrainerConfig adds ref/dpo_beta
_DPO_DEFAULTS: dict[str, Any] = {
    "ref_ckpt": None, "dpo_beta": 0.1,
}

# PPOTrainerConfig adds PPO-specific fields
_PPO_DEFAULTS: dict[str, Any] = {
    "reward_ckpt": None, "critic_ckpt": None, "role_device": None,
    "critic_lr": 1e-5, "kl_coef": 0.02, "use_kl_loss": True,
    "kl_loss_coef": 0.001, "kl_loss_type": "low_var_kl",
    "clip_eps": 0.2, "clip_ratio_c": 3.0, "value_clip_eps": 0.5,
    "value_loss_coef": 0.5, "gamma": 1.0, "lam": 0.95,
    "critic_warmup_steps": 20,
}


def _field_defaults_for_algo(algo: str) -> dict[str, Any]:
    """Return the merged default dict for the config class matching *algo*."""
    defaults = dict(_BASE_DEFAULTS)
    if algo in RL_ALGOS:
        defaults.update(_ROLLOUT_DEFAULTS)
        defaults.update(_POLICY_DEFAULTS)
    if algo == "ppo":
        defaults.update(_PPO_DEFAULTS)
    if algo == "dpo":
        defaults.update(_DPO_DEFAULTS)
    return defaults


def _is_valid_field(algo: str, field_name: str) -> bool:
    """Check if *field_name* exists on the config class for *algo*."""
    return field_name in _field_defaults_for_algo(algo)


# == optional areno integration =============================================

def _try_areno_available() -> bool:
    try:
        import areno  # noqa: F401
        return True
    except ImportError:
        return False


def _validate_with_areno(algo: str, config_kwargs: dict[str, Any], warnings: list[str]) -> None:
    """When areno is installed, validate the config against the real dataclass."""
    try:
        from areno.api.trainer_config import (
            DPOTrainerConfig, PolicyTrainerConfig, PPOTrainerConfig, TrainerConfig,
        )
        cls_map = {
            "sft": TrainerConfig, "dpo": DPOTrainerConfig,
            "gspo": PolicyTrainerConfig, "grpo": PolicyTrainerConfig,
            "ppo": PPOTrainerConfig,
        }
        cls = cls_map[algo]
        import dataclasses as _dc
        valid = {k: v for k, v in config_kwargs.items() if k in {f.name for f in _dc.fields(cls)}}
        cls(**valid)
    except Exception as exc:
        warnings.append(f"[areno validation] {exc}")


# == memory estimation (pure, CPU-testable) =================================

DEFAULT_KV_BLOCK_SIZE = 256


@dataclass(frozen=True)
class GpuMemoryInfo:
    """Probed GPU memory snapshot.

    When multiple devices are probed, *total_bytes* and *free_bytes* are the
    minimum across all probed cards (conservative). *device_ids* lists the
    actual device indices that were probed.
    """

    total_bytes: int
    free_bytes: int
    device_count: int
    device_ids: "Optional[List[int]]" = None


class GpuProbe(Protocol):
    def probe(self) -> "GpuMemoryInfo | None": ...


class RealGpuProbe:
    """Probe CUDA devices via ``torch.cuda.mem_get_info``.

    When *gpu_count* is 0 or None, probes all visible devices (aggregates the
    minimum free and total across all cards so the estimate is conservative).
    When *gpu_count* is a positive integer, probes only the first *gpu_count*
    devices and reports the minimum free / total among them.
    """

    def __init__(self, gpu_count: int = 0) -> None:
        self._gpu_count = gpu_count

    def probe(self) -> "GpuMemoryInfo | None":
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            total_devices = torch.cuda.device_count()
            # If user specified a count, cap at that (and at available).
            n = self._gpu_count if self._gpu_count > 0 else total_devices
            n = min(n, total_devices)
            if n == 0:
                return None
            # Probe each device; use the minimum free and total across cards
            # so the estimate is conservative (worst-fit card drives the budget).
            min_free = None
            min_total = None
            device_ids = list(range(n))
            for dev in device_ids:
                free, total = torch.cuda.mem_get_info(dev)
                free = int(free)
                total = int(total)
                if min_free is None or free < min_free:
                    min_free = free
                if min_total is None or total < min_total:
                    min_total = total
            return GpuMemoryInfo(
                total_bytes=min_total or 0,
                free_bytes=min_free or 0,
                device_count=n,
                device_ids=device_ids,
            )
        except Exception:
            return None


class FakeGpuProbe:
    """Deterministic GPU probe for CPU tests.

    Returns a fixed ``GpuMemoryInfo`` so memory-fit assertions are reproducible.
    """

    def __init__(self, info: GpuMemoryInfo) -> None:
        self._info = info

    def probe(self) -> "GpuMemoryInfo | None":
        return self._info


@dataclass(frozen=True)
class ModelShape:
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    dtype_bytes: int = 2
    tie_word_embeddings: bool = False
    is_moe: bool = False
    num_experts: "int | None" = None
    moe_intermediate_size: int = 0
    # When set, weight/optimizer estimates use this param count directly
    # instead of computing from architecture fields. Used when the shape
    # was inferred from a model name and the architecture fields are
    # approximate.
    param_count_override: "int | None" = None


def model_shape_from_config(config: Any) -> ModelShape:
    """Build a ModelShape from an areno ModelConfig dataclass."""
    import torch

    dtype = getattr(config, "dtype", torch.bfloat16)
    dtype_bytes = 4 if dtype in (torch.float32, torch.float64) else 2
    return ModelShape(
        num_layers=int(config.num_hidden_layers),
        num_attention_heads=int(config.num_attention_heads),
        num_kv_heads=int(config.num_key_value_heads),
        head_dim=int(config.head_dim),
        hidden_size=int(config.hidden_size),
        intermediate_size=int(config.intermediate_size),
        vocab_size=int(config.vocab_size),
        dtype_bytes=dtype_bytes,
        tie_word_embeddings=bool(config.tie_word_embeddings),
        is_moe=getattr(config, "enable_moe_block", False),
        num_experts=getattr(config, "num_experts", None),
        moe_intermediate_size=getattr(config, "moe_intermediate_size", 0),
    )


import re as _re


def parse_param_count_from_name(ckpt: str) -> "int | None":
    """Extract approximate parameter count (in absolute units) from a model name.

    Supports common naming patterns::

        Qwen/Qwen3-0.6B        -> 600_000_000
        Qwen3-1.7B             -> 1_700_000_000
        Qwen2.5-7B-Instruct    -> 7_000_000_000
        Qwen3-30B-A3B          -> 30_000_000_000  (30B total, MoE)
        Qwen3-235B-A22B        -> 235_000_000_000
        Llama-3-8B             -> 8_000_000_000
        gemma-2-2b             -> 2_000_000_000

    Returns ``None`` if no parameter-size pattern is found.
    """
    match = _re.search(r"(\d+(?:\.\d+)?)\s*([bBmM])", ckpt)
    if not match:
        return None
    num = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "b":
        return int(num * 1e9)
    return int(num * 1e6)


# Known model architectures for approximate shape inference from param count.
# Each entry: (model_family_prefix, [(param_count_approx, ModelShape), ...])
# Sorted by param count so the closest match below the actual count is used.
_KNOWN_SHAPES: list[tuple[str, list[tuple[int, ModelShape]]]] = [
    ("qwen3", [
        (600_000_000,   ModelShape(28, 16, 8,  128, 1024,  3072,  151936)),
        (1_700_000_000, ModelShape(28, 16, 8,  128, 2048,  6144,  151936)),
        (4_000_000_000, ModelShape(36, 32, 8,  128, 2560,  9728,  152064)),
        (7_000_000_000, ModelShape(28, 28, 4,  128, 3584,  18944, 152064)),
        (14_000_000_000, ModelShape(40, 40, 8,  128, 5120,  13824, 152064)),
        (32_000_000_000, ModelShape(64, 64, 8,  128, 5120,  27648, 152064)),
        (72_000_000_000, ModelShape(80, 64, 8,  128, 8192,  29568, 152064)),
    ]),
    ("llama", [
        (1_000_000_000,  ModelShape(16, 16, 4, 128, 2048,  5632,  128256)),
        (8_000_000_000,  ModelShape(32, 32, 8, 128, 4096,  14336, 128256)),
        (70_000_000_000, ModelShape(80, 64, 8, 128, 8192,  28672, 128256)),
    ]),
    ("gemma", [
        (2_000_000_000,  ModelShape(26, 8,  1, 256, 2304,  9216,  256000)),
        (9_000_000_000,  ModelShape(42, 16, 1, 256, 3072,  24576, 256000)),
        (27_000_000_000, ModelShape(46, 32, 16,256, 4608,  36864, 256000)),
    ]),
]


def infer_shape_from_name(ckpt: str) -> "tuple[ModelShape | None, str | None]":
    """Infer an approximate ModelShape from a checkpoint name.

    Parses the parameter count from the name (e.g. ``Qwen3-0.6B`` -> 0.6B),
    then looks up the closest known architecture for that model family.

    Returns ``(shape, note)`` where *note* is a human-readable caveat string
    when the estimate is approximate, or ``None`` when exact.
    """
    param_count = parse_param_count_from_name(ckpt)
    if param_count is None:
        return None, None

    ckpt_lower = ckpt.lower()
    for prefix, shapes in _KNOWN_SHAPES:
        if prefix not in ckpt_lower:
            continue
        # Find the closest param count (smallest absolute difference).
        best = min(shapes, key=lambda s: abs(s[0] - param_count))
        shape = ModelShape(
            num_layers=best[1].num_layers,
            num_attention_heads=best[1].num_attention_heads,
            num_kv_heads=best[1].num_kv_heads,
            head_dim=best[1].head_dim,
            hidden_size=best[1].hidden_size,
            intermediate_size=best[1].intermediate_size,
            vocab_size=best[1].vocab_size,
            dtype_bytes=2,
            tie_word_embeddings=param_count < 1_000_000_000,
        )
        note = (
            f"Inferred architecture from model name '{ckpt}' "
            f"(~{param_count / 1e9:.1f}B params, closest match: {prefix} "
            f"{best[0] / 1e9:.1f}B). Weight/optimizer estimates are exact; "
            f"KV-cache and activation estimates are approximate."
        )
        # For MoE names like "30B-A3B", mark as MoE.
        moe_match = _re.search(r"(\d+(?:\.\d+)?)B\s*[-/]?\s*A(\d+(?:\.\d+)?)B", ckpt, _re.IGNORECASE)
        if moe_match:
            total_b = float(moe_match.group(1))
            active_b = float(moe_match.group(2))
            shape = ModelShape(
                num_layers=shape.num_layers,
                num_attention_heads=shape.num_attention_heads,
                num_kv_heads=shape.num_kv_heads,
                head_dim=shape.head_dim,
                hidden_size=shape.hidden_size,
                intermediate_size=shape.intermediate_size,
                vocab_size=shape.vocab_size,
                dtype_bytes=2,
                tie_word_embeddings=shape.tie_word_embeddings,
                is_moe=True,
            )
            note += f" MoE detected: {total_b}B total, {active_b}B active."
        return shape, note

    # Unknown family but we have a param count — use a generic dense shape.
    # Rough heuristic: hidden ~ 128 * ceil(param_count / 50M), layers ~ 28.
    hidden = max(512, 128 * ((param_count // 50_000_000) or 1))
    layers = 28 if param_count < 5_000_000_000 else 64
    shape = ModelShape(
        num_layers=layers,
        num_attention_heads=max(8, hidden // 128),
        num_kv_heads=max(2, (hidden // 128) // 4),
        head_dim=128,
        hidden_size=hidden,
        intermediate_size=hidden * 3,
        vocab_size=152000,
        dtype_bytes=2,
        tie_word_embeddings=param_count < 1_000_000_000,
        # Weight/optimizer use the exact param count from the name,
        # not the approximate architecture fields.
        param_count_override=param_count,
    )
    note = (
        f"Inferred architecture from param count (~{param_count / 1e9:.1f}B) "
        f"with generic heuristic shapes. Weight/optimizer estimates are exact; "
        f"KV-cache and activation estimates are very approximate. "
        f"Provide a local checkpoint path for accurate estimation."
    )
    return shape, note


def estimate_param_count(shape: ModelShape) -> int:
    """Estimate total parameter count for a dense or MoE transformer."""
    if shape.param_count_override is not None:
        return shape.param_count_override
    embedding = shape.vocab_size * shape.hidden_size
    attn = (2 * shape.num_attention_heads + 2 * shape.num_kv_heads) * shape.head_dim * shape.hidden_size
    if shape.is_moe and shape.num_experts:
        expert_ffn = 3 * shape.hidden_size * shape.moe_intermediate_size * shape.num_experts
        shared_ffn = 3 * shape.hidden_size * shape.intermediate_size if shape.intermediate_size else 0
        mlp = expert_ffn + shared_ffn
    else:
        mlp = 3 * shape.hidden_size * shape.intermediate_size
    norms = 2 * shape.hidden_size
    per_layer = attn + mlp + norms
    lm_head = 0 if shape.tie_word_embeddings else shape.vocab_size * shape.hidden_size
    return embedding + shape.num_layers * per_layer + lm_head


def estimate_weight_bytes(shape: ModelShape) -> int:
    return estimate_param_count(shape) * shape.dtype_bytes


def estimate_optimizer_bytes(param_count: int, dtype_bytes: int = 2, *, adam_8bit: bool = False) -> int:
    if adam_8bit:
        return param_count * (dtype_bytes + 6)
    return param_count * (dtype_bytes + 12)


def estimate_kv_cache_bytes(
    shape: ModelShape, *, tp_size: int, max_running_seqs: int,
    max_cache_len: int, block_size: int = DEFAULT_KV_BLOCK_SIZE,
    dtype_bytes: "int | None" = None,
) -> int:
    db = dtype_bytes or shape.dtype_bytes
    local_kv_heads = max(1, shape.num_kv_heads // tp_size)
    max_blocks_per_seq = (max_cache_len + block_size - 1) // block_size
    num_blocks = max(max_running_seqs * max_blocks_per_seq, 1)
    total_blocks = num_blocks + 1
    per_tensor = total_blocks * block_size * local_kv_heads * shape.head_dim * db
    return shape.num_layers * 2 * per_tensor


@dataclass(frozen=True)
class MemoryBreakdown:
    weights: int
    optimizer: int
    kv_cache: int
    activations: int
    total: int
    per_gpu_total: int
    per_gpu_free: "int | None" = None
    headroom_bytes: "int | None" = None

    @property
    def headroom_ok(self) -> bool:
        if self.headroom_bytes is None:
            return True
        return self.headroom_bytes >= 0


def estimate_activation_bytes(
    shape: ModelShape, *, mini_bs: int, seq_len: int, activation_checkpointing: bool = True
) -> int:
    if mini_bs <= 0 or seq_len <= 0:
        return 0
    base = 34 * shape.hidden_size * seq_len * mini_bs * shape.dtype_bytes
    if activation_checkpointing:
        return int(base * 0.3)
    return int(base)


def estimate_memory(
    shape: ModelShape, *, tp_size: int, dp_size: int, max_running_seqs: int,
    max_cache_len: int, mini_bs: int, adam_8bit: bool = False,
    activation_checkpointing: bool = True, gpu_info: "GpuMemoryInfo | None" = None,
    block_size: int = DEFAULT_KV_BLOCK_SIZE,
    num_extra_trainable: int = 0, num_extra_frozen: int = 0,
) -> MemoryBreakdown:
    param_count = estimate_param_count(shape)
    per_gpu_weights = estimate_weight_bytes(shape) // tp_size
    per_gpu_opt = estimate_optimizer_bytes(param_count, shape.dtype_bytes, adam_8bit=adam_8bit) // tp_size

    # Extra trainable models (e.g. PPO critic): weights + optimizer per model.
    extra_trainable_weights = per_gpu_weights * num_extra_trainable
    extra_trainable_opt = per_gpu_opt * num_extra_trainable

    # Extra frozen models (e.g. PPO/DPO ref model): weights only, no optimizer.
    extra_frozen_weights = per_gpu_weights * num_extra_frozen

    # KV cache: prompts are split across DP ranks, so each GPU only holds
    # max_running_seqs / dp_size concurrent sequences (not the full batch).
    seqs_per_gpu = max(1, max_running_seqs // dp_size)
    per_gpu_kv = estimate_kv_cache_bytes(
        shape, tp_size=tp_size, max_running_seqs=seqs_per_gpu,
        max_cache_len=max_cache_len, block_size=block_size,
    )

    # Activations are only present during the train phase.
    per_gpu_act = estimate_activation_bytes(
        shape, mini_bs=mini_bs, seq_len=max_cache_len,
        activation_checkpointing=activation_checkpointing,
    )

    # In AReno's RL loop, rollout and train alternate. During rollout, the
    # model offloads train weights but holds KV cache. During train, KV cache
    # is released and train weights + optimizer + activations are loaded.
    # Peak memory is therefore the LARGER of the two phases, not their sum.
    #
    # Multi-role algorithms (PPO, DPO) add extra model weights:
    # - PPO: critic (trainable) + ref (frozen) [+ reward (frozen)]
    # - DPO: ref (frozen)
    # Frozen models are present in both phases; trainable extras have
    # optimizer state in train phase and infer weights in rollout phase.
    all_weights = per_gpu_weights + extra_trainable_weights + extra_frozen_weights
    rollout_phase = per_gpu_kv + all_weights
    train_phase = (
        per_gpu_weights + extra_trainable_weights  # trainable weights
        + per_gpu_opt + extra_trainable_opt        # optimizer for trainable models
        + extra_frozen_weights                      # frozen models stay loaded
        + per_gpu_act
    )
    total = max(rollout_phase, train_phase)

    per_gpu_total = gpu_info.total_bytes if gpu_info else 0
    per_gpu_free = gpu_info.free_bytes if gpu_info else None
    headroom = (per_gpu_free - total) if per_gpu_free is not None else None
    return MemoryBreakdown(
        weights=per_gpu_weights, optimizer=per_gpu_opt, kv_cache=per_gpu_kv,
        activations=per_gpu_act, total=total, per_gpu_total=per_gpu_total,
        per_gpu_free=per_gpu_free, headroom_bytes=headroom,
    )


# == context-length splitting ================================================

def split_context_len(algo: str, context_len: int) -> "tuple[int, int]":
    if context_len <= 0:
        raise ValueError("context_len must be positive")
    if algo in OFFLINE_ALGOS:
        if algo == "sft":
            return context_len, 0
        half = context_len // 2
        return half, context_len - half
    prompt = min(1024, context_len // 4)
    return prompt, context_len - prompt


# == auto-detection =========================================================


def _best_tp_for_gpus(gpus: int) -> int:
    """Pick the largest power-of-2 that divides gpus and is <= 8."""

    for tp in (8, 4, 2, 1):
        if gpus % tp == 0:
            return tp
    return 1

@dataclass(frozen=True)
class AutoParams:
    """Parameters auto-derived from GPU probe and model size."""

    gpus: int
    tp_size: int
    context_len: int
    batch_size: int
    note: str


def auto_detect_params(
    gpu_info: GpuMemoryInfo,
    param_count: int,
    algo: str,
    *,
    mem_frac: float = 0.85,
    n_samples: int = 4,
) -> AutoParams:
    """Derive GPU count, tp_size, context_len, and batch_size from hardware.

    Heuristics:
    - gpus = detected device count
    - tp_size = largest power-of-2 that divides gpus and keeps per-GPU
      weights under ~40% of single-GPU VRAM (so optimizer+activations fit)
    - context_len = 4096 for small models on 80GB, scaled down for tighter VRAM
    - batch_size = largest power-of-2 where memory fits within mem_frac
    """
    gpus = gpu_info.device_count
    vram_per_gpu = gpu_info.total_bytes
    model_bytes_bf16 = param_count * 2
    # Adam fp32 states: param_count * (2 + 12) = param_count * 14
    opt_bytes = param_count * 14

    # --- tp_size: pick the largest tp that divides gpus and keeps
    # per-GPU weight+optimizer under 50% of VRAM ---
    best_tp = 1
    for tp in [8, 4, 2, 1]:
        if gpus % tp == 0:
            per_gpu_model = (model_bytes_bf16 + opt_bytes) // tp
            if per_gpu_model < vram_per_gpu * 0.5:
                best_tp = tp
                break
    # Fallback if even tp=1 exceeds 50% (very large model on small GPU)
    if best_tp == 1 and model_bytes_bf16 + opt_bytes > vram_per_gpu * 0.8:
        best_tp = max(tp for tp in [8, 4, 2, 1] if gpus % tp == 0)

    # --- context_len: start from 4096, scale down if VRAM is tight ---
    if param_count < 2_000_000_000:
        ctx = 8192 if vram_per_gpu >= 70_000_000_000 else 4096
    elif param_count < 10_000_000_000:
        ctx = 4096 if vram_per_gpu >= 70_000_000_000 else 2048
    else:
        ctx = 2048 if vram_per_gpu >= 70_000_000_000 else 1024

    # --- batch_size: binary-search the largest power-of-2 that fits ---
    # Estimate KV + activation per batch unit at the chosen context_len.
    max_cache_len = ctx  # prompt + response ~= ctx
    # Use a rough per-seq KV estimate: layers * 2 * ceil(ctx/256) * 256 * kv_heads/tp * head_dim * 2
    # We don't know exact layers/heads here, so approximate from param_count.
    approx_layers = 28 if param_count < 5_000_000_000 else 64
    approx_kv_heads = 8
    approx_head_dim = 128
    local_kv = max(1, approx_kv_heads // best_tp)
    blocks_per_seq = (max_cache_len + 255) // 256
    kv_per_seq = approx_layers * 2 * blocks_per_seq * 256 * local_kv * approx_head_dim * 2
    # Activation per seq (with checkpointing): 34 * hidden * ctx * 2 * 0.3
    approx_hidden = max(512, int((param_count / approx_layers ** 1.5) ** 0.5) * 128)
    act_per_seq = int(34 * approx_hidden * max_cache_len * 2 * 0.3)

    per_gpu_fixed = (model_bytes_bf16 + opt_bytes) // best_tp
    target_budget = int(vram_per_gpu * mem_frac)
    available_for_batch = target_budget - per_gpu_fixed

    # For RL, each prompt generates n_samples sequences, so batch contributes
    # batch_size * n_samples to KV cache.
    n_samples_eff = n_samples if algo in RL_ALGOS else 1
    per_batch_unit = n_samples_eff * (kv_per_seq + act_per_seq)

    if per_batch_unit <= 0:
        batch = 1
    else:
        batch = 1
        while batch * 2 * per_batch_unit < available_for_batch:
            batch *= 2
        batch = max(batch, 1)
        # Cap at 64 for sanity.
        batch = min(batch, 64)

    note = (
        f"Auto-detected {gpus} GPUs ({vram_per_gpu / 1e9:.0f} GB each). "
        f"Derived tp_size={best_tp}, context_len={ctx}, batch_size={batch} "
        f"for ~{param_count / 1e9:.1f}B param model with {algo}."
    )
    return AutoParams(
        gpus=gpus, tp_size=best_tp, context_len=ctx, batch_size=batch, note=note,
    )



@dataclass(frozen=True)
class ConfigValue:
    value: Any
    source: str  # "default" | "derived" | "explicit"


@dataclass(frozen=True)
class RecipeResult:
    algo: str
    command: str
    config: dict
    memory: "MemoryBreakdown | None"
    warnings: list
    dataset_rows: "int | None"

    def to_dict(self) -> dict:
        mem = None
        if self.memory is not None:
            mem = {
                "weights_bytes": self.memory.weights,
                "optimizer_bytes": self.memory.optimizer,
                "kv_cache_bytes": self.memory.kv_cache,
                "activations_bytes": self.memory.activations,
                "total_estimated_bytes": self.memory.total,
                "per_gpu_total_bytes": self.memory.per_gpu_total,
                "per_gpu_free_bytes": self.memory.per_gpu_free,
                "headroom_bytes": self.memory.headroom_bytes,
                "headroom_ok": self.memory.headroom_ok,
            }
        return {
            "algo": self.algo,
            "command": self.command,
            "config": {k: {"value": v.value, "source": v.source} for k, v in self.config.items()},
            "memory": mem,
            "warnings": list(self.warnings),
            "dataset_rows": self.dataset_rows,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


def _field_to_cli_name(field_name: str) -> str:
    return _FIELD_TO_CLI.get(field_name, field_name).replace("_", "-")


# Per-algo command-required field sets.
_BASE_REQUIRED: frozenset[str] = frozenset({
    "ckpt", "dataset_path", "tp_size", "world_size",
    "batch_size", "mini_bs", "max_prompt_tokens", "max_new_tokens",
})
_RL_REQUIRED: frozenset[str] = _BASE_REQUIRED | frozenset({
    "n_samples", "reward_fn_path", "max_running_prompts",
})
_DPO_REQUIRED: frozenset[str] = _BASE_REQUIRED | frozenset({
    "ref_ckpt", "dpo_beta",
})


def _required_fields_for(algo: str) -> frozenset[str]:
    if algo in RL_ALGOS:
        return _RL_REQUIRED
    if algo == "dpo":
        return _DPO_REQUIRED
    return _BASE_REQUIRED


def _build_command(algo: str, config_map: dict) -> str:
    """Render a copyable areno train command with key inputs + explicit overrides."""
    required = _required_fields_for(algo)
    parts = ["areno train", f"--algo {algo}"]

    # Order fields following TRAIN_OPTION_GROUPS sections.
    ordered: list[str] = []
    for _section, opts in _TRAIN_OPTION_GROUPS:
        for opt in opts:
            fn = _CLI_TO_FIELD.get(opt, opt)
            if fn in config_map and fn not in ordered and fn != "algo":
                ordered.append(fn)
    for key in config_map:
        if key not in ordered and key != "algo":
            ordered.append(key)

    for fn in ordered:
        cv = config_map[fn]
        val = cv.value
        if val is None:
            continue
        is_required = fn in required
        is_explicit = cv.source == "explicit"
        if not is_required and not is_explicit:
            continue
        if fn == "chat_template_enable_thinking":
            if val is False and is_explicit:
                parts.append("--disable-thinking")
            continue
        cli = _field_to_cli_name(fn)
        if isinstance(val, bool):
            if fn == "keep_rollout_state" and not val:
                parts.append("--drop-rollout-state")
                continue
            if val and is_explicit:
                parts.append(f"--{cli}")
            continue
        parts.append(f"--{cli} {val}")
    return " ".join(parts)


def _count_dataset_rows(dataset_path: str) -> "int | None":
    """Count rows in a dataset file or directory.

    Supports .jsonl, .ndjson, .json (list), .csv files and directories of .jsonl files.
    Uses generators for large files to avoid loading entire file into memory.
    """
    if not dataset_path:
        return None
    p = Path(dataset_path)
    if not p.exists():
        return None
    if p.is_file():
        if p.suffix in {".jsonl", ".ndjson"}:
            # Generator-based counting for memory efficiency with large files
            with p.open("r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        if p.suffix == ".json":
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return len(data) if isinstance(data, list) else None
        if p.suffix == ".csv":
            # Generator-based counting for memory efficiency with large CSV files
            with p.open("r", encoding="utf-8") as f:
                return max(sum(1 for _ in f) - 1, 0)
        return None
    if p.is_dir():
        count = 0
        for child in sorted(p.iterdir()):
            if child.suffix in {".jsonl", ".ndjson"} and child.is_file():
                with child.open("r", encoding="utf-8") as f:
                    count += sum(1 for _ in f)
        return count or None
    return None


def generate_recipe(
    *, algo: str, gpus: int = 0, tp_size: int = 0, context_len: int = 0,
    batch_size: int = 0,
    ckpt: "str | None" = None, dataset_path: "str | None" = None,
    reward_fn_path: "str | None" = None, overrides: "dict | None" = None,
    gpu_probe: "GpuProbe | None" = None, n_samples: int = 4,
    mini_bs: "int | None" = None, mem_frac: float = 0.9,
    adam_8bit: bool = False, activation_checkpointing: bool = True,
    block_size: int = DEFAULT_KV_BLOCK_SIZE,
    model_shape: "ModelShape | None" = None,
    auto: bool = False,
    agent_fn: "str | None" = None, dataset_loader_fn: "str | None" = None,
    agent_timeout_s: "float | None" = None, train_tool_results: bool = False,
    lr: "float | None" = None,
) -> RecipeResult:
    """Generate a complete training recipe.

    When *auto* is True and gpus/tp_size/context_len/batch_size are 0 (unset),
    the generator probes the local GPU and derives these values automatically
    from detected VRAM and the model's parameter count.
    """

    overrides = overrides or {}
    warnings: list[str] = []

    # 1. validate algorithm
    if algo not in VALID_ALGOS:
        raise ValueError(
            f"[stage=algo-resolution] unknown algorithm '{algo}'; registered: {', '.join(sorted(VALID_ALGOS))}"
        )

    # 2. auto-detect GPU params if requested
    if auto:
        probe = gpu_probe or RealGpuProbe(gpu_count=gpus)
        gpu_info = probe.probe()
        if gpu_info is None:
            warnings.append("--auto requested but no GPU detected; falling back to defaults.")
            gpus = gpus or 8
            # When no GPU, pick a tp_size that divides gpus.
            if tp_size <= 0 or gpus % tp_size != 0:
                tp_size = _best_tp_for_gpus(gpus)
            context_len = context_len or 4096
            batch_size = batch_size or 32
        else:
            # Need param count for auto-detection.
            param_count = None
            shape_tmp: "ModelShape | None" = model_shape
            if ckpt and shape_tmp is None:
                try:
                    from areno.models.registry import config_from_hf
                    shape_tmp = model_shape_from_config(config_from_hf(ckpt))
                except Exception:
                    pass
            if shape_tmp is None and ckpt and ckpt != "<your-ckpt>":
                inferred, _ = infer_shape_from_name(ckpt)
                shape_tmp = inferred
            if shape_tmp is not None:
                param_count = estimate_param_count(shape_tmp)
            else:
                # No model info — use conservative defaults.
                param_count = 1_000_000_000
                warnings.append("--auto: could not determine model size; assuming 1B params for sizing.")

            auto_params = auto_detect_params(gpu_info, param_count, algo, mem_frac=mem_frac, n_samples=n_samples)
            warnings.append(auto_params.note)
            # If user specified --gpus, honor it; otherwise use detected count.
            gpus = gpus or auto_params.gpus
            tp_size = tp_size or auto_params.tp_size
            context_len = context_len or auto_params.context_len
            batch_size = batch_size or auto_params.batch_size
    else:
        # Use conservative defaults when not auto and value is 0.
        # These are lower than areno's CLI defaults (which assume 8x80GB GPUs
        # running 7B+ models) so that small-model recipes don't start with
        # excessive KV-cache allocations.
        gpus = gpus or 8
        tp_size = tp_size or 4
        context_len = context_len or 2048
        batch_size = batch_size or 8

    # 3. validate parallelism
    if gpus <= 0:
        raise ValueError("[stage=parallelism] --gpus must be positive")
    if tp_size <= 0:
        raise ValueError("[stage=parallelism] --tp-size must be positive")
    if gpus % tp_size != 0:
        raise ValueError("[stage=parallelism] --world-size must be divisible by --tp-size")
    dp_size = gpus // tp_size

    # 4. validate sizing
    if context_len <= 0:
        raise ValueError("[stage=sizing] --context-len must be positive")
    if batch_size <= 0:
        raise ValueError("[stage=sizing] --batch-size must be positive")
    if algo in RL_ALGOS and n_samples < 1:
        raise ValueError("[stage=sizing] --n-samples must be >= 1 for RL algorithms")
    if mini_bs is not None and mini_bs > batch_size:
        warnings.append(f"--mini-bs ({mini_bs}) > --batch-size ({batch_size}); will be capped at batch_size.")
        mini_bs = min(mini_bs, batch_size)

    # 4. validate overrides against real option names
    valid_options = _valid_option_names()
    for key in overrides:
        if key not in valid_options:
            known = ", ".join(sorted(valid_options))
            raise ValueError(f"[stage=override] unknown option '{key}'; valid: {known}")
    overrides_field: dict[str, Any] = {}
    for key, val in overrides.items():
        if key in _INVERTED_BOOL:
            overrides_field[_INVERTED_BOOL[key]] = not val
        elif key == "disable_thinking":
            overrides_field["chat_template_enable_thinking"] = not val if val else None
        else:
            overrides_field[_CLI_TO_FIELD.get(key, key)] = val

    # 4b. validate enum-valued overrides
    _ENUM_VALIDATORS: dict[str, set[str]] = {
        "attn_backend": {"flash", "native"},
        "model_hub": {"hf", "modelscope"},
        "lr_decay_style": {"cosine", "linear", "constant"},
    }
    for field_name, valid_values in _ENUM_VALIDATORS.items():
        val = overrides_field.get(field_name)
        if val is not None and val not in valid_values:
            raise ValueError(
                f"[stage=override] {field_name}='{val}' is invalid; must be one of: {', '.join(sorted(valid_values))}"
            )

    # 5. split context_len
    max_prompt_tokens, max_new_tokens = split_context_len(algo, context_len)
    derived: dict[str, str] = {
        "world_size": "gpus",
        "tp_size": "explicit" if "tp_size" in overrides_field else "derived",
        "batch_size": "explicit" if "batch_size" in overrides_field else "derived",
        "mini_bs": "derived from batch_size",
        "max_prompt_tokens": "context_len",
        "max_new_tokens": "context_len",
        "adam_8bit": "recipe input",
        "activation_checkpointing": "recipe input",
        "model_hub": "recipe default",
    }
    if algo in RL_ALGOS:
        max_running_seqs = batch_size * n_samples
        derived["max_running_prompts"] = "batch_size * n_samples"
        derived["n_samples"] = "recipe input"
        derived["keep_rollout_state"] = "recipe input"
    else:
        max_running_seqs = batch_size
    if algo == "dpo":
        derived["dpo_beta"] = "recipe input"

    resolved_mini_bs = mini_bs or min(batch_size, 16)

    # 6. probe GPU memory
    # When the user specified --gpus explicitly, probe only that many cards.
    # When gpu_probe is injected (tests), use it directly.
    probe = gpu_probe or RealGpuProbe(gpu_count=gpus)
    gpu_info = probe.probe()
    if gpu_info is not None:
        dev_info = f" (devices {gpu_info.device_ids})" if gpu_info.device_ids else ""
        warnings.append(
            f"GPU probe: {gpu_info.device_count} devices{dev_info}, "
            f"{gpu_info.free_bytes / 1e9:.1f} GB free / {gpu_info.total_bytes / 1e9:.1f} GB total (min across cards)"
        )
    else:
        warnings.append("No GPU detected -- memory estimates only; cannot verify fit.")

    # 7. load model config and estimate memory
    memory: "MemoryBreakdown | None" = None
    shape: "ModelShape | None" = model_shape
    if ckpt and shape is None:
        try:
            from areno.models.registry import config_from_hf

            model_config = config_from_hf(ckpt)
            shape = model_shape_from_config(model_config)
        except Exception as exc:
            if ckpt != "<your-ckpt>":
                warnings.append(f"Could not read model config from '{ckpt}': {exc}")

    # 7b. If still no shape and we have a model name, infer from naming.
    if shape is None and ckpt and ckpt != "<your-ckpt>":
        inferred_shape, note = infer_shape_from_name(ckpt)
        if inferred_shape is not None:
            shape = inferred_shape
            if note:
                warnings.append(note)

    if shape is not None:
        max_cache_len = max_prompt_tokens + max_new_tokens
        # Compute extra model roles for multi-role algorithms.
        # PPO: critic (trainable) + ref (frozen) [+ reward (frozen) if reward_ckpt]
        # DPO: ref (frozen)
        num_extra_trainable = 1 if algo == "ppo" else 0
        num_extra_frozen = 0
        if algo == "ppo":
            num_extra_frozen = 1  # ref model
            if overrides_field.get("reward_ckpt"):
                num_extra_frozen += 1  # reward model
        elif algo == "dpo":
            num_extra_frozen = 1  # ref model
        if num_extra_trainable or num_extra_frozen:
            role_desc = []
            if num_extra_trainable:
                role_desc.append(f"{num_extra_trainable} trainable (critic)")
            if num_extra_frozen:
                role_desc.append(f"{num_extra_frozen} frozen (ref/reward)")
            warnings.append(
                f"Multi-role algo '{algo}': extra memory for {', '.join(role_desc)} included in estimate."
            )

        memory = estimate_memory(
            shape, tp_size=tp_size, dp_size=dp_size, max_running_seqs=max_running_seqs,
            max_cache_len=max_cache_len, mini_bs=resolved_mini_bs, adam_8bit=adam_8bit,
            activation_checkpointing=activation_checkpointing, gpu_info=gpu_info,
            block_size=block_size,
            num_extra_trainable=num_extra_trainable, num_extra_frozen=num_extra_frozen,
        )

        # Auto-adjust batch_size to fit within free VRAM.
        # Only adjust when user didn't explicitly set batch_size and we have
        # GPU memory info.  This is the key step: instead of warning "exceeds
        # VRAM", we actually shrink batch_size to fit.
        # We use free_vram * 0.95 as the safe limit -- the remaining 5% is
        # reserved for CUDA runtime, workspace, and fragmentation overhead.
        SAFE_VRAM_RATIO = 0.95
        effective_free = int(gpu_info.free_bytes * SAFE_VRAM_RATIO) if gpu_info else None

        if (
            effective_free is not None
            and "batch_size" not in overrides_field
        ):
            # Check if current setting already fits; if not, try to adjust.
            # Also try to *increase* batch_size when there's headroom to use
            # more of the available VRAM productively.
            needs_adjust = (
                memory.headroom_bytes is not None and memory.headroom_bytes < 0
            )
            if needs_adjust:
                original_batch = batch_size
                best_batch = batch_size
                for candidate in [64, 32, 16, 8, 4, 2, 1]:
                    if algo in RL_ALGOS:
                        cand_seqs = candidate * n_samples
                    else:
                        cand_seqs = candidate
                    cand_mini = min(candidate, 16)
                    cand_mem = estimate_memory(
                        shape, tp_size=tp_size, dp_size=dp_size,
                        max_running_seqs=cand_seqs, max_cache_len=max_cache_len,
                        mini_bs=cand_mini, adam_8bit=adam_8bit,
                        activation_checkpointing=activation_checkpointing,
                        gpu_info=gpu_info, block_size=block_size,
                        num_extra_trainable=num_extra_trainable, num_extra_frozen=num_extra_frozen,
                    )
                    if cand_mem.total <= effective_free:
                        best_batch = candidate
                        memory = cand_mem
                        break
                else:
                    # Even batch_size=1 doesn't fit -- keep smallest and warn.
                    best_batch = 1
                    cand_seqs = n_samples if algo in RL_ALGOS else 1
                    memory = estimate_memory(
                        shape, tp_size=tp_size, dp_size=dp_size,
                        max_running_seqs=cand_seqs, max_cache_len=max_cache_len,
                        mini_bs=1, adam_8bit=adam_8bit,
                        activation_checkpointing=activation_checkpointing,
                        gpu_info=gpu_info, block_size=block_size,
                        num_extra_trainable=num_extra_trainable, num_extra_frozen=num_extra_frozen,
                    )

                if best_batch < original_batch:
                    batch_size = best_batch
                    max_running_seqs = batch_size * n_samples if algo in RL_ALGOS else batch_size
                    resolved_mini_bs = mini_bs or min(batch_size, 16)
                    warnings.append(
                        f"Auto-adjusted batch_size {original_batch} -> {best_batch} "
                        f"to fit within {effective_free / 1e9:.1f} GB usable VRAM "
                        f"({SAFE_VRAM_RATIO:.0%} of {gpu_info.free_bytes / 1e9:.1f} GB free)."
                    )

        if memory.headroom_bytes is not None and memory.total > effective_free:
            warnings.append(
                f"WARNING: Estimated memory ({memory.total / 1e9:.1f} GB) exceeds usable VRAM "
                f"({effective_free / 1e9:.1f} GB = {SAFE_VRAM_RATIO:.0%} of free) "
                f"even at batch_size=1. Consider increasing --tp-size or --gpus."
            )
        if shape.is_moe:
            warnings.append("MoE model detected -- weight estimate includes all experts.")

    # 8. probe dataset size
    dataset_rows: "int | None" = None
    if dataset_path:
        dataset_rows = _count_dataset_rows(dataset_path)
        if dataset_rows is not None:
            steps_per_epoch = max(dataset_rows // batch_size, 1)
            warnings.append(f"Dataset has {dataset_rows} rows -> ~{steps_per_epoch} steps/epoch at batch_size={batch_size}.")
        elif dataset_path != "<your-dataset>":
            warnings.append(f"Could not count dataset rows for '{dataset_path}' (unsupported format or remote ref).")

    # 9. algorithm-specific requirement checks
    if algo == "sft" and not dataset_loader_fn and not overrides_field.get("dataset_loader_fn"):
        warnings.append("--algo sft requires --dataset-loader-fn at run time.")
    if algo in RL_ALGOS and not reward_fn_path and not overrides_field.get("reward_ckpt"):
        warnings.append(f"--algo {algo} requires --reward-fn-path or --reward-ckpt at run time.")
    if algo == "ppo" and not overrides_field.get("critic_ckpt"):
        warnings.append("--algo ppo requires --critic-ckpt at run time.")
    if algo == "dpo" and not overrides_field.get("ref_ckpt"):
        warnings.append("--algo dpo requires --ref-ckpt at run time (or it defaults to actor ckpt).")
    if agent_fn:
        if agent_timeout_s is not None and agent_timeout_s < 60:
            warnings.append(f"--agent-timeout-s={agent_timeout_s} is very short for multi-turn agent rollouts.")

    # 10. build config dict (standalone, no areno dependency)
    config_values: dict[str, Any] = dict(_field_defaults_for_algo(algo))
    config_values.update(
        algo=algo,
        ckpt=ckpt or "<your-ckpt>",
        dataset_path=dataset_path or "<your-dataset>",
        world_size=gpus,
        tp_size=tp_size,
        batch_size=batch_size,
        mini_bs=resolved_mini_bs,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=max_new_tokens,
        adam_8bit=adam_8bit,
        activation_checkpointing=activation_checkpointing,
    )
    if algo in RL_ALGOS:
        config_values["n_samples"] = n_samples
        config_values["reward_fn_path"] = reward_fn_path
        config_values["keep_rollout_state"] = True
    if algo == "dpo":
        config_values["dpo_beta"] = 0.1

    # Agentic RL parameters
    derived_agentic: list[str] = []
    if agent_fn:
        config_values["agent_fn"] = agent_fn
        derived_agentic.append("agent_fn")
    if dataset_loader_fn:
        config_values["dataset_loader_fn"] = dataset_loader_fn
        derived_agentic.append("dataset_loader_fn")
    if agent_timeout_s is not None:
        config_values["agent_timeout_s"] = agent_timeout_s
        derived_agentic.append("agent_timeout_s")
    if train_tool_results:
        config_values["train_tool_results"] = True
        derived_agentic.append("train_tool_results")
    if lr is not None:
        config_values["optimizer_lr"] = lr
        derived_agentic.append("optimizer_lr")

    explicit: dict[str, Any] = {}
    for key, val in overrides_field.items():
        if _is_valid_field(algo, key):
            config_values[key] = val
            explicit[key] = val
    # Agentic parameters passed via dedicated CLI flags are also explicit.
    for key in derived_agentic:
        explicit[key] = config_values[key]

    # Optional: validate with real areno dataclass when available
    if _try_areno_available():
        _validate_with_areno(algo, config_values, warnings)

    # 11. build provenance
    provenance: dict[str, ConfigValue] = {}
    for field_name, default_val in _field_defaults_for_algo(algo).items():
        val = config_values.get(field_name, default_val)
        source = "default"
        if field_name in explicit:
            source = "explicit"
        elif field_name in derived:
            source = "derived"
        provenance[field_name] = ConfigValue(value=val, source=source)

    # 12. build command
    command = _build_command(algo, provenance)

    return RecipeResult(
        algo=algo, command=command, config=provenance, memory=memory,
        warnings=warnings, dataset_rows=dataset_rows,
    )


# == CLI =====================================================================

def _parse_set(value: str) -> "tuple[str, Any]":
    if "=" not in value:
        raise ValueError(f"--set expects key=value, got: {value}")
    key, raw = value.split("=", 1)
    key = key.strip()
    try:
        return key, int(raw)
    except ValueError:
        pass
    try:
        return key, float(raw)
    except ValueError:
        pass
    if raw.lower() in ("true", "false"):
        return key, raw.lower() == "true"
    return key, raw.strip()


def main(argv: "Optional[List[str]]" = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate an areno training recipe with memory estimation and per-value provenance."
    )
    parser.add_argument("--algo", required=True, choices=list(VALID_ALGOS), help="Training algorithm.")
    parser.add_argument("--ckpt", default=None, help="Model checkpoint path or name.")
    parser.add_argument("--gpus", type=int, default=0, help="Total GPU count. 0 = auto-detect with --auto, else 8.")
    parser.add_argument("--tp-size", type=int, default=0, help="Tensor-parallel size. 0 = auto or 4.")
    parser.add_argument("--context-len", type=int, default=0, help="Total context length. 0 = auto or 4096.")
    parser.add_argument("--batch-size", type=int, default=0, help="Target batch size. 0 = auto or 32.")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-detect GPUs and derive tp_size/context_len/batch_size from hardware.")
    parser.add_argument("--dataset-path", default=None, help="Dataset path for row counting.")
    parser.add_argument("--reward-fn-path", default=None, help="Python file defining reward_fn(record).")
    parser.add_argument("--agent-fn", default=None, help="Python file defining async run_agent(ctx, batch) for agentic RL.")
    parser.add_argument("--dataset-loader-fn", default=None, help="Python file defining load_training_dataset() for custom dataset formats.")
    parser.add_argument("--agent-timeout-s", type=float, default=None, help="Timeout in seconds for each agent rollout.")
    parser.add_argument("--train-tool-results", action="store_true", help="Include tool result tokens in loss (agentic RL only).")
    parser.add_argument("--n-samples", type=int, default=4, help="Rollout samples per prompt (RL only).")
    parser.add_argument("--mini-bs", type=int, default=None, help="Training microbatch size.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (maps to optimizer_lr).")
    parser.add_argument("--mem-frac", type=float, default=0.9, help="Target GPU memory fraction (advisory).")
    parser.add_argument("--adam-8bit", action="store_true", help="Use 8-bit Adam optimizer states.")
    parser.add_argument("--no-activation-checkpointing", action="store_true", help="Disable activation checkpointing.")
    parser.add_argument("--block-size", type=int, default=DEFAULT_KV_BLOCK_SIZE, help="KV cache block size.")
    parser.add_argument("--set", action="append", default=[], help="Explicit override as key=value.")
    parser.add_argument("--format", choices=["cli", "json", "both"], default="both", help="Output format.")
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {}
    for item in args.set:
        key, val = _parse_set(item)
        overrides[key] = val

    try:
        result = generate_recipe(
            algo=args.algo, gpus=args.gpus, tp_size=args.tp_size,
            context_len=args.context_len, batch_size=args.batch_size,
            ckpt=args.ckpt, dataset_path=args.dataset_path,
            reward_fn_path=args.reward_fn_path, overrides=overrides,
            n_samples=args.n_samples, mini_bs=args.mini_bs,
            mem_frac=args.mem_frac, adam_8bit=args.adam_8bit,
            activation_checkpointing=not args.no_activation_checkpointing,
            block_size=args.block_size, auto=args.auto,
            agent_fn=args.agent_fn, dataset_loader_fn=args.dataset_loader_fn,
            agent_timeout_s=args.agent_timeout_s, train_tool_results=args.train_tool_results,
            lr=args.lr,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.format in ("cli", "both"):
        print(result.command)
        print()
        for w in result.warnings:
            print(f"# {w}")
        if result.memory is not None:
            m = result.memory
            print("# memory estimate (per GPU):")
            print(f"#   weights:      {m.weights / 1e9:.2f} GB")
            print(f"#   optimizer:    {m.optimizer / 1e9:.2f} GB")
            print(f"#   kv_cache:     {m.kv_cache / 1e9:.2f} GB")
            print(f"#   activations:  {m.activations / 1e9:.2f} GB")
            print(f"#   total:        {m.total / 1e9:.2f} GB")
            if m.per_gpu_free is not None:
                print(f"#   free VRAM:    {m.per_gpu_free / 1e9:.2f} GB")
                print(f"#   headroom:     {(m.headroom_bytes or 0) / 1e9:+.2f} GB")
        print()
        print("# provenance:")
        for key, cv in sorted(result.config.items()):
            if cv.value is not None:
                print(f"#   {key} = {cv.value}  [{cv.source}]")

    if args.format in ("json", "both"):
        if args.format == "both":
            print("\n--- JSON ---")
        print(result.to_json())

    return 0


if __name__ == "__main__":
    sys.exit(main())
