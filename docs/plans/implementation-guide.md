# Training Recipe Generator — Implementation Guide

> This document describes the implemented training-recipe generator for the
> `areno-run-training` skill. It is an English companion to the Chinese design

## 1. Overview

The recipe generator is a standalone Python script that accepts high-level
training inputs — algorithm mode, GPU count, context length, and target batch
size — and produces:

1. A copy-pasteable `areno train` launch command containing only key recipe
   parameters and explicit overrides (not every dataclass default).
2. A structured JSON config with per-value provenance (`default` / `derived` /
   `explicit`).
3. An optional per-GPU memory breakdown (weights, optimizer, KV-cache,
   activations) compared against probed free VRAM, with a headroom warning when
   the estimate exceeds available memory.

The script runs without the `areno` package, `torch`, or CUDA installed. It
embeds field definitions and defaults synced from
`areno/api/trainer_config.py`. When `areno` is importable, the generator
additionally validates the recipe against the real dataclass `__post_init__`
constraints.

## 2. File Layout

```
skills/areno-run-training/
  SKILL.md                     # User-facing skill documentation with examples
  scripts/
    generate_recipe.py         # Generator (executable CLI + importable API)
tests/
  test_run_training_recipe_cpu.py   # 55 CPU tests, all passing
docs/plans/
  training-recipe-generator.md     # Chinese design document
  implementation-guide.md          # This file
```

## 3. Input Contract

| Input | Flag | Type | Default | Description |
|---|---|---|---|---|
| Algorithm | `--algo` | `sft\|dpo\|gspo\|grpo\|ppo` | required | Validated against embedded `VALID_ALGOS`. |
| GPU count | `--gpus` | int | 8 | Maps to `world_size`. |
| Tensor-parallel | `--tp-size` | int | 4 | Must divide `--gpus`. |
| Context length | `--context-len` | int | 4096 | Split into `max_prompt_tokens` + `max_new_tokens`. |
| Target batch | `--batch-size` | int | 32 | Maps to `batch_size`. |
| Checkpoint | `--ckpt` | str | none | Model path for architecture-based memory estimation. |
| Dataset | `--dataset-path` | str | none | Local `.jsonl`/`.json`/`.csv` for row counting. |
| Reward function | `--reward-fn-path` | str | none | Required at run time for RL algorithms. |
| Rollout samples | `--n-samples` | int | 8 | Samples per prompt (RL only). |
| Microbatch | `--mini-bs` | int | min(batch,16) | Training microbatch size. |
| 8-bit Adam | `--adam-8bit` | flag | off | Affects optimizer memory estimate. |
| Activation checkpointing | `--no-activation-checkpointing` | flag | on | Affects activation memory estimate. |
| KV block size | `--block-size` | int | 256 | KV cache block size. |
| Explicit override | `--set key=value` | repeatable | none | Passed through to real CLI option names. |
| Output format | `--format` | `cli\|json\|both` | `both` | `cli` = command only; `json` = structured output. |
| Model shape | `model_shape` (API only) | `ModelShape` | none | Allows memory estimation without a checkpoint. |

### Context-Length Splitting

The total `--context-len` is split into `max_prompt_tokens` and `max_new_tokens`
per algorithm:

| Algorithm | Split |
|---|---|
| SFT | prompt = context_len, response = 0 |
| DPO | prompt = 50%, response = 50% |
| GSPO / GRPO / PPO | prompt = min(1024, 25%), response = remainder |

Override with `--set max_new_tokens=2048`.

## 4. Memory Estimation

When `--ckpt` points to a loadable checkpoint (or `model_shape` is supplied),
the generator estimates four memory components per GPU:

| Component | Formula |
|---|---|
| **Weights** | `param_count * dtype_bytes`. Parameter count follows the GQA + SwiGLU layout: `embedding + layers * (q + k + v + o + gate + up + down + norms) + lm_head`. Supports MoE (all experts included). |
| **Optimizer** | fp32 Adam: `params * (grad_dtype + 12)`. 8-bit Adam: `params * (grad_dtype + 6)`. |
| **KV cache** | `num_layers * 2 * (num_blocks + 1) * block_size * local_kv_heads * head_dim * dtype_bytes`, mirroring `Model.allocate_kv_caches` in `areno/engine/inference.py`. |
| **Activations** | `34 * hidden * seq * mini_bs * dtype_bytes`, reduced by 70% with activation checkpointing (coarse estimate). |

Key derivations:

- Weights and optimizer states are TP-sharded: `bytes_per_gpu = total // tp_size`.
- KV cache and activations are per-GPU.
- `local_kv_heads = num_key_value_heads // tp_size`.
- `max_blocks_per_seq = ceil(max_cache_len / block_size)`.
- `num_blocks = max_running_seqs * max_blocks_per_seq`.
- RL: `max_running_seqs = batch_size * n_samples`. Offline: `max_running_seqs = batch_size`.
- `headroom_bytes = free_vram - total_estimated`. Negative values trigger a warning.

GPU probing is isolated behind a `GpuProbe` protocol. The real implementation
calls `torch.cuda.mem_get_info`. CPU tests inject `FakeGpuProbe` /
`FakeNoneProbe` with deterministic values.

## 5. CLI Option Name Mapping

Six fields have CLI option names that differ from their config dataclass field
names. These mappings are synced from `_trainer_config_from_args` in
`areno/cli/train.py:595-791`.

| CLI option | Config field | Mapping type |
|---|---|---|
| `lr` | `optimizer_lr` | Direct |
| `min_lr` | `optimizer_min_lr` | Direct |
| `adam_beta1` | `optimizer_beta1` | Direct |
| `adam_beta2` | `optimizer_beta2` | Direct |
| `disable_thinking` | `chat_template_enable_thinking` | Ternary: no flag -> None, `--disable-thinking` -> False |
| `drop_rollout_state` | `keep_rollout_state` | Inverted boolean: `--drop-rollout-state` -> False |

### CLI-Only Options Excluded from `--set`

Five CLI flags appear in `TRAIN_OPTION_GROUPS` but have no corresponding
`TrainerConfig` field. They control the train command's own pre-flight helpers
(memory tuning, smoke tests) and cannot be used as `--set` override targets:

- `tune_params`, `mem_frac`, `tune_max_samples`, `smoke_infer`, `smoke_train`

The function `_valid_option_names()` returns all real CLI option names minus
these five.

## 6. Validation Pipeline

All validation runs before any expensive model or worker initialization:

1. **Algorithm resolution** (`stage=algo-resolution`): `algo` must be in
   `VALID_ALGOS`. Error lists all registered algorithms.
2. **Parallelism** (`stage=parallelism`): `gpus > 0`, `tp_size > 0`,
   `gpus % tp_size == 0`.
3. **Sizing** (`stage=sizing`): `context_len > 0`, `batch_size > 0`.
4. **Override** (`stage=override`): every `--set` key must be in
   `_valid_option_names()`. Error lists all valid options.

Errors are formatted as `[stage=<stage>] <message>` and contain only the
affected input and the reason — no dataset samples or full stack traces.

## 7. Runtime Requirement Warnings

The generator checks algorithm-specific required parameters and appends
warnings when they are missing (generation is not interrupted):

| Algorithm | Required parameter | Warning |
|---|---|---|
| SFT | `dataset_loader_fn` | `--algo sft requires --dataset-loader-fn at run time.` |
| GSPO/GRPO/PPO | `reward_fn_path` or `reward_ckpt` | `--algo <algo> requires --reward-fn-path or --reward-ckpt at run time.` |
| PPO | `critic_ckpt` | `--algo ppo requires --critic-ckpt at run time.` |
| DPO | `ref_ckpt` | `--algo dpo requires --ref-ckpt at run time (or it defaults to actor ckpt).` |

## 8. Provenance Tracking

Every config field is annotated with one of three sources:

| Source | Meaning |
|---|---|
| `default` | Value comes from the embedded default table (`_field_defaults_for_algo`). |
| `derived` | Value computed from `--gpus`, `--context-len`, or `--batch-size` (e.g., `world_size`, `max_prompt_tokens`, RL's `n_samples`, DPO's `dpo_beta`). |
| `explicit` | Value provided via `--set key=value`. |

Provenance is emitted in both the CLI comment block and the JSON `config`
object.

## 9. Command Rendering

`_build_command()` outputs only:

1. **Required fields** (per-algorithm subsets):
   - Base (all algos): `ckpt`, `dataset_path`, `tp_size`, `world_size`,
     `batch_size`, `mini_bs`, `max_prompt_tokens`, `max_new_tokens`.
   - RL adds: `n_samples`, `reward_fn_path`, `max_running_prompts`.
   - DPO adds: `ref_ckpt`, `dpo_beta`.
2. **Explicit overrides**: fields with `source == "explicit"`.

Non-required defaults (e.g., `epochs`, `score_micro_bs`, `optimizer_lr`) do not
appear in the command but are present in the full provenance / JSON output.
Option ordering follows `_TRAIN_OPTION_GROUPS` section order.

## 10. Output Formats

### Human-readable (`--format cli`)

```
areno train --algo gspo --ckpt <your-ckpt> --dataset-path <your-dataset> \
  --world-size 8 --tp-size 4 --batch-size 32 --n-samples 8 \
  --max-prompt-tokens 1024 --max-new-tokens 3072 \
  --reward-fn-path reward.py --mini-bs 16

# No GPU detected -- memory estimates only; cannot verify fit.
# --algo gspo requires --reward-fn-path or --reward-ckpt at run time.
# provenance:
#   world_size = 8  [derived]
#   tp_size = 4  [derived]
#   batch_size = 32  [derived]
#   ...
```

### Structured (`--format json`)

```json
{
  "algo": "gspo",
  "command": "areno train --algo gspo ...",
  "config": {
    "world_size": {"value": 8, "source": "derived"},
    "optimizer_lr": {"value": 1e-06, "source": "default"},
    "reward_fn_path": {"value": "reward.py", "source": "default"}
  },
  "memory": {
    "weights_bytes": 375812096,
    "optimizer_bytes": 2630684672,
    "kv_cache_bytes": 7523532800,
    "activations_bytes": 235929600,
    "total_estimated_bytes": 109083068,
    "per_gpu_total_bytes": 80000000000,
    "per_gpu_free_bytes": 70000000000,
    "headroom_bytes": 59091693200,
    "headroom_ok": true
  },
  "warnings": ["No GPU detected -- ..."],
  "dataset_rows": null
}
```

## 11. Test Suite

`tests/test_run_training_recipe_cpu.py` contains 55 CPU tests (all passing on
Python 3.9 + pytest 8.4). No GPU, network, or external database required.

| Test class | Count | Coverage |
|---|---|---|
| `TestParamCount` | 2 | Dense parameter formula, tied vs. untied embeddings |
| `TestWeightBytes` | 2 | bf16 and fp32 weight bytes |
| `TestOptimizerBytes` | 2 | fp32 and 8-bit Adam state bytes |
| `TestKvCacheBytes` | 2 | KV cache formula matches `allocate_kv_caches`, TP sharding halves KV |
| `TestEstimateMemory` | 3 | Sum check, positive/negative headroom |
| `TestSplitContextLen` | 5 | SFT/DPO/RL splits, small context, zero-value error |
| `TestRecipeSuccess` | 5 | SFT/DPO/GSPO/PPO recipes, offline has no rollout fields |
| `TestRecipeDeterminism` | 1 | Same input produces identical output |
| `TestRecipeValidation` | 7 | Invalid algo/tp/gpus/context_len/batch/override/CLI-only option |
| `TestRecipeOverrides` | 4 | lr/min_lr mapping, drop_rollout_state inversion, disable_thinking |
| `TestProvenance` | 4 | default/derived/RL-derived/explicit markers |
| `TestOutputFields` | 11 | JSON fields, command option names, excluded defaults, memory estimate, warnings (RL/PPO/DPO/SFT), reward_ckpt alternative |
| `TestDatasetCounting` | 2 | jsonl row count, remote ref returns null |
| `TestCliOverrideParsing` | 5 | int/float/bool/string/invalid parsing |

Run them with:

```bash
python3 -m pytest tests/test_run_training_recipe_cpu.py -v
```

## 12. Compatibility

- The generator adds a new skill and script. It does not modify
  `areno/cli/train.py`, `TrainerConfig` dataclasses, or `pyproject.toml`.
- The script runs on Python 3.9+ with only the standard library (no areno,
  torch, or CUDA required).
- Existing `areno train` behavior and the existing CPU test suite are
  unaffected.
- Tests verified on Python 3.9.6 + pytest 8.4.2: **55 passed, 0 failed**.
