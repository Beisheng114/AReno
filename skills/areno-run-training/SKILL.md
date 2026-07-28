---
name: areno-run-training
description: Use this skill to generate a complete, runnable areno training recipe (config + launch command) from high-level inputs like algorithm mode, GPU count, context length, and target batch. Includes memory estimation with model-name fallback and per-value provenance.
---

# areno Run Training: Recipe Generator

Generate a directly runnable training recipe for `areno train` from
high-level inputs. The generator estimates weight, optimizer, KV-cache, and
activation memory, compares against probed free VRAM, and warns if the recipe
may not fit.

## Quick Start

```bash
python skills/areno-run-training/scripts/generate_recipe.py \
    --algo gspo --gpus 8 --tp-size 4 \
    --context-len 4096 --batch-size 32 \
    --ckpt Qwen/Qwen3-0.6B \
    --reward-fn-path examples/math/math_verify_reward.py
```

Output (example on a machine without GPU):

```
areno train --algo gspo --ckpt Qwen/Qwen3-0.6B ...

# No GPU detected -- memory estimates only; cannot verify fit.
# Inferred architecture from model name 'Qwen/Qwen3-0.6B' (~0.6B params,
#   closest match: qwen3 0.6B). Weight/optimizer estimates are exact;
#   KV-cache and activation estimates are approximate.
# memory estimate (per GPU):
#   weights:      0.30 GB
#   optimizer:    2.09 GB
#   kv_cache:     30.07 GB
#   activations:  1.37 GB
#   total:        33.83 GB
# provenance:
#   world_size = 8  [derived]
#   tp_size = 4  [derived]
#   batch_size = 32  [derived]
#   ...
```

## Input Contract

| Input | Flag | Type | Default | Description |
|---|---|---|---|---|
| Algorithm | `--algo` | `sft\|dpo\|gspo\|grpo\|ppo` | required | Training algorithm, validated against built-in algorithm list. |
| GPU count | `--gpus` | int | 8 | Total device count, maps to `world_size`. |
| Tensor-parallel | `--tp-size` | int | 4 | Must divide `--gpus`. |
| Context length | `--context-len` | int | 4096 | Total prompt + response token budget. Split per algorithm. |
| Target batch | `--batch-size` | int | 32 | Prompt/pair batch size. |
| Checkpoint | `--ckpt` | str | none | Model path or name for memory estimation. |
| Dataset | `--dataset-path` | str | none | Dataset path for row counting (`.jsonl`, `.json`, `.csv`, directory). |
| Reward function | `--reward-fn-path` | str | none | Required at run time for RL algorithms. |
| Rollout samples | `--n-samples` | int | 8 | Samples per prompt (RL only). |
| Microbatch | `--mini-bs` | int | min(batch_size, 16) | Training microbatch size. |
| 8-bit Adam | `--adam-8bit` | flag | off | Use 8-bit optimizer states. |
| Activation checkpointing | `--no-activation-checkpointing` | flag | on | Disable layer activation recompute. |
| KV block size | `--block-size` | int | 256 | KV cache block size (must match runtime). |
| Overrides | `--set key=value` | repeatable | none | Explicit overrides for any real `areno train` option. |
| Output format | `--format` | `cli\|json\|both` | `both` | `cli` = command only; `json` = structured output. |

## Output Fields

### Human-readable (`--format cli`)

A copyable `areno train ...` command followed by provenance and memory
estimates as comments.

### Structured (`--format json`)

```jsonc
{
  "algo": "gspo",
  "command": "areno train --algo gspo ...",
  "config": {
    "world_size": {"value": 8, "source": "derived"},
    "tp_size":    {"value": 4, "source": "derived"},
    "batch_size": {"value": 32, "source": "derived"},
    "optimizer_lr": {"value": 1e-06, "source": "default"}
    // ...
  },
  "memory": {
    "weights_bytes": 300000000,
    "optimizer_bytes": 2090000000,
    "kv_cache_bytes": 30070000000,
    "activations_bytes": 1370000000,
    "total_estimated_bytes": 33830000000,
    "per_gpu_total_bytes": 0,
    "per_gpu_free_bytes": null,
    "headroom_bytes": null,
    "headroom_ok": true
  },
  "warnings": [
    "No GPU detected -- memory estimates only; cannot verify fit.",
    "Inferred architecture from model name 'Qwen/Qwen3-0.6B' ..."
  ],
  "dataset_rows": null
}
```

Each `config` entry has a `source` field:
- `default` -- the value comes from the embedded default table (synced from `TrainerConfig`).
- `derived` -- computed from `--gpus`, `--context-len`, or `--batch-size`.
- `explicit` -- set via `--set key=value`.

## Context-Length Splitting

The total `--context-len` is split into `max_prompt_tokens` and
`max_new_tokens` per algorithm:

| Algorithm | Split |
|---|---|
| SFT | prompt = context_len, response = 0 |
| DPO | prompt = 50%, response = 50% |
| GSPO / GRPO / PPO | prompt = min(1024, 25%), response = remainder |

Override with `--set max_new_tokens=2048`.

## Memory Estimation

The generator uses a three-tier fallback to estimate memory:

### Tier 1: Local checkpoint (most accurate)

When `--ckpt` points to a loadable local checkpoint, the generator reads its
HF `config.json` via `areno.models.registry.config_from_hf` and estimates:

1. **Weights**: parameter count times `dtype_bytes` (bf16 = 2 bytes/param).
2. **Optimizer**: Adam master copy (fp32) + moments (fp32 or 8-bit).
3. **KV cache**: `num_layers * 2 * (num_blocks+1) * block_size * local_kv_heads * head_dim * dtype_bytes`, mirroring `Model.allocate_kv_caches`.
4. **Activations**: `34 * hidden * seq * batch * dtype_bytes`, reduced by 70% with activation checkpointing.

### Tier 2: Model name inference (approximate)

When the checkpoint is not locally loadable (e.g. a HuggingFace repo ID),
the generator parses the parameter count from the model name
(e.g. `Qwen3-0.6B` -> 0.6B params) and looks up the closest known
architecture from a built-in table covering Qwen3, Llama, and Gemma families.

- **Weight and optimizer estimates are exact** because they only depend on parameter count.
- **KV cache and activation estimates are approximate** because the architecture details (layers, heads, hidden size) come from a closest-match lookup, not the actual config.
- For MoE names like `Qwen3-30B-A3B`, the generator detects the MoE pattern and marks the shape accordingly.

When tier 2 is used, a warning is emitted:
```
# Inferred architecture from model name 'Qwen3-7B' (~7.0B params, closest
#   match: qwen3 7.0B). Weight/optimizer estimates are exact; KV-cache and
#   activation estimates are approximate.
```

### Tier 3: Unknown model name (coarse heuristic)

When the model name contains a parameter count but does not match any known
model family (e.g. `custom-42B`), the generator uses a generic heuristic to
derive a plausible architecture. Weight and optimizer estimates remain exact
(via `param_count_override`), but KV cache and activation estimates are very
approximate.

When tier 3 is used, a warning is emitted:
```
# Inferred architecture from param count (~42.0B) with generic heuristic
#   shapes. Weight/optimizer estimates are exact; KV-cache and activation
#   estimates are very approximate. Provide a local checkpoint path for
#   accurate estimation.
```

### Fallback Guidance for Agent

When the generator cannot produce a memory estimate at all (e.g. the model
name contains no parameter-size pattern, or `--ckpt` is omitted), the
`memory` field will be `null` and a warning will be emitted. In this
situation, the calling agent should take the following steps:

1. **Try to find the model's parameter count.** Search the web or the model's
   HuggingFace/ModelScope page for the parameter count, then pass it as part
   of the `--ckpt` name (e.g. prepend or use the canonical name like
   `Qwen/Qwen3-0.6B`).

2. **Manually supply a `ModelShape` via the Python API.** If the agent can
   determine the architecture parameters (num_layers, hidden_size,
   num_attention_heads, num_kv_heads, head_dim, intermediate_size,
   vocab_size) from the model's documentation or config, call the generator
   function directly:

   ```python
   from skills.areno_run_training.scripts.generate_recipe import generate_recipe, ModelShape

   shape = ModelShape(
       num_layers=32, num_attention_heads=32, num_kv_heads=8,
       head_dim=128, hidden_size=4096, intermediate_size=14336,
       vocab_size=128256,
   )
   result = generate_recipe(
       algo="gspo", gpus=8, tp_size=4, context_len=4096,
       batch_size=32, ckpt="Llama-3-8B",
       reward_fn_path="reward.py", model_shape=shape,
   )
   ```

3. **Use the LLM's own knowledge as a last resort.** If the agent (LLM)
   cannot find the exact config but knows the approximate architecture of
   the model family, it should construct a best-guess `ModelShape` and add a
   warning note in the output. For example, if the agent knows that a
   "Llama-3-8B" model typically has 32 layers, 4096 hidden size, and 32
   attention heads, it should supply those values directly and note:

   ```
   # NOTE: Architecture manually inferred by agent from model family knowledge.
   #       Verify against the actual model config.json before relying on
   #       KV-cache and activation estimates.
   ```

4. **Always recommend `--tune-params` for production runs.** Regardless of
   which tier produced the estimate, analytical estimates can deviate from
   actual peak memory due to kernel workspace, CUDA graph capture, and
   framework overhead. The agent should always append this advisory:

   ```
   # For exact peak memory, run: areno train --tune-params <same-args>
   ```

### GPU VRAM Probing

The generator probes free VRAM via `torch.cuda.mem_get_info` when available.
When no GPU is detected, `headroom_bytes` is `null` and a warning is emitted.
CPU tests inject `FakeGpuProbe` with deterministic values.

## CLI Option Name Mapping

Some CLI flags map to non-obvious config field names:

| CLI flag | Config field | Notes |
|---|---|---|
| `--lr` | `optimizer_lr` | Direct rename |
| `--min-lr` | `optimizer_min_lr` | Direct rename |
| `--adam-beta1` | `optimizer_beta1` | Direct rename |
| `--adam-beta2` | `optimizer_beta2` | Direct rename |
| `--disable-thinking` | `chat_template_enable_thinking` | Ternary: no flag -> None, flag -> False |
| `--drop-rollout-state` | `keep_rollout_state` | Inverted boolean |

CLI-only utility flags (`--tune-params`, `--mem-frac`, `--smoke-infer`,
`--smoke-train`, `--tune-max-samples`) cannot be used with `--set`.

## Runtime Requirement Warnings

The generator checks algorithm-specific required parameters and warns when
missing (generation is not interrupted):

| Algorithm | Required parameter |
|---|---|
| SFT | `--dataset-loader-fn` |
| GSPO/GRPO/PPO | `--reward-fn-path` or `--reward-ckpt` |
| PPO | `--critic-ckpt` |
| DPO | `--ref-ckpt` |

## Limitations

- Without a GPU, memory estimates cannot be verified against real free VRAM.
- MoE weight estimates include all experts; actual activation memory for MoE
  routing may differ.
- Dataset row counting supports local `.jsonl`, `.json`, `.csv` files and
  directories of `.jsonl`. Remote HF/ModelScope refs return `null`.
- The generated command uses placeholder `<your-ckpt>` / `<your-dataset>` when
  those are not provided. Replace them before running.
- Model-name inference covers Qwen3 (0.6B–72B), Llama (1B–70B), and Gemma
  (2B–27B). Other model families use a coarse generic heuristic.

## Minimal Runnable Example

```bash
# SFT recipe with explicit override -- no GPU needed
python skills/areno-run-training/scripts/generate_recipe.py \
    --algo sft --gpus 8 --tp-size 4 \
    --context-len 2048 --batch-size 16 \
    --format json
```

```bash
# DPO recipe with memory estimation from model name
python skills/areno-run-training/scripts/generate_recipe.py \
    --algo dpo --gpus 4 --tp-size 2 \
    --context-len 4096 --batch-size 8 \
    --ckpt Qwen/Qwen3-0.6B \
    --set dpo_beta=0.05 --set lr=5e-7
```

```bash
# GSPO recipe with MoE model name detection
python skills/areno-run-training/scripts/generate_recipe.py \
    --algo gspo --gpus 8 --tp-size 4 \
    --context-len 4096 --batch-size 4 \
    --ckpt Qwen/Qwen3-30B-A3B \
    --reward-fn-path examples/math/math_verify_reward.py \
    --format cli
```
