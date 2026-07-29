# Training Recipe Generator

Generates directly runnable AReno training recipes with memory estimation and per-value provenance.

## Quick Start

```bash
python .agents/skills/areno-run-training/scripts/generate_recipe.py \
    --algo gspo --gpus 8 --tp-size 4 \
    --context-len 4096 --batch-size 32 \
    --ckpt Qwen/Qwen3-0.6B \
    --reward-fn-path examples/math/math_verify_reward.py
```

Example output (no GPU):

```
areno train --algo gspo --ckpt Qwen/Qwen3-0.6B ...

# No GPU detected -- memory estimates only; cannot verify fit.
# Inferred architecture from model name 'Qwen/Qwen3-0.6B' (~0.6B params,
#   closest match: qwen3 0.6B). Weight/optimizer estimates are exact;
#   KV-cache and activation estimates are approximate.
# memory estimate (per GPU):
#   weights:      0.60 GB
#   optimizer:   10.08 GB
#   kv_cache:    15.04 GB
#   activations:  2.24 GB
#   total:       27.96 GB

# provenance:
#   world_size = 8  [derived]
#   tp_size = 4  [derived]
#   batch_size = 32  [derived]
#   optimizer_lr = 1e-06  [default]
#   ...
```

## Input Contract

### Required Parameters

| Parameter | Flag | Type | Description |
|-----------|------|------|-------------|
| Algorithm | `--algo` | string | `sft`, `dpo`, `gspo`, `grpo`, `ppo` |

### Optional Parameters

| Parameter | Flag | Type | Default | Description |
|-----------|------|------|---------|-------------|
| GPU count | `--gpus` | int | 8 (auto-detected with `--auto`) | Total GPU count, maps to `--world-size` |
| Tensor-parallel | `--tp-size` | int | 4 | Must divide `--gpus` |
| Context length | `--context-len` | int | 2048 | Total prompt + response token budget |
| Target batch | `--batch-size` | int | 8 | Prompt/pair batch size |
| Checkpoint | `--ckpt` | string | none | Model path or name for memory estimation |
| Dataset | `--dataset-path` | string | none | Dataset path for row counting |
| Reward function | `--reward-fn-path` | string | none | Python file defining reward_fn(record) |
| Rollout samples | `--n-samples` | int | 4 | Samples per prompt (RL only) |
| Microbatch | `--mini-bs` | int | min(batch_size,16) | Training microbatch size; uses 16 as upper bound unless explicitly set |
| 8-bit Adam | `--adam-8bit` | flag | off | Use 8-bit Adam optimizer states |
| Activation checkpointing | `--no-activation-checkpointing` | flag | on | Disable layer activation recompute |
| KV block size | `--block-size` | int | 256 | KV cache block size |
| Output format | `--format` | string | `both` | `cli`, `json`, or `both` |
| Auto-detect | `--auto` | flag | off | Auto-detect GPUs and derive parameters |
| Memory fraction | `--mem-frac` | float | 0.9 | Target GPU memory fraction |
| Overrides | `--set key=value` | repeatable | none | Explicit override for any real `areno train` option |

## Output Fields

### CLI Format (`--format cli`)

A copyable `areno train ...` command followed by memory estimates and provenance comments.

### JSON Format (`--format json`)

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
    "weights_bytes": 600000000,
    "optimizer_bytes": 10080000000,
    "kv_cache_bytes": 15040000000,
    "activations_bytes": 2240000000,
    "total_estimated_bytes": 27960000000,
    "per_gpu_total_bytes": 0,
    "per_gpu_free_bytes": null,
    "headroom_bytes": null,
    "headroom_ok": true
  },
  "warnings": [
    "No GPU detected -- memory estimates only; cannot verify fit."
  ],
  "dataset_rows": null
}
```

**Config source values:**
- `default` — value from the embedded default table (synced from `TrainerConfig`)
- `derived` — computed from `--gpus`, `--context-len`, or `--batch-size`
- `explicit` — set via `--set key=value`

## Context-Length Splitting

The total `--context-len` is split into `max_prompt_tokens` and `max_new_tokens` per algorithm:

| Algorithm | Split |
|-----------|-------|
| SFT | prompt = context_len, response = 0 |
| DPO | prompt = 50%, response = 50% |
| GSPO / GRPO / PPO | prompt = min(1024, 25%), response = remainder |

Override with `--set max_new_tokens=2048`.

## Memory Estimation

### Three-Tier Fallback

**Tier 1: Local checkpoint (most accurate)**

When `--ckpt` points to a loadable local checkpoint, read its `config.json` and estimate:
- **Weights**: parameter count × `dtype_bytes` (bf16 = 2 bytes/param)
- **Optimizer**: Adam master copy (fp32) + moments (fp32 or 8-bit)
- **KV cache**: `num_layers * 2 * (num_blocks+1) * block_size * local_kv_heads * head_dim * dtype_bytes`
- **Activations**: `34 * hidden * seq * batch * dtype_bytes`, reduced by 70% with activation checkpointing

**Tier 2: Model name inference (approximate)**

When the checkpoint is not locally loadable (e.g., a HuggingFace repo ID), parse the parameter count from the model name (e.g., `Qwen3-0.6B` → 0.6B params) and look up the closest known architecture:
- **Weight and optimizer estimates are exact** because they only depend on parameter count
- **KV cache and activation estimates are approximate** because architecture details come from a closest-match lookup
- For MoE names like `Qwen3-30B-A3B`, the generator detects the MoE pattern and marks the shape accordingly

**Tier 3: Unknown model name (coarse heuristic)**

When the model name contains a parameter count but does not match any known model family, use a generic heuristic to derive a plausible architecture.

### GPU VRAM Probing

Probes free VRAM via `torch.cuda.mem_get_info` when available. When no GPU is detected, `headroom_bytes` is `null` and a warning is emitted. CPU tests inject `FakeGpuProbe` with deterministic values.

## Limitations

- Without a GPU, memory estimates cannot be verified against real free VRAM
- MoE weight estimates include all experts; actual activation memory for MoE routing may differ
- Dataset row counting supports local `.jsonl`, `.json`, `.csv` files and directories of `.jsonl`
- The generated command uses placeholder `<your-ckpt>` / `<your-dataset>` when those are not provided
- Model-name inference covers Qwen3 (0.6B–72B), Llama (1B–70B), and Gemma (2B–27B)
- `--auto` works best with CUDA available; falls back gracefully on CPU-only environments

## Minimal Runnable Examples

### SFT Recipe

```bash
python .agents/skills/areno-run-training/scripts/generate_recipe.py \
    --algo sft --gpus 8 --tp-size 4 \
    --context-len 2048 --batch-size 16 \
    --format json
```

### DPO Recipe with Memory Estimation

```bash
python .agents/skills/areno-run-training/scripts/generate_recipe.py \
    --algo dpo --gpus 4 --tp-size 2 \
    --context-len 4096 --batch-size 8 \
    --ckpt Qwen/Qwen3-0.6B \
    --set dpo_beta=0.05 --set lr=5e-7
```

### GSPO Recipe with MoE Model Detection

```bash
python .agents/skills/areno-run-training/scripts/generate_recipe.py \
    --algo gspo --gpus 8 --tp-size 4 \
    --context-len 4096 --batch-size 4 \
    --ckpt Qwen/Qwen3-30B-A3B \
    --reward-fn-path examples/math/math_verify_reward.py \
    --format cli
```

### Using Auto-Detection

```bash
python .agents/skills/areno-run-training/scripts/generate_recipe.py \
    --algo gspo --auto \
    --ckpt Qwen/Qwen3-0.6B \
    --reward-fn-path examples/math/math_verify_reward.py
```

## Runtime Requirement Warnings

The generator checks algorithm-specific required parameters and warns when missing (generation is not interrupted). These warnings indicate parameters needed at training runtime:

| Algorithm | Required parameter |
|-----------|-------------------|
| SFT | `--dataset-loader-fn` (for custom dataset loading) |
| GSPO/GRPO/PPO | `--reward-fn-path` or `--reward-ckpt` |
| PPO | `--critic-ckpt` |
| DPO | `--ref-ckpt` |

## CLI Option Name Mapping

Some CLI flags map to non-obvious config field names:

| CLI flag | Config field | Notes |
|----------|-------------|-------|
| `--lr` | `optimizer_lr` | Direct rename |
| `--min-lr` | `optimizer_min_lr` | Direct rename |
| `--adam-beta1` | `optimizer_beta1` | Direct rename |
| `--adam-beta2` | `optimizer_beta2` | Direct rename |
| `--disable-thinking` | `chat_template_enable_thinking` | Ternary: no flag → None, flag → False |
| `--drop-rollout-state` | `keep_rollout_state` | Inverted boolean |

CLI-only utility flags (`--tune-params`, `--mem-frac`, `--smoke-infer`, `--smoke-train`, `--tune-max-samples`) cannot be used with `--set`.

## Troubleshooting

### Memory estimates don't match actual usage

1. Try providing a local checkpoint path instead of a HuggingFace ID
2. Use `--auto` to let the generator probe actual GPU VRAM
3. Manually provide a `ModelShape` via the Python API

### Batch doesn't fit in VRAM

The generator automatically shrinks `batch_size` to fit within available VRAM (95% safety threshold). To adjust manually:

```bash
python .agents/skills/areno-run-training/scripts/generate_recipe.py \
    --algo gspo --gpus 8 --tp-size 4 \
    --context-len 4096 --batch-size 4 \
    --set batch_size=2
```

### Model name not recognized

The generator uses regex to parse parameter counts. Supported formats:
- `Qwen/Qwen3-0.6B` → 0.6B
- `Qwen3-1.7B` → 1.7B
- `Llama-3-8B` → 8B
- `gemma-2-2b` → 2B

If not recognized, provide a local `--ckpt` path or manually override relevant parameters using `--set`.