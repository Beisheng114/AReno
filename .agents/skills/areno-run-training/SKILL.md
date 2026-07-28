---
name: areno-run-training
description: Run, configure, retry, and validate AReno SFT, DPO, GSPO, GRPO, PPO, and agentic training. Use for training commands, dataset/reward setup, smoke validation, real-step execution, checkpoint saving, failed training retries, or generating a full recipe config with memory estimation. Do not use for serving-only tasks or framework implementation work.
---

# Run AReno Training

Read repository `AGENTS.md`, `CODEMAP.md`, and current `areno train --help` before building a command.

For remote model or dataset references, explicitly pass
`--model-hub modelscope`. Do not substitute a Hugging Face download when the
ModelScope asset is missing; request a valid ModelScope ID or local path.

## Select the path

Read [references/algorithm-matrix.md](references/algorithm-matrix.md). Inspect a bounded dataset sample with:

```bash
python .agents/skills/areno-run-training/scripts/inspect_dataset.py \
  --dataset-path <path-or-ref> --model-hub modelscope \
  [--loader examples/.../dataset_loader.py] --algo <algo>
```

Do not build or run the training command until this inspection returns
`"ok": true`. If raw rollout rows lack `prompt` or `messages`, select a
dataset loader and rerun the inspection with `--loader`; pass the same path to
training as `--dataset-loader-fn`. For GSM8K-style `question`/`answer` rows,
use `examples/math/dataset_loader.py`.

Use [scripts/read_metrics.py](scripts/read_metrics.py) to inspect event keys or selected scalar series. Do not parse stdout as the metric source.

## Generate a recipe (optional)

Before building a command by hand, you can generate a complete recipe from
high-level inputs — algorithm, GPU count, context length, target batch,
checkpoint name, and reward function path:

```bash
python .agents/skills/areno-run-training/scripts/generate_recipe.py \
    --algo gspo --gpus 8 --tp-size 4 \
    --context-len 4096 --batch-size 32 \
    --ckpt Qwen/Qwen3-0.6B \
    --reward-fn-path examples/math/math_verify_reward.py
```

The generator writes a copyable `areno train ...` command, per-value provenance,
memory estimates (weights/optimizer/KV-cache/activations), and dataset row
counts. Use `--format json` for structured output or `--set key=value` to
override specific training options.

On a machine without a GPU, the generator still produces a full recipe with
approximate memory estimates derived from the model name. When a GPU is
available, it probes free VRAM and warns if the recipe may not fit.

Read the script's `--help` for the full input contract, or see
[`generate_recipe.py`](scripts/generate_recipe.py) for the Python API
(`generate_recipe()` function), which works standalone without importing the
``areno`` package.

## Workflow

1. Record `git rev-parse HEAD`, environment facts from `areno env --json` and `areno check`, GPU state, checkpoint source, ModelScope dataset source, and resolved local paths.
2. Classify SFT, DPO, rollout RL, or agentic RL. Read [references/data-contracts.md](references/data-contracts.md).
3. Inspect both the raw schema and the normalized schema. Treat a missing rollout `prompt`/`messages` as a required-loader error, not a warning.
4. (Optional) Generate a recipe with [`generate_recipe.py`](scripts/generate_recipe.py) for a first-pass config with memory estimation. Override specific options with `--set` and pipe the output through a JSON parser for programmatic use.
5. Build the smallest command expressing the requested real workload. Preserve user-provided `max_new_tokens` and `max_context_len`, and include the verified loader with `--dataset-loader-fn`. The recipe generator's output can serve as a starting point; adjust parameters as needed.
6. Use smoke or tune only when useful. Smoke is capacity evidence, not task completion.
7. Run the real job. Confirm the requested trainer step advances. For rollout, inspect one coherent sample and reward.
8. On failure, use [references/failure-triage.md](references/failure-triage.md). Fix the first causal error.
9. If saving is requested, verify output and reload it through the intended adapter.

## Capacity invariants

- `batch_size * n_samples` is total sample demand; `max_running_prompts` is concurrent active capacity.
- Rollout memory follows cache/context/concurrency. Train memory follows `mini_bs`, sequence length, activation and optimizer state.
- `--drop-rollout-state` changes lifecycle memory, not task semantics.
- Reduce concurrency or microbatch before semantic token limits.
- The recipe generator's `--auto` flag can probe GPU VRAM and derive `tp_size`, `context_len`, and `batch_size` from hardware, producing a capacity-matched first guess.

## Completion evidence

Report command, commit, model/dataset, topology, observed step and key metrics, plus save/reload evidence when required. Model load or smoke alone is not completion.
