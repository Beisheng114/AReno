# 方案：run-training 训练配方（recipe）生成器

> 对应 issue：为 areno-run-training skill 开发训练配方生成器。
> 实现已完成，代码在 `skills/areno-run-training/`，测试 55 passed。

## 1. 目标与非目标

### 1.1 目标

- 提供一个可执行的 recipe 生成器，输入 **模式（SFT/DPO/online-RL）、GPU 数、上下文长度、目标 batch**，输出：
  1. 一条可直接复制运行的 `areno train ...` 启动命令（只含关键参数 + 显式覆盖，不含全部默认值）；
  2. 一份可编辑的结构化配置（含**逐值来源 provenance**：`default` / `derived` / `explicit`）；
  3. 当提供 checkpoint 路径或 `model_shape` 时，预估权重/优化器/KV-cache/激活显存，并与 GPU 实时剩余显存对比，给出 headroom 警告。
- **独立运行**：脚本内嵌字段定义和默认值表（从 `areno/api/trainer_config.py` 同步），不依赖 areno 包即可运行。当 areno 可导入时，额外做 dataclass `__post_init__` 校验。
- 真实 CLI 选项名：内嵌 `_TRAIN_OPTION_GROUPS`（从 `areno/cli/train.py` 同步），排除 CLI-only 标志（`tune_params`、`mem_frac` 等）。
- 失败时清晰标出受影响阶段与非法输入，不暴露训练样本，不隐藏原始错误。

### 1.2 非目标

- 不替换 trainer、rollout 引擎、dashboard 存储、公共 SDK 架构。
- 不引入外部数据库/托管控制面/重量级依赖。
- 不自动修改用户现有配置、删除 artifacts、终止无关进程。

## 2. 选址与命名

仓库现状：不存在 `.agents/` 目录，现有 skill 位于顶层 `skills/areno-model-adaptation/`，含 `SKILL.md` + `scripts/`。

> 注：issue 文案提到 `.agents/skills/` 与 `.agents/scripts/`，但本仓库实际采用 `skills/<skill-name>/` 约定。本方案对齐现有 `skills/` 结构，避免引入平行的第二套 skill 加载机制。

落盘结构：

```
skills/
  areno-run-training/
    SKILL.md                              # skill 用法文档（含可复制示例）
    scripts/
      generate_recipe.py                  # 生成器主体（可独立执行 + 可 import）
  areno-model-adaptation/                 # 已存在
tests/
  test_run_training_recipe_cpu.py         # CPU 测试套件（55 个测试）
docs/
  plans/
    training-recipe-generator.md          # 本文件
```

## 3. 输入契约

生成器接收显式输入，全部带默认值，未启用时行为不变。

| 输入 | CLI 名 | 类型 | 默认 | 说明 |
|---|---|---|---|---|
| 模式 | `--algo` | `sft\|dpo\|gspo\|grpo\|ppo` | — | 必填，校验内嵌 `VALID_ALGOS` |
| GPU 数 | `--gpus` | int | 8 | 映射为 `world_size` |
| 张量并行 | `--tp-size` | int | 4 | 必须整除 `--gpus` |
| 上下文长度 | `--context-len` | int | 4096 | 拆为 `max_prompt_tokens` + `max_new_tokens` |
| 目标 batch | `--batch-size` | int | 32 | 直接对应 `--batch-size` |
| 模型 | `--ckpt` | str | — | checkpoint 路径，用于架构读取和显存预估 |
| 数据集 | `--dataset-path` | str | — | 支持本地 `.jsonl`/`.json`/`.csv` 行数统计 |
| 奖励函数 | `--reward-fn-path` | str | — | RL 模式运行时必需 |
| 采样数 | `--n-samples` | int | 8 | RL 的每 prompt 采样数 |
| 微批次 | `--mini-bs` | int | min(batch,16) | 训练微批次大小 |
| 8-bit Adam | `--adam-8bit` | flag | off | 影响 optimizer 显存预估 |
| 激活检查点 | `--no-activation-checkpointing` | flag | on | 影响激活显存预估 |
| KV block | `--block-size` | int | 256 | KV cache 块大小 |
| 显式覆盖 | `--set <opt>=<val>...` | 重复 | — | 透传到真实 CLI 选项名，校验合法性 |
| 输出格式 | `--format` | `cli\|json\|both` | `both` | `cli`=启动命令；`json`=结构化配置+provenance |
| 模型架构 | `model_shape` (API 参数) | `ModelShape` | — | 无 ckpt 时直接传入模型架构做显存预估 |

> `--context-len` 拆分策略（可被 `--set max_new_tokens=...` 覆盖）：
> SFT：`max_prompt_tokens = context_len`，`max_new_tokens = 0`（监督路径不生成）；
> DPO：`max_prompt_tokens = context_len // 2`，`max_new_tokens = context_len // 2`；
> online-RL（gspo/grpo/ppo）：`max_prompt_tokens = min(1024, context_len // 4)`，`max_new_tokens = context_len - max_prompt_tokens`。
> 拆分值始终标注 `provenance=derived`，显式覆盖后改为 `explicit`。

## 3.5 显存预估模型（已实现）

当 `--ckpt` 指向可加载的本地 checkpoint 时，生成器通过 `areno.models.registry.config_from_hf` 读取 HF `config.json` 获取架构参数。若 areno 不可用或无 ckpt，可通过 `model_shape` 参数直接传入 `ModelShape` 进行预估。

| 组件 | 公式 | 来源 |
|---|---|---|
| **权重** | `param_count * dtype_bytes`；param_count 按 GQA + SwiGLU 布局逐项计算（embedding + layers*(q+k+v+o+gate+up+down+norms) + lm_head），支持 MoE | `ModelConfig` 字段 |
| **优化器** | fp32 Adam: `params * (grad_dtype + 12)`；8-bit: `params * (grad_dtype + 6)` | `adam_8bit` 标志 |
| **KV cache** | `num_layers * 2 * (num_blocks+1) * block_size * local_kv_heads * head_dim * dtype_bytes` | 对齐 `Model.allocate_kv_caches` + `InferenceBatchState` |
| **激活** | `34 * hidden * seq * mini_bs * dtype_bytes`，开启检查点时 ×0.3 | 标准估算 |

- 权重和优化器被 `tp_size` 切分（TP-sharded），DP 复制。
- KV cache 和激活是 per-GPU。
- `local_kv_heads = num_key_value_heads // tp_size`（TP 分 shard KV heads）。
- `num_blocks = max_running_seqs * max_blocks_per_seq`，`max_blocks_per_seq = ceil(max_cache_len / block_size)`，`max_cache_len = max_prompt_tokens + max_new_tokens`。
- RL 模式 `max_running_seqs = batch_size * n_samples`；离线模式 `max_running_seqs = batch_size`。
- GPU 探测通过 `GpuProbe` 协议隔离——真实实现用 `torch.cuda.mem_get_info`，CPU 测试用 fake 注入确定性值。
- `headroom_bytes = free_vram - total_estimated`，负值触发警告 `"Estimated memory exceeds free VRAM by X GB"`。

## 4. 输出契约

### 4.1 人类可读（`--format cli`）

一条等价于 `areno train ...` 的命令字符串，仅含关键参数（必填字段 + 按算法分组的必需字段）和显式覆盖，不含全部默认值。命令后附 `#` 注释块说明 warnings 和 provenance。可直接复制到 shell 运行（替换占位符后）。

### 4.2 结构化（`--format json`）

```jsonc
{
  "algo": "gspo",
  "command": "areno train --algo gspo --ckpt <your-ckpt> ...",
  "config": {
    "world_size": {"value": 8, "source": "derived"},
    "tp_size":    {"value": 4, "source": "derived"},
    "batch_size": {"value": 32, "source": "derived"},
    "max_prompt_tokens": {"value": 1024, "source": "derived"},
    "max_new_tokens":    {"value": 3072, "source": "derived"},
    "reward_fn_path":    {"value": "reward.py", "source": "default"},
    "optimizer_lr":      {"value": 1e-06, "source": "default"}
    // ...
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
  "warnings": [
    "No GPU detected -- memory estimates only; cannot verify fit."
  ],
  "dataset_rows": null
}
```

每个 config 项：`{"value": <T>, "source": "default|derived|explicit"}`。`warnings` 列出非致命提示。`memory` 在有 `model_shape` 或可读 ckpt 时非 null。

## 5. 实现要点

### 5.1 独立运行（不引入硬依赖 areno）

脚本内嵌以下数据（从源码同步），使其在无 areno/torch 的环境也能独立运行：

- **`VALID_ALGOS`**：算法名 → config 类名映射，同步自 `areno.api.algorithms._register_builtin`。
- **`_TRAIN_OPTION_GROUPS`**：CLI 选项分组，同步自 `areno/cli/train.py:48-133`。
- **`_BASE_DEFAULTS` / `_ROLLOUT_DEFAULTS` / `_POLICY_DEFAULTS` / `_DPO_DEFAULTS` / `_PPO_DEFAULTS`**：config 字段默认值，同步自 `areno/api/trainer_config.py:19-190`。
- **`_field_defaults_for_algo(algo)`**：按算法合并最窄 config 类的全部字段默认值。
- **`_is_valid_field(algo, field_name)`**：检查字段是否属于该算法的 config 类。

当 areno 可导入时，`_validate_with_areno()` 额外构造真实 dataclass 做 `__post_init__` 校验，失败仅 warn 不中断。

### 5.2 CLI 选项名映射

以下字段的 CLI 选项名与 config 字段名不同，需显式映射（同步自 `_trainer_config_from_args`，`train.py:595-791`）：

| CLI 选项名 | config 字段名 | 说明 |
|---|---|---|
| `lr` | `optimizer_lr` | 直接映射 |
| `min_lr` | `optimizer_min_lr` | 直接映射 |
| `adam_beta1` | `optimizer_beta1` | 直接映射 |
| `adam_beta2` | `optimizer_beta2` | 直接映射 |
| `disable_thinking` | `chat_template_enable_thinking` | 三值映射：无 flag → None，`--disable-thinking` → False |
| `drop_rollout_state` | `keep_rollout_state` | 反转布尔：`--drop-rollout-state` → `keep_rollout_state=False` |

### 5.3 CLI-only 选项排除

以下 CLI 标志出现在 `TRAIN_OPTION_GROUPS` 中但没有对应的 config dataclass 字段，它们控制 train 命令自身的预检工具（调参、冒烟测试），不能通过 `--set` 覆盖：

- `tune_params`、`mem_frac`、`tune_max_samples`、`smoke_infer`、`smoke_train`

`_valid_option_names()` 从白名单中排除这些标志。

### 5.4 校验顺序（输入在昂贵初始化前完成）

1. `algo` ∈ `VALID_ALGOS` → 否则 `ValueError`（标阶段 `algo-resolution`，列已注册算法）。
2. `gpus > 0`、`tp_size > 0`、`gpus % tp_size == 0` → 否则报 `--world-size must be divisible by --tp-size`（标阶段 `parallelism`）。
3. `context_len > 0`、`batch_size > 0` → 否则正值校验（标阶段 `sizing`）。
4. `--set` 键 ∈ `_valid_option_names()` → 否则报 `unknown option '<key>'; valid: ...`（标阶段 `override`，列全部合法选项）。
5. CLI 选项名 → config 字段名转换（含反转布尔和三值映射）。

> 失败信息格式：`[stage=<stage>] <message>`，仅含受影响输入与原因，不打印数据集样本或完整 trace。

### 5.5 算法特定的运行时必需参数警告

生成器对每种算法检查运行时必需参数是否缺失，缺失时追加 warning（不中断生成）：

| 算法 | 必需参数 | 警告文案 |
|---|---|---|
| SFT | `dataset_loader_fn` | `--algo sft requires --dataset-loader-fn at run time.` |
| GSPO/GRPO/PPO | `reward_fn_path` 或 `reward_ckpt` | `--algo <algo> requires --reward-fn-path or --reward-ckpt at run time.` |
| PPO | `critic_ckpt` | `--algo ppo requires --critic-ckpt at run time.` |
| DPO | `ref_ckpt` | `--algo dpo requires --ref-ckpt at run time (or it defaults to actor ckpt).` |

### 5.6 provenance 记录

- `default`：字段值来自 `_field_defaults_for_algo(algo)` 中的默认值。
- `derived`：由 `--gpus`/`--context-len`/`--batch-size` 推导出的值（`world_size`、`tp_size`、`batch_size`、`mini_bs`、`max_prompt_tokens`、`max_new_tokens`、RL 的 `n_samples`/`keep_rollout_state`、DPO 的 `dpo_beta` 等）。
- `explicit`：通过 `--set` 提供的值。

### 5.7 命令渲染

`_build_command()` 只输出以下两类参数：

1. **必需字段**（按算法分组）：`_BASE_REQUIRED` 对所有算法、`_RL_REQUIRED` 对 RL、`_DPO_REQUIRED` 对 DPO。
2. **显式覆盖**：`source == "explicit"` 的字段。

非必需的默认值不出现在命令中（如 `epochs`、`score_micro_bs`、`optimizer_lr` 等），但完整出现在 provenance / JSON 输出中。

### 5.8 确定性

- 不读取时间、随机数、环境 GPU 状态（`GpuProbe` 可注入 fake）；输入相同 → 输出字符串逐字节相同（测试断言）。
- 字段排序固定：按 `_TRAIN_OPTION_GROUPS` 的分组与顺序输出命令行选项。

## 6. 测试计划

新增 `tests/test_run_training_recipe_cpu.py`，纯 CPU，无 GPU/网络，**55 个测试全部通过**。

| 测试类 | 测试数 | 覆盖内容 |
|---|---|---|
| `TestParamCount` | 2 | 参数量公式（dense + tied embeddings） |
| `TestWeightBytes` | 2 | bf16/fp32 权重字节数 |
| `TestOptimizerBytes` | 2 | fp32/8-bit Adam 状态字节数 |
| `TestKvCacheBytes` | 2 | KV cache 公式 + TP sharding |
| `TestEstimateMemory` | 3 | 总和校验 + headroom 正/负 |
| `TestSplitContextLen` | 5 | SFT/DPO/RL 拆分 + 边界 |
| `TestRecipeSuccess` | 5 | SFT/DPO/GSPO/PPO 配方 + offline 无 rollout 字段 |
| `TestRecipeDeterminism` | 1 | 同输入同输出 |
| `TestRecipeValidation` | 7 | 非法 algo/tp/gpus/context_len/batch/override/CLI-only 选项 |
| `TestRecipeOverrides` | 4 | lr/min_lr/drop_rollout_state/disable_thinking |
| `TestProvenance` | 4 | default/derived/RL derived/explicit |
| `TestOutputFields` | 11 | JSON 字段、命令选项名合法性、排除无关默认值、显存预估、警告（RL/PPO/DPO/SFT）、reward_ckpt 替代 |
| `TestDatasetCounting` | 2 | jsonl 行数统计 + 远程 ref 返回 None |
| `TestCliOverrideParsing` | 5 | int/float/bool/string/invalid |

最小确定性 fixture：所有测试均以常量输入运行，无需外部数据库/沙箱。

## 7. 文档

- `skills/areno-run-training/SKILL.md`：描述输入契约、默认值、输出字段、限制、可复制示例（SFT/DPO/GSPO 各一）、显存预估说明。
- 本文档 `docs/plans/training-recipe-generator.md`：完整设计方案与实现记录。

## 8. 兼容性

- 新增 skill 与脚本，不改 `areno/cli/train.py`、不改 `TrainerConfig` 数据类、不加依赖。
- 脚本可独立运行（仅需 Python 3.9+ 标准库），不需要 areno/torch/CUDA。
- 现有 `areno train` 行为与现有 `pytest tests/ -k cpu` 不受影响。
- 测试已在 Python 3.9 + pytest 8.4 下运行通过（`55 passed`）。

## 9. 评审清单

- [x] 三模式生成可直接运行且确定的 recipe，支持显式覆盖，校验真实选项名。
- [x] 复用现有契约（内嵌同步自 `trainer_config.py` 和 `train.py`），无外部数据库/强制沙箱。
- [x] 默认行为向后兼容。
- [x] CPU 测试覆盖成功、非法输入、边界/失败路径，断言输出字段与错误信息（55 个测试全部通过）。
- [x] SKILL.md 含最小可运行示例并解释可观测输出。
