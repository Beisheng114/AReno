# PR: feat(areno-run-training): 训练配方生成器 (Training Recipe Generator)

---

## 一、对 PR 任务的理解

### 1.1 当前代码存在的问题 / 缺少的能力

在 `areno-run-training` skill 中，用户要构建一条完整的 `areno train` 命令，需要手动查阅以下信息：

- **算法配置类层级** — SFT 用 `TrainerConfig`，DPO 用 `DPOTrainerConfig`，GSPO/GRPO 用 `PolicyTrainerConfig`，PPO 用 `PPOTrainerConfig`，每个类有不同字段集和默认值。
- **CLI 选项名与配置字段名的映射** — `--lr` 映射到 `optimizer_lr`，`--disable-thinking` 映射到 `chat_template_enable_thinking`，`--drop-rollout-state` 是 `keep_rollout_state` 的布尔反转。
- **上下文长度拆分规则** — SFT 全部分配给 prompt，DPO 对半分，RL 算法取 `min(1024, context_len // 4)` 给 prompt。
- **显存估算** — 需要知道权重、优化器、KV-cache、激活各自占用多少字节，以及是否超出了 GPU 可用显存。
- **多角色算法的额外内存** — PPO 有 critic（可训练）+ ref（冻结），DPO 有 ref（冻结），这些额外模型的内存必须计入。
- **Agentic RL 参数** — `--agent-fn`、`--agent-timeout-s`、`--train-tool-results` 等参数何时需要、如何传入。

整个过程高度依赖经验，容易出错（如 tp_size 不能整除 gpus、batch_size 超显存、mini_bs > batch_size 等），且缺少一个"给出高层输入 → 得到完整可运行命令 + 内存估算 + 每个值的来源标注"的工具。

### 1.2 本 PR 的目标

为本 skill 添加一个独立的训练配方生成器脚本 `generate_recipe.py`，实现：

- 接受算法（SFT/DPO/GSPO/GRPO/PPO）、GPU 数量、张量并行大小、上下文长度、目标 batch size 等高层参数
- 输出可直接复制的 `areno train` 命令
- 对每个配置值标注来源：`default`（内置默认）/ `derived`（从输入推导）/ `explicit`（用户显式覆盖）
- 三级内存估算降级：本地 checkpoint 配置 → 模型名推断 → 通用参数量启发式
- GPU 显存探测后自动调低 batch_size 以适配可用 VRAM
- 支持 `--auto` 模式从硬件自动推导 tp_size / context_len / batch_size
- 支持 `--set key=value` 显式覆盖任意配置项
- 支持 `--format cli|json|both` 三种输出格式

### 1.3 本 PR 明确不处理的内容

- **不修改 `areno` 核心包** — 不改动 `areno/api/trainer_config.py`、`areno/cli/train.py` 或任何核心数据类
- **不添加新依赖** — 脚本仅使用 Python 标准库（`argparse`、`json`、`re`、`pathlib`、`dataclasses`、`typing`）
- **不执行训练** — 生成器只产出配置和命令，不启动训练进程
- **不覆盖 `areno train --tune-params` 的功能** — `--tune-params` 是运行时内存探测调优，本脚本是启动前的静态配方生成
- **不支持 GPU 上的实际前向推理验证** — 内存估算是基于架构参数的解析公式，不加载真实权重

### 1.4 修改影响的模块、接口和使用场景

| 模块/文件 | 影响类型 | 说明 |
|-----------|----------|------|
| `.agents/skills/areno-run-training/scripts/generate_recipe.py` | 新增 | 主脚本，独立运行，不依赖 `areno` 包 |
| `.agents/skills/areno-run-training/SKILL.md` | 修改 | 新增 "Generate a recipe (optional)" 章节和工作流第 4 步 |
| `.agents/skills/areno-run-training/references/recipe-generator.md` | 新增 | 英文用户文档 |
| `tests/test_recipe_generator_cpu.py` | 新增 | 77 个 CPU 测试用例 |

**使用场景：**
- 用户在启动训练前快速生成一条可用命令
- CI/CD 管道中程序化生成配方（JSON 输出）
- 无 GPU 环境下预估训练内存需求
- Skill agent 自动为用户推导配置参数

### 1.5 验收标准

1. `generate_recipe.py` 可在无 `areno` 包、无 GPU 的环境下独立运行
2. 对 SFT/DPO/GSPO/GRPO/PPO 五种算法均能生成包含必需字段的命令
3. 内存估算三级降级全部可用且有对应测试
4. GPU 探测可用时自动调低 batch_size 以适配 VRAM
5. 所有 CPU 测试通过（`pytest tests/test_recipe_generator_cpu.py -v`）
6. SKILL.md 更新包含配方生成步骤
7. 不改变 `areno train` CLI 或核心 API 的任何行为

---

## 二、实现思路

### 2.1 修改涉及的主要文件和模块

- **`generate_recipe.py`**（1354 行）— 配方生成器主脚本
- **`test_recipe_generator_cpu.py`**（816 行）— CPU 测试套件
- **`SKILL.md`** — skill 描述和工作流更新
- **`recipe-generator.md`** — 用户文档

### 2.2 核心流程和数据流

```
用户输入 (CLI args)
    │
    ▼
┌─────────────────────────────────────┐
│ 1. 算法验证 (VALID_ALGOS)            │
│ 2. --auto 模式: GPU 探测 → 推导参数    │
│ 3. 并行度验证 (gpus % tp_size == 0)   │
│ 4. 尺寸验证 (context_len, batch_size) │
│ 4b. 覆盖验证 + 枚举值验证              │
│ 5. 上下文拆分 (split_context_len)     │
│ 6. GPU 显存探测 (GpuProbe)            │
│ 7. 模型配置加载 → ModelShape           │
│ 7b. 模型名推断 → ModelShape (降级)     │
│ 8. 内存估算 (estimate_memory)         │
│    └─ batch_size 自动调低适配 VRAM     │
│ 9. 数据集行数统计                      │
│ 10. 构建配置字典 + provenance         │
│ 11. 构建 areno train 命令             │
│ 12. 可选: areno 数据类验证            │
└─────────────────────────────────────┘
    │
    ▼
RecipeResult { command, config, memory, warnings, dataset_rows }
    │
    ├── --format cli  → 可复制命令 + 注释
    ├── --format json → JSON 对象
    └── --format both → cli + json
```

### 2.3 关键数据结构

#### `ModelShape`（frozen dataclass）

```python
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
    param_count_override: "int | None" = None  # 模型名推断时使用精确参数量
```

`param_count_override` 是关键设计：当从模型名推断 shape 时，架构字段（layers/heads 等）是近似值，但参数量是从名称精确解析的（如 `Qwen3-0.6B` → 600,000,000）。此时权重和优化器估算使用 override 值（精确），KV-cache 和激活估算使用架构字段（近似）。

#### `MemoryBreakdown`（frozen dataclass）

```python
@dataclass(frozen=True)
class MemoryBreakdown:
    weights: int           # per-GPU weight bytes
    optimizer: int          # per-GPU optimizer bytes
    kv_cache: int           # per-GPU KV cache bytes
    activations: int        # per-GPU activation bytes
    total: int              # max(rollout_phase, train_phase)
    per_gpu_total: int
    per_gpu_free: "int | None"
    headroom_bytes: "int | None"
```

`total` 取 rollout 和 train 两个阶段的较大值，而非之和 — 因为 AReno 的 RL loop 中 rollout 和 train 交替执行，权重/优化器在 train 阶段加载，KV-cache 在 rollout 阶段加载，不同时占用。

#### `ConfigValue` + `RecipeResult`

```python
@dataclass(frozen=True)
class ConfigValue:
    value: Any
    source: str  # "default" | "derived" | "explicit"

@dataclass(frozen=True)
class RecipeResult:
    algo: str
    command: str
    config: dict           # field_name -> ConfigValue
    memory: "MemoryBreakdown | None"
    warnings: list
    dataset_rows: "int | None"
```

### 2.4 关键算法

#### 内存估算公式

| 组件 | 公式 | 说明 |
|------|------|------|
| 权重 | `param_count * dtype_bytes / tp_size` | bf16 = 2 bytes/param |
| 优化器 | `param_count * (dtype_bytes + 12) / tp_size` | Adam: 2 momentum + 2 variance + 8 master fp32 |
| 优化器(8bit) | `param_count * (dtype_bytes + 6) / tp_size` | 8bit Adam: 1+1+4 |
| KV-cache | `layers * 2 * blocks * block_size * (kv_heads/tp) * head_dim * dtype` | blocks = ceil(max_cache_len/256) * seqs_per_gpu |
| 激活 | `34 * hidden * seq_len * mini_bs * dtype * (0.3 if ckpt else 1.0)` | 34 = 每层激活系数 |

#### 三级模型 shape 降级

```
1. 本地 checkpoint → areno.models.registry.config_from_hf(ckpt) → model_shape_from_config()
   (需要 areno 包安装)
2. 模型名推断 → parse_param_count_from_name() + _KNOWN_SHAPES 匹配 → ModelShape
   (支持 Qwen3/Llama/Gemma 系列，精确参数量 + 近似架构)
3. 通用启发式 → 只用 param_count_override
   (任何含 "XXb"/"XXm" 的名称)
```

#### batch_size 自动调低

当 `memory.headroom_bytes < 0` 且 batch_size 未被显式覆盖时，从 `[64, 32, 16, 8, 4, 2, 1]` 依次尝试，找到第一个 `cand_mem.total <= free_vram * 0.95` 的值。

#### 上下文长度拆分

| 算法 | max_prompt_tokens | max_new_tokens |
|------|-------------------|----------------|
| SFT | context_len | 0 |
| DPO | context_len // 2 | context_len - half |
| RL (GSPO/GRPO/PPO) | min(1024, context_len // 4) | context_len - prompt |

### 2.5 重要设计选择及理由

**选择 1：内置默认值而非运行时导入 `areno` 包**

理由：脚本需要在没有安装 `areno` 的环境（如 CI、本地开发机）下独立运行。默认值从 `trainer_config.py` 手动同步为 `_BASE_DEFAULTS`、`_ROLLOUT_DEFAULTS` 等字典。当 `areno` 可导入时，额外运行 `_validate_with_areno()` 进行真实数据类验证。

**选择 2：`Protocol` 抽象 GPU 探测**

理由：`GpuProbe` 是 `Protocol`，`RealGpuProbe` 和 `FakeGpuProbe` 都满足接口。测试注入 `FakeGpuProbe` 实现确定性验证，生产代码用 `RealGpuProbe` 调用 `torch.cuda.mem_get_info`。

**选择 3：多卡取最小值而非平均值**

理由：`RealGpuProbe.probe()` 在多卡场景取所有卡的最小 free 和最小 total（保守策略），确保配方在任何一张卡上都不会 OOM。

**选择 4：rollout/train 取 max 而非 sum**

理由：AReno 的 RL loop 中，rollout 阶段模型持有 KV-cache + 推理权重，train 阶段释放 KV-cache 并加载训练权重 + 优化器 + 激活。两阶段交替，峰值取较大者。`generate_recipe.py:566-584` 有详细注释说明。

**选择 5：`param_count_override` 机制**

理由：从模型名推断 shape 时，参数量是精确的（`Qwen3-0.6B` → 600M），但架构字段（layers=28, hidden=1024）是从 `_KNOWN_SHAPES` 表查到的近似值。权重/优化器用精确参数量，KV-cache/ activations 用近似架构字段，在精度和可用性之间取得平衡。

### 2.6 未采用的方案

| 方案 | 不采用原因 |
|------|-----------|
| 运行时导入 `areno` 包获取默认值 | 破坏独立运行需求；CI 环境无 `areno` |
| 用 `subprocess` 调用 `areno train --help` 解析选项 | 过于脆弱，依赖 CLI 输出格式稳定性 |
| 实际加载模型权重验证显存 | 等同于 `--tune-params` 功能，超出本脚本定位 |
| 使用 `tomllib` 解析 `pyproject.toml` 获取依赖 | 脚本无新依赖要求；默认值已手动同步 |

### 2.7 兼容性、性能、异常处理考虑

- **兼容性**：不修改任何 `areno` 核心 API 或 CLI；skill 描述更新是纯增量
- **性能**：所有计算为纯 Python 解析公式，无 I/O 密集操作；数据集行数统计使用生成器避免大文件全加载
- **异常处理**：每个验证阶段用 `[stage=xxx]` 前缀标记错误来源；`areno` 导入失败时静默降级；模型配置加载失败时降级到模型名推断；模型名无法解析时降级到通用启发式
- **Python 3.9 兼容**：类型注解使用 `Optional[List[int]]` 而非 `list[int] | None`（3.9 dataclass 不支持后者）

---

## 三、对自己代码的 Review

### 3.1 正确性

| 检查项 | 结论 | 依据 |
|--------|------|------|
| SFT 命令生成 | ✓ 正确 | `--max-new-tokens 0`（SFT 无生成），`--max-prompt-tokens` 等于 context_len |
| DPO 命令生成 | ✓ 正确 | `--dpo-beta 0.1` 出现在命令中，`--ref-ckpt` 不在命令中（默认为 None，运行时用 actor ckpt） |
| GSPO 命令生成 | ✓ 正确 | `--reward-fn-path` 出现，`--n-samples 4` 出现，`max_running_prompts` 为 None（运行时填充） |
| 上下文拆分边界 | ✓ 正确 | 128k 极端值测试通过：SFT→(131072,0)，DPO→(65536,65536)，RL→(1024,130048) |
| batch_size=1 | ✓ 正确 | `mini_bs` 被 cap 到 `min(1, 16) = 1` |
| gpus=0 | ✓ 正确 | 0 被默认为 8，不报错 |
| gpus=-1 | ✓ 正确 | 默认不触发（-1 != 0），后续验证报 `[stage=parallelism] --gpus must be positive` |
| 负数 context_len | ✓ 正确 | 0 被默认为 2048，但 -1 不被默认，`split_context_len` 报 `context_len must be positive` |
| 显式 batch_size 不被自动调低 | ✓ 正确 | `test_explicit_batch_not_auto_adjusted` 验证 overrides 中 batch_size=64 在 VRAM 不足时仍保持 64 |

### 3.2 可读性

| 检查项 | 结论 | 依据 |
|--------|------|------|
| 命名 | ✓ 清晰 | `estimate_kv_cache_bytes`、`split_context_len`、`_field_defaults_for_algo` 等，职责可从名称推断 |
| 注释 | ✓ 充分 | `MemoryBreakdown` 取 max 的理由在 `generate_recipe.py:566-583` 有 8 行注释；`_KNOWN_SHAPES` 表有结构说明 |
| 函数职责 | ✓ 单一 | 每个函数做一件事：`parse_param_count_from_name` 只解析名称，`infer_shape_from_name` 只做 shape 推断，`estimate_memory` 只做内存估算 |
| 常量分组 | ✓ 清晰 | `== constants ==`、`== memory estimation ==`、`== context-length splitting ==` 等分区注释 |

### 3.3 复用性

| 检查项 | 结论 | 依据 |
|--------|------|------|
| 重复代码 | ✓ 无明显重复 | `estimate_weight_bytes`、`estimate_optimizer_bytes`、`estimate_kv_cache_bytes` 各自独立实现，但都是不同公式，不存在可合并的重复 |
| 数据集行数统计 | ✓ 使用生成器 | `.jsonl` 和 `.csv` 都用 `sum(1 for _ in f)` 避免全加载 |
| 默认值字典合并 | ✓ 复用 `_field_defaults_for_algo` | 基类默认值 + 算法特定默认值一次合并，`_is_valid_field` 和 provenance 构建都调用它 |

### 3.4 兼容性

| 检查项 | 结论 | 依据 |
|--------|------|------|
| 已有默认行为 | ✓ 未改变 | 不修改 `areno train` CLI 选项、`TrainerConfig` 数据类或任何公开 API |
| SKILL.md 工作流 | ✓ 向后兼容 | 新增第 4 步标注 "(Optional)"，原有步骤 1-3 和 5-9 保持不变 |
| 公开接口 | ✓ 纯增量 | 新增脚本和文档，不删除或重命名任何现有内容 |

### 3.5 异常处理

| 检查项 | 结论 | 依据 |
|--------|------|------|
| 未知算法 | ✓ 有 `[stage=algo-resolution]` 前缀 | `generate_recipe.py:891-893` |
| 并行度不整除 | ✓ 有 `[stage=parallelism]` 前缀 | `generate_recipe.py:949-950` |
| 未知覆盖字段 | ✓ 有 `[stage=override]` 前缀 + 合法值列表 | `generate_recipe.py:967-969` |
| 枚举值非法 | ✓ 有字段名 + 合法值列表 | `generate_recipe.py:985-990` |
| areno 导入失败 | ✓ 静默降级 | `_try_areno_available()` 返回 False，跳过验证 |
| 模型配置加载失败 | ✓ 降级到模型名推断 | `generate_recipe.py:1040-1042` 捕获异常并 warning |
| GPU 探测失败 | ✓ 降级到估算模式 | `generate_recipe.py:1029` warning "No GPU detected" |
| CLI 错误退出码 | ✓ exit code 1 | `main()` 中 `ValueError` 被捕获，打印到 stderr，返回 1 |

### 3.6 测试

| 检查项 | 结论 | 依据 |
|--------|------|------|
| 新增逻辑测试覆盖 | ✓ 77 个测试 | 见下方测试分类表 |
| 原有测试不受影响 | ✓ 无关联 | 本 PR 只新增文件，不修改 `areno` 核心代码 |
| 测试无 GPU 依赖 | ✓ 全部 CPU | `FakeGpuProbe` 注入确定性 VRAM 值 |
| 测试无 areno 包依赖 | ✓ 独立运行 | `test_no_areno_import_dependency` 验证 |

**测试分类（77 个用例）：**

| 测试类 | 用例数 | 覆盖范围 |
|--------|--------|----------|
| `TestSFTRecipe` | 4 | 命令生成、上下文拆分、溯源、占位符 |
| `TestDPORecipe` | 3 | 命令生成、上下文拆分、显式覆盖 |
| `TestGSPORecipe` | 4 | RL 命令、上下文拆分、奖励函数警告、n_samples |
| `TestInvalidInputs` | 11 | 未知算法、不整除、零/负 GPU、零/负 context_len、零/负 batch_size、未知覆盖、CLI-only 选项 |
| `TestDeterminism` | 2 | 相同输入相同输出、多次运行命令稳定 |
| `TestMemoryEstimation` | 7 | 权重/优化器/8bit/KV-cache/激活检查点/完整字段/负 headroom |
| `TestAutoBatchAdjust` | 3 | VRAM 不足时缩小、充足时保持、显式时不调整 |
| `TestModelNameInference` | 7 | Qwen3 参数解析、百万级、无法解析、shape 推断、MoE 检测、未知系列、精确权重 |
| `TestDatasetRowCount` | 6 | jsonl/json/csv/不存在/目录/steps_per_epoch |
| `TestCLI` | 8 | JSON 输出、CLI 输出、无效算法、并行度错误、--set 覆盖、无 areno 依赖、--auto 有效、--auto 无 GPU 降级 |
| `TestBoundaryValues` | 10 | 235B MoE、60M 小模型、128k 上下文、batch=1、batch=1024、单 GPU、128 GPU、tp=gpus、空数据集、长模型名、特殊字符 |
| `TestAgenticParams` | 3 | 全 agentic 参数组合、短超时警告、无 agent-fn 默认 |
| `TestMultiRoleMemory` | 3 | PPO > GRPO 内存、多角色警告、直接 estimate_memory 测试 |
| `TestParameterValidation` | 3 | n_samples=0、mini_bs > batch_size、非法 attn_backend |
| `TestLearningRateFlag` | 2 | --lr 出现且 explicit、--lr 默认 |
| `TestAutoDetectNSamples` | 1 | n_samples 影响 batch_size 推导 |

### 3.7 性能

| 检查项 | 结论 | 依据 |
|--------|------|------|
| 额外计算 | ✓ 可忽略 | 所有估算为 O(1) 解析公式；无循环遍历模型权重 |
| 内存开销 | ✓ 极小 | 数据集行数统计用生成器；无大对象分配 |
| I/O 开销 | ✓ 最小 | 仅 GPU 探测（`torch.cuda.mem_get_info`）和数据集行数统计（文件读取） |

### 3.8 提交范围

| 检查项 | 结论 | 依据 |
|--------|------|------|
| 无关格式化 | ✓ 无 | 本 PR 只新增文件和更新 SKILL.md 描述 |
| 文件修改 | ✓ 精确 | 4 个文件，共 2507 insertions, 6 deletions（SKILL.md 的 6 行删除是原文替换） |

### 3.9 Review 后实际发现并处理的问题

| 问题 | 处理方式 |
|------|----------|
| Python 3.9 dataclass 与 `from __future__ import annotations` 不兼容 | 类型注解从 `"list[int] | None"` 改为 `"Optional[List[int]]"` |
| 测试加载模块时 dataclass 内省失败 | 在 `exec_module` 前注册 `sys.modules[_gen_name] = gen` |
| `auto_detect_params` 硬编码 n_samples=8 | 修改为接受 `n_samples` 参数，RL 算法用实际值 |
| CSV 行数统计未排除表头 | 修改为 `max(sum(1 for _ in f) - 1, 0)` |
| `--auto` 无 GPU 时缺少降级路径 | 增加 fallback 到默认值 + warning |
| DPO `ref_ckpt` 缺失时缺少运行时提示 | 增加算法特定 requirement 检查（warning 而非 error） |

---

## 四、遇到的问题、挑战与解决方法

### 问题 1：Python 3.9 dataclass 与 `from __future__ import annotations` 不兼容

**现象：**

```
AttributeError: 'NoneType' object has no attribute '__dict__'
```

在测试文件中通过 `importlib.util.spec_from_file_location` 加载 `generate_recipe.py` 时，`@dataclass(frozen=True)` 装饰器抛出异常。

**定位过程：**

1. 查看 traceback，发现错误发生在 `@dataclass` 装饰器内部
2. 确认本地 Python 版本为 3.9.6（`python3 --version`）
3. 分析：`from __future__ import annotations` 使所有注解变为字符串，dataclass 在 3.9 中通过 `typing.get_type_hints()` 解析字符串注解时需要访问模块的 `__dict__`
4. 用 `importlib.util` 动态加载的模块未在 `sys.modules` 中注册，导致 `get_type_hints()` 找不到模块

**根因：**

Python 3.9 的 `dataclasses` 模块在处理 `from __future__ import annotations` 时，需要通过 `sys.modules` 查找模块以解析字符串类型注解。`importlib.util.module_from_spec()` 创建的模块在调用 `exec_module` 前未注册到 `sys.modules`。

**解决方法：**

在测试文件中，`exec_module` 前注册模块：

```python
_spec = importlib.util.spec_from_file_location("generate_recipe", SCRIPT)
gen = importlib.util.module_from_spec(_spec)
import sys as _sys
_sys.modules[_spec.name] = gen   # <-- 关键：先注册再执行
_spec.loader.exec_module(gen)
```

同时，将脚本中的类型注解从 `"list[int] | None"` 改为 `"Optional[List[int]]"`，确保 3.9 兼容。

**验证方式：**

```bash
python3 -m pytest tests/test_recipe_generator_cpu.py -v
# 77 passed in 7.16s
```

**经验总结：**

动态加载含 dataclass + `from __future__ import annotations` 的模块时，必须在 `exec_module` 前注册到 `sys.modules`。以后在测试中加载非包内模块时，应优先检查 Python 版本兼容性。

---

### 问题 2：`auto_detect_params` 硬编码 n_samples

**现象：**

`auto_detect_params` 函数在估算 batch_size 时，RL 算法的 KV-cache 需求按 `batch_size * 8` 计算（硬编码 n_samples=8），但用户传入 `n_samples=4` 时，推导出的 batch_size 偏小。

**定位过程：**

1. 审查 `auto_detect_params` 函数，发现 `n_samples_eff = 8 if algo in RL_ALGOS else 1`
2. 与 `generate_recipe` 函数的 `n_samples` 参数（默认 4）对比，发现不一致
3. 编写 `TestAutoDetectNSamples` 测试验证：n_samples=8 推导出的 batch_size 应 <= n_samples=4 的

**根因：**

`auto_detect_params` 在实现时直接硬编码了 8，忽略了函数参数 `n_samples`。

**解决方法：**

```python
# 修改前
n_samples_eff = 8 if algo in RL_ALGOS else 1

# 修改后
n_samples_eff = n_samples if algo in RL_ALGOS else 1
```

**验证方式：**

```python
def test_auto_detect_uses_custom_n_samples(self):
    params_4 = gen.auto_detect_params(gpu, param_count, "grpo", n_samples=4)
    params_8 = gen.auto_detect_params(gpu, param_count, "grpo", n_samples=8)
    assert params_8.batch_size <= params_4.batch_size
```

**经验总结：**

函数有参数时，应始终使用参数值而非硬编码。编写测试时，应覆盖不同参数值的行为差异。

---

### 问题 3：多角色算法内存估算遗漏

**现象：**

PPO 的内存估算只计算了 actor 模型，未计算 critic（可训练，有优化器）和 ref（冻结，无优化器）的额外内存。

**定位过程：**

1. 分析 PPO 训练流程：actor + critic（均需优化器状态）+ ref（冻结权重）
2. 对比 GRPO（只有 actor），PPO 内存应约为 2x
3. 编写 `test_ppo_memory_higher_than_grpo` 测试，断言 `1.8 < ratio < 2.5`
4. 初版未通过 — PPO 和 GRPO 内存几乎相同

**根因：**

`estimate_memory` 函数没有 `num_extra_trainable` 和 `num_extra_frozen` 参数。

**解决方法：**

在 `estimate_memory` 中增加两个参数：

```python
def estimate_memory(
    shape, *, ..., num_extra_trainable: int = 0, num_extra_frozen: int = 0,
) -> MemoryBreakdown:
    extra_trainable_weights = per_gpu_weights * num_extra_trainable
    extra_trainable_opt = per_gpu_opt * num_extra_trainable
    extra_frozen_weights = per_gpu_weights * num_extra_frozen
    # ... 加入 rollout_phase 和 train_phase 计算
```

在 `generate_recipe` 中根据算法设置：

```python
num_extra_trainable = 1 if algo == "ppo" else 0  # critic
num_extra_frozen = 0
if algo == "ppo":
    num_extra_frozen = 1  # ref
    if overrides_field.get("reward_ckpt"):
        num_extra_frozen += 1
elif algo == "dpo":
    num_extra_frozen = 1  # ref
```

**验证方式：**

```python
def test_ppo_memory_higher_than_grpo(self):
    grpo_result = gen.generate_recipe(algo="grpo", **common)
    ppo_result = gen.generate_recipe(algo="ppo", **common)
    assert ppo_result.memory.total > grpo_result.memory.total
    ratio = ppo_result.memory.total / grpo_result.memory.total
    assert 1.8 < ratio < 2.5
```

**经验总结：**

实现内存估算时，必须考虑算法的多角色特性。PPO 不只是"多一个模型"，而是"一个可训练（有优化器）+ 一个冻结（无优化器）"，两者的内存开销不同。

---

## 五、分步骤运行结果证明

### 步骤 1：环境准备

**目的：** 确认测试环境可用。

**命令：**

```bash
python3 --version
```

**输出：**

```
Python 3.9.6
```

**解释：** 本机为 macOS，Python 3.9.6。脚本设计兼容 3.9+，无需 GPU 或 `areno` 包。

---

### 步骤 2：运行完整 CPU 测试套件

**目的：** 验证全部 77 个测试用例通过。

**命令：**

```bash
python3 -m pytest tests/test_recipe_generator_cpu.py -v --tb=short
```

**关键输出（末尾）：**

```
tests/test_recipe_generator_cpu.py::TestSFTRecipe::test_generates_runnable_command PASSED [  1%]
tests/test_recipe_generator_cpu.py::TestSFTRecipe::test_context_len_split_all_prompt PASSED [  2%]
tests/test_recipe_generator_cpu.py::TestSFTRecipe::test_provenance_sources PASSED [  3%]
tests/test_recipe_generator_cpu.py::TestSFTRecipe::test_uses_placeholders_when_none PASSED [  5%]
...
tests/test_recipe_generator_cpu.py::TestAutoDetectNSamples::test_auto_detect_uses_custom_n_samples PASSED [100%]

============================== 77 passed in 7.16s ==============================
```

**解释：** 77 个测试全部通过，耗时 7.16 秒。覆盖 SFT/DPO/GSPO 成功路径、无效输入、边界值、确定性、内存估算、自动 batch 调整、模型名推断、数据集行数、CLI 输出、多角色内存、参数验证等。

---

### 步骤 3：CLI 示例 — GSPO 配方生成

**目的：** 验证 GSPO 算法生成完整命令 + 内存估算 + provenance。

**命令：**

```bash
python3 .agents/skills/areno-run-training/scripts/generate_recipe.py \
    --algo gspo --gpus 8 --tp-size 4 \
    --context-len 4096 --batch-size 32 \
    --ckpt Qwen/Qwen3-0.6B \
    --reward-fn-path examples/math/math_verify_reward.py \
    --format both
```

**关键输出（CLI 部分）：**

```
areno train --algo gspo --ckpt Qwen/Qwen3-0.6B --dataset-path <your-dataset> --world-size 8 --tp-size 4 --batch-size 32 --n-samples 4 --max-prompt-tokens 1024 --max-new-tokens 3072 --reward-fn-path examples/math/math_verify_reward.py --mini-bs 16

# No GPU detected -- memory estimates only; cannot verify fit.
# Could not read model config from 'Qwen/Qwen3-0.6B': No module named 'areno'
# Inferred architecture from model name 'Qwen/Qwen3-0.6B' (~0.6B params, closest match: qwen3 0.6B). Weight/optimizer estimates are exact; KV-cache and activation estimates are approximate.
# memory estimate (per GPU):
#   weights:      0.30 GB
#   optimizer:    2.09 GB
#   kv_cache:     7.52 GB
#   activations:  1.37 GB
#   total:        7.82 GB

# provenance:
#   batch_size = 32  [derived]
#   max_prompt_tokens = 1024  [derived]
#   max_new_tokens = 3072  [derived]
#   world_size = 8  [derived]
#   tp_size = 4  [derived]
#   mini_bs = 16  [derived]
#   optimizer_lr = 1e-06  [default]
#   epochs = 10  [default]
```

**解释：**
- 生成了完整的 `areno train` 命令，包含所有必需字段
- 三级降级生效：`areno` 包未安装 → 模型名推断 `Qwen3-0.6B` → shape 匹配 qwen3 0.6B
- 上下文拆分：RL 算法 `min(1024, 4096//4)=1024` 给 prompt，3072 给 new tokens
- provenance 标注清晰：derived（从输入推导）vs default（内置默认）
- 内存估算：weights 0.30 GB（600M * 2 bytes / 4 TP），optimizer 2.09 GB，total 7.82 GB（不含 GPU 探测所以 headroom 为 null）

---

### 步骤 4：错误处理 — 并行度不整除

**目的：** 验证 gpus 不能被 tp_size 整除时的错误信息。

**命令：**

```bash
python3 .agents/skills/areno-run-training/scripts/generate_recipe.py \
    --algo sft --gpus 6 --tp-size 4 \
    --context-len 1024 --batch-size 8 \
    --format json
```

**输出：**

```
error: [stage=parallelism] --world-size must be divisible by --tp-size
```

**退出码：** 1

**解释：** 6 % 4 != 0，`[stage=parallelism]` 标签清晰标明错误来源阶段。错误输出到 stderr，退出码为 1。

---

### 步骤 5：--set 显式覆盖

**目的：** 验证 `--set lr=5e-7` 覆盖 optimizer_lr 且标注为 explicit。

**命令：**

```bash
python3 .agents/skills/areno-run-training/scripts/generate_recipe.py \
    --algo sft --gpus 8 --tp-size 4 \
    --context-len 2048 --batch-size 16 \
    --ckpt Qwen/Qwen3-0.6B \
    --set lr=5e-7 \
    --format cli
```

**关键输出：**

```
areno train --algo sft --ckpt Qwen/Qwen3-0.6B --dataset-path <your-dataset> --world-size 8 --tp-size 4 --batch-size 16 --max-prompt-tokens 2048 --max-new-tokens 0 --mini-bs 16 --lr 5e-07

# provenance:
#   optimizer_lr = 5e-07  [explicit]
```

**解释：** `--lr 5e-07` 出现在命令中（因为 `source == "explicit"`），provenance 标注为 `explicit`。银行业务标准的 `--lr` CLI 名正确映射到 `optimizer_lr` 配置字段名。

---

### 步骤 6：提交范围验证

**目的：** 确认本次 PR 的文件变更范围。

**命令：**

```bash
git diff --stat 1cf3d79~1 HEAD
```

**输出：**

```
 .agents/skills/areno-run-training/SKILL.md         |   42 +-
 .../references/recipe-generator.md                 |  301 +++++
 .../areno-run-training/scripts/generate_recipe.py  | 1354 ++++++++++++++++++++
 tests/test_recipe_generator_cpu.py                 |  816 ++++++++++++
 4 files changed, 2507 insertions(+), 6 deletions(-)
```

**解释：** 4 个文件变更，2507 行新增，6 行删除（SKILL.md 的工作流描述替换）。无无关文件修改。

---

## 文件变更总览

| 文件 | 类型 | 行数 | 说明 |
|------|------|------|------|
| `.agents/skills/areno-run-training/scripts/generate_recipe.py` | 新增 | 1354 | 配方生成器主脚本 |
| `tests/test_recipe_generator_cpu.py` | 新增 | 816 | 77 个 CPU 测试用例 |
| `.agents/skills/areno-run-training/references/recipe-generator.md` | 新增 | 301 | 英文用户文档 |
| `.agents/skills/areno-run-training/SKILL.md` | 修改 | +36/-6 | 新增配方生成章节和工作流步骤 |

**提交历史：**

```
c538056 feat(recipe): add agentic RL params, multi-role memory estimation, --lr flag
ea11f60 docs(recipe): fix documentation accuracy issues
6f06480 test(recipe-generator): add boundary value tests
671f5a8 fix(recipe-generator): add --auto mode tests and optimize CSV handling
1cf3d79 feat(areno-run-training): add training recipe generator
```