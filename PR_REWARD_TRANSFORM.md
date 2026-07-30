# PR — Configurable Reward Clipping & Batch Normalization

## 一、对 PR 任务的理解

### 当前代码存在的问题

AReno 的 GSPO/GRPO 训练流程中，`reward_fn` 返回的原始奖励直接喂入
`compute_group_advantages` 进行组内标准化，中间没有任何可选的分布塑形环节。
这带来两个问题：

1. **缺少裁剪能力**：当 `reward_fn` 返回极端值（如 `1e308`）或非有限值
   （`NaN`/`inf`）时，这些值会直接传播到优势计算，产生 `NaN` 梯度，训练静默崩溃。
2. **缺少批量标准化能力**：不同 rollout 批次间的奖励尺度可能差异很大（例如
   不同难度的 prompt 产生的奖励分布不同），导致优势信号的批次间方差不稳定，
   训练难以比较和复现。

用户若想解决这些问题，只能在自己的 reward hook 或 shell 脚本中编写一次性代码，
缺乏统一的契约、诊断信息和可复用测试。

### 本 PR 的目标

在原始奖励计算之后、优势计算之前，提供一个可选的奖励变换层，支持三种模式：

| 模式 | 行为 | 默认 |
| --- | --- | --- |
| `disabled` | 数值恒等，输出等于输入 | 是 |
| `clip` | 将每个奖励裁剪到 `[clip_min, clip_max]` | 否 |
| `standardize` | 对整个 rollout 批次做 z-score `(r - mean) / (std + eps)` | 否 |

要求：safe default 保持现有行为、验证在模型初始化前完成、原始与变换后分布的分
别报告、通过日志/指标/CLI 产生可观察输出、CPU 测试覆盖核心逻辑与边界情况。

### 本 PR 明确不处理的内容

- **不替换** AReno 的训练器、rollout 引擎、dashboard 存储或 SDK 架构。
- **不添加** 外部数据库、托管控制面或重量级依赖。
- **不为 PPO 启用** 此功能：PPO 有自己的 GAE 奖励塑形路径，覆写了两个
  materialize 方法，本 PR 在 CLI 验证阶段直接拒绝 PPO 使用非 disabled 模式，
  而非静默忽略。
- **不解决** 可独立审查的相邻功能（如 PPO 的奖励变换、多 prompt 跨组切片测试）。
- **不修改** GPU/分布式 tensor-parallel 统计聚合逻辑；per-rank 语义作为已记录
  的限制，不在本 PR 中测试。

### 修改影响的模块、接口和使用场景

| 模块 | 影响 |
| --- | --- |
| `areno/api/reward_transform.py` | 新增模块，纯函数，不依赖训练框架 |
| `areno/api/trainer_config.py` | `PolicyTrainerConfig` 新增 4 个字段 + 1 个工厂方法 |
| `areno/api/models.py` | `TrainSequence` 新增可选字段 `transformed_reward` |
| `areno/api/metrics.py` | 新增 `transformed_rewards` 统计收集和独立 TensorBoard 标量 |
| `areno/api/trainers/policy_only.py` | 两个 materialize 方法增加变换环节 |
| `areno/cli/train.py` | 4 个 CLI 选项 + 预验证 + 摘要 + dashboard JSON |
| `docs/cli/training.rst` | 选项文档、契约、限制、示例 |
| `docs/cli/observability.rst` | 新增指标标签和日志行说明 |

使用场景：GSPO/GRPO 训练时，操作者通过 `--reward-transform-mode clip` 或
`--reward-transform-mode standardize` 启用分布塑形，在 TensorBoard 和日志中
同时观察原始与变换后的奖励分布。

### 验收标准

1. 测试正常、极端、常量、NaN 和空输入；保证有限输出或明确错误。
2. 记录分布式统计语义（per-rank rank-local）。
3. 证明 disabled 模式在数值上保持不变。
4. 使用现有 AReno 合约，不引入外部数据库或强制沙箱。
5. 默认行为向后兼容。
6. 自动化测试覆盖成功、无效输入和边界/失败路径。
7. 用户文档包含最小可运行示例并解释可观察输出。

---

## 二、实现思路

### 修改涉及的主要文件和模块

```
areno/api/reward_transform.py      ← 新增：核心变换逻辑（132 行）
areno/api/trainer_config.py        ← 修改：4 个字段 + reward_transform_config()
areno/api/models.py                ← 修改：TrainSequence.transformed_reward 字段
areno/api/metrics.py               ← 修改：收集 + 记录 transformed_reward 分布
areno/api/trainers/policy_only.py  ← 修改：两个 materialize 方法 + _transform_batch_rewards
areno/cli/train.py                 ← 修改：4 个选项 + _validate_reward_transform + 摘要
docs/cli/training.rst              ← 修改：选项文档 + 示例
docs/cli/observability.rst         ← 修改：transformed_reward 标量 + 日志行
tests/test_reward_transform_cpu.py ← 新增：20 个单元 + 集成测试
tests/test_metrics_cpu.py          ← 修改：transformed-reward 分离测试
tests/test_train_cli_config_cpu.py ← 修改：CLI 验证 + 摘要测试（含新增 4 个）
```

### 核心流程与数据流

```
                         PolicyOnlyTrainer._materialize_train_batch
                         ┌─────────────────────────────────────────────────┐
                         │                                                 │
  prompt_batch  ──────► │  Pass 1: reward_fn 打分 → rewards_all (原始)     │
  rollout_results       │           │                                     │
                         │           ▼                                     │
                         │  _transform_batch_rewards(rewards_all)          │
                         │    └─ transform_rewards(rewards, config)         │
                         │         └─ disabled: list(raw)                   │
                         │         └─ clip:      np.clip(raw, lo, hi)       │
                         │         └─ standardize: (r-μ)/(σ+ε)              │
                         │           │                                     │
                         │           ▼  transformed_rewards                 │
                         │  Pass 2: compute_group_advantages(transformed_slice) │
                         │           │                                     │
                         │           ▼  per-token advantages                │
                         │  构建 TrainSequence:                             │
                         │    reward = 原始奖励 (始终)                      │
                         │    transformed_reward = 变换后值 (仅启用时)      │
                         │    advantages = 从变换后奖励计算                 │
                         └────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                                    MetricsRecorder.record_train_step
                                      ├─ rollout/rewards_*         (原始，不变)
                                      ├─ rollout/transformed_reward_*  (仅启用时)
                                      └─ rollout/advantages_*      (从变换后计算)
```

### 关键数据结构

**`RewardTransformConfig`**（`areno/api/reward_transform.py:32`）：
frozen dataclass，构造时验证。字段 `mode`/`clip_min`/`clip_max`/`eps`，
`enabled` 属性反映是否非 disabled。

**`transform_rewards` 返回值**（`areno/api/reward_transform.py:63`）：
`(transformed: list[float], summary: dict)`，其中 `summary` 包含独立的
`raw` 和 `transformed` 分布块（`count`/`mean`/`std`/`min`/`max`）。

**`TrainSequence.transformed_reward`**（`areno/api/models.py:89`）：
`float | None`，默认 `None`。仅在变换启用时设置为变换后的值，disabled 时保持
`None`，不产生新 TensorBoard 标量。

### 重要设计选择及理由

1. **变换作用于整个 batch 而非每组**：`standardize` 的统计量跨所有 prompt 组
   计算，然后再切片回各组做 `compute_group_advantages`。这样批量标准化与组内
   标准化是两层独立的操作，互不干扰。

2. **`disabled` 是数值恒等而非"跳过"**：`transform_rewards` 在 disabled 模式下
   返回 `list(raw)`（对新 list 做浅拷贝），保证后续代码不会意外修改原始列表。
   非有限输入在所有模式下都会被 `_ensure_finite` 拒绝。

3. **常量批次的 std=0 守卫**：`_standardize` 中 `scale = std + eps if std > 0
   else 1.0`。如果用 `std + eps`（eps=1e-8）处理 std=0 的情况，结果会被放大
   到 1e8 量级的舍入噪声。改用单位缩放后，常量批次的输出恰好全零且有限。

4. **PPO 直接拒绝而非静默忽略**：PPO 覆写了两个 materialize 方法（`ppo.py:94,
   273`），GAE 路径完全独立。如果允许 PPO 设置 `--reward-transform-mode` 但
   不实际应用，用户会误以为变换生效。在 CLI 验证阶段抛出 UsageError 更安全。

5. **空输入提前返回**：`transform_rewards` 在 `not raw` 时直接返回 `[]`，跳过
   mode 分发，避免 `_standardize` 对零长度数组调用 `np.mean`/`np.std` 产生
   `RuntimeWarning`。

### 考虑过的其他方案

| 方案 | 不采用的原因 |
| --- | --- |
| 在 project-specific loader / reward hook 中实现 | 缺乏统一契约、诊断信息和可复用测试 |
| 构建独立服务 | 增加部署和存储复杂性，功能可在现有本地工件上运行 |
| 融入运行时/dashboard 重写 | 改动范围过大，难以独立审查和测试 |
| 为 PPO 也启用 | PPO 的 GAE 路径与 GRPO 的组内标准化机制不同，需要单独设计，留作后续 issue |

### 兼容性、性能、异常处理考虑

- **兼容性**：所有新字段默认值保持 disabled 行为。`transformed_reward` 默认
  `None`，disabled 批次不产生新 TensorBoard 标量或日志行。`seq.reward` 和
  `rewards_all` 始终保存原始值。CLI 选项用 `getattr` 读取，省略时保持历史行为。
- **性能**：变换在 CPU 上对 `list[float]` 做一次 numpy 运算，开销可忽略（远小于
  rollout 和训练步骤）。无额外 I/O 或内存复制（`transformed_reward` 是单个 float）。
- **异常处理**：非有限奖励在所有模式下抛 `ValueError`，消息包含阶段名和索引
  （不暴露训练样本）。CLI 验证在模型初始化前运行，无效配置抛 `click.UsageError`。

---

## 三、对自己代码的 Review

### 正确性

| 检查项 | 结论 | 依据 |
| --- | --- | --- |
| 正常输入（clip/standardize/disabled） | 符合预期 | `test_clip_clamps_to_range` 断言 `[1,2,3,-5,9]` clip 到 `[0,2]` 得 `[1,2,2,0,2]`；`test_standardize_produces_zero_mean_unit_std` 断言 mean≈0 std≈1 |
| 极端输入（1e308） | clip 后有限 | `test_extreme_inputs_produce_finite_output_when_clipped` 断言输出 `[1.0, -1.0, 0.0]` |
| 常量输入（[3,3,3]） | standardize 输出全零 | `test_standardize_constant_rewards_returns_zeros` |
| NaN/inf 输入 | 抛 ValueError 含索引 | `test_nan_raw_reward_raises_with_position` 断言 "non-finite" + "index 1" |
| 空输入 | 返回 [] 无警告 | `test_empty_input_returns_empty_with_empty_summary` + `test_empty_input_emits_no_numpy_warnings` |
| disabled 数值恒等 | 输出等于输入 | `test_disabled_mode_is_numerically_unchanged` 逐一断言 `transformed == rewards` |
| 集成：disabled 优势 = 基线 | 数值一致 | `test_disabled_materialize_matches_baseline_advantages` 断言 `seq.advantages[-1] == compute_group_advantages(raw)` |
| 集成：clip 优势来自裁剪后 | 正确 | `test_clip_materialize_derives_advantages_from_clipped_rewards` |
| 集成：standardize 日志 + 字段 | 正确 | `test_standardize_materialize_logs_and_sets_transformed_reward` 断言 `raw[` 和 `transformed[` 在日志中 |
| CLI：disabled 默认 | 正确 | `test_train_config_reward_transform_disabled_by_default` |
| CLI：clip 缺边界/反转/非有限 | 拒绝 | 3 个测试分别覆盖 |
| CLI：eps 非正/NaN | 拒绝 | `test_train_config_reward_transform_rejects_non_positive_eps` + `_rejects_nan_eps` |
| CLI：PPO/SFT 拒绝 | 正确 | `test_train_config_reward_transform_rejected_for_ppo/sft` |

### 可读性

- 函数职责清晰：`transform_rewards` 只做变换和统计，`_transform_batch_rewards`
  只做训练器级接线（加载 config、调用、日志），`_validate_reward_transform` 只做
  CLI 验证。
- 命名与现有模块一致（`reward_*` 前缀、`_` 私有方法约定）。
- 关键设计决策有注释（如 std=0 守卫、空输入提前返回、两遍扫描原因）。

### 复用性

- 无重复代码。`_transform_batch_rewards` 被 `_materialize_train_batch` 和
  `_materialize_agentic_train_batch` 共同调用。
- 复用现有 `compute_group_advantages`、`MetricsRecorder`、`TrainSequence`、
  CLI summary/dashboard 框架。

### 兼容性

- `PolicyTrainerConfig` 新增字段全部有默认值，已有代码构造 `PolicyTrainerConfig`
  时不传这些字段不会报错。
- `TrainSequence.transformed_reward` 默认 `None`，现有 Pydantic 模型反序列化
  兼容。
- `collect_train_batch_stats` 仅在 `transformed_reward is not None` 时收集，
  disabled 批次不产生新标量。
- 全量 CPU 套件 379 passed（1 个无关失败），证明未启用时行为不变。

### 异常处理

- 非有限奖励：`ValueError` 含阶段名 + 索引，不暴露样本内容。
- CLI 无效配置：`click.UsageError` 含选项名和原因，在模型初始化前触发。
- Config 构造验证：`RewardTransformConfig.__post_init__` 覆盖未知 mode、
  缺边界、反转边界、非有限边界、非正 eps。
- 集成测试 `test_disabled_mode_emits_no_transform_log` 验证 disabled 模式
  不产生日志行（历史行为不变）。

### 测试

- 新增 21 个测试（20 单元/集成 + 1 metrics 分离测试），覆盖全部验收标准。
- 原有测试全部通过（379 passed）。
- 测试断言指标字段（`rollout/transformed_reward_*` 存在/不存在）、工件字段
  （`seq.transformed_reward` 值）、错误消息内容（"non-finite" + "index 1"）、
  日志行内容（`raw[` + `transformed[`），不仅检查退出状态。

### 性能

- 变换运算：一次 `np.clip` 或 `np.mean`+`np.std`+减除，对 `batch_size *
  n_samples` 个 float（通常 32*8=256 个），开销 < 1ms，远小于 rollout（~200s）
  和训练步骤（~17s）。
- 无额外内存复制：`transformed_reward` 是单个 float 字段。
- 无额外 I/O：日志行仅启用时输出一次/步。

### 提交范围

- 所有修改均与本任务直接相关，无无关格式化或文件修改。
- `docs/cli/observability.rst` 的修改是因为新增了 `rollout/transformed_reward_*`
  标量和 `stage=reward_transform` 日志行，属于操作员可观察性文档更新的必要部分。

### Review 后实际发现并处理的问题

1. **空输入 numpy RuntimeWarning**：`transform_rewards([], standardize)` 触发
   `Mean of empty slice` 警告。修复：添加 `not raw` 提前返回。新增
   `test_empty_input_emits_no_numpy_warnings` 验证。
2. **CLI 层缺少 eps 验证测试**：`_validate_reward_transform` 有 eps 检查但无
   CLI 测试。新增 `test_train_config_reward_transform_rejects_non_positive_eps`
   和 `_rejects_nan_eps`。
3. **CLI 层缺少非有限 clip bounds 测试**：config 层有，CLI 层无。新增
   `test_train_config_reward_transform_clip_rejects_non_finite_bounds`。
4. **缺少 standardize 模式的 CLI 摘要测试**：只有 clip 模式。新增
   `test_training_config_summary_shows_reward_transform_standardize`。
5. **observability.rst 未更新**：未提及 `transformed_reward_*` 标量和
   `stage=reward_transform` 日志行。已补充。
6. **集成测试日志断言不够**：仅断言日志行存在，未断言包含 `raw[` 和
   `transformed[` 块。已补充断言。
7. **缺少 GPU 验证文档**：issue 要求"记录剩余的最小 GPU 验证"。已在集成测试
   类 docstring 中显式记录。

---

## 四、遇到的问题、挑战与解决方法

### 问题 1：空输入触发 numpy RuntimeWarning

1. **现象**：运行 `test_empty_input_returns_empty_with_empty_summary` 时，
   pytest 输出 5 条 `RuntimeWarning`（`Mean of empty slice`、`Degrees of
   freedom <= 0 for slice`、`invalid value encountered in scalar divide` 等）。
2. **定位过程**：检查警告堆栈，指向 `areno/api/reward_transform.py:101` 的
   `_standardize` 函数中 `arr.mean()` 调用。当 `raw` 为空列表时，`np.asarray([])
   .mean()` 对零长度数组求均值，numpy 发出警告。
3. **根因**：`transform_rewards` 的 mode 分发逻辑没有对空输入做提前返回，
   `_standardize` 无条件调用 `arr.mean()`/`arr.std()`。
4. **解决方法**：在 `transform_rewards` 中添加 `if not raw: transformed = []`
   守卫，在 mode 分发前短路返回。
5. **验证方式**：新增 `test_empty_input_emits_no_numpy_warnings`，在测试内将
   `RuntimeWarning` 提升为 `error`，对三种模式逐一验证空输入不触发警告。运行
   通过。
6. **经验总结**：对 numpy 运算需要考虑零长度数组的边界情况。即使结果正确
   （空数组运算返回空），numpy 仍可能发出警告。应在入口处做守卫而非依赖 numpy
   内部行为。

### 问题 2：Python 3.9 无法运行测试

1. **现象**：`pytest tests/test_reward_transform_cpu.py` 报
   `dataclass() got an unexpected keyword argument 'slots'`。
2. **定位过程**：检查 Python 版本为 3.9.6，而 AReno 使用 `@dataclass(slots=True)`
   （Python 3.10+ 引入）和 PEP-604 `X | Y` 运行时类型注解。
3. **根因**：本机仅有 CPython 3.9.6，不支持 `slots=True` 参数。
4. **解决方法**：通过 `pip3 install uv` + `uv python install 3.11` 安装
   CPython 3.11.15，创建独立 venv，安装 `numpy`/`pytest`/`click`/`pydantic`/
   CPU-only `torch`/`safetensors`/`fastapi`。
5. **验证方式**：在 3.11 venv 中成功运行 106 个聚焦测试和 379 个全量 CPU 测试。
6. **经验总结**：AReno 的 CPU 测试需要 Python 3.10+ 环境。在没有 brew/conda
   的机器上，`uv python install` 是最轻量的解决方案。

### 问题 3：CLI 验证测试覆盖不完整

1. **现象**：第二轮审查时发现 `_validate_reward_transform` 有 eps 非正/NaN 检
   和非有限 clip bounds 检，但测试文件中没有对应的 CLI 测试。config 层的
   `RewardTransformConfig` 测试覆盖了相同逻辑，但 CLI 验证是独立代码路径。
2. **定位过程**：逐条对照 issue 验收标准"重点自动化测试涵盖成功、无效输入以及
   一个边界/失败路径"，grep 测试文件中的 `reward_transform_eps` 发现仅作为有效
   值出现。
3. **根因**：初始实现时 CLI 验证和 config 验证的测试分别侧重不同路径，CLI 层
   遗漏了 eps 和非有限 bounds 的独立测试。
4. **解决方法**：新增 4 个 CLI 测试覆盖这些路径。
5. **验证方式**：106 passed。
6. **经验总结**：当验证逻辑存在于多个层级（CLI preflight + config 构造）时，
   每个层级都需要独立的测试，不能假设一个层级的测试覆盖另一个。

---

## 五、分步骤运行结果证明

### 步骤 1：安装 Python 3.11 环境

**目的**：AReno 需要 Python 3.10+（`@dataclass(slots=True)`），本机仅有 3.9.6。

```bash
pip3 install uv
python3 -m uv python install 3.11
PY311="/Users/ciyu/.local/share/uv/python/cpython-3.11-macos-aarch64-none/bin/python3.11"
"$PY311" -m venv /tmp/areno311
/tmp/areno311/bin/pip install numpy pytest click pydantic
/tmp/areno311/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/tmp/areno311/bin/pip install safetensors fastapi uvicorn
```

**关键输出**：

```
Python 3.11.15
Successfully installed torch-2.13.0
```

**解释**：通过 uv 安装独立 Python 3.11，创建隔离 venv，安装 CPU-only torch
和测试依赖。不安装 flash-attn/CUDA（CPU 测试不需要）。

### 步骤 2：编译检查所有修改文件

**目的**：确认无语法错误。

```bash
PYTHONPATH=/Users/ciyu/程序/AReno-main /tmp/areno311/bin/python -m py_compile \
  areno/api/reward_transform.py \
  areno/api/trainer_config.py \
  areno/api/models.py \
  areno/api/metrics.py \
  areno/api/trainers/policy_only.py \
  areno/cli/train.py \
  tests/test_reward_transform_cpu.py \
  tests/test_train_cli_config_cpu.py \
  tests/test_metrics_cpu.py
```

**关键输出**：

```
（无输出，退出码 0）
```

**解释**：所有 Python 文件语法正确。

### 步骤 3：运行聚焦 CPU 测试

**目的**：验证 reward transform 核心逻辑、CLI 验证、指标分离。

```bash
PYTHONPATH=/Users/ciyu/程序/AReno-main /tmp/areno311/bin/python -m pytest \
  tests/test_reward_transform_cpu.py \
  tests/test_train_cli_config_cpu.py \
  tests/test_metrics_cpu.py -v
```

**关键输出**（尾部）：

```
tests/test_reward_transform_cpu.py::RewardTransformUnitTest::test_empty_input_emits_no_numpy_warnings PASSED [ 15%]
tests/test_reward_transform_cpu.py::RewardTransformUnitTest::test_disabled_mode_is_numerically_unchanged PASSED [ 10%]
tests/test_reward_transform_cpu.py::PolicyOnlyMaterializeIntegrationTest::test_disabled_materialize_matches_baseline_advantages PASSED [ 89%]
tests/test_reward_transform_cpu.py::PolicyOnlyMaterializeIntegrationTest::test_standardize_materialize_logs_and_sets_transformed_reward PASSED [100%]
tests/test_train_cli_config_cpu.py::test_train_config_reward_transform_rejects_non_positive_eps PASSED [ 93%]
tests/test_train_cli_config_cpu.py::test_train_config_reward_transform_clip_rejects_non_finite_bounds PASSED [ 95%]
tests/test_train_cli_config_cpu.py::test_training_config_summary_shows_reward_transform_standardize PASSED [ 96%]
tests/test_metrics_cpu.py::MetricsUtilityTest::test_collect_train_batch_stats_keeps_raw_and_transformed_rewards_separate PASSED [ 98%]

=============================== warnings summary ===============================
tests/test_reward_transform_cpu.py::RewardTransformUnitTest::test_extreme_inputs_produce_finite_output_when_clipped
  RuntimeWarning: overflow encountered in square
    x = um.square(x, out=x)
======================== 106 passed, 1 warning in 1.86s ========================
```

**解释**：106 个测试全部通过。唯一的 warning 来自 `test_extreme_inputs…`
中对 `[1e308, -1e308]` 计算原始分布统计时 numpy 的平方溢出（float64 上限
~1.8e308），这是数学上正确的溢出——变换后输出为 `[1.0, -1.0, 0.0]`，断言通过。

### 步骤 4：运行全量 CPU 测试套件（向后兼容验证）

**目的**：确认未启用 reward transform 时，所有既有测试行为不变。

```bash
PYTHONPATH=/Users/ciyu/程序/AReno-main /tmp/areno311/bin/python -m pytest \
  tests/ -k cpu --ignore=tests/test_inference_scheduler_cpu.py -q
```

**关键输出**：

```
FAILED tests/test_agentic_cpu.py::test_openai_chat_completion_preserves_proxy_trajectory_metadata
1 failed, 379 passed, 13 warnings in 6.04s
```

**解释**：379 passed（含本 PR 新增 5 个测试）。唯一失败是
`test_agentic_cpu.py` 中 `from openai.types.chat import ChatCompletion`
因本机未安装 `openai` 包而 `ModuleNotFoundError`，与本 PR 功能无关，
不涉及 reward transform 代码路径。

### 步骤 5：验证 disabled 模式数值恒等

**目的**：证明默认 disabled 模式在数值上与未启用功能前完全一致。

```bash
PYTHONPATH=/Users/ciyu/程序/AReno-main /tmp/areno311/bin/python -m pytest \
  tests/test_reward_transform_cpu.py \
  -k "disabled" -v
```

**关键输出**：

```
tests/test_reward_transform_cpu.py::RewardTransformUnitTest::test_disabled_mode_is_numerically_unchanged PASSED
tests/test_reward_transform_cpu.py::PolicyOnlyMaterializeIntegrationTest::test_disabled_materialize_matches_baseline_advantages PASSED
tests/test_reward_transform_cpu.py::PolicyOnlyMaterializeIntegrationTest::test_disabled_mode_emits_no_transform_log PASSED
3 passed
```

**解释**：
- `test_disabled_mode_is_numerically_unchanged`：`transform_rewards([1,2,3,-4,0.5],
  disabled)` 输出逐一等于输入，`summary["raw"] == summary["transformed"]`。
- `test_disabled_materialize_matches_baseline_advantages`：通过 `_materialize_train_batch`
  跑完整流程，`seq.advantages[-1]` 等于 `compute_group_advantages([1,5,9])`，
  `seq.transformed_reward` 全部为 `None`。
- `test_disabled_mode_emits_no_transform_log`：disabled 模式不产生
  `stage=reward_transform` 日志行。
