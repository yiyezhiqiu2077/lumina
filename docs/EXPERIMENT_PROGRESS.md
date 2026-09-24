# Historical MagicBrush-only v3：Lumina 8-GPU 实验运行与进度记录

> 本文是历史 MagicBrush-only v3 record，不是当前 MagicBrush + RefEdit mixed 2×3 recipe。
> 它记录本轮 objective ablation 从首次运行、问题定位、代码与训练 recipe 修正、验证实验，到当前正式训练的完整过程。
>
> **最终更新：2026-09-19 10:07 UTC。fresh `3e-6` 正式 v3 串行队列已全部完成并科学成功：Attention、GCE、CE 均有 2,750 条连续 metrics、10 个完整且 objective 内 fingerprint 一致的 checkpoint、objective exit code 0 和 `quality_status.json: SUCCEEDED`；串行总队列 `all.exit_code=0`。三项最终 50-step 平均 `L_gen` 分别为 `2.790620`、`2.757802`、`2.790224`，均低于各自 baseline，参数全程有限、无 clipping、`bad_windows=0`。训练进程和 `lumina_all` 已退出，八卡回到每卡约 1,605 MiB 的运行前基线。**

## 1. 实验目标与固定约束

本轮实验比较三个 objective：

1. Attention
2. GCE
3. CE

正式实验必须严格串行执行 **Attention → GCE → CE**，不能同时训练。共同配置为：

| 配置项 | 值 |
| --- | --- |
| GPU | 8 张，`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` |
| 每卡 batch size | 4 |
| gradient accumulation | 1 |
| global batch size | 32 |
| precision | BF16 |
| 数据 | MagicBrush 官方完整 train，8,807 条记录、26,421 张图片 |
| sampler | `DistributedSampler(drop_last=True)` |
| 每 rank 每 epoch 样本数 | 1,100 |
| 每 epoch 全局实际样本数 | 8,800 |
| 每 rank 每 epoch micro-batch | 275 |
| optimizer steps / epoch | 275 |
| epoch 数 | 10 |
| 正式总步数 | 2,750 / objective |
| checkpoint 周期 | 每 275 步 |
| LoRA | rank 16、alpha 16、dropout 0.05 |
| LoRA targets | `q_proj`、`k_proj`、`v_proj`、`attn_out` |
| optimizer | AdamW，betas `(0.9, 0.95)`，weight decay 0.1 |
| 最近一次失败正式 peak LR | `5e-6`（已在 Attention step 1450 证实不稳定） |
| 当前正式 peak LR | `3e-6`（三个 1650-step matched 长 probes 均已验证，已写入正式 YAML） |
| warmup | 20 steps |
| gradient clip | 4.0 |
| generation loss | sample-mean CE + `1e-5 ×` FP32 raw logit z-loss |

每个 objective 的预期 checkpoint 为：

```text
275, 550, 825, 1100, 1375, 1650, 1925, 2200, 2475, 2750
```

## 2. 数据与资产准备

首先按 `docs/EXPERIMENTS.md` 准备运行环境和数据资产：

- 安装并使用 `uv 0.12.15`。
- 建立 Python 3.10 环境并安装项目依赖。
- 匿名取得 `Alpha-VLLM/Lumina-DiMOO` 完整模型资产；未使用或暴露 `HF_TOKEN`。
- 导出 MagicBrush 官方完整训练集：
  - 8,807 条记录；
  - 26,421 张 source/target/mask 图片。
- 完成 deterministic 512×512 shared geometry。
- 完成 32×32 VQ grid tokenization。
- 完成 GCE cluster 构建。
- 完成数据、模型、geometry、token 和 cluster audit。

已成功生成的资产没有因后续训练失败而删除或重复生成。

主要资产路径：

```text
<asset-root>/lumina_assets
```

MagicBrush 训练 manifest：

```text
<asset-root>/lumina_assets/datasets/lumina_edit/magicbrush/tokens/train/manifest.jsonl
```

## 3. 首次正式运行（v1）

### 3.1 如何启动

首次将原两卡设置改为八卡，并保持：

```text
8 GPUs × batch 4/GPU × accumulation 1 = global batch 32
```

随后通过 detached tmux 启动严格串行队列，语义等价于：

```bash
bash scripts/train/run_tmux.sh all
```

首次输出目录：

```text
<experiment-root>/formal/lumina_objective_ablation_8g_v1
```

当时也建立了持续更新的 `progress.log`，并监控 pipeline、Smoke、各 objective、tmux 和八张 GPU。

### 3.2 v1 实际结果

| Objective | Metrics | 进程结果 | 最终 generation loss | 科学判定 |
| --- | ---: | --- | ---: | --- |
| Attention | 2,750 条，连续 | exit code 0 | `21.3233` | 失败：明显发散 |
| GCE | 2,750 条，连续 | exit code 0 | `8.2705` | 失败：明显回归 |
| CE | 795 条，连续 | exit code 130 | `10.3596` | 失败：在确认复现发散趋势后主动停止 |

总队列 `all.exit_code=141`。旧 Smoke 虽记录 exit code 0，但日志中存在：

- 2 条 CUDA out-of-memory；
- 4 条 `NCCL WARN`。

所以 v1 的 Smoke 也不能作为科学成功依据。

### 3.3 为什么 v1 没有形成有效完整结果

v1 的关键问题不是“程序完全无法运行”，而是：

- Attention 和 GCE 可以机械地运行到 2,750 步并返回 0；
- 但 generation loss 已持续恶化，结果在科学上无效；
- CE 共享相同的 generation/LoRA 训练路径，并复现了类似发散轨迹；
- 因而在 step 795 主动停止 CE，避免继续消耗八卡算力产生已知无效结果。

旧 v1 输出树被完整保留作为失败证据，后续没有覆盖，也没有将其 checkpoint 用作 fresh 正式训练的起点。

## 4. 定位出的原始问题

### 4.1 长 schedule 下 `1e-5` peak LR 过高

早期的 825-step probe 曾看起来稳定，但 probe 错误地同时把：

- 实际停止步数设为 825；
- cosine scheduler 总长度也设为 825。

这会让 LR 在 probe 内提前快速衰减，因此不能代表正式 2,750-step scheduler 的前 825 步。

修复 stop horizon 和 scheduler horizon 后，在正式 2,750-step LR 轨迹上重跑 `1e-5` CE probe，generation loss 从约 step 550 开始恶化，并在 step 700 被质量门判定失败：

```text
generation-loss window 6.86417 > baseline 2.77652 × 1.75
for 2 consecutive non-overlapping windows
```

对比表明，旧压缩 scheduler 在 step 700 的 LR 仅约 `0.583e-6`，而正确正式 scheduler 当时仍约为 `8.55e-6`。因此先前 probe 的“稳定”主要来自错误的快速降 LR。

### 4.2 Attention 辅助损失使用了 post-RoPE Q/K

原实现用经过 RoPE 位置旋转后的 Q/K 计算 semantic localization auxiliary loss，使预期的语义目标混入位置编码。

修复后：

- Attention auxiliary 使用 pre-RoPE semantic Q/K；
- 真正 self-attention 仍使用 post-RoPE Q/K。

### 4.3 CE/GCE reduction 对不同监督 token 数的样本权重不稳定

MagicBrush 各样本的有效监督 token 数不同。原 token-mean 语义会让监督 token 更多的样本在 batch 内占更大权重。

正式配置改为 `sample_mean`：

1. 先对每个样本的有效监督位置求均值；
2. 再对有效样本求均值。

CE 和 GCE 使用相同 reduction 语义。

### 4.4 缺少科学质量门

原路径主要依据进程 exit code 判断成功。这样会把“训练程序正常退出但 loss 已发散”错误地视为成功。

v1 Attention 和 GCE 正是这种情况：exit code 0，但科学上失败。

### 4.5 Smoke 太短且旧 Smoke 存在 OOM/NCCL 异常

20-step Smoke 只能验证启动和 objective 路径，无法跨过约 step 500–700 才出现的延迟发散。旧 Smoke 还实际包含 OOM 和 NCCL warning，因此不能用于证明稳定性。

### 4.6 checkpoint/resume 语义不够严格

旧 checkpoint 不具备当前要求的完整 run fingerprint、质量状态和严格 metrics continuity 验证。尤其不能安全地：

- 从旧两卡 rank-specific checkpoint 恢复八卡训练；
- 从已确认发散的 checkpoint 延续同一 recipe；
- 在 loss、scheduler、optimizer、数据或代码语义改变后继续旧状态。

## 5. 后续代码与训练流程改进

### 5.1 分离停止步数与 scheduler horizon

新增独立的：

- `max_optimizer_steps`：本次进程实际停止位置；
- `scheduler_horizon_steps`：cosine scheduler 的完整训练长度。

现在 825-step probe 虽在 step 825 停止，但 scheduler horizon 仍是 2,750，所以 probe 前 825 步与正式实验具有完全相同的 LR 轨迹。

### 5.2 修复 objective/loss 语义

完成以下修改：

- Attention localization 改用 pre-RoPE Q/K。
- CE 和 GCE 统一使用 sample-mean supervised reduction。
- 在有效监督 target positions 上计算 FP32 raw logit z-loss。
- 正式 z-loss 权重设为 `1e-5`。
- 显式固定 LoRA targets、AdamW betas 和各 objective 的独立配置。
- 保持 BF16 主模型执行、FP32 LoRA 参数和 FP32 loss 关键计算。

### 5.3 增加诊断指标

每步或按固定周期记录：

- `L_gen`、`L_total`、raw/weighted z-loss；
- 有效监督 token 数；
- max absolute valid logit；
- pre/post-clip gradient norm 和 clipping 状态；
- Q/K LoRA gradient norm；
- 总体及按 target 划分的 LoRA parameter norm；
- optimizer update norm / update ratio；
- 参数有限性；
- LR、step time、peak VRAM；
- quality baseline、window loss、clip fraction、bad windows；
- Attention/GCE generation 与 auxiliary 梯度分解。

### 5.4 增加 fail-fast 科学质量门

质量门现在会检查：

- loss、梯度、参数是否有限；
- logit 是否失控；
- clipping 是否长期过高；
- optimizer update ratio 是否异常；
- generation loss 是否相对早期 baseline 持续回归。

正式配置使用 step 101–400 建立 baseline，之后按 50-step 非重叠窗口检查。连续两个窗口超过 baseline 的 1.75 倍时判定 `QUALITY_FAILED`。

串行队列只有在以下两项同时满足后才启动下一个 objective：

1. 进程 exit code 为 0；
2. `quality_status.json` 为 `SUCCEEDED`。

### 5.5 加强 checkpoint 和 resume 安全

当前完整 checkpoint 必须包含：

```text
checkpoint-NNNNNN/
  _SUCCESS
  checkpoint_meta.json
  lora.pt
  lora_config.json
  training_state.rank00.pt
  ...
  training_state.rank07.pt
```

run fingerprint 会覆盖：

- objective 和 world size；
- effective/global batch；
- 模型、配置、数据和 GCE cluster 身份；
- optimizer、LoRA 和 loss 配置；
- stop horizon 与 scheduler horizon；
- sampler/cursor；
- quality gate state；
- source SHA256。

恢复时还要求 `train_metrics.jsonl` 从 step 1 到 checkpoint step 完整连续，避免重复或跳步记录。

### 5.6 改善 tmux 与状态脚本

- 使用 exact tmux session matching，避免 prefix 误匹配。
- 所有 Smoke、probe、正式训练和 progress watcher 都在 detached tmux 中运行。
- launcher 记录真实 Python/torchrun exit code，而不是 `tee` 的状态。
- fresh run 检测已有 metrics、checkpoint、日志或 marker 时拒绝覆盖。
- 状态区分 `RUNNING`、`PROCESS_FAILED`、`QUALITY_FAILED` 和 `SUCCEEDED`。

## 6. 修复后运行的验证实验

### 6.1 正确 formal-horizon 的 `1e-5` CE probe

输出目录：

```text
<experiment-root>/probes/lumina_objective_ablation_8g_probe_ce_formalhorizon_20260918
```

结果：

- scheduler horizon 正确保持 2,750；
- step 550 后开始明显回归；
- step 700 触发 `QUALITY_FAILED`；
- 该失败 root 被保留；
- 没有从其 checkpoint 继续同一发散 recipe。

### 6.2 单变量降低 peak LR

在其他正式配置保持不变的情况下，只将 peak LR 从 `1e-5` 调整为 `5e-6`。

选择依据包括：

- 正式 horizon 下 step 400 LR 约 `4.76e-6`；
- step 550 LR 约 `4.55e-6`；
- 低于旧错误 probe step 400 的 `5.44e-6`；
- 约为失败 probe 发散阶段 LR 的一半。

### 6.3 三个 matched 825-step probes

三个 probe 均从 base model 在各自 fresh root 启动，并保持正式 2,750-step scheduler horizon。

| Objective | 结果 | Quality | Checkpoints |
| --- | --- | --- | --- |
| CE | 825 条连续 metrics，exit 0 | `SUCCEEDED` | 275、550、825 完整 |
| Attention | 825 条连续 metrics，exit 0 | `SUCCEEDED` | 275、550、825 完整 |
| GCE | 825 条连续 metrics，exit 0 | `SUCCEEDED` | 275、550、825 完整 |

对应 roots：

```text
<experiment-root>/probes/lumina_objective_ablation_8g_probe_ce_lr5e6_formalhorizon_20260918
<experiment-root>/probes/lumina_objective_ablation_8g_probe_attention_lr5e6_formalhorizon_retry1_20260918
<experiment-root>/probes/lumina_objective_ablation_8g_probe_gce_lr5e6_formalhorizon_20260918
```

第一次 Attention `5e-6` probe 在 step 0 遇到 torchrun rendezvous 端口占用；没有训练状态可恢复，因此保留失败 root，并从 base model 在 fresh retry root 重启，最终成功。

### 6.4 fresh 三目标 Smoke

输出目录：

```text
<experiment-root>/smokes/lumina_objective_ablation_8g_smoke_lr5e6_20260918
```

严格串行运行 Attention → GCE → CE，每项 20 steps。结果：

- `smokes.exit_code=0`；
- 三项 metrics 连续；
- 三项 quality 均为 `SUCCEEDED`；
- 三个 `checkpoint-000020` 均完整；
- 无 OOM、无 `NCCL WARN`；
- 完成后 GPU 回到约 1,605 MiB/卡的运行前已有基线占用。

### 6.5 自动化测试

修复后执行：

- 初轮 focused tests：11 passed；
- 三个 `3e-6` 长 probes 全部成功并更新正式 YAML 后，focused tests：53 passed；
- 最新 full test suite：79 passed；
- shell syntax checks：通过；
- `git diff --check`：通过。

随后在 fresh root `<experiment-root>/smokes/lumina_objective_ablation_8g_smoke_lr3e6_20260918` 完成了 `3e-6` 的 Attention → GCE → CE 串行 Smoke。三项均有 20 条连续 metrics、exit code 0、quality `SUCCEEDED`、完整 checkpoint-000020 和 8 个 rank state；总 `smokes.exit_code=0`，日志无 OOM、`NCCL WARN`、traceback 或质量失败。完成后无 Lumina torchrun 或训练 tmux，八卡均回到约 1,605 MiB/卡。

已于 2026-09-18 22:05 UTC 左右从 base model 在全新 root `<experiment-root>/formal/lumina_objective_ablation_8g_v3_lr3e6_20260918` 启动正式 Attention → GCE → CE 串行队列；未使用任何 probe、v1 或 v2 checkpoint。detached `lumina_all` 和 `lumina_v3_progress` 正常运行，初始 Attention metrics 连续且日志无错误。

没有创建 git commit。

## 7. fresh v2 正式实验及延迟发散

正式输出 root：

```text
<experiment-root>/formal/lumina_objective_ablation_8g_v2_lr5e6_20260918
```

启动方式：

```bash
bash scripts/train/run_tmux.sh all
```

启动时间约为 2026-09-18 11:49 UTC，队列固定为 Attention → GCE → CE。Attention 前 825 步稳定，证明短 probe 和 Smoke 的启动、loss 与 objective 路径正确；但该结果没有覆盖更晚的失稳区间。

### 7.1 最终状态

| Objective | 状态 | Metrics | Quality | 完整 checkpoint |
| --- | --- | ---: | --- | --- |
| Attention | 已停止，exit 1 | 1–1450 连续 | `QUALITY_FAILED` | 275、550、825、1100、1375 |
| GCE | NOT STARTED | 0 | 尚无 | 无 |
| CE | NOT STARTED | 0 | 尚无 | 无 |

串行队列 `all.exit_code=1`，质量门失败原因：

```text
quality gate failed at step 1450:
generation-loss window 6.49544 > baseline 2.78457 × 1.75
for 2 consecutive non-overlapping windows
```

关键非重叠窗口：

| Steps | 平均 `L_gen` | 相对 baseline | 判定 |
| --- | ---: | ---: | --- |
| 1301–1350 | 4.24864 | 1.53× | 尚未超过门限 |
| 1351–1400 | 5.56718 | 2.00× | 第一个 bad window |
| 1401–1450 | 6.49544 | 2.33× | 第二个 bad window，失败 |

同时观察到 generation gradient 渐进增长，但不是一次性机械或数值故障：

- 窗口平均 grad norm 从 step 1051–1100 的约 `0.116` 增至 step 1401–1450 的约 `2.841`；
- update ratio 始终很小，全程峰值约 `0.000454`；
- max absolute logit 约为 141–152；
- 参数始终有限；
- 没有 OOM 或 NCCL 故障。

因此它属于 **延迟 generation optimization 失稳**，不能把已有 checkpoint 的机械完整性当作科学有效性。step 1375 已在明显恶化区，恢复它不安全；从 step 1100 恢复同一 recipe 也只会继续相同轨迹。v2 root 被保留，未 resume，GCE 和 CE 没有启动。

这次结果同时推翻了“825-step probe 足以证明 2,750-step 稳定”的假设：`5e-6` 只能视为通过短控制，不能作为最终正式 LR。

## 8. 当前修复验证：`3e-6` / 1650-step matched 长 probes

为跨过已知 step-1450 失稳区间，只改变 peak LR：从 `5e-6` 降至 `3e-6`。其他模型、数据、batch、loss、optimizer、warmup 和 scheduler 语义均保持不变。三个 objective 分别从相同 base model 在 fresh root 启动，不能互相继承 checkpoint。

共同关键字段：

```text
learning_rate=3e-6
max_optimizer_steps=1650
scheduler_horizon_steps=2750
checkpoint_every_steps=275
8 GPUs × batch 4 × accumulation 1 = global batch 32
```

### 8.1 Attention：已科学成功

输出 root：

```text
<experiment-root>/probes/lumina_objective_ablation_8g_probe_attention_lr3e6_h1650_20260918
```

结果：

- metrics 为 step 1–1650，完整连续；
- exit code 为 0；
- quality 为 `SUCCEEDED`；
- step 101–400 generation-loss baseline 为 `2.802851`；
- 已知风险区 step 1301–1650 的 50-step 窗口平均 `L_gen` 保持在约 `2.70–2.83`，没有持续回归；
- 参数始终有限、无 gradient clipping，max absolute logit 约 145；
- checkpoint 275、550、825、1100、1375、1650 均有 `_SUCCESS` 和 8 个 rank state；
- 无 OOM、`NCCL WARN` 或 traceback；完成后 GPU 回到约 1,605 MiB/卡的运行前已有基线。

这证明 `3e-6` Attention 已跨过 v2 的 step-1450 风险点，但不能单独证明 GCE、CE 或完整 2,750-step 正式训练稳定。

### 8.2 GCE：已科学成功

Attention 完成并释放八卡后，GCE 从 base model 在下列 fresh root 启动：

```text
<experiment-root>/probes/lumina_objective_ablation_8g_probe_gce_lr3e6_h1650_20260918
```

生成配置：

```text
probe/configs/mb_gce_1650.yaml
```

最终结果：

- metrics 为 step 1–1650，完整连续；
- exit code 为 0；
- quality 为 `SUCCEEDED`；
- step 101–400 generation-loss baseline 为 `2.798524`；
- step 1301–1650 的所有完整 50-step 窗口平均 `L_gen` 为 `2.672–2.801`，相对 baseline 为 `0.955–1.001×`，没有持续回归；
- GCE loss、gradient、logit、update ratio 和参数始终有限，`bad_windows=0`；
- checkpoint 275、550、825、1100、1375、1650 均有 `_SUCCESS` 和 8 个 rank state；
- 日志无端口冲突、OOM、`NCCL WARN`、traceback 或质量失败；训练退出后 GPU 回到约 1,605 MiB/卡的运行前已有基线。

### 8.3 CE：已科学成功

CE 已从相同 base model、独立 fresh root 启动匹配的 `3e-6` / 1650-step probe，不继承 Attention 或 GCE checkpoint：

```text
<experiment-root>/probes/lumina_objective_ablation_8g_probe_ce_lr3e6_h1650_20260918
```

启动前已确认无 Lumina torchrun 或训练 tmux，且八卡已释放到约 1,605 MiB/卡。生成配置保持 stop horizon 1650、scheduler horizon 2750、每 275 步 checkpoint、8×4×1 global batch 32 和 peak LR `3e-6`。最终结果：

- metrics 为 step 1–1650，完整连续；
- exit code 为 0；
- quality 为 `SUCCEEDED`；
- fingerprint digest 为 `beb40f2ef6ed2396f7ab55f775f21525ccb1d0a5e9be7a8ff597680a2bfe11c5`；
- step 101–400 generation-loss baseline 为 `2.802612`；
- step 1401–1450、1451–1500、1501–1550、1551–1600、1601–1650 的窗口平均 `L_gen` 分别为 `2.701210`、`2.767142`、`2.718401`、`2.743615`、`2.760247`，均不高于 baseline，没有持续回归；
- 参数始终有限、`bad_windows=0`；
- checkpoint 275、550、825、1100、1375、1650 均有 `_SUCCESS`、8 个 rank state 及完整公共文件；
- 日志无端口冲突、OOM、`NCCL WARN`、traceback 或质量失败；
- 训练退出后无 Lumina torchrun 或训练 tmux，八卡均回到约 1,605 MiB/卡的已有基线。

Attention、GCE、CE 三个 matched 1650-step 长 probes 已全部科学成功，因此正式 recipe 已选择 `3e-6` 并写入三个正式 YAML。长 probe 仍不能替代完整 2750-step 正式实验；正式队列仍须从 base model 在全新 root 严格串行运行。

> `quality_status.json` 在训练进行时可能显示 `status=RUNNING, step=0`。这是运行期状态文件的写入语义，不表示训练停滞；实时步数以 `train_metrics.jsonl` 为准。

## 9. fresh v3 正式实验最终状态

正式输出 root：

```text
<experiment-root>/formal/lumina_objective_ablation_8g_v3_lr3e6_20260918
```

该队列于 2026-09-18 22:05 UTC 左右从 base model 在 fresh root 启动，没有继承 v1、v2、Smoke 或 probe checkpoint。固定顺序为 Attention → GCE → CE。

截至 2026-09-19 10:07 UTC，正式串行队列已完成：

| Objective | 最终状态 | Metrics | Quality / exit | 完整 checkpoints |
| --- | --- | --- | --- | --- |
| Attention | 科学成功 | 1–2750，连续 | `SUCCEEDED` / `0` | 275、550、825、1100、1375、1650、1925、2200、2475、2750 |
| GCE | 科学成功 | 1–2750，连续 | `SUCCEEDED` / `0` | 275、550、825、1100、1375、1650、1925、2200、2475、2750 |
| CE | 科学成功 | 1–2750，连续 | `SUCCEEDED` / `0` | 275、550、825、1100、1375、1650、1925、2200、2475、2750 |

Attention 最终验收结果：

- 2,750 条 metrics 完整连续；
- 十个预期 checkpoint 全部具有 `_SUCCESS`、8 个 rank state 和完整公共文件；
- step 101–400 baseline 为 `2.802851`；
- 最后一个窗口 step 2701–2750 的平均 `L_gen` 为 `2.790620`，是 baseline 的 `0.996×`；
- 全程参数有限，`bad_windows=0`；
- objective exit code 为 0，quality 为 `SUCCEEDED`；
- 首尾 checkpoint 与最终 quality 状态使用一致的 fingerprint：`c0d1b030bc6a3c576fa81313dda9bc3fb6da99fde7c168ecd0f99d1c93c13b4d`。

GCE 当前科学状态：

- step 101–400 baseline 已形成，为 `2.798524`；
- step 401–450 平均 `L_gen=2.742221`，为 baseline 的 `0.980×`；
- step 451–500 平均 `L_gen=2.674513`，为 baseline 的 `0.956×`；
- step 501–550 平均 `L_gen=2.698362`，为 baseline 的 `0.964×`；
- step 551–600 平均 `L_gen=2.740567`，为 baseline 的 `0.979×`；
- step 601–650、651–700、701–750、751–800 的平均 `L_gen` 分别为 `2.849402`、`2.800466`、`2.747329`、`2.723102`，相对 baseline 为 `1.018×`、`1.001×`、`0.982×`、`0.973×`；
- step 1001–1050、1051–1100、1101–1150 的平均 `L_gen` 分别为 `2.723564`、`2.735456`、`2.754890`，相对 baseline 为 `0.973×`、`0.977×`、`0.984×`；
- step 1151–1200、1201–1250、1251–1300、1301–1350 的平均 `L_gen` 分别为 `2.787550`、`2.715417`、`2.719374`、`2.711389`，相对 baseline 为 `0.996×`、`0.970×`、`0.972×`、`0.969×`；
- step 1351–1400、1401–1450、1451–1500、1501–1550、1551–1600 的平均 `L_gen` 分别为 `2.800601`、`2.672134`、`2.740134`、`2.688490`、`2.718327`，相对 baseline 为 `1.001×`、`0.955×`、`0.979×`、`0.961×`、`0.971×`；
- step 2601–2650、2651–2700、2701–2750 的平均 `L_gen` 分别为 `2.677098`、`2.676058`、`2.757802`，相对 baseline 为 `0.957×`、`0.956×`、`0.985×`；最后一个窗口平均 `gce_loss=3.679529`、平均 grad norm `0.220235`；
- 全程参数保持有限，`bad_windows=0`；
- 2,750 条 metrics 为 step 1–2750 完整连续，objective exit code 为 0，quality 为 `SUCCEEDED`；
- 十个预期 checkpoint 均已验证有 `_SUCCESS`、8 个 rank state，各 12 个文件；所有 checkpoint fingerprint 一致，为 `be46268bf22ff28cd802f87a0b138df80825a4ce118c8fbea9546962e7e3c394`；
- GCE 验收通过后队列才启动 CE，二者没有训练重叠；CE 当前 metrics 为 step 1–410 完整连续；
- CE step 101–400 baseline 已完整形成，为 `2.802612`；其六个 50-step 子窗口均值为 `2.793988`、`2.813210`、`2.845511`、`2.785154`、`2.762218`、`2.815594`，相对 baseline 为 `0.997–1.015×`，没有回归趋势；
- CE baseline 后 step 401–450、451–500、501–550 的平均 `L_gen` 分别为 `2.749348`、`2.681411`、`2.705355`，相对 baseline 为 `0.981×`、`0.957×`、`0.965×`；对应平均 grad norm 为 `0.095110`、`0.085883`、`0.084985`，均无 clipping，`bad_windows=0`；
- CE step 551–600、601–650、651–700、701–750、751–800 的平均 `L_gen` 分别为 `2.748738`、`2.859009`、`2.813184`、`2.761491`、`2.739007`，相对 baseline 为 `0.981×`、`1.020×`、`1.004×`、`0.985×`、`0.977×`；没有持续回归，均无 clipping，`bad_windows=0`；
- CE step 851–900、901–950、951–1000、1001–1050、1051–1100 的平均 `L_gen` 分别为 `2.803363`、`2.780227`、`2.750433`、`2.747730`、`2.759557`，相对 baseline 为 `1.000×`、`0.992×`、`0.981×`、`0.980×`、`0.985×`；没有持续回归，均无 clipping，`bad_windows=0`；
- CE step 1101–1150、1151–1200、1201–1250、1251–1300、1301–1350 的平均 `L_gen` 分别为 `2.778818`、`2.810690`、`2.739758`、`2.747060`、`2.738425`，相对 baseline 为 `0.992×`、`1.003×`、`0.978×`、`0.980×`、`0.977×`；没有持续回归，均无 clipping，`bad_windows=0`；
- CE step 1351–1400、1401–1450、1451–1500、1501–1550、1551–1600、1601–1650 的平均 `L_gen` 分别为 `2.828228`、`2.701210`、`2.767142`、`2.718401`、`2.743615`、`2.760247`，相对 baseline 为 `1.009×`、`0.964×`、`0.987×`、`0.970×`、`0.979×`、`0.985×`；CE 已健康通过 v2 的 step-1450 风险点和 1650-step 长 probe 覆盖区间，均无 clipping，`bad_windows=0`；
- CE step 1651–1700、1701–1750、1751–1800、1801–1850、1851–1900、1901–1950 的平均 `L_gen` 分别为 `2.704358`、`2.796480`、`2.849538`、`2.798476`、`2.798083`、`2.760346`，相对 baseline 为 `0.965×`、`0.998×`、`1.017×`、`0.999×`、`0.998×`、`0.985×`；没有持续回归，均无 clipping，`bad_windows=0`；
- CE step 1951–2000、2001–2050、2051–2100、2101–2150、2151–2200 的平均 `L_gen` 分别为 `2.715086`、`2.716701`、`2.783429`、`2.720446`、`2.774554`，相对 baseline 为 `0.969×`、`0.969×`、`0.993×`、`0.971×`、`0.990×`；没有持续回归，均无 clipping，`bad_windows=0`；
- CE step 2201–2250、2251–2300、2301–2350、2351–2400、2401–2450、2451–2500 的平均 `L_gen` 分别为 `2.851369`、`2.717051`、`2.730264`、`2.718014`、`2.797784`、`2.829249`，相对 baseline 为 `1.017×`、`0.969×`、`0.974×`、`0.970×`、`0.998×`、`1.010×`；没有持续回归，均无 clipping，`bad_windows=0`，最大 logit 为 145，最大 update ratio 继续下降；
- CE 最终 50-step 窗口 step 2701–2750 的平均 `L_gen` 为 `2.790224`，为 baseline `2.802612` 的 `0.996×`；最后一步 `L_gen=3.107358`，平均 grad norm `0.096658`，无 clipping，最大 logit 145，`bad_windows=0`；
- CE 的 2,750 条 metrics 为 step 1–2750 完整连续，所有 loss 和参数有限，objective exit code 为 0，quality 为 `SUCCEEDED`；
- CE checkpoint 275、550、825、1100、1375、1650、1925、2200、2475 和 2750 均已验证有 `_SUCCESS`、8 个 rank state 和 12 个文件，fingerprint 一致，为 `17fd51bf3d7c3b7d6e27dd9393af6a35511039bcdb2d3ce79c735357ddc168b6`；
- 严格串行总队列于 2026-09-19 10:07 UTC 完成，`all.exit_code=0`；当前 v3 日志错误匹配数为 0，无 OOM、`NCCL WARN`、traceback、`QUALITY_FAILED` 或 `PROCESS_FAILED`。

最终运行状态：

- `lumina_all` 已正常退出，三个 objective 严格按 Attention → GCE → CE 串行完成；
- 已确认无残留 Lumina torchrun 或训练 worker；
- 八张 GPU 均已回到约 1,605 MiB 的运行前基线占用；
- `ttadk`、`tunnel` 及其他无关进程未触碰；
- session-only 定期监控任务已在全部验收通过后停止。

最终产物：

- 已按 `scripts/eval/plot_training_curves.py` 生成 14 张 PNG：Attention 7 张、GCE 4 张、CE 2 张，以及三项 `L_gen` 横向比较图 1 张；
- comparison 图：`<experiment-root>/formal/lumina_objective_ablation_8g_v3_lr3e6_20260918/comparison/compare_generation_loss.png`；
- 轻量结果包：`<experiment-root>/archives/lumina_objective_ablation_8g_v3_lr3e6_20260918_results.tar.gz`；
- 生成后人工检查曲线时发现原 rolling mean 在首尾使用零填充，造成边界假性下跌；已改为按实际可用样本数归一化，补充边界单测（`tests/eval/test_plot_training_curves.py`，6 passed），重新生成全部曲线和归档；
- 最终结果包大小为 8,059,221 bytes，SHA-256 为 `e2f21b9fe2cdcff7f2df197667e1efc3b1e0b40991c17b828313e5116e2bd5f0`；
- 归档共 59 个 entries，经检查不含 checkpoint 目录、`.pt`、`.pth` 或 `.safetensors` 权重；保留了 metrics、quality 状态、日志、exit code、曲线、正式 configs 和 provenance 文本；
- 工作树包含本轮未提交源码与配置修改，因此结果包同时保存了 `git_status.txt` 和实际使用的三份 YAML；`git_commit.txt` 不能单独代表这些未提交改动。

旧 v1 watcher 中的 `lumina_all RUNNING` 来自全局同名的当前 v3 tmux session，不表示 v1 被恢复。v1 仍仅作为失败证据保留。

## 10. 后续验收与操作原则

每个正式 objective 必须满足：

- 2,750 条连续 metrics；
- 十个预期 checkpoint 全部完整；
- objective exit code 为 0；
- `quality_status.json` 为 `SUCCEEDED`；
- generation loss 有限且没有持续回归。

全部三项完成后还需：

- `all.exit_code=0`；
- 无残留 Lumina 训练进程或训练 tmux；
- GPU 回到约 1,605 MiB/卡的已有基线；
- 为三个 objective 生成训练曲线；
- 生成三组 `L_gen` comparison；
- 创建不包含 checkpoint、模型和数据资产的轻量结果包。

如果后续失败：

- 保留当前 output root，不覆盖证据；
- 先定位根因；
- 只有 checkpoint 完整、fingerprint 一致、metrics 连续，并且 checkpoint 位于可判定的科学安全区间时才允许 resume；
- 若 recipe、代码或 loss 语义变化，则必须使用 fresh root 从 base model 重启；
- 不从旧两卡或已确认发散的 checkpoint 恢复。

## 11. 结果目录与查看方式

所有 Lumina objective ablation 结果现已统一整理到：

```text
<experiment-root>
```

目录分层如下：

- `formal/`：正式串行 Attention → GCE → CE 结果；
- `probes/`：长探针结果；
- `smokes/`：Smoke 结果；
- `archives/`：轻量归档。

当前正式 v3 root：

```text
<experiment-root>/formal/lumina_objective_ablation_8g_v3_lr3e6_20260918
```

正式训练已经完成，因此这里查看的是最终结果快照，而不是仍在滚动的 live watcher。

### 查看正式总览

```bash
cat <experiment-root>/formal/lumina_objective_ablation_8g_v3_lr3e6_20260918/logs/progress.log
```

### 查看最终 CE 日志尾部

```bash
tail -n 40 <experiment-root>/formal/lumina_objective_ablation_8g_v3_lr3e6_20260918/logs/ce.log
```

### 查看最终一条 CE 指标

```bash
tail -n 1 <experiment-root>/formal/lumina_objective_ablation_8g_v3_lr3e6_20260918/MB-CE-8G-B4-A1-S42/train_metrics.jsonl
```

### 查看 comparison 图与轻量结果包

```bash
ls <experiment-root>/formal/lumina_objective_ablation_8g_v3_lr3e6_20260918/comparison
ls <experiment-root>/archives
```

截至 2026-09-21，旧的 `lumina_progress` 与 `lumina_formal_progress` watcher 已停止；v1、v2、Smoke、probes 与 v3 正式结果都已按上述目录结构归档保留。v1 和 v2 的 `progress.log` 继续仅作为失败实验的历史证据，不作为当前正式队列的实时判断依据。
