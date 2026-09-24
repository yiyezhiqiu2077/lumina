# Lumina-DiMOO：Mixed editing objective ablation

## Repository Layout

- 实现：`src/`
- 可执行工作流：`scripts/`
- 实验与运行配置：`configs/`
- 文档：`docs/`
- 静态资产：`assets/`
- vendored third-party：`third_party/`

当前依赖方向和 package 职责见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，环境说明见
[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。当前主工作流是 MagicBrush + RefEdit 的 2×3
controlled ablation；旧 MagicBrush-only v3 配置仍保留在 `configs/train/ablation/mb_*.yaml`，仅用于
追溯历史结果，不是当前 mixed recipe。

| objective | `full_target` | `edit_region_hardlock` |
| --- | --- | --- |
| CE | `mixed_ce_8g_b4_a1.yaml` | `mixed_ce_editregion_8g_b4_a1.yaml` |
| Attention | `mixed_attention_postrope_region_8g_b4_a1.yaml` | `mixed_attention_editregion_8g_b4_a1.yaml` |
| GCE | `mixed_gce_8g_b4_a1.yaml` | `mixed_gce_editregion_8g_b4_a1.yaml` |

三种 objective 互斥：CE 为 `L_gen + 1e-5 × L_z`；Attention 为
`L_gen + 1e-5 × L_z + 0.3 × L_attn`；GCE 为
`L_gen + 1e-5 × L_z + 1.0 × L_gce`（levels 1024 / 512）。它们共享 padding correctness、
data corruption、LoRA、DDP、gradient accumulation、checkpoint 和 resume 实现。

## 环境与资产

```bash
git clone git@github.com:yiyezhiqiu2077/lumina.git
cd lumina
uv sync --extra dev

bash scripts/setup_local_assets.sh model /path/to/Lumina-DiMOO
bash scripts/setup_local_assets.sh magicbrush /path/to/magicbrush-token-root
bash scripts/setup_local_assets.sh refedit /path/to/refedit-final-mask
bash scripts/setup_local_assets.sh mixed /path/to/mixed-token-root
```

本项目通过 Git 忽略的 `local_assets/` 使用机器本地资产：
`local_assets/models/Lumina-DiMOO`、`local_assets/datasets/magicbrush`、
`local_assets/datasets/refedit`、`local_assets/datasets/mixed` 与
`local_assets/experiments`。模型权重、token data、GCE cluster、logs 与 checkpoint 均不在 Git 中。
mixed manifest 的 `token_file` 相对其自身父目录解析；具体 dataset revision 和审计结果以该目录的
`dataset_meta.json` / mixed metadata 为准。

当前审计的 mixed manifest 为 MagicBrush 8807 + RefEdit 7804 = 16611 样本。8 GPU、batch/GPU 4、
accum 1 时，实际每 epoch step 与总 step 由 manifest、sampler 与 loader 自动计算；当前该 manifest
对应 519 step/epoch、5190 step/10 epoch，不是硬编码常数。

## 当前 mixed 训练 recipe

| 项目 | 值 |
| --- | --- |
| GPU / batch / accumulation / global batch | 8 / 4 / 1 / 32 |
| seed / precision / max sequence | 42 / BF16 / 5120 |
| optimizer | AdamW，lr=3e-6，betas=(0.9, 0.95)，wd=0.1，clip=4.0，warmup=20 |
| generation reduction | global `token_mean`，`z-loss=1e-5` |
| LoRA | r=16，alpha=16，dropout=0.05；q/k/v/attn_out |
| Attention | post-RoPE、`region_mass`、layers 24–27、weight=0.3 |
| GCE | weight=1.0，levels 1024 / 512 |

`full_target` 保持 Lumina 原有的 whole-target random masked corruption。`edit_region_hardlock` 在
GT mask 外使用 source codes，在 mask 内保留 target codes，并只将当前 cosine corruption 随机选出的
subset 置为 MASK；generation loss 只监督这个 subset。Attention loss 始终使用完整 GT edit mask，
对所有 source spatial tokens softmax：`P_G = sum_{j in M} p_j`，
`L_attn = -log(P_G.clamp_min(eps))`。它不是 hard attention mask，也不只在 mask 内 softmax。

在 `edit_region_hardlock + token_mean` 下，mask 较大的 sample 自然贡献更多 supervised tokens，
因此 MagicBrush / RefEdit 的 sample proportion 不等于 supervised-token proportion。日志会记录
`valid_target_token_sum` 与 `supervised_token_share`；这只是当前 recipe 的可解释性质，不参与 loss。

## 启动与验证

当前 4×A800 只用于 correctness：已完成 20-step full-target smoke、20-step editregion smoke，及
Attention/GCE accum=1 diagnostic。不要在当前机器启动 8-GPU formal run。

```bash
# 4-GPU correctness；默认/兼容旧调用为 full_target
bash scripts/train/run_mixed_smokes.sh full --print-command
bash scripts/train/run_mixed_smokes.sh editregion --print-command
bash scripts/train/run_mixed_smokes.sh all --print-command

# 未来 8-GPU 机器：只打印正式命令；--run 要求 clean Git worktree
bash scripts/train/run_mixed_formal.sh \
  configs/train/formal/mixed_ce_8g_b4_a1.yaml --print-command
```

每个 formal model 必须从同一 original Lumina base 和 fresh LoRA 开始，使用独立 output root；不得
resume 4-GPU smoke、旧 v3、另一个 objective 或另一个 corruption mode。checkpoint fingerprint 覆盖
corruption、训练语义与 runtime semantic source，错误 resume 会被拒绝。run provenance 记录 Git SHA、
dirty status 与 `uv.lock` SHA256。

## 评测与工具

GT-mask hard-lock generation 的主 evaluator 是
[`scripts/eval/evaluate_gt_mask_editing.py`](scripts/eval/evaluate_gt_mask_editing.py)。它支持
Source-copy Token Accuracy、Edit Token Accuracy、Changed-token Accuracy、Inside L1、Inside PSNR、
Inside L1 vs Target Reconstruction、boundary leakage/seam diagnostic 和 oracle hard-lock diagnostic。
这里的 GT mask 是 inference input，不是 free-edit localization benchmark；未实现的 LPIPS/DINO/CLIP
不应被表述为已有正式结果。

数据准备、RefEdit 审计和 manifest 构建见 `scripts/data/`；GCE cluster 是独立资产，工具位于
`scripts/tools/gce/`。更多面向新集群的一条命令流程、环境检查、smoke、formal、评测与打包见
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

```bash
uv run pytest -q
```

多节点启动、FP8、FSDP 和 FlashAttention 性能优化不属于当前工作流的承诺范围。
