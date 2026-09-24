# Lumina-DiMOO Mixed Editing Experiments

本仓库的正式实验使用 MagicBrush + RefEdit mixed 数据，统一比较 CE、Attention、GCE 三种
训练目标，以及 `full_target`、`edit_region_hardlock` 两种 target corruption。

## 环境安装

```bash
git clone git@github.com:yiyezhiqiu2077/lumina.git
cd lumina
git checkout <FORMAL_EXPERIMENT_CODE_SHA>
uv sync --frozen --extra dev --extra upstream --extra analysis
```

## 模型与数据

使用本机 `local_assets/` 软链保存模型和数据，不提交这些资产、checkpoint 或输出。

```bash
export PROJECT_ROOT=/path/to/lumina
cd "$PROJECT_ROOT"

bash scripts/setup_local_assets.sh model /path/to/Lumina-DiMOO
bash scripts/setup_local_assets.sh magicbrush /path/to/magicbrush-token-root
bash scripts/setup_local_assets.sh refedit /path/to/refedit-final-mask
bash scripts/setup_local_assets.sh mixed /path/to/mixed-token-root

export MODEL_PATH="$PROJECT_ROOT/local_assets/models/Lumina-DiMOO"
export DATA_ROOT="$PROJECT_ROOT/local_assets/datasets/mixed"
export DATA_CONFIG="$DATA_ROOT/manifest.jsonl"
export GCE_CLUSTER_PATH=/path/to/gce_clusters_1024_512.pt
export OUTPUT_ROOT=/path/to/experiments/lumina_mixed_2x3
mkdir -p "$OUTPUT_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

当前 audited mixed manifest 包含 MagicBrush 8807 条、RefEdit 7804 条，共 16611 条。

## 正式配置

| 项目 | 值 |
| --- | --- |
| GPU | 8 |
| Batch / GPU | 4 |
| Gradient Accumulation | 1 |
| Global Batch | 32 |
| Precision | BF16 |
| Learning Rate | 3e-6 |
| LoRA | r16 / alpha16 / dropout0.05 |
| Loss Reduction | `token_mean` |
| Epochs | 10 |
| Steps / Epoch | 519 |
| Formal Steps | 5190 |

`519 / 5190` 是当前 manifest 的计算结果；实际运行由 `load_train_config()` 动态解析，manifest
变化时会自动适配。

## 六组实验

| Objective | `full_target` | `edit_region_hardlock` |
| --- | --- | --- |
| CE | `mixed_ce_8g_b4_a1.yaml` | `mixed_ce_editregion_8g_b4_a1.yaml` |
| Attention | `mixed_attention_postrope_region_8g_b4_a1.yaml` | `mixed_attention_editregion_8g_b4_a1.yaml` |
| GCE | `mixed_gce_8g_b4_a1.yaml` | `mixed_gce_editregion_8g_b4_a1.yaml` |

- CE：`L_total = L_gen + 1e-5 L_z`
- Attention：`L_total = L_gen + 1e-5 L_z + 0.3 L_attn`；使用 post-RoPE Q/K、`region_mass`、layers 24–27。
- GCE：`L_total = L_gen + 1e-5 L_z + 1.0 L_gce`；levels 为 1024 / 512。

`full_target` 使用全目标随机 masked corruption。`edit_region_hardlock` 中 GT mask 外使用 source token，
GT mask 内的本轮选中位置使用 MASK，未选中位置保留 target token；generation loss 仅监督本轮 MASK 的
GT-region token。Attention loss 使用完整 GT mask，而不是随机 subset。

## 训练

先只打印并检查全部六组命令：

```bash
bash scripts/train/run_mixed_2x3_formal.sh --print-command
```

确认环境与输出目录为空后运行：

```bash
bash scripts/train/run_mixed_2x3_formal.sh --run
```

运行顺序为 CE full → CE editregion → Attention full → Attention editregion → GCE full → GCE editregion。
六组共用一个 `OUTPUT_ROOT`，通过各自 `output_name` 写入独立子目录。每组完成后必须同时满足 quality status、
最后一条 metric 和最终 checkpoint success marker 的 success gate，否则 pipeline 立即停止。

## 评测

GT-mask hard-lock evaluator：

```bash
uv run python scripts/eval/evaluate_gt_mask_editing.py --help
```

当前实现的核心指标包括：

- Source-copy Token Accuracy
- Edit Token Accuracy
- Changed-token Accuracy
- Inside L1 / MSE / PSNR
- Inside L1 vs Target Reconstruction
- Boundary L1
- Full L1 / MSE / PSNR
- Oracle hard-lock diagnostic

六组评测须使用相同 held-out samples、GT masks、timesteps、CFG 和 seeds。

## 测试

```bash
uv run pytest -q
```
