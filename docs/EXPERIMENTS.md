# Objective ablation 实验协议

本文定义第一轮 Lumina-DiMOO / MagicBrush objective ablation 的可复现实验协议；它只描述
配置、数据 contract、命令与记录规范，不预设任何 objective 的结果优劣。

## 目标与环境

第一阶段在同一 MagicBrush token manifest 上比较：E0=CE、E1=Attention、E2=GCE。三者
共享数据顺序、seed、BF16、LoRA、AdamW、DDP 与 checkpoint 机制；唯一 objective 差异为
辅助损失项。推荐环境：

```bash
cd /path/to/lumina
uv sync --extra dev --extra upstream
# 画图时额外安装：uv sync --extra dev --extra upstream --extra analysis

export CUDA_VISIBLE_DEVICES=0,1
export MODEL_PATH=/path/to/Lumina-DiMOO
export DATA_ROOT=/data02/<user>/datasets/lumina_edit
export DATA_CONFIG=$DATA_ROOT/magicbrush/tokens/train/manifest.jsonl
export OUTPUT_ROOT=/data02/<user>/experiments/lumina_objective_ablation_v1
export GCE_CLUSTER_PATH=$DATA_ROOT/artifacts/gce_clusters_1024_512.pt
```

第一轮可直接使用现有 token manifest，避免为目录命名复制大型数据：

```bash
export DATA_CONFIG=/data02/zhangyuyang/experiments/lumina_dimoo_magicbrush_a0_a1_v2_20260909/dataset/official_tokens/train/manifest.jsonl
wc -l "$DATA_CONFIG"  # 必须为 8807
```

推荐的长期数据布局为：

```text
/data02/<user>/datasets/lumina_edit/
├── magicbrush/{raw,prepared,tokens/train/{manifest.jsonl,files/}}
├── senior/{raw,prepared,tokens/train/{manifest.jsonl,files/}}
├── mixed/{magicbrush_senior_1to1.jsonl,magicbrush_senior_all.jsonl}
└── artifacts/gce_clusters_1024_512.pt
```

## 数据 contract 与检查

训练输入是 canonical token manifest；每行须能定位 source/target/edit mask 与 token 文件，并
保持 source spatial token、target token 和 mask 的既有语义。训练前运行：

```bash
uv run python scripts/data/check_magicbrush.py all \
  --manifest "$DATA_CONFIG" --model "$MODEL_PATH" --output "$OUTPUT_ROOT/data_audit"
```

Senior 原始数据尚不接入 trainer。未来 adapter 的 canonical raw manifest 每行必须至少为：

```json
{"index": 0, "sample_key": "senior/...", "session_id": "...", "instruction": "...", "source": "...", "target": "...", "mask_edit": "..."}
```

`source`、`target`、`mask_edit` 必须空间对齐；mask 的 `1/255` 表示编辑区域、`0` 表示未变
区域；`sample_key` 必须在跨数据集范围全局唯一。MagicBrush 使用 `magicbrush/...` 前缀，
Senior 使用 `senior/...` 前缀。拿到真实 Senior metadata 后，按“审计 schema → 转 canonical
raw manifest → shared geometry → VQ tokenization → 验证 mask → 建 token manifest → 构建固定
1:1 mixed manifest”的顺序实现，不把 Senior 专用 schema 写入 trainer。

GCE cluster 是 VQ codebook 的资产，不依赖某个数据集。存在时先检查；不存在时构建一次：

```bash
uv run python scripts/tools/gce/build_clusters.py \
  --model "$MODEL_PATH" --output "$GCE_CLUSTER_PATH" \
  --device cuda:0 --levels 1024 512 --seed 0
```

## Matched training configuration

正式 YAML 位于 `configs/train/ablation/`。共同设置为 2 GPU、batch/GPU=8、accum=2、
global batch=32、seed=42、BF16、`max_seq_len=5120`、AdamW (`1e-5`, betas `[0.9,0.95]`,
weight decay `0.1`)、clip `4.0`、warmup 20、LoRA rank/alpha/dropout=`16/16/0.05`，及
`condition_dropout=0.1`。

MagicBrush token manifest 有 8807 行。`DistributedSampler(drop_last=True)` 与
`DataLoader(drop_last=True)` 下，每 rank 每 epoch 有 550 micro-batches；accum=2，故每 epoch
为 275 optimizer steps。三组均训练 10 epochs=2750 steps，每 275 steps 保存 checkpoint：
275、550、825、1100、1375、1650、1925、2200、2475、2750。

| Run | Config | Objective |
| --- | --- | --- |
| E0 | `mb_ce_2g_b8_a2.yaml` | `L_total=L_gen` |
| E1 | `mb_attention_2g_b8_a2.yaml` | `L_gen + 0.1 * L_attn`，layers 24–27 |
| E2 | `mb_gce_2g_b8_a2.yaml` | `L_gen + 1.0 * L_gce`，levels 1024/512 |

正式顺序是 Attention → GCE → CE；不要在本协议阶段把 smoke YAML 提交或启动完整训练。

```bash
uv run python scripts/train/train.py --config configs/train/ablation/mb_attention_2g_b8_a2.yaml
uv run python scripts/train/train.py --config configs/train/ablation/mb_gce_2g_b8_a2.yaml
uv run python scripts/train/train.py --config configs/train/ablation/mb_ce_2g_b8_a2.yaml
```

## Loss curves 与 checkpoint/validation

trainer 在 `$OUTPUT_ROOT/<output_name>/train_metrics.jsonl` 记录 optimizer-step 指标；checkpoint
为 `checkpoint-000275/` 等。绘图只读取该 JSONL，不改训练过程：

```bash
uv run python scripts/eval/plot_training_curves.py \
  --input "$OUTPUT_ROOT/MB-ATTN-2G-B8-A2-S42/train_metrics.jsonl" \
  --objective attention --steps-per-epoch 275 --smooth-window 25 \
  --output "$OUTPUT_ROOT/MB-ATTN-2G-B8-A2-S42/curves"

uv run python scripts/eval/plot_training_curves.py compare \
  --ce "$OUTPUT_ROOT/MB-CE-2G-B8-A2-S42/train_metrics.jsonl" \
  --attention "$OUTPUT_ROOT/MB-ATTN-2G-B8-A2-S42/train_metrics.jsonl" \
  --gce "$OUTPUT_ROOT/MB-GCE-2G-B8-A2-S42/train_metrics.jsonl" \
  --steps-per-epoch 275 --smooth-window 25 --output "$OUTPUT_ROOT/comparison"
```

每图显示低透明度 raw curve 与 rolling mean，x 轴为 optimizer step，并在每个 epoch boundary
标记虚线。Attention 输出 generation、raw/weighted attention、total、ratio、localization、entropy
与 layer 24–27 localization 图；GCE 输出 generation、GCE、各 level、total；CE 输出 generation
与 total。compare 只比较三组 `L_gen`，不能比较 `L_total`，因为 objective 的 total 定义不同。

## Senior mixed experiments

第二阶段定义 M0=CE、M1=Attention、M2=GCE，首版采用 MagicBrush:Senior=1:1。若 Senior 也有
8807 个样本，mixed manifest 为 17614 行，约 550 optimizer steps/epoch、10 epochs 为 5500
steps；否则必须依最终 manifest 行数重新计算，不能沿用该估算。

## Experiment record

每次运行记录：Run ID、Git SHA、manifest/样本数、objective、GPU IDs/type/count、batch/GPU、
accumulation、global batch、seed、max sequence length、steps/epochs、LR、LoRA rank、auxiliary
weight、开始/结束时间、wall time、peak VRAM、best checkpoint 与 notes。训练前不得在文档中宣称
Attention 或 GCE 优于 CE。
