# Lumina-DiMOO MagicBrush Experiments

本仓库用于在 **Lumina-DiMOO** 上进行 MagicBrush 图像编辑实验，统一支持 CE、Attention 和 GCE 三种训练目标，并提供数据处理、LoRA 微调、DDP 训练、checkpoint 和评测流程。

## 项目结构

```text
lumina/
├── configs/        # 数据、训练、分布式配置
├── docs/           # 环境与实验说明
├── scripts/        # train / eval / data / inference
├── src/            # dataset / model / training 实现
├── tests/          # 单元测试
├── pyproject.toml
└── uv.lock
```

详细结构见 `docs/ARCHITECTURE.md`。

## 环境安装

项目使用 Python 3.10 和 `uv` 管理依赖：

```bash
git clone git@github.com:yiyezhiqiu2077/lumina.git
cd lumina

uv sync --frozen --extra dev
```

检查环境：

```bash
uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

详细环境说明见 `docs/ENVIRONMENT.md`。

## 模型与数据

模型、数据和实验输出不提交到 Git。

推荐设置：

```bash
export PROJECT_ROOT=/path/to/lumina
export ASSET_ROOT=/path/to/lumina_assets

export MODEL_PATH="$ASSET_ROOT/models/Lumina-DiMOO"
export DATA_ROOT="$ASSET_ROOT/datasets/lumina_edit"
export DATA_CONFIG="$DATA_ROOT/magicbrush/tokens/train/manifest.jsonl"
export GCE_CLUSTER_PATH="$DATA_ROOT/artifacts/gce_clusters_1024_512.pt"

export OUTPUT_ROOT=/path/to/experiments/lumina/example_run
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

MagicBrush 数据处理流程：

```text
raw images
→ shared geometry
→ VQ pre-tokenization
→ manifest.jsonl
→ training
```

完整数据准备流程见 `docs/EXPERIMENTS.md`。

## 正式实验

当前 matched experiments：

```text
configs/train/ablation/
├── mb_ce_8g_b4_a1.yaml
├── mb_attention_8g_b4_a1.yaml
└── mb_gce_8g_b4_a1.yaml
```

共同配置：

| 参数 | 值 |
| --- | --- |
| GPU | 8 |
| Batch / GPU | 4 |
| Global Batch | 32 |
| Precision | BF16 |
| Learning Rate | `3e-6` |
| LoRA Rank | 16 |
| Training Steps | 2750 |
| Checkpoint Interval | 275 |

三种 objective：

```text
CE:
L_total = L_gen

Attention:
L_total = L_gen + λ_attn L_attn

GCE:
L_total = L_gen + λ_gce L_gce
```

## 训练

统一训练入口：

```bash
uv run python scripts/train/train.py --config <CONFIG>
```

推荐使用 tmux workflow：

```bash
bash scripts/train/run_tmux.sh all
```

按顺序运行：

```text
Attention → GCE → CE
```

也可以单独运行：

```bash
bash scripts/train/run_tmux.sh attention
bash scripts/train/run_tmux.sh gce
bash scripts/train/run_tmux.sh ce
```

推荐正式实验流程：

```text
Smoke → 1650-step Probe → 2750-step Formal Run
```

详细说明见 `docs/EXPERIMENTS.md`。

## 分布式训练

当前正式支持：

```text
Single-node Multi-GPU DDP
```

当前验证配置：

```text
1 node × 8 GPUs
```

每张 GPU 一个 process，使用 `DistributedSampler` 划分数据，并通过 PyTorch DDP 同步梯度。

当前尚未提供正式的 multi-node DDP 启动接口。

## 测试

```bash
uv run pytest -q
```

## 文档

```text
docs/ARCHITECTURE.md        # 代码结构
docs/ENVIRONMENT.md         # 环境配置
docs/EXPERIMENTS.md         # 数据与实验流程
docs/EXPERIMENT_PROGRESS.md # 实验记录
```



见 `LICENSE`。
