# Lumina-DiMOO：统一 MagicBrush 训练

## Repository Layout

- 实现：`src/`
- 可执行工作流：`scripts/`
- 实验与运行配置：`configs/`
- 文档：`docs/`
- 静态资产：`assets/`
- vendored third-party：`third_party/`

当前依赖方向和 package 职责见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，环境说明见
[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。

本仓库的 `main` 只维护一份 MagicBrush 训练框架，通过唯一的 objective 模式隔离三种
实验：

| 模式 | 总损失 | 额外依赖 |
| --- | --- | --- |
| `ce` | `L_gen` | 无 |
| `attention` | `L_gen + 0.1 × L_attn` | attention auxiliary |
| `gce` | `L_gen + 1.0 × L_gce` | `GCE_CLUSTER_PATH` |

三种模式互斥：attention 不加载或实例化 GCE；GCE 不请求 attention Q/K auxiliary；CE
只计算官方 generation CE。它们共享 padding correctness、数据 corruption、LoRA、DDP、
gradient accumulation、checkpoint 和 resume 实现。

## 环境与资产

```bash
git clone git@github.com:yiyezhiqiu2077/lumina.git
cd lumina
git switch main
uv sync --extra dev

export MODEL_PATH=/path/to/Lumina-DiMOO
export DATA_ROOT=/path/to/MagicBrush
export OUTPUT_ROOT=/path/to/lumina_outputs
```

`uv sync` 会按项目锁定的 PyTorch CUDA 12.1 wheel index 安装依赖。数据根目录应包含
`official_tokens/train/manifest.jsonl`；模型权重、MagicBrush 数据、GCE cluster、日志和
checkpoint 均不在 Git 中。

## 三个启动入口

```bash
# B0：与 GCE 设置匹配的 CE baseline（4 GPU × batch 4 × accum 2）
uv run python scripts/train/train.py --config configs/train/magicbrush_ce.yaml

# A1：attention supervision（2 GPU × batch 8 × accum 4）
uv run python scripts/train/train.py --config configs/train/magicbrush_attention.yaml

# G1：GCE（4 GPU × batch 4 × accum 2）
export GCE_CLUSTER_PATH=/path/to/gce_clusters_1024_512.pt
uv run python scripts/train/train.py --config configs/train/magicbrush_gce.yaml
```

三个配置均由唯一入口 [train.py](scripts/train/train.py) 读取。该入口会读取各训练配置
显式引用的 dataset / distributed 配置，在启动前校验 global batch，然后使用
`torch.distributed.run` 启动同一脚本的 worker 模式。差异只在 `objective.mode` 与各自真实
超参数。CE/GCE 使用 global batch 32；attention 使用 global batch 64，因此 CE 不应被
表述为与 attention 的严格 matched baseline。

`DATA_CONFIG` 可覆盖 dataset 配置中的默认 manifest，`OUTPUT_ROOT` 指定输出根目录；
`--resume-from-checkpoint` 可从新进程恢复。例如：

```bash
uv run python scripts/train/train.py --config configs/train/magicbrush_gce.yaml \
  --resume-from-checkpoint /path/to/checkpoint-000500
```

## 配置与验证

- [configs/train/magicbrush_ce.yaml](configs/train/magicbrush_ce.yaml)
- [configs/train/magicbrush_attention.yaml](configs/train/magicbrush_attention.yaml)
- [configs/train/magicbrush_gce.yaml](configs/train/magicbrush_gce.yaml)

## Matched Objective Ablation

严格 matched 的 CE / Attention / GCE 对照实验位于
[`configs/train/ablation/`](configs/train/ablation/)，统一使用 2 GPU、每卡 batch 8、gradient
accumulation 2、global batch 32 和相同的训练 schedule。完整的环境、资产准备、前台 smoke、
一条命令 tmux 正式启动、resume、曲线与结果打包流程见
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)。

`configs/train/magicbrush_*.yaml` 是既有实验配置；它们保留用于历史实验，不构成这三种
objective 的严格 matched comparison。

## 数据与工具

数据准备和 tokenization 共用 `src/dataset/` 的实现：

```bash
# 从原始 MagicBrush manifest 创建固定的 train/val/probe manifest
uv run python scripts/data/preprocess_magicbrush.py prepare \
  --train-manifest /path/to/train.jsonl --output /path/to/prepared

# 用本地 Lumina VQ-VAE 预编码上述 manifest
uv run python scripts/data/preprocess_magicbrush.py pretokenize \
  --manifest /path/to/prepared/train.jsonl --model "$MODEL_PATH" --output /path/to/tokens

# 检查图像/VQ 几何、序列契约和非方形 token 布局
uv run python scripts/data/check_magicbrush.py all \
  --manifest /path/to/tokens/manifest.jsonl --model "$MODEL_PATH" --output /path/to/audit
```

GCE cluster 是独立数据资产，不会提交到 Git。使用工具生成和检查：

```bash
uv run python scripts/tools/gce/build_clusters.py --model "$MODEL_PATH" --output /path/to/gce_clusters_1024_512.pt
uv run python scripts/tools/gce/inspect_clusters.py --model "$MODEL_PATH" --clusters /path/to/gce_clusters_1024_512.pt
uv run python scripts/tools/gce/probe_scale.py --model "$MODEL_PATH" --manifest /path/to/tokens/manifest.jsonl --clusters /path/to/gce_clusters_1024_512.pt
```

attention layer calibration 位于 `scripts/tools/attention/calibrate_layers.py`。hard-lock 评测与
评测 manifest 预处理位于 `scripts/eval/`。

## Upstream SFT 与推理

原始 Lumina SFT 依赖单独的 upstream extra：

```bash
uv sync --extra upstream
uv run python scripts/train/upstream/train.py --help
```

上游 SFT 的示例数据配置在 `configs/upstream/data.yaml`。推理入口为：

```bash
uv run python scripts/inference/t2i.py --help
uv run python scripts/inference/i2i.py --help
uv run python scripts/inference/mmu.py --help
uv run python scripts/inference/t2i_ddp.py --help
```

## Repository Structure

- `src/dataset/`：MagicBrush 训练数据与稳定 corruption。
- `src/models/`：Lumina 模型、attention 与 GCE objective 的唯一实现。
- `src/training/`：LoRA、objective dispatch、checkpoint、DDP 与配置加载。
- `scripts/train/train.py`：唯一正式 MagicBrush 训练入口。
- `scripts/data/`：MagicBrush 数据准备与契约检查 CLI。
- `scripts/tools/gce/`：GCE cluster 构建、检查和尺度诊断 CLI；损失实现不在脚本中。
- `scripts/eval/`：MagicBrush 评测工作流。
- `configs/`：训练、数据和 distributed specification。
- `tests/`：按 data/models/objectives/training 分层的 regression tests。

```bash
uv run pytest -q
```

