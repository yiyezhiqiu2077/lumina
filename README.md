# Lumina-DiMOO：MagicBrush CE/GCE 实验

本分支用于在 MagicBrush 上复现并比较两组受控训练：

- `B0-CE`：官方 image-only masked-token CE。
- `G1-GCE`：在相同训练设置上加入 image-only Grouped Cross-Entropy。

共同实验基线为 `1475c9739cd6f5e2dacabbd82f907e71504e7109`。本分支不包含
attention supervision、full-attention supervision 或 ML-Cache 的改动。

## 环境与前置资产

推荐 Python 3.10、BF16 和 `uv`。在目标集群上先安装与 CUDA/驱动匹配的 PyTorch
wheel，再执行 `uv sync --extra dev`。完整依赖版本、当前验证环境及 H100 使用建议见
[ENVIRONMENT.md](ENVIRONMENT.md)。首轮迁移不要启用 FP8。

以下资产不在 Git 中，需自行准备并通过环境变量提供路径：

- `MODEL_PATH`：Lumina-DiMOO 预训练模型目录。
- `DATA_ROOT`：MagicBrush 已处理数据根目录。
- `OUTPUT_ROOT`：本次运行的输出根目录。
- `GCE_CLUSTER_PATH`：GCE 的 8192-codebook 聚类文件；仅 GCE 运行需要。

数据目录、manifest、512→32×32 VQ token 以及 mask 的约定见
[docs/DATASET.md](docs/DATASET.md)。请先阅读 [ASSETS.md](ASSETS.md) 并在目标集群
重新获取资产；不要将权重、数据、cluster `.pt`、checkpoint 或输出提交到 Git。

## 快速开始

```bash
git clone git@github.com:yiyezhiqiu2077/lumina.git
cd lumina
git switch experiment/gce

# 在已安装 CUDA 版 PyTorch 的环境中解析其余依赖。
uv sync --extra dev

export MODEL_PATH=/path/to/Lumina-DiMOO
export DATA_ROOT=/path/to/MagicBrush
export OUTPUT_ROOT=/path/to/lumina_outputs
export GCE_CLUSTER_PATH=/path/to/gce_clusters_1024_512.pt

# 检查 manifest、token 文件与 32×32 VQ 网格。
uv run bash scripts/check_dataset.sh

# 10 个 optimizer step，保存后用新进程恢复并完成 step 11--12。
uv run bash scripts/smoke_gce.sh
```

`uv sync` 会创建项目虚拟环境，因此命令使用 `uv run bash ...`。若改用 conda，请先
激活对应环境，再以 `bash ...` 执行同一脚本。

## 正式训练与恢复

```bash
# B0：CE 基线
uv run bash scripts/train_magicbrush_ce.sh

# G1：GCE
uv run bash scripts/train_magicbrush_gce.sh
```

脚本默认读取
`$DATA_ROOT/official_tokens/train/manifest.jsonl`，也可显式覆盖
`DATA_CONFIG`。默认输出为 `$OUTPUT_ROOT/B0-CE` 和 `$OUTPUT_ROOT/G1-GCE`；可用
`OUTPUT_DIR` 覆盖。恢复训练始终从新进程启动，例如：

```bash
RESUME_FROM_CHECKPOINT=/path/to/checkpoint-000500 \
uv run bash scripts/train_magicbrush_gce.sh
```

每个启动脚本都会打印实验名、commit、主机、GPU、配置及资产路径，但不会打印密钥。
正式配置和 CE/GCE 差异见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) 与：

- [configs/experiments/magicbrush_ce.yaml](configs/experiments/magicbrush_ce.yaml)
- [configs/experiments/magicbrush_gce.yaml](configs/experiments/magicbrush_gce.yaml)

## 迁移说明

- 本分支的单节点默认是 4 GPU、每卡 batch 4、累积 2，global batch 为 32。
- 多节点启动尚待与谭樾对齐；在此之前请不要自行假设 rank、共享文件系统或
  rendezvous 配置。详见 [docs/DISTRIBUTED.md](docs/DISTRIBUTED.md)。
- 路径模板见 [configs/paths.example.yaml](configs/paths.example.yaml)。
- 训练和 smoke 都会在各自输出目录写入日志和 checkpoint；不要复用不匹配的
  配置或 GCE cluster 文件恢复。

## 提交前的轻量验证

```bash
uv run bash scripts/check_dataset.sh
uv run pytest -q tests/test_gce_loss.py
bash -n scripts/*.sh
```
