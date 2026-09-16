# Lumina-DiMOO：统一 MagicBrush 训练

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

请先按 [ENVIRONMENT.md](ENVIRONMENT.md) 安装 CUDA 对应的 PyTorch。数据根目录应包含
`official_tokens/train/manifest.jsonl`；模型权重、MagicBrush 数据、GCE cluster、日志和
checkpoint 均不在 Git 中。

## 三个启动入口

```bash
# B0：与 GCE 设置匹配的 CE baseline（4 GPU × batch 4 × accum 2）
uv run bash scripts/train_magicbrush_ce.sh

# A1：attention supervision（2 GPU × batch 8 × accum 4）
uv run bash scripts/train_magicbrush_attention.sh

# G1：GCE（4 GPU × batch 4 × accum 2）
export GCE_CLUSTER_PATH=/path/to/gce_clusters_1024_512.pt
uv run bash scripts/train_magicbrush_gce.sh
```

三个 launcher 都调用 [train/train_magicbrush.py](train/train_magicbrush.py)，差异只在
`--objective {ce,attention,gce}` 与各自真实超参数。CE/GCE 使用 global batch 32；attention
使用 global batch 64，因此 CE 不应被表述为与 attention 的严格 matched baseline。

`DATA_CONFIG` 可覆盖默认 manifest，`OUTPUT_DIR` 可覆盖默认输出目录，
`RESUME_FROM_CHECKPOINT` 可从新进程恢复。例如：

```bash
RESUME_FROM_CHECKPOINT=/path/to/checkpoint-000500 \
uv run bash scripts/train_magicbrush_gce.sh
```

## 配置与验证

- [configs/experiments/magicbrush_ce.yaml](configs/experiments/magicbrush_ce.yaml)
- [configs/experiments/magicbrush_attention.yaml](configs/experiments/magicbrush_attention.yaml)
- [configs/experiments/magicbrush_gce.yaml](configs/experiments/magicbrush_gce.yaml)

```bash
uv run pytest -q
bash -n scripts/*.sh
```

多节点启动、FP8、FSDP 和 FlashAttention 性能优化不属于当前 main 的承诺范围。
