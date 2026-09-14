# Lumina-DiMOO：MagicBrush attention-supervision 实验

本分支是可迁移的 MagicBrush A1 attention-supervision 训练包。共同实验基线为
`1475c9739cd6f5e2dacabbd82f907e71504e7109`：保留该基线中的 source-only attention
supervision，并加入后续的 batch padding attention 修复。

本分支不包含 GCE、full-attention supervision 或 ML-Cache 代码。

## 实验定义

训练目标为：

```text
L = L_gen + 0.1 * L_attn
```

`L_gen` 保持官方 masked target-token CE。`L_attn` 保持既有定义：在 layers 24--27
中使用真实 instruction-token Q、clean source-image spatial K 和 GT edit mask 的
source-only normalized CE。未修改 attention loss 数学形式、LoRA target 或模型权重
初始化。

padding 修复使用 `config.pad_token_id`，并把变长 batch 的 `attention_bias` 真实传入
SDPA，以阻止真实 token 访问 padding K/V；正常非 padding token 的 attention 行为不变。

## 环境与前置资产

推荐 Python 3.10、BF16 和 `uv`。目标集群上先安装与 CUDA/驱动匹配的 PyTorch wheel，
再执行 `uv sync --extra dev`。完整依赖、已验证环境和 H100 使用建议见
[ENVIRONMENT.md](ENVIRONMENT.md)；首轮迁移不要启用 FP8。

以下资产不在 Git 中，需在目标集群自行准备：

- `MODEL_PATH`：Lumina-DiMOO 预训练模型目录。
- `DATA_ROOT`：MagicBrush 已处理数据根目录。
- `OUTPUT_ROOT`：本次运行的输出根目录。

目录约定、manifest、512→32×32 VQ token 和 mask 处理细节见
[docs/DATASET.md](docs/DATASET.md)。资产来源与校验说明见 [ASSETS.md](ASSETS.md)。
请勿将数据、权重、checkpoint、日志、cache 或密钥提交到 Git。

## 快速开始

```bash
git clone git@github.com:yiyezhiqiu2077/lumina.git
cd lumina
git switch experiment/attention

# 在已安装 CUDA 版 PyTorch 的环境中解析其余依赖。
uv sync --extra dev

export MODEL_PATH=/path/to/Lumina-DiMOO
export DATA_ROOT=/path/to/MagicBrush
export OUTPUT_ROOT=/path/to/lumina_outputs

# 检查 manifest、token 文件与 32×32 VQ 网格。
uv run bash scripts/check_dataset.sh

# 10 个 optimizer step，保存后用新进程恢复并完成 step 11--12。
uv run bash scripts/smoke_attention_loss.sh
```

`uv sync` 会创建项目虚拟环境，因此命令使用 `uv run bash ...`。若改用 conda，请先
激活对应环境，再以 `bash ...` 执行同一脚本。

## 正式训练与恢复

默认训练设置为 2 GPU、每卡 batch 8、gradient accumulation 4，global batch 为 64。

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
uv run bash scripts/train_magicbrush_attention_loss.sh
```

脚本默认读取 `$DATA_ROOT/official_tokens/train/manifest.jsonl`，并输出到
`$OUTPUT_ROOT/A1-ATTN`。可使用 `DATA_CONFIG`、`OUTPUT_DIR`、`NPROC_PER_NODE` 和
`CUDA_VISIBLE_DEVICES` 显式覆盖。恢复训练总是从新进程启动：

```bash
RESUME_FROM_CHECKPOINT=/path/to/checkpoint-000275 \
uv run bash scripts/train_magicbrush_attention_loss.sh
```

启动时会打印实验名、commit、主机、GPU、配置及资产路径，但不会打印密钥。完整超参数
和日志定义见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) 与
[configs/experiments/magicbrush_attention.yaml](configs/experiments/magicbrush_attention.yaml)。

## 迁移说明

- attention trainer 当前保留数据集的 `max_seq_len=5120` 默认值；审计过的样本序列约为
  2.1k token。
- 多节点启动尚待与谭樾对齐；在此之前请不要自行假设 rank、共享文件系统或 rendezvous
  配置。详见 [docs/DISTRIBUTED.md](docs/DISTRIBUTED.md)。
- 路径模板见 [configs/paths.example.yaml](configs/paths.example.yaml)。
- 不要用 lambda=0.3/1.0、full-attention 或 ML-Cache 实验的 checkpoint 恢复本分支。

## 提交前的轻量验证

```bash
uv run bash scripts/check_dataset.sh
uv run pytest -q tests/test_attention_loss.py tests/test_checkpoint_auxiliary.py tests/test_lora.py
bash -n scripts/*.sh
```
