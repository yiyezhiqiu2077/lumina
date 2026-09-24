# Lumina-DiMOO Mixed Editing Experiments

本仓库用于在 Lumina-DiMOO 上进行图像编辑实验，当前统一支持 MagicBrush + RefEdit、
CE / Attention / GCE 三种训练目标，以及 `full_target` / `edit_region_hardlock` 两种 target
corruption。提供数据处理、LoRA 微调、DDP、checkpoint 和 GT-mask 评测流程。

详细实验设计见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 项目结构

```text
lumina/
├── configs/
├── docs/
├── scripts/
├── src/
├── tests/
├── pyproject.toml
└── uv.lock
```

详细结构见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 环境安装

```bash
git clone git@github.com:yiyezhiqiu2077/lumina.git
cd lumina
uv sync --frozen --extra dev

uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

## 模型与数据

使用 `PROJECT_ROOT/local_assets/` 保存本机软链，不提交模型、数据、checkpoint 或实验输出。

```text
local_assets/
├── models/
│   └── Lumina-DiMOO
├── datasets/
│   ├── magicbrush/
│   ├── refedit/
│   └── mixed/
└── experiments/
```

```bash
bash scripts/setup_local_assets.sh model /path/to/Lumina-DiMOO
bash scripts/setup_local_assets.sh magicbrush /path/to/magicbrush-token-root
bash scripts/setup_local_assets.sh refedit /path/to/refedit-root
bash scripts/setup_local_assets.sh mixed /path/to/mixed-token-root
```

数据流程为 MagicBrush + RefEdit → shared preprocessing → VQ pre-tokenization → mixed manifest →
training。详细准备步骤见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 正式实验

| Objective | `full_target` | `edit_region_hardlock` |
| --- | --- | --- |
| CE | `mixed_ce_8g_b4_a1.yaml` | `mixed_ce_editregion_8g_b4_a1.yaml` |
| Attention | `mixed_attention_postrope_region_8g_b4_a1.yaml` | `mixed_attention_editregion_8g_b4_a1.yaml` |
| GCE | `mixed_gce_8g_b4_a1.yaml` | `mixed_gce_editregion_8g_b4_a1.yaml` |

```text
configs/train/formal/
├── mixed_ce_8g_b4_a1.yaml
├── mixed_attention_postrope_region_8g_b4_a1.yaml
└── mixed_gce_8g_b4_a1.yaml

configs/train/ablation/
├── mixed_ce_editregion_8g_b4_a1.yaml
├── mixed_attention_editregion_8g_b4_a1.yaml
└── mixed_gce_editregion_8g_b4_a1.yaml
```

| 项目 | 值 |
| --- | --- |
| GPU | 8 |
| Batch / GPU | 4 |
| Gradient Accumulation | 1 |
| Global Batch | 32 |
| Precision | BF16 |
| Learning Rate | 3e-6 |
| LoRA Rank | 16 |
| Loss Reduction | `token_mean` |
| Training Epochs | 10 |
| Steps / Epoch | 519 |
| Formal Steps | 5190 |

`519 / 5190` 对应当前 audited 的 16611-sample mixed manifest，不是硬编码通用值。

- CE：`L_total = L_gen + 1e-5 L_z`
- Attention：`L_total = L_gen + 1e-5 L_z + 0.3 L_attn`
- GCE：`L_total = L_gen + 1e-5 L_z + 1.0 L_gce`

Attention 使用 post-RoPE Q/K + region-mass supervision；详细定义见
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 训练

六组正式训练入口：

```bash
bash scripts/train/run_mixed_2x3_formal.sh --print-command
bash scripts/train/run_mixed_2x3_formal.sh --run
```

脚本按 CE full → CE editregion → Attention full → Attention editregion → GCE full → GCE editregion
串行执行，并在每组结束后检查 quality、metrics 和最终 checkpoint。

## 分布式训练

Single-node Multi-GPU DDP：当前正式 topology 为 `1 node × 8 GPUs`。数据加载使用
`DistributedSampler`，训练使用 PyTorch DDP。

## 评测

GT-mask hard-lock evaluator：

```text
scripts/eval/evaluate_gt_mask_editing.py
```

当前核心指标：

- Source-copy Token Accuracy
- Edit Token Accuracy
- Changed-token Accuracy
- Inside L1
- Inside PSNR
- Inside L1 vs Target Reconstruction
- Oracle hard-lock diagnostic
- ROI / Full LPIPS
- ROI / Full DINO-I
- ROI / Full CLIP-I

详细评测协议见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 测试

```bash
uv run pytest -q
```

## 文档

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)：当前 mixed 2×3 workflow
