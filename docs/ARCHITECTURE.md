# 仓库架构

本仓库采用纯 `src` layout：可复用实现位于 `src/`，可执行入口位于
`scripts/`，参数位于 `configs/`，文档位于 `docs/`，静态文件位于 `assets/`。

## 实现

- `src/dataset/`：MagicBrush sequence、稳定 corruption、几何预处理、tokenization 与审计。
- `src/models/lumina/`：Lumina 模型；`src/models/objectives/`：attention 与 GCE objective。
- `src/training/`：配置、DDP、LoRA、checkpoint 与 objective dispatch。
- `src/utils/`：token 常量、prompt 和图像/生成工具。
- `src/generators/`：T2I、I2I、hard-lock editing 和 MMU 采样实现。
- `src/xllmx/`：原始 Lumina SFT 所需的上游 package。

## 入口与配置

- `scripts/train/train.py`：唯一 mixed CE/attention/GCE 训练入口。
- `scripts/train/upstream/`：保留的原始 Lumina SFT CLI。
- `scripts/inference/`：T2I、I2I、MMU 与 DDP T2I CLI。
- `scripts/data/`：MagicBrush 和通用 pre-tokenizer CLI。
- `configs/train/`、`configs/dataset/`、`configs/distributed/`：mixed formal/validation 配置；
  `configs/upstream/`：上游 SFT 示例。

## 依赖方向

`scripts -> src`，`tests -> src`，`dataset -> utils`，`training -> dataset/models`，
`generators -> models/utils`。

`third_party/VLMEvalKit` 是完整 vendored project，保留其内部目录和依赖管理，
不作为主项目 package 或默认依赖。
