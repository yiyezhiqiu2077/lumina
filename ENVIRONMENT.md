# 环境

使用 Python 3.10 和 BF16。先按目标集群 CUDA/驱动策略安装 PyTorch CUDA wheel，再执行：

```bash
uv sync --extra dev
```

已验证的源环境为 PyTorch 2.3.1 + CUDA 12.1、Transformers 4.46.2、Diffusers 0.34.0
与 4×A800 80GB。迁移到 H100 时首轮保持 BF16；不要自动启用 FP8。`flash-attn` 不是
前置依赖，基础 smoke 通过前不要额外安装。
