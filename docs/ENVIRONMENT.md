# 环境

使用 Python 3.10 和 BF16。唯一验证过的安装路径是让 `uv` 从项目的 PyTorch CUDA 12.1
explicit index 安装锁定依赖：

```bash
uv sync --extra dev
```

已验证的组合为 PyTorch 2.3.1 + CUDA runtime 12.1、Transformers 4.46.2、Diffusers
0.34.0 与 4×A800 80GB。完成同步后请先确认：

```bash
uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

迁移到 H100 时首轮保持 BF16；不要自动启用 FP8。`flash-attn` 不是前置依赖，基础 smoke
通过前不要额外安装。
