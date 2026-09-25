# 环境

使用 Python 3.10 和 BF16。安装路径是让 `uv` 从项目的 PyTorch CUDA 12.1
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
