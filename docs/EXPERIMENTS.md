# Objective ablation 一条命令实验指南

本轮 MagicBrush objective ablation 采用相同数据、seed、LoRA、optimizer、batch 与 schedule，比较 CE、Attention、GCE 三个 objective。正式运行顺序固定为 Attention → GCE → CE，三者均使用 GPU 0/1，不能并发。

```bash
bash scripts/train/run_tmux.sh attention
bash scripts/train/run_tmux.sh gce
bash scripts/train/run_tmux.sh ce
```

这些命令创建 detached tmux session。不需要先进入 tmux、在 tmux 内重新 export 环境变量，或手工 Ctrl+B D。

## 1. 环境与路径

```bash
export PROJECT_ROOT=/path/to/lumina
cd "$PROJECT_ROOT"
uv sync --extra dev --extra upstream --extra analysis

export ASSET_ROOT=/path/to/lumina_assets
export DATA_ROOT="$ASSET_ROOT/datasets/lumina_edit"
export MODEL_PATH="$ASSET_ROOT/models/Lumina-DiMOO"
export OUTPUT_ROOT=/path/to/experiments/lumina_objective_ablation_v1
export MAGICBRUSH_DATA_CONFIG="$DATA_ROOT/magicbrush/tokens/train/manifest.jsonl"
export DATA_CONFIG="$MAGICBRUSH_DATA_CONFIG"
export GCE_CLUSTER_PATH="$DATA_ROOT/artifacts/gce_clusters_1024_512.pt"
export CUDA_VISIBLE_DEVICES=0,1
```

启动 helper 前，在普通 shell 中一次性设置这些变量。helper 会用安全 shell quoting 将调用时刻的 `PROJECT_ROOT`、`ASSET_ROOT`、`DATA_ROOT`、`MODEL_PATH`、`OUTPUT_ROOT`、`MAGICBRUSH_DATA_CONFIG`、`DATA_CONFIG`、`GCE_CLUSTER_PATH`、`CUDA_VISIBLE_DEVICES` snapshot 到 launcher，不依赖旧 tmux server 的环境。

检查：

```bash
command -v tmux && tmux -V
test -f "$MODEL_PATH/config.json" && echo "MODEL CONFIG OK"
test -d "$MODEL_PATH/vqvae" && echo "VQVAE OK"
test -f "$DATA_CONFIG" && echo "MANIFEST OK"
test -f "$GCE_CLUSTER_PATH" && echo "GCE CLUSTER OK"
wc -l "$DATA_CONFIG" # 必须为 8807

uv run python scripts/data/check_magicbrush.py all \
  --manifest "$DATA_CONFIG" --model "$MODEL_PATH" \
  --output "$OUTPUT_ROOT/data_audit"
```

正式 launcher 不下载权重、数据或 GCE cluster。若复制 manifest 后 `token_file` 指向旧绝对路径，只修正 manifest 内该路径，不必重新 tokenize。

### 2.1 已有或下载 Lumina-DiMOO 权重

已有模型时直接设置：

```bash
export MODEL_PATH=/path/to/existing/Lumina-DiMOO
```

没有模型时：

```bash
export MODEL_PATH="$ASSET_ROOT/models/Lumina-DiMOO"
mkdir -p "$MODEL_PATH"
uv run hf download Alpha-VLLM/Lumina-DiMOO --local-dir "$MODEL_PATH"
```

网络受限时可先设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

然后重试下载。不要只获取单个主模型文件；训练、预处理和推理还需要 config、tokenizer 与 `vqvae/`。

### 2.2 MagicBrush：已有 token data 或从官方数据准备

已有 tokenized MagicBrush 时，直接设置已有 manifest：

```bash
export MAGICBRUSH_DATA_CONFIG=/path/to/magicbrush/tokens/train/manifest.jsonl
export DATA_CONFIG="$MAGICBRUSH_DATA_CONFIG"
wc -l "$DATA_CONFIG"  # 必须为 8807
```

没有 MagicBrush token data 时，按 raw → shared geometry → VQ tokenize → manifest 的顺序准备**官方完整 train（8807）**。本轮不建立 10% validation split：

```bash
mkdir -p "$DATA_ROOT/magicbrush/raw/images"
mkdir -p "$DATA_ROOT/magicbrush/prepared"
mkdir -p "$DATA_ROOT/magicbrush/tokens/train"

uv run --with "datasets>=3,<5" python - <<'PY'
import json
import os
from pathlib import Path
from datasets import load_dataset

root = Path(os.environ["DATA_ROOT"]) / "magicbrush" / "raw"
images = root / "images"
images.mkdir(parents=True, exist_ok=True)
dataset = load_dataset("osunlp/MagicBrush", split="train")
if len(dataset) != 8807:
    raise RuntimeError(f"Expected 8807 MagicBrush train samples, got {len(dataset)}")

with (root / "train.jsonl").open("w", encoding="utf-8") as handle:
    for index, row in enumerate(dataset):
        image_id, turn = str(row["img_id"]), int(row["turn_index"])
        prefix = f"{index:06d}_{image_id}_t{turn}"
        source, target, mask = (images / f"{prefix}_{name}.png" for name in ("source", "target", "mask"))
        row["source_img"].convert("RGB").save(source)
        row["target_img"].convert("RGB").save(target)
        row["mask_img"].convert("L").save(mask)
        handle.write(json.dumps({
            "index": index, "sample_key": f"magicbrush/{image_id}/{turn}",
            "session_id": image_id, "img_id": image_id, "turn_index": turn,
            "instruction": row["instruction"], "source": str(source),
            "target": str(target), "mask_edit": str(mask),
        }, ensure_ascii=False) + "\n")
PY
```

创建 deterministic shared geometry：

```bash
uv run python - <<'PY'
import os
from pathlib import Path
from dataset.geometry import enrich_geometry, resolve_record_paths
from dataset.utils import read_jsonl, write_jsonl

root = Path(os.environ["DATA_ROOT"]) / "magicbrush"
raw, prepared = root / "raw/train.jsonl", root / "prepared/train.jsonl"
rows = [resolve_record_paths(row, raw) for row in read_jsonl(raw)]
if len(rows) != 8807:
    raise RuntimeError(f"Expected 8807 raw samples, got {len(rows)}")
write_jsonl(prepared, enrich_geometry(rows, seed=42, target_size=512))
print(prepared)
PY

wc -l "$DATA_ROOT/magicbrush/prepared/train.jsonl" # 必须为 8807
```

进行 VQ tokenize：

```bash
uv run python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/data/preprocess_magicbrush.py pretokenize \
  --manifest "$DATA_ROOT/magicbrush/prepared/train.jsonl" \
  --model "$MODEL_PATH" \
  --output "$DATA_ROOT/magicbrush/tokens/train"

export MAGICBRUSH_DATA_CONFIG="$DATA_ROOT/magicbrush/tokens/train/manifest.jsonl"
export DATA_CONFIG="$MAGICBRUSH_DATA_CONFIG"
wc -l "$DATA_CONFIG" # 必须为 8807
```

### 2.3 GCE cluster

已有 cluster 时设置 `GCE_CLUSTER_PATH`。没有时只需构建一次：

```bash
mkdir -p "$DATA_ROOT/artifacts"
uv run python scripts/tools/gce/build_clusters.py \
  --model "$MODEL_PATH" \
  --output "$DATA_ROOT/artifacts/gce_clusters_1024_512.pt" \
  --device cuda:0 --levels 1024 512 --seed 0
export GCE_CLUSTER_PATH="$DATA_ROOT/artifacts/gce_clusters_1024_512.pt"
```

## 2. 固定 matched 配置

| 项目 | 值 |
| --- | --- |
| configs | `configs/train/ablation/mb_{ce,attention,gce}_2g_b8_a2.yaml` |
| GPU / batch / accumulation / global batch | 2 / 8 / 2 / 32 |
| seed / precision / max sequence | 42 / BF16 / 5120 |
| optimizer | AdamW，lr=1e-5，warmup=20，clip=4.0 |
| LoRA | rank=16，alpha=16，dropout=0.05 |
| MagicBrush / epochs / steps per epoch | 8807 / 10 / 275 |
| total steps / checkpoint interval | 2750 / 275 |

`DistributedSampler(drop_last=True)` 和 `DataLoader(drop_last=True)` 下每 rank 每 epoch 是 550 micro-batches；accum=2，所以为 275 optimizer steps。checkpoint steps：275、550、825、1100、1375、1650、1925、2200、2475、2750。tmux helper 不改这三份 YAML。

## 3. Smoke：前台运行

从正式 YAML 复制一个 `/tmp` smoke config，只改输出名、`training.max_optimizer_steps=20` 和 `training.checkpoint_every_steps=20`；不要提交 smoke YAML。

```bash
uv run python - <<'PY'
from pathlib import Path
import yaml

src = Path("configs/train/ablation/mb_attention_2g_b8_a2.yaml")
dst = Path("/tmp/mb_attention_smoke.yaml")
cfg = yaml.safe_load(src.read_text())
cfg["experiment_name"] = cfg["output_name"] = "SMOKE-MB-ATTN-2G-B8-A2-S42"
cfg["training"]["max_optimizer_steps"] = 20
cfg["training"]["checkpoint_every_steps"] = 20
dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(dst)
PY
uv run python scripts/train/train.py --config /tmp/mb_attention_smoke.yaml
```

GCE smoke 按相同方式使用 `mb_gce_2g_b8_a2.yaml`、`SMOKE-MB-GCE-2G-B8-A2-S42`、`/tmp/mb_gce_smoke.yaml`。两个 smoke 通过后再开始正式训练。

## 4. 正式 tmux 训练

```bash
# Fresh Attention
bash scripts/train/run_tmux.sh attention
# Attention 的 exit code 为 0 后：
bash scripts/train/run_tmux.sh gce
# Attention 和 GCE 的 exit code 均为 0 后：
bash scripts/train/run_tmux.sh ce
```

fresh run 发现对应 objective output 下已有 `train_metrics.jsonl`、`checkpoint-*`、`experiment_config.json` 或 `lora_report.json` 时会拒绝启动，避免污染旧实验。请换用新的 `OUTPUT_ROOT`；若确实要继续同一 run，只能显式 resume：

```bash
# Resume Attention
bash scripts/train/run_tmux.sh attention \
  --resume-from-checkpoint \
  "$OUTPUT_ROOT/MB-ATTN-2G-B8-A2-S42/checkpoint-001375"

# Resume GCE
bash scripts/train/run_tmux.sh gce \
  --resume-from-checkpoint \
  "$OUTPUT_ROOT/MB-GCE-2G-B8-A2-S42/checkpoint-001375"

# Resume CE
bash scripts/train/run_tmux.sh ce \
  --resume-from-checkpoint \
  "$OUTPUT_ROOT/MB-CE-2G-B8-A2-S42/checkpoint-001375"
```

resume checkpoint 必须存在，且必须位于该 objective 的正确实验目录中。resume 同样执行 session / GPU 冲突检查。

| Objective | tmux session | Config | Log | Exit code |
| --- | --- | --- | --- | --- |
| attention | `lumina_attn` | `mb_attention_2g_b8_a2.yaml` | `$OUTPUT_ROOT/logs/attention.log` | `$OUTPUT_ROOT/logs/attention.exit_code` |
| gce | `lumina_gce` | `mb_gce_2g_b8_a2.yaml` | `$OUTPUT_ROOT/logs/gce.log` | `$OUTPUT_ROOT/logs/gce.exit_code` |
| ce | `lumina_ce` | `mb_ce_2g_b8_a2.yaml` | `$OUTPUT_ROOT/logs/ce.log` | `$OUTPUT_ROOT/logs/ce.exit_code` |

状态、日志和可选 attach：

```bash
bash scripts/train/run_tmux.sh status
bash scripts/train/run_tmux.sh logs attention
bash scripts/train/run_tmux.sh logs gce
bash scripts/train/run_tmux.sh logs ce

tmux attach -t lumina_attn
tmux attach -t lumina_gce
tmux attach -t lumina_ce
```

启动前会检查 tmux、环境变量、模型 `config.json`/`vqvae`、可创建 `OUTPUT_ROOT`、存在且为 8807 行的 manifest、目标 YAML、`CUDA_VISIBLE_DEVICES`；GCE 还检查 `GCE_CLUSTER_PATH`。同一 session 存在时会报 `session already exists`；任一另一个 Lumina session 存在时会报 `another Lumina training session is already active`。不会创建 `-2` 等替代 session。

每次启动写入 `$OUTPUT_ROOT/logs/<objective>.command.sh`（owner-only）。launcher 显式 export snapshot，写入 date、hostname、pwd、Git SHA、GPU、资产路径和 config；stdout/stderr 写入 objective log。它用 `set -o pipefail` 和 `${PIPESTATUS[0]}` 记录真实 Python/torchrun exit code，而不是 `tee` 的返回值。

## 5. 曲线与比较

```bash
uv run python scripts/eval/plot_training_curves.py \
  --input "$OUTPUT_ROOT/MB-ATTN-2G-B8-A2-S42/train_metrics.jsonl" \
  --objective attention --steps-per-epoch 275 --smooth-window 25 \
  --output "$OUTPUT_ROOT/MB-ATTN-2G-B8-A2-S42/curves"

uv run python scripts/eval/plot_training_curves.py \
  --input "$OUTPUT_ROOT/MB-GCE-2G-B8-A2-S42/train_metrics.jsonl" \
  --objective gce --steps-per-epoch 275 --smooth-window 25 \
  --output "$OUTPUT_ROOT/MB-GCE-2G-B8-A2-S42/curves"

uv run python scripts/eval/plot_training_curves.py \
  --input "$OUTPUT_ROOT/MB-CE-2G-B8-A2-S42/train_metrics.jsonl" \
  --objective ce --steps-per-epoch 275 --smooth-window 25 \
  --output "$OUTPUT_ROOT/MB-CE-2G-B8-A2-S42/curves"

uv run python scripts/eval/plot_training_curves.py compare \
  --ce "$OUTPUT_ROOT/MB-CE-2G-B8-A2-S42/train_metrics.jsonl" \
  --attention "$OUTPUT_ROOT/MB-ATTN-2G-B8-A2-S42/train_metrics.jsonl" \
  --gce "$OUTPUT_ROOT/MB-GCE-2G-B8-A2-S42/train_metrics.jsonl" \
  --steps-per-epoch 275 --smooth-window 25 --output "$OUTPUT_ROOT/comparison"
```

横向只比较三者 `L_gen`，不比较 `L_total`。Attention 曲线含 raw/weighted attention、ratio、localization、entropy 和 layers 24–27；GCE 含 GCE 与各 cluster level。

## 6. 打包轻量结果

```bash
git rev-parse HEAD > "$OUTPUT_ROOT/git_commit.txt"
git status --short > "$OUTPUT_ROOT/git_status.txt"
mkdir -p "$OUTPUT_ROOT/configs_used"
cp configs/train/ablation/mb_{ce,attention,gce}_2g_b8_a2.yaml "$OUTPUT_ROOT/configs_used/"
wc -l "$MAGICBRUSH_DATA_CONFIG" > "$OUTPUT_ROOT/dataset_count.txt"
readlink -f "$MAGICBRUSH_DATA_CONFIG" > "$OUTPUT_ROOT/dataset_manifest_path.txt"
readlink -f "$MODEL_PATH" > "$OUTPUT_ROOT/model_path.txt"
cd "$(dirname "$OUTPUT_ROOT")"
tar --exclude='checkpoint*' --exclude='*.pt' --exclude='*.pth' --exclude='*.safetensors' \
  -czf lumina_objective_ablation_v1_results.tar.gz "$(basename "$OUTPUT_ROOT")"
```

轻量包包含 configs、metrics、曲线、比较图、Git 信息、dataset/model path 与 logs/exit codes；不包含 checkpoint、权重或数据资产。
