# Objective ablation 一条命令实验指南

本轮 MagicBrush objective ablation 采用相同数据、seed、LoRA、optimizer、batch 与 schedule，比较 CE、Attention、GCE 三个 objective。正式运行顺序固定为 Attention → GCE → CE，三者均使用 GPU 0–7，不能并发。

```bash
bash scripts/train/run_tmux.sh all

# 也可手工串行：
# bash scripts/train/run_tmux.sh attention
# bash scripts/train/run_tmux.sh gce
# bash scripts/train/run_tmux.sh ce
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
export EXPERIMENT_ROOT=/path/to/experiments/lumina
export OUTPUT_ROOT="$EXPERIMENT_ROOT/formal/lumina_objective_ablation_8g_v3_lr3e6_20260918"
export MAGICBRUSH_DATA_CONFIG="$DATA_ROOT/magicbrush/tokens/train/manifest.jsonl"
export DATA_CONFIG="$MAGICBRUSH_DATA_CONFIG"
export GCE_CLUSTER_PATH="$DATA_ROOT/artifacts/gce_clusters_1024_512.pt"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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

推荐目录组织如下：`$EXPERIMENT_ROOT/formal/` 保存正式 Attention → GCE → CE 串行结果，`$EXPERIMENT_ROOT/probes/` 保存长探针，`$EXPERIMENT_ROOT/smokes/` 保存 Smoke，`$EXPERIMENT_ROOT/archives/` 保存轻量打包结果。`OUTPUT_ROOT` 必须始终是其中某个单独 run 的根目录，不能直接设为 `$EXPERIMENT_ROOT`。

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
| configs | `configs/train/ablation/mb_{ce,attention,gce}_8g_b4_a1.yaml` |
| GPU / batch / accumulation / global batch | 8 / 4 / 1 / 32 |
| seed / precision / max sequence | 42 / BF16 / 5120 |
| optimizer | AdamW，lr=3e-6，betas=(0.9, 0.95)，warmup=20，clip=4.0 |
| generation loss | sample-mean CE + `1e-5 × z-loss`（supervised target positions，FP32） |
| LoRA | rank=16，alpha=16，dropout=0.05；q/k/v/out projection |
| quality gate | early baseline、50-step disjoint windows、loss/logit/update/clipping checks |
| MagicBrush / epochs / steps per epoch | 8807 / 10 / 275 |
| total steps / checkpoint interval | 2750 / 275 |

`DistributedSampler(drop_last=True)` 下每 rank 每 epoch 是 1100 个样本（全局丢弃 7 个）；`DataLoader(batch=4, drop_last=True)` 产生 275 micro-batches。accum=1，所以仍为 275 optimizer steps。checkpoint steps：275、550、825、1100、1375、1650、1925、2200、2475、2750。全局 batch 仍为 32，三组实验的训练量与 schedule 保持 matched。

## 3. Smoke 与长稳定性验证

短 smoke 串行执行 Attention → GCE → CE，各跑 20 step，用于验证启动、objective 隔离、checkpoint 和清理；它不代表长期稳定：

```bash
bash scripts/train/run_smokes_tmux.sh start
bash scripts/train/run_smokes_tmux.sh status
bash scripts/train/run_smokes_tmux.sh logs
```

在正式实验前，必须让 CE、Attention 和 GCE 分别从 base model、各自全新的 `OUTPUT_ROOT` 运行到 step 1650，跨过已知的 step-1450 延迟失稳区：

```bash
bash scripts/train/run_probe_tmux.sh attention 1650
bash scripts/train/run_probe_tmux.sh status
bash scripts/train/run_probe_tmux.sh logs

# Attention 完成并释放八卡后，在新的 OUTPUT_ROOT 运行：
bash scripts/train/run_probe_tmux.sh gce 1650

# GCE 完成并释放八卡后，再在新的 OUTPUT_ROOT 运行：
bash scripts/train/run_probe_tmux.sh ce 1650
```

`5e-6` 虽曾通过 825-step probes 和 20-step Smoke，却在 fresh 正式 Attention 的 step 1450 发生延迟 generation-loss 发散，因此不能作为正式 recipe。只将 peak LR 单变量降低到 `3e-6` 后，Attention、GCE、CE 三个 matched 1650-step 长探针均达到 `exit_code=0`、`quality_status.json=SUCCEEDED`、metrics 1–1650 连续、六个 checkpoint 完整，且 step 1301–1650 的 generation-loss 窗口无持续回归。因此正式 recipe 固定使用经验证的 `3e-6` peak LR。

长探针的 `max_optimizer_steps=1650` 只限定实际停止步数；生成配置显式保留 `scheduler_horizon_steps=2750`，所以其前 1650 步 LR 与正式 recipe 完全一致。探针 fingerprint 同时记录停止步数和 scheduler horizon。探针不得从旧版或已判为失败的 checkpoint resume，也不得把探针 checkpoint 用作正式 2750-step run 的恢复点。短 Smoke 只验证启动和 objective 隔离，不能替代长稳定性验证。

## 4. 正式 tmux 训练

```bash
# Fresh Attention
bash scripts/train/run_tmux.sh attention
# Attention 的 exit code 为 0 后：
bash scripts/train/run_tmux.sh gce
# Attention 和 GCE 的 exit code 均为 0 后：
bash scripts/train/run_tmux.sh ce
```

fresh run 发现对应 objective output、quality 文件、launcher/log 或 checkpoint 已存在时会拒绝启动，避免污染旧实验。每个诊断 cohort 与正式重跑都应使用新的 `OUTPUT_ROOT`；若确实要继续同一 run，只能显式 resume：

```bash
# Resume Attention
bash scripts/train/run_tmux.sh attention \
  --resume-from-checkpoint \
  "$OUTPUT_ROOT/MB-ATTN-8G-B4-A1-S42/checkpoint-001375"

# Resume GCE
bash scripts/train/run_tmux.sh gce \
  --resume-from-checkpoint \
  "$OUTPUT_ROOT/MB-GCE-8G-B4-A1-S42/checkpoint-001375"

# Resume CE
bash scripts/train/run_tmux.sh ce \
  --resume-from-checkpoint \
  "$OUTPUT_ROOT/MB-CE-8G-B4-A1-S42/checkpoint-001375"
```

resume checkpoint 必须存在且位于该 objective 的正确实验目录中，并具有 `_SUCCESS`、`checkpoint_meta.json`、LoRA 权重和全部 8 个 rank state。trainer 还会核验 run fingerprint 与 `train_metrics.jsonl` 的 1..step 连续性；任何 loss semantics、schedule、模型、数据或代码差异都会拒绝恢复。resume 同样执行 session / GPU 冲突检查。

| Objective | tmux session | Config | Log | Exit code | Quality |
| --- | --- | --- | --- | --- | --- |
| attention | `lumina_attn` | `mb_attention_8g_b4_a1.yaml` | `$OUTPUT_ROOT/logs/attention.log` | `$OUTPUT_ROOT/logs/attention.exit_code` | `$OUTPUT_ROOT/MB-ATTN-8G-B4-A1-S42/quality_status.json` |
| gce | `lumina_gce` | `mb_gce_8g_b4_a1.yaml` | `$OUTPUT_ROOT/logs/gce.log` | `$OUTPUT_ROOT/logs/gce.exit_code` | `$OUTPUT_ROOT/MB-GCE-8G-B4-A1-S42/quality_status.json` |
| ce | `lumina_ce` | `mb_ce_8g_b4_a1.yaml` | `$OUTPUT_ROOT/logs/ce.log` | `$OUTPUT_ROOT/logs/ce.exit_code` | `$OUTPUT_ROOT/MB-CE-8G-B4-A1-S42/quality_status.json` |

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

每次启动写入 `$OUTPUT_ROOT/logs/<objective>.command.sh`（owner-only）。launcher 显式 export snapshot，写入 date、hostname、pwd、Git SHA、GPU、资产路径和 config；stdout/stderr 写入 objective log。它用 `set -o pipefail` 和 `${PIPESTATUS[0]}` 记录真实 Python/torchrun exit code，而不是 `tee` 的返回值。串行队列只有在该 objective 的进程退出码为 0 且 quality 状态为 `SUCCEEDED` 时才继续；`QUALITY_FAILED` 或 `PROCESS_FAILED` 都会停止队列。

## 5. 曲线与比较

```bash
uv run python scripts/eval/plot_training_curves.py \
  --input "$OUTPUT_ROOT/MB-ATTN-8G-B4-A1-S42/train_metrics.jsonl" \
  --objective attention --steps-per-epoch 275 --smooth-window 25 \
  --output "$OUTPUT_ROOT/MB-ATTN-8G-B4-A1-S42/curves"

uv run python scripts/eval/plot_training_curves.py \
  --input "$OUTPUT_ROOT/MB-GCE-8G-B4-A1-S42/train_metrics.jsonl" \
  --objective gce --steps-per-epoch 275 --smooth-window 25 \
  --output "$OUTPUT_ROOT/MB-GCE-8G-B4-A1-S42/curves"

uv run python scripts/eval/plot_training_curves.py \
  --input "$OUTPUT_ROOT/MB-CE-8G-B4-A1-S42/train_metrics.jsonl" \
  --objective ce --steps-per-epoch 275 --smooth-window 25 \
  --output "$OUTPUT_ROOT/MB-CE-8G-B4-A1-S42/curves"

uv run python scripts/eval/plot_training_curves.py compare \
  --ce "$OUTPUT_ROOT/MB-CE-8G-B4-A1-S42/train_metrics.jsonl" \
  --attention "$OUTPUT_ROOT/MB-ATTN-8G-B4-A1-S42/train_metrics.jsonl" \
  --gce "$OUTPUT_ROOT/MB-GCE-8G-B4-A1-S42/train_metrics.jsonl" \
  --steps-per-epoch 275 --smooth-window 25 --output "$OUTPUT_ROOT/comparison"
```

横向只比较三者 `L_gen`，不比较 `L_total`。Attention 曲线含 raw/weighted attention、ratio、localization、entropy 和 layers 24–27；GCE 含 GCE 与各 cluster level。

## 6. 打包轻量结果

```bash
git rev-parse HEAD > "$OUTPUT_ROOT/git_commit.txt"
git status --short > "$OUTPUT_ROOT/git_status.txt"
mkdir -p "$OUTPUT_ROOT/configs_used" "$EXPERIMENT_ROOT/archives"
cp configs/train/ablation/mb_{ce,attention,gce}_8g_b4_a1.yaml "$OUTPUT_ROOT/configs_used/"
wc -l "$MAGICBRUSH_DATA_CONFIG" > "$OUTPUT_ROOT/dataset_count.txt"
readlink -f "$MAGICBRUSH_DATA_CONFIG" > "$OUTPUT_ROOT/dataset_manifest_path.txt"
readlink -f "$MODEL_PATH" > "$OUTPUT_ROOT/model_path.txt"
cd "$(dirname "$OUTPUT_ROOT")"
tar --exclude='checkpoint*' --exclude='*.pt' --exclude='*.pth' --exclude='*.safetensors' \
  -czf "$EXPERIMENT_ROOT/archives/$(basename "$OUTPUT_ROOT")_results.tar.gz" "$(basename "$OUTPUT_ROOT")"
```

轻量包包含 configs、metrics、曲线、比较图、Git 信息、dataset/model path 与 logs/exit codes；不包含 checkpoint、权重或数据资产。
