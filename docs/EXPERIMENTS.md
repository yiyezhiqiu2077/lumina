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

当前服务器已有资产可直接使用：

```bash
export PROJECT_ROOT=/data02/zhangyuyang/github_push_stage.FshNx1/lumina
export ASSET_ROOT=/data02/zhangyuyang
export DATA_ROOT=/data02/zhangyuyang/experiments/lumina_dimoo_magicbrush_a0_a1_v2_20260909/dataset
export MODEL_PATH=/data02/zhangyuyang/Lumina-DiMOO/models/Lumina-DiMOO
export OUTPUT_ROOT=/data02/zhangyuyang/experiments/lumina_objective_ablation_v1
export MAGICBRUSH_DATA_CONFIG="$DATA_ROOT/official_tokens/train/manifest.jsonl"
export DATA_CONFIG="$MAGICBRUSH_DATA_CONFIG"
export GCE_CLUSTER_PATH=/data02/zhangyuyang/experiments/lumina_dimoo_gce_assets_20260912/gce_clusters_1024_512.pt
export CUDA_VISIBLE_DEVICES=0,1
cd "$PROJECT_ROOT"
```

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

helper 不下载权重、数据或 GCE cluster。若复制 manifest 后 `token_file` 指向旧绝对路径，只修正 manifest 内该路径，不必重新 tokenize。

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
bash scripts/train/run_tmux.sh attention
# Attention 的 exit code 为 0 后：
bash scripts/train/run_tmux.sh gce
# Attention 和 GCE 的 exit code 均为 0 后：
bash scripts/train/run_tmux.sh ce
```

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
