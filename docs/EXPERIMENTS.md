# Lumina-DiMOO Mixed Editing Experiments

本页给出空服务器上的正式 Mixed 2×3 实验流程。除官方 MagicBrush TEST 外，模型、训练数据、token、GCE cluster 和评测权重均由仓库脚本下载或生成；不会复制旧服务器资产。

## 环境安装

```bash
git clone git@github.com:yiyezhiqiu2077/lumina.git
cd lumina
uv sync --frozen --extra dev --extra upstream --extra analysis --extra eval --extra data
```

网络受限可显式设 `HF_ENDPOINT=https://hf-mirror.com`；repo id、不可变 revision 和最终 hash 不变。

## 空服务器目录

```bash
export PROJECT_ROOT=$PWD
export ASSET_ROOT="$PROJECT_ROOT/local_assets"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

```text
local_assets/
├── models/
├── datasets/{magicbrush,refedit,mixed,magicbrush-test}/
├── artifacts/
├── experiments/
└── logs/
```

全部 pinned public assets 位于 `configs/formal_assets.yaml`，禁止用 `main` 或短 revision。

## 下载正式模型

```bash
uv run python scripts/setup/download_formal_models.py --assets-root "$ASSET_ROOT"
```

下载器仅在完整 snapshot 且 `download_manifest.json` identity 一致时跳过；半完成目录不会被当作可用资产。

## 准备 MagicBrush Train

```bash
uv run python scripts/data/download_magicbrush_train.py --assets-root "$ASSET_ROOT"
uv run python scripts/data/download_magicbrush_train.py --assets-root "$ASSET_ROOT" --geometry-only
torchrun --nproc_per_node=8 scripts/data/preprocess_magicbrush.py pretokenize \
  --manifest "$ASSET_ROOT/datasets/magicbrush/prepared/train.jsonl" \
  --model "$ASSET_ROOT/models/Lumina-DiMOO" --output "$ASSET_ROOT/datasets/magicbrush/tokens/train"
```

脚本读取 pinned `osunlp/MagicBrush` `train` split 的官方字段，严格要求 8807 条；训练不拆 MagicBrush train/dev/val。

## 准备 RefEdit

```bash
uv run python scripts/data/download_refedit.py --assets-root "$ASSET_ROOT"
uv run python scripts/data/preprocess_refedit.py audit --raw-root "$ASSET_ROOT/datasets/refedit/raw" --output "$ASSET_ROOT/datasets/refedit/audit"
uv run python scripts/data/preprocess_refedit.py tokenize --raw-root "$ASSET_ROOT/datasets/refedit/raw" \
  --model "$ASSET_ROOT/models/Lumina-DiMOO" --output "$ASSET_ROOT/datasets/refedit/tokens/train" --seed 42 --target-size 512
```

RefEdit 固定 revision，strict parser 后必须恰为 7804 条。

## 构建 Mixed Tokens

```bash
uv run python scripts/data/build_mixed_edit_manifest.py \
  --magicbrush-manifest "$ASSET_ROOT/datasets/magicbrush/tokens/train/manifest.jsonl" \
  --refedit-manifest "$ASSET_ROOT/datasets/refedit/tokens/train/manifest.jsonl" --output "$ASSET_ROOT/datasets/mixed"
```

正式 manifest 必须为 MagicBrush 8807 + RefEdit 7804 = 16611，sample key 唯一、payload 均为 32×32。

## 构建 GCE Cluster

```bash
uv run python scripts/tools/gce/build_clusters.py --model "$ASSET_ROOT/models/Lumina-DiMOO" \
  --output "$ASSET_ROOT/artifacts/gce_clusters_1024_512.pt" --levels 1024 512 --seed 0
```

## 准备 MagicBrush TEST

TEST archive 需要从官方 MagicBrush GitHub 渠道手工下载（密码 `MagicBrush`），不能上传或再分发：

```bash
export MAGICBRUSH_TEST_ROOT=/path/to/official/unpacked/test
uv run python scripts/data/prepare_magicbrush_test.py --test-root "$MAGICBRUSH_TEST_ROOT" --output "$ASSET_ROOT/datasets/magicbrush-test/canonical"
uv run python scripts/eval/prepare_magicbrush_eval.py --manifest "$ASSET_ROOT/datasets/magicbrush-test/canonical/manifest.jsonl" --output "$ASSET_ROOT/datasets/magicbrush-test/geometry.jsonl"
torchrun --nproc_per_node=8 scripts/data/preprocess_magicbrush.py pretokenize --manifest "$ASSET_ROOT/datasets/magicbrush-test/geometry.jsonl" --model "$ASSET_ROOT/models/Lumina-DiMOO" --output "$ASSET_ROOT/datasets/magicbrush-test/tokens"
```

Importer 优先读取真实 `edit_sessions.json` 和 archive 内路径。若不是 535 sessions / 1053 turns，或存在重复、缺图、空 mask，审计明确报 `REAL TEST ARCHIVE NOT VERIFIED`。每个 turn 独立使用官方提供 source，不串接模型上一轮输出。

## 准备评测权重

```bash
uv run python scripts/setup/download_formal_models.py --assets-root "$ASSET_ROOT" --only dino --only clip
uv run python scripts/setup/prefetch_lpips.py --assets-root "$ASSET_ROOT"
```

## 资产审计

一键准备支持失败停止、阶段 `_SUCCESS` marker、验证后跳过与任意 cwd：

```bash
bash scripts/setup/run_formal_prepare.sh --print-command
bash scripts/setup/run_formal_prepare.sh --run
```

顺序为 environment、models、MagicBrush train、geometry、tokens、RefEdit、mixed、GCE、metrics、LPIPS、TEST、TEST token、final audit。`formal_assets.json` 记录 code SHA、dirty、`uv.lock`、assets、token、GCE 和 metrics 的 identity/hash。

## 六组正式训练

```bash
export MODEL_PATH="$ASSET_ROOT/models/Lumina-DiMOO"
export DATA_ROOT="$ASSET_ROOT/datasets/mixed"
export DATA_CONFIG="$DATA_ROOT/train/manifest.jsonl"
export GCE_CLUSTER_PATH="$ASSET_ROOT/artifacts/gce_clusters_1024_512.pt"
export OUTPUT_ROOT="$ASSET_ROOT/experiments/lumina_mixed_2x3"
bash scripts/train/run_mixed_2x3_formal.sh --print-command
bash scripts/train/run_mixed_2x3_formal.sh --run
```

顺序固定 CE full → CE editregion → Attention full → Attention editregion → GCE full → GCE editregion。六份 YAML 的 LR、batch、epochs、`token_mean`、loss 权重及 corruption semantics 均不由该流程改变。

## MagicBrush TEST 评测

```bash
export EVAL_OUTPUT_ROOT="$ASSET_ROOT/experiments/lumina_mixed_2x3_eval"
export MAGICBRUSH_TEST_CANONICAL_MANIFEST="$ASSET_ROOT/datasets/magicbrush-test/canonical/manifest.jsonl"
export MAGICBRUSH_TEST_TOKEN_MANIFEST="$ASSET_ROOT/datasets/magicbrush-test/tokens/manifest.jsonl"
export MAGICBRUSH_TEST_SUBSET="$ASSET_ROOT/datasets/magicbrush-test/eval_subset.jsonl"
export DINO_MODEL_PATH="$ASSET_ROOT/models/dinov2-base"
export CLIP_MODEL_PATH="$ASSET_ROOT/models/clip-vit-large-patch14"
bash scripts/eval/run_mixed_2x3_eval.sh --print-command
bash scripts/eval/run_mixed_2x3_eval.sh --run
```

Full LPIPS 为 canonical `LPIPS(alex, spatial=False)`；ROI LPIPS 为 `spatial=True` map 的 GT-mask 均值。DINO/CLIP ROI 采用同一 GT-mask bbox 加 10% context。

## 结果比较

评测输出六组 summary、per-sample JSONL 与 comparison JSON/CSV。只使用同一 1053-turn TEST、相同 seed、sampling、metric weights 与 ROI padding。

## 轻量结果打包

```bash
uv run python scripts/tools/package_formal_results.py --train-root "$OUTPUT_ROOT" --eval-root "$EVAL_OUTPUT_ROOT" --output "$ASSET_ROOT/archives/lumina_mixed_2x3_results.tar.gz"
```

仅白名单配置、provenance、quality、summary、gzip JSONL、comparison 和 manifest；checkpoint、weights、token、raw data、images、cache 均排除。超过 50 MiB 明确失败。

## 测试

```bash
uv run pytest -q
uv run python -m compileall src scripts tests
```
