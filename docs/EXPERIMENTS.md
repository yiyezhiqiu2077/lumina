# Lumina-DiMOO Mixed Editing Experiments

正式训练使用 MagicBrush train 8807 + RefEdit 7804；正式评测只使用官方 MagicBrush TEST。
MagicBrush TEST 绝不进入训练，也不使用 MagicBrush train、dev 或 mixed training manifest 做正式评测。

## 环境安装

```bash
git clone git@github.com:yiyezhiqiu2077/lumina.git
cd lumina
git checkout <FORMAL_EXPERIMENT_CODE_SHA>
uv sync --frozen --extra dev --extra upstream --extra analysis --extra eval
```

## 本地资产

模型、数据、权重与输出均保留在本机 `local_assets/` 或外部实验目录，不进入 Git。

```bash
export PROJECT_ROOT=/path/to/lumina
cd "$PROJECT_ROOT"

bash scripts/setup_local_assets.sh model /path/to/Lumina-DiMOO
bash scripts/setup_local_assets.sh mixed /path/to/mixed-token-root
bash scripts/setup_local_assets.sh magicbrush-test /path/to/MagicBrush-test
bash scripts/setup_local_assets.sh dino /path/to/dinov2-base
bash scripts/setup_local_assets.sh clip /path/to/clip-vit-large-patch14

export MODEL_PATH="$PROJECT_ROOT/local_assets/models/Lumina-DiMOO"
export DATA_ROOT="$PROJECT_ROOT/local_assets/datasets/mixed"
export DATA_CONFIG="$DATA_ROOT/train/manifest.jsonl"
export GCE_CLUSTER_PATH=/path/to/gce_clusters_1024_512.pt
export OUTPUT_ROOT=/path/to/experiments/lumina_mixed_2x3
export EVAL_OUTPUT_ROOT=/path/to/experiments/lumina_mixed_2x3_eval
export DINO_MODEL_PATH="$PROJECT_ROOT/local_assets/models/dinov2-base"
export CLIP_MODEL_PATH="$PROJECT_ROOT/local_assets/models/clip-vit-large-patch14"
mkdir -p "$OUTPUT_ROOT" "$EVAL_OUTPUT_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

当前 audited mixed manifest 为 MagicBrush 8807、RefEdit 7804、合计 16611。正式 launcher 会再次检查
这三个数、`sample_key` 唯一性和每个 token file；`519 / 5190` 则仍由 `load_train_config()` 动态计算。

## MagicBrush TEST 准备

从官方渠道手工下载并解压 test archive：

```bash
export MAGICBRUSH_TEST_ROOT=/path/to/unpacked/MagicBrush-test
export MAGICBRUSH_TEST_RAW="$PROJECT_ROOT/local_assets/datasets/magicbrush-test/canonical"
export MAGICBRUSH_TEST_GEOMETRY="$PROJECT_ROOT/local_assets/datasets/magicbrush-test/geometry.jsonl"
export MAGICBRUSH_TEST_TOKENS="$PROJECT_ROOT/local_assets/datasets/magicbrush-test/tokens"

uv run python scripts/data/prepare_magicbrush_test.py \
  --test-root "$MAGICBRUSH_TEST_ROOT" --output "$MAGICBRUSH_TEST_RAW"
uv run python scripts/eval/prepare_magicbrush_eval.py \
  --manifest "$MAGICBRUSH_TEST_RAW/manifest.jsonl" --output "$MAGICBRUSH_TEST_GEOMETRY"
uv run python scripts/data/preprocess_magicbrush.py pretokenize \
  --manifest "$MAGICBRUSH_TEST_GEOMETRY" --model "$MODEL_PATH" --output "$MAGICBRUSH_TEST_TOKENS"

export MAGICBRUSH_TEST_CANONICAL_MANIFEST="$MAGICBRUSH_TEST_RAW/manifest.jsonl"
export MAGICBRUSH_TEST_TOKEN_MANIFEST="$MAGICBRUSH_TEST_TOKENS/manifest.jsonl"
export MAGICBRUSH_TEST_SUBSET="$PROJECT_ROOT/local_assets/datasets/magicbrush-test/eval_subset.jsonl"
uv run python scripts/eval/evaluate_gt_mask_editing.py \
  --manifest "$MAGICBRUSH_TEST_TOKEN_MANIFEST" --subset "$MAGICBRUSH_TEST_SUBSET" \
  --prepare-subset-only --limit 0
```

`limit=0` 冻结所有 eligible test turns，并保留 canonical manifest 顺序。每个 multi-turn edit 都使用官方提供的
该 turn source + instruction → target，当前协议不把模型上一轮输出接入下一轮。

## 正式配置

| 项目 | 值 |
| --- | --- |
| GPU | 8 |
| Batch / GPU | 4 |
| Gradient Accumulation | 1 |
| Global Batch | 32 |
| Precision | BF16 |
| Learning Rate | 3e-6 |
| LoRA | r16 / alpha16 / dropout0.05 |
| Loss Reduction | `token_mean` |
| Epochs | 10 |

| Objective | `full_target` | `edit_region_hardlock` |
| --- | --- | --- |
| CE | `mixed_ce_8g_b4_a1.yaml` | `mixed_ce_editregion_8g_b4_a1.yaml` |
| Attention | `mixed_attention_postrope_region_8g_b4_a1.yaml` | `mixed_attention_editregion_8g_b4_a1.yaml` |
| GCE | `mixed_gce_8g_b4_a1.yaml` | `mixed_gce_editregion_8g_b4_a1.yaml` |

CE、Attention、GCE 的科学 recipe 固定在六份 YAML 中。Attention 使用 post-RoPE Q/K + `region_mass`；GCE
使用 levels 1024 / 512。`edit_region_hardlock` 只监督本轮 MASK 的 GT-region token，Attention 使用完整 GT mask。

## 资产审计与训练

`run_mixed_2x3_formal.sh --run` 会先生成 `$OUTPUT_ROOT/formal_assets.json`，记录 mixed train manifest、
Lumina/VQ tokenizer/weight SHA256、GCE cluster SHA256，以及当前 VQ codebook 的 GCE inspection。

```bash
bash scripts/train/run_mixed_2x3_formal.sh --print-command
bash scripts/train/run_mixed_2x3_formal.sh --run
```

顺序固定为 CE full → CE editregion → Attention full → Attention editregion → GCE full → GCE editregion；
任一组未通过 quality、last metric 和 final checkpoint success gate，pipeline 立即停止。

## 正式评测

正式 evaluation launcher 固定复用同一个 `MAGICBRUSH_TEST_TOKEN_MANIFEST`、`MAGICBRUSH_TEST_SUBSET`、
sampling 参数、DINO checkpoint、CLIP checkpoint、LPIPS alex 和 ROI padding 0.10。

```bash
bash scripts/eval/run_mixed_2x3_eval.sh --print-command
bash scripts/eval/run_mixed_2x3_eval.sh --run
```

它会评测六个动态解析出的 final checkpoint，并在 `$EVAL_OUTPUT_ROOT` 写入独立目录和 comparison。评测时会更新
`$OUTPUT_ROOT/formal_assets.json`：MagicBrush TEST canonical/token/subset SHA256、样本数、DINO/CLIP 的固定 model id、
local revision/config/weight SHA256、LPIPS 版本与 AlexNet checkpoint SHA256。

Token：Source-copy Token Accuracy、Edit Token Accuracy、Changed-token Accuracy。

Pixel：Inside L1 / MSE / PSNR、Inside L1 vs Target Reconstruction、Boundary L1、Full L1 / MSE / PSNR。

Perceptual / Semantic：ROI LPIPS ↓、ROI DINO-I ↑、ROI CLIP-I ↑、Full LPIPS ↓、Full DINO-I ↑、Full CLIP-I ↑。
ROI 是 GT-mask hard-lock 的主要感知/语义比较：LPIPS 在空间距离图的 mask 内均值；DINO/CLIP 使用同一 GT-mask bbox
加 10% context crop。Full 指标保留完整图像参考。所有权重均本地加载，禁止 floating 或随机 LPIPS backbone。

## 测试

```bash
uv run pytest -q
```
