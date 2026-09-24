# Mixed editing 2×3 ablation 一条命令实验指南

本指南适用于当前 MagicBrush + RefEdit mixed editing 工作流：2 个 target corruption × CE、Attention、GCE
3 个 objective。当前 4-GPU 服务器只运行 tests、20-step correctness smoke、必要的 3–5 step diagnostic 与
evaluator correctness；六组 8-GPU formal experiment 留给新集群。历史 MagicBrush-only v3（sample_mean、
2750-step）仍可通过 `configs/train/ablation/mb_*.yaml` 和旧 helpers 追溯，但不是本指南的当前 recipe。

## 1. 环境与路径

```bash
export PROJECT_ROOT=/path/to/lumina
cd "$PROJECT_ROOT"
uv sync --extra dev --extra upstream --extra analysis

bash scripts/setup_local_assets.sh model /path/to/Lumina-DiMOO
bash scripts/setup_local_assets.sh magicbrush /path/to/magicbrush-token-root
bash scripts/setup_local_assets.sh refedit /path/to/refedit-final-mask
bash scripts/setup_local_assets.sh mixed /path/to/mixed-token-root

export MODEL_PATH="$PROJECT_ROOT/local_assets/models/Lumina-DiMOO"
export DATA_ROOT="$PROJECT_ROOT/local_assets/datasets/mixed"
export DATA_CONFIG="$DATA_ROOT/manifest.jsonl"
export OUTPUT_ROOT="$PROJECT_ROOT/local_assets/experiments/mixed_2x3"
export GCE_CLUSTER_PATH=/path/to/gce_clusters_1024_512.pt
```

`local_assets/` 是机器本地软链目录，不进入 Git。不要把真实服务器路径、用户名或 staging 路径写入配置。
mixed manifest 的 `token_file` 相对 manifest.parent 解析；数据 revision、tokenization 和审计结果以数据目录
中的 `dataset_meta.json` 与 mixed metadata 为准。

## 2. 固定 matched 配置

| 项目 | 值 |
| --- | --- |
| GPU / batch / accumulation / global batch | 8 / 4 / 1 / 32 |
| seed / precision / max sequence | 42 / BF16 / 5120 |
| optimizer | AdamW，lr=3e-6，betas=(0.9, 0.95)，wd=0.1，clip=4.0，warmup=20 |
| generation reduction | global `token_mean`，`z-loss=1e-5` |
| LoRA | r=16，alpha=16，dropout=0.05；q/k/v/attn_out |
| epochs / steps | 10 mixed epochs；由真实 manifest、sampler、loader 自动推导 |

当前 audited manifest 是 MagicBrush 8807 + RefEdit 7804 = 16611。按 8×4×1 与
`DistributedSampler(drop_last=True)` / `DataLoader(drop_last=True)`，当前为 519 step/epoch 与 5190
total steps；这不是可移植的硬编码值。

## 3. 六组配置与 objective

| objective | `full_target` | `edit_region_hardlock` |
| --- | --- | --- |
| CE | `configs/train/formal/mixed_ce_8g_b4_a1.yaml` | `configs/train/ablation/mixed_ce_editregion_8g_b4_a1.yaml` |
| Attention | `configs/train/formal/mixed_attention_postrope_region_8g_b4_a1.yaml` | `configs/train/ablation/mixed_attention_editregion_8g_b4_a1.yaml` |
| GCE | `configs/train/formal/mixed_gce_8g_b4_a1.yaml` | `configs/train/ablation/mixed_gce_editregion_8g_b4_a1.yaml` |

CE 是 `L_gen + 1e-5 × L_z`；Attention 是 `L_gen + 1e-5 × L_z + 0.3 × L_attn`；GCE 是
`L_gen + 1e-5 × L_z + 1.0 × L_gce`，levels 为 1024 / 512。paired comparison 是同 objective 的 full
↔ editregion；也可在 editregion 内比较 CE ↔ Attention ↔ GCE。所有正式 run 均从同一 original Lumina
base 与 fresh LoRA 开始，不得 resume smoke、旧 v3、另一个 objective 或另一个 corruption mode。

Attention 使用 real instruction Q、clean source spatial K、post-RoPE Q/K、layers 24–27。softmax 覆盖全部
source spatial positions，完整 GT mask `M` 只用于 `P_G = sum_{j in M} p_j` 与
`L_region = -log(P_G.clamp_min(eps))`；它不是 hard attention mask，也不只在 mask 内 softmax。

## 4. Target corruption 与 token weighting

`full_target` 保持原 whole-target random masked corruption。`edit_region_hardlock` 为：

```text
x_j = MASK       j ∈ S ⊆ M
      target_j   j ∈ M \ S
      source_j   j ∉ M
```

`M` 是完整 GT edit token region，`S` 是本轮 cosine corruption 选择的 masked subset；generation labels
只监督 `S`。Attention loss 始终使用完整 `M`，不是 `S`。

`token_mean` 下 supervised token 更多的 sample 自然有更大 generation-gradient 贡献。因此 MagicBrush /
RefEdit 的 sample proportion 不等于 supervised-token proportion，尤其 mask area 不同时。此性质不应通过
sample mean、dataset reweight、mask reweight 或 sampler 改写。日志为 overall / magicbrush / refedit 记录
`sample_count`、`mean_edit_token_count`、`mean_edit_fraction`、`mean_valid_target_count`、
`valid_target_token_sum` 与 `supervised_token_share`；两个数据集的 share 应约为 1。

## 5. 4-GPU correctness workflow

先只打印命令，确认环境与 config routing：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/train/run_mixed_smokes.sh full --print-command
bash scripts/train/run_mixed_smokes.sh editregion --print-command
bash scripts/train/run_mixed_smokes.sh all --print-command
```

默认旧调用 `bash scripts/train/run_mixed_smokes.sh --run` 仍代表 full-target。`editregion --run` 串行 CE →
Attention → GCE，并要求新的 `OUTPUT_ROOT`。当前已完成 20-step full-target 与 editregion correctness；若
corruption 单测未被改坏，不要无目的重跑它们。

本轮可额外运行 fresh 4-GPU、batch/GPU=4、accum=1、3-step 的 Attention/GCE diagnostic，
`gradient_decomposition_every_steps=1`。它只验证 finite、NCCL、无 deadlock 与 diagnostic：Attention 每步
`weighted_auxiliary_gradient_norm ≈ 0.3 × raw_auxiliary_gradient_norm`，GCE 每步相等；不据此做效果结论。

## 6. 8-GPU formal workflow

以下命令只为新集群准备；不要在当前 4-GPU 机器运行。formal launcher 可从任意 cwd 接受 repo-relative 或
absolute config path，`--run` 会拒绝 dirty Git worktree。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /tmp
/path/to/lumina/scripts/train/run_mixed_formal.sh \
  configs/train/formal/mixed_ce_8g_b4_a1.yaml --print-command
```

每个 run 使用独立 output root。checkpoint fingerprint 会阻止 full_target ↔ editregion 或其它训练语义的错误
resume；run provenance 同时写入 Git commit、dirty status 和 `uv.lock` SHA256。

## 7. GT-mask hard-lock evaluation

```bash
uv run python scripts/eval/evaluate_gt_mask_editing.py --help
```

对同一 held-out samples、masks、timesteps、CFG 与 seed 比较。现有主要指标为 Changed-token Accuracy、
Edit Token Accuracy、Source-copy Token Accuracy、Inside L1、Inside PSNR 与 Inside L1 vs Target
Reconstruction。oracle 是 diagnostic upper bound；boundary metric 只用于 leakage / seam diagnostic。GT mask
是 inference input，当前不是 free-edit localization benchmark。

## 8. 打包与历史 v3

轻量包只收录 configs、metrics、curves、Git 信息、provenance 与 logs；不包含 checkpoint、模型权重、数据集或
GCE cluster。历史 MagicBrush-only v3 结果与命令必须保持可追溯，但不能描述为当前 mixed 2×3 recipe 或当前
正式结论。

```bash
git rev-parse HEAD
git status --short
uv run pytest -q
```
