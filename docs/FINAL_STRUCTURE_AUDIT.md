# Final structure audit

Audit branch: `cleanup/final-structure`
Audit base: `2f48da2e665c69797b7c68203ab1fb0bf68f9dee`

## Scope and invariants

This document records the pre-cleanup dependency graph only.  No training,
objective, model, optimizer, scheduler, data-corruption, padding, DDP, or
checkpoint-resume behavior was changed while preparing this audit.

The canonical implementation boundary is intended to be:

```text
src/lumina_dimoo/    MagicBrush data, objectives, LoRA, configuration, runner
model/               upstream/checkpoint `trust_remote_code` compatibility
datasets/            retained general preprocessing helpers while tools need them
scripts/             public launcher and thin convenience wrappers
```

## Launcher and configuration audit

`scripts/train/train.py` currently loads YAML and calls `runner.run(...)`
directly.  `runner.run()` immediately calls `dist.init_process_group("nccl")`
and reads `LOCAL_RANK`; therefore a normal `python scripts/train/train.py ...`
invocation advertised in the README is inconsistent with the worker's DDP
requirements.

The train YAML files already declare the intended local GPU counts and global
batches, but `load_train_config()` does not currently consume these values for
launching or validate their arithmetic.  The required values are:

| Config | `nproc_per_node` | batch/GPU | accumulation | computed global batch | declared |
| --- | ---: | ---: | ---: | ---: | ---: |
| `magicbrush_ce.yaml` | 4 | 4 | 2 | 32 | 32 |
| `magicbrush_attention.yaml` | 2 | 8 | 4 | 64 | 64 |
| `magicbrush_gce.yaml` | 4 | 4 | 2 | 32 | 32 |

`configs/runtime/single_node.yaml` is presently not read by the config loader.
`configs/dataset/magicbrush.yaml` is also not read by it; its default manifest,
VQ-grid, and corruption-seed description currently duplicate documentation.

## Legacy and canonical import map

| Old path | Current user/importer | Canonical path | Safe to remove now? |
| --- | --- | --- | --- |
| `attention_supervision/attention_loss.py` | `model/modeling_llada.py` | `lumina_dimoo.objectives.attention` | No; first change the root-model import, then remove the shim. |
| `attention_supervision/lora.py` | old trainer and hard-lock evaluator | `lumina_dimoo.training.lora` | No; update/move the evaluator and remove the old trainer first. |
| `objectives/gce.py` | `src/lumina_dimoo/training/runner.py`, `model/gce_loss.py` | `lumina_dimoo.objectives.gce` | No; first repair both imports while retaining GCE lazy import. |
| `training/lora.py` | no non-shim runtime importer found | `lumina_dimoo.training.lora` | Yes after final reference scan. |
| `training/objective_dispatch.py` | no non-shim runtime importer found | `lumina_dimoo.training.objective` | Yes after final reference scan. |
| `datasets/magicbrush_tokens.py` | old trainer and several MagicBrush audit/calibration tools | `lumina_dimoo.data.magicbrush` | No; update/move the tools and delete the old trainer first. |
| `datasets/magicbrush_dataset.py` | data preparation, tokenization, hard-lock evaluation, geometry audit, plus `read_jsonl` used by canonical data | `lumina_dimoo.data` for the minimal trainer helper | No; it is an active general preprocessing utility. |
| `train/train_magicbrush_attention.py` | `train/run_a0_a1_long1500.sh` only | `scripts/train/launch.py` plus unified runner | Yes after the new launcher is tested. |
| `train/run_a0_a1_long1500.sh` | old A0/A1 launch path only | `scripts/train/launch.py` | Yes after the new launcher is tested. |

The two concrete canonical-import fixes required before deleting shims are:

```text
src/lumina_dimoo/training/runner.py
    objectives.gce -> lumina_dimoo.objectives.gce  (still lazy for GCE only)

model/modeling_llada.py
    attention_supervision.attention_loss -> lumina_dimoo.objectives.attention
```

`model/gce_loss.py` also imports the GCE compatibility shim and must be made
canonical before `objectives/` can disappear.

## Model and dataset boundaries

The root `model/` package remains necessary.  The pretrained model declares
checkpoint auto-map modules under the root model implementation, and
`src/lumina_dimoo/models/__init__.py` deliberately re-exports the small public
surface (`LLaDAForMultiModalGeneration`, `MagicBrushModelOutput`).  The three
one-line `src/lumina_dimoo/models/*.py` star-import facades are redundant and
can be removed after their absence is checked.

`src/lumina_dimoo/data/magicbrush.py` is canonical for the training dataset but
currently imports only `read_jsonl` from `datasets.magicbrush_dataset`.  That
small helper can be migrated locally without altering MagicBrush data behavior.
The root `datasets/` package must remain while preprocessing and evaluation
tools retain their geometry and manifest utilities.

## Root `train/` classification

| File | Classification | Action in cleanup |
| --- | --- | --- |
| `train/train.py` | upstream training entrypoint | retain |
| `train/train.sh` | upstream wrapper | retain |
| `train/train_magicbrush_attention.py` | legacy MagicBrush A0/A1 trainer introduced at the experimental baseline | remove after replacement validation |
| `train/run_a0_a1_long1500.sh` | legacy launcher for the old trainer | remove after replacement validation |

## MagicBrush tools classification

The following files were introduced with the MagicBrush experiment baseline,
not upstream model training.  They are relocation candidates, not blanket
deletion targets:

| Current tools file | Intended destination |
| --- | --- |
| `prepare_magicbrush.py`, `prepare_magicbrush_eval.py`, `pretokenize_magicbrush.py` | `scripts/data/` |
| `audit_magicbrush_geometry.py`, `audit_magicbrush_sequences.py`, `audit_non_square_token_layout.py` | `scripts/tools/attention/` |
| `calibrate_attention_layers.py` | `scripts/tools/attention/` |
| `evaluate_magicbrush_hardlock.py` | `scripts/eval/` |
| `run_dev528_hardlock_8gpu.sh` | legacy, absolute-path experiment launcher; remove rather than promote |

## Packaging and ignore audit

`pyproject.toml` currently installs both `src` and root packages, including the
deprecated shim packages.  This is appropriate for root `model`, `utils`, and
`xllmx` checkpoint/upstream compatibility, but not for shims once their users
are gone.  The `.gitignore` currently contains only `__pycache__/` and
`output/`; it needs targeted virtual-environment, build, test-cache, and
experiment-output rules without ignoring source-bearing `datasets/`, `data/`,
`models/`, or `configs/` directories.

## Required post-change evidence

Before this branch can be proposed for merge, rerun the import scan used for
this audit, verify the config-driven launcher and its global-batch validation,
exercise CE/attention/GCE numeric oracles, verify launcher-based checkpoint
resume, and perform a fresh default-branch clone test.  The legacy experiment
branches and all safety tags are out of scope and must be retained.

## Superseding model/checkpoint compatibility audit

The final-layout migration request supersedes the earlier proposal to retain
`src/lumina_dimoo`.  The following evidence was collected before moving any
model file.

### Checkpoint result

The local pretrained checkpoint has no Python source files.  Its `config.json`
contains:

```json
"auto_map": {
  "AutoConfig": "configuration_llada.LLaDAConfig",
  "AutoModel": "modeling_llada.LLaDAModelLM",
  "AutoModelForCausalLM": "modeling_llada.LLaDAModelLM"
}
```

Consequently, a local-only
`AutoConfig.from_pretrained(..., trust_remote_code=True)` fails because
`configuration_llada.py` is absent from the checkpoint directory.  The
currently validated training route does **not** depend on that failing dynamic
path: it imports the repository's static
`model.modeling_xllmx_dimoo.LLaDAForMultiModalGeneration` and loads the
checkpoint weights through that class.

Therefore the checkpoint has no hard `model.*` `auto_map` dependency, but a
source compatibility boundary is still necessary for existing upstream scripts,
tests, and external callers that import `model`.  It must be a shim only, never
a second implementation.

### Exact relocation plan (not executed in this audit)

1. Create canonical packages `src/models/lumina`, `src/models/objectives`,
   `src/dataset`, `src/training`, and `src/utils`.
2. Move the complete implementations from `model/configuration_llada.py`,
   `model/modeling_llada.py`, and `model/modeling_xllmx_dimoo.py` to
   `src/models/lumina/`, preserving their relative imports and exact model
   behavior.
3. Change only the canonical model's attention-helper import to
   `models.objectives.attention`.  Move the existing attention and GCE
   objective implementations to `src/models/objectives/` unchanged.
4. Replace the three root `model/*.py` implementation files with thin
   re-exports from `models.lumina.*`; retain `model/__init__.py` as the same
   public compatibility surface.  This keeps direct legacy imports working
   while ensuring a single source of model logic.
5. Move canonical data and training implementations out of
   `src/lumina_dimoo`, migrate the minimal `read_jsonl` helper into
   `src/dataset/utils.py`, and remove the superseded package rather than adding
   wrappers.
6. Update internal imports to `dataset`, `models`, and `training`; only the
   root `model` shims may import canonical `models.lumina` modules.  Remove the
   attention/objectives/training shims and `datasets/magicbrush_tokens.py`
   only after a reference scan proves they have no remaining runtime users.
7. Keep static model loading in the final training path.  Do not claim that the
   supplied local checkpoint is independently `trust_remote_code` loadable
   until checkpoint source files or its `auto_map` are supplied separately.

This plan preserves all numerical, DDP, padding, LoRA, corruption, and
checkpoint-resume invariants because it changes module locations and imports,
not their executable logic.
