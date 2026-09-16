# Root layout audit

审计日期：2026-09-16。审计基线为 `layout/before-root-consolidation`
（`af4a4c8d4a67f1c051446d7ba6b1735cecbc4b01`）。本文件记录迁移前的
依赖关系；它不是当前架构文档。

| Old path | Type | Inbound users / imports | Candidate destination | Action | Behavioral risk | Required validation |
| --- | --- | --- | --- | --- | --- | --- |
| `config.py` | shared constants/defaults | MagicBrush dataset/GCE/audits, inference | `src/utils/constants.py`; unused defaults removed | move/delete | token-id drift | canonical import + three loss oracles |
| `data/item_processor.py` | VQ pre-tokenizer implementation | `pre_tokenizer/pre_tokenize.py` only | `src/xllmx/data/dimoo_item_processor.py` | move | VQ crop/tokenization contract | pre-tokenizer parser/import |
| `datasets/magicbrush_dataset.py` | MagicBrush geometry/manifest helpers | dataset preprocessing, eval, audit CLIs | `src/dataset/{geometry,utils}.py` | split/move | geometry and split determinism | data tests + audit CLI |
| `generators/` | reusable sampling implementation | inference/evaluation | `src/generators/` | move | generation import path only | inference parser/import |
| `inference/` | executable CLIs | direct user entrypoints | `scripts/inference/` | move/rename | root `sys.path` hack | four `--help` checks |
| `model/` | compatibility shim | root train/inference/generator | remove after callers migrate | delete | checkpoint/model import | canonical model import + smoke |
| `pre_tokenizer/` | executable CLIs | direct user entrypoint | `scripts/data/pre_tokenizer/` | move | relative launcher paths | parser + `bash -n` |
| `train/` | upstream SFT entrypoint | direct user entrypoint | `scripts/train/upstream/` | move | SFT imports/launcher path | `--help` + `bash -n` |
| `xllmx/` | reusable upstream package | upstream SFT | `src/xllmx/` | move as package | package discovery | external-cwd imports |
| `examples/` | static images | documentation only | `assets/examples/` | move | Markdown links | link scan |
| `ht_grpo/` | auxiliary documentation/assets | documentation only | `docs/ht_grpo/`, `assets/ht_grpo/` | split/move | relative image links | Markdown inspection |
| `VLMEvalKit/` | vendored third-party project | independent vendored project | `third_party/VLMEvalKit/` | move intact | its internal relative paths | no internal rewrite |
| `configs/data.yaml` | upstream SFT example | `train/train.sh` | `configs/upstream/data.yaml` | move | launcher path | `bash -n` |
| `requirements.txt` | legacy dependency list | upstream SFT/inference/pre-tokenizer | `pyproject` extras | classify/delete | missing optional dependency | uv lock + CLI imports |
| `ENVIRONMENT.md` | documentation | README | `docs/ENVIRONMENT.md` | move | link path | README link |
| `Technical-Report.pdf` | documentation asset | README/docs if linked | `docs/Technical-Report.pdf` | move | link path | file presence |

## Dependency findings

- `data/item_processor.py` and `xllmx/data/item_processor.py` have different
  responsibilities. The first defines `DimooItemProcessor` plus VQ image
  encoding/crop helpers; the second defines upstream conversation-processing
  abstractions. Only the root pre-tokenizer imports the former. They must stay
  separate modules under `src/xllmx/data/`.
- `datasets/magicbrush_dataset.py` owns shared geometry, manifest I/O and
  split helpers. Its functions will be split between `dataset.geometry` and
  `dataset.utils`; no copies are needed.
- `GENERATION_CONFIG`, `IMAGE_CONFIG`, and `EDIT_TYPE_CONFIG` have no runtime
  users. `SPECIAL_TOKENS` and `VISUAL_CODEBOOK_SIZE` have runtime users and
  move together to `utils.constants`. Prompt templates already have a
  canonical factory in `utils.prompt_utils` and will not be duplicated.
- Root `model/` is only a thin shim. Its users are root train/inference and one
  generator; all can use `models.lumina` after migration.
- `requirements.txt` contains MagicBrush core dependencies already covered by
  `pyproject`, upstream-SFT-only dependencies (`pandas`, `tensorboard`,
  `fairscale`, `h5py`, `bitsandbytes`, `torchao`), inference/UI-only
  dependencies (`gradio`, `httpx[socks]`), and developer tooling
  (`pre-commit`). Vendored VLMEvalKit keeps its own requirements and is not a
  main-project extra.

## Migration guards

The migration is path/import/package-only. CE, attention, GCE, LoRA,
optimizer/scheduler, corruption, DDP, checkpoint/resume, token/mask
semantics, sampling, upstream SFT, and inference algorithms are out of scope.
After each package move, canonical imports must work without `PYTHONPATH` or a
repository-root `sys.path` injection.
