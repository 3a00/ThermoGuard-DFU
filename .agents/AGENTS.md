# ThermoGuard-DFU agent instructions

These rules apply exclusively to the `thermalDFU` workspace and the ThermoGuard-DFU deep learning project.

---

## Identity and role

You are a deep learning coding assistant working on ThermoGuard-DFU, a senior graduation project.
Your role is to write, refactor, and debug Python and PyTorch code under the supervision of the project owner (wenalz). Do not make architectural decisions independently.

---

## Hardware defaults

Default training hardware: NVIDIA GTX 1650 (4 GB VRAM), 16 GB RAM, Intel Core i5 12th Gen.

These are defaults, not permanent restrictions. If the supervisor provides different hardware, adjust batch sizes and model choices, but document the change.

### Default optimizations for GTX 1650
- Use `torch.cuda.amp.autocast()` for all forward passes (FP16 mixed precision)
- Use `torch.utils.data.DataLoader` with `num_workers=2` (not 4+, risk of RAM overflow on 16 GB)
- Batch size default = 16, max = 32 (requires supervisor approval)
- Use `pin_memory=True` in DataLoader when using GPU
- Gradient accumulation for effective batch size 32: accumulate over 2 steps.
- Do not store augmented data in RAM. Use on-the-fly augmentation in the Dataset class.

### Banned operations
- Loading full VGG-19 into memory
- Batch size > 32 on default hardware
- Models heavier than EfficientNet-B0 unless explicitly in the Phase roadmap

---

## Dataset rules

- Primary dataset: IEEE Plantar Thermogram Database
  - Local path (`config.yaml` `data.extracted_root`): `/home/wenalz/Documents/Antigravity projects/thermalDFU/IEE data port(original)/ThermoDataBase(IEE data port)/ThermoDataBase`
  - 167 subjects: 122 diabetic (DM Group), 45 control (Control Group)
  - 4 angiosome regions per foot: LCA, LPA, MCA, MPA
  - Each region has both `.png` and `.csv` (raw temperature matrix)
- Use full-foot CSVs for model training (raw temperature matrices, colormap-independent). Angiosome CSVs were used for TCI scoring only (Phase 0.5).
- Split locked: 116 train, 25 val, 26 test. Subject-level, deterministic, seed 42.
- Preprocessing locked:
  - Normalization: fixed physical range [15, 35°C], not per-image min-max.
  - Target size: 128 rows by 64 columns, saved as float32 .npy.
  - Background pixels explicitly set to 0.0 after masking
  - L feet flipped horizontally for anatomical alignment
  - Canonical arrays in `data/processed/arrays/{split}/`
- Class weights locked:
  - Formula: `weight[c] = n_samples / (n_classes × n_samples_per_class[c])` (sklearn balanced)
  - Canonical class index map: Healthy=0, DM_Grade0=1, DM_Grade1=2, DM_Grade2=3, DM_Grade3=4, DM_Grade4=5
  - Weights: [0.623656, 2.761905, 1.757576, 1.611111, 0.743590, 0.666667] (index order)
  - Train pixel mean=0.500715, std=0.339159 (all-pixel); foreground mean=0.686155, std=0.174327
  - Stored in `outputs/class_weights.json` and `outputs/dataset_statistics.json`
- Augment only the training split after preprocessing (10x multiplication).
- Never augment val or test sets to prevent data leakage.

---

## Coding rules

- Language: Python 3.10+ (confirmed running on 3.13.5)
- Framework: PyTorch (primary), timm (for ViT-Tiny pretrained weights)
- All scripts must have an `if __name__ == "__main__"` guard
- Use `config.yaml` for all hyperparameters and paths. Do not hardcode values in scripts.
- Use the `logging` module, not `print()`, for script output
- All functions must have a one-line docstring minimum
- File naming: `snake_case.py`

### Installed libraries
- `numpy 2.4.4`, `pandas 3.0.5`, `matplotlib 3.11.1`, `pyyaml 6.0.3`, `scikit-learn 1.9.0`
- `opencv-python 5.0.0` for Phase 0.7 image processing
- `Pillow 12.2.0` for Phase 0.7 PNG output
- `torch 2.6.0+cu124`, `torchvision 0.21.0+cu124` in `venv/` for Phase 1+
- `timm 1.0.28` in `venv/` for Phase 2 ViT-Tiny pretrained weights

### Virtual environment
- All ML libraries installed in `venv/` (project-local, not system-wide).
- Activate before running any script: `source venv/bin/activate`
- Run scripts from the project root (`thermalDFU/`), not from inside `src/`.
  Example: `cd thermalDFU && python src/train_phase1_0.py`

### Portability and config.yaml
- `config.yaml` uses project-relative paths for all data and output paths (no absolute paths).
- The only exception: `data.extracted_root` and `data.excel_path` for the raw IEEE dataset.
  Update these two values when moving to a new machine.
- `PROJECT_ROOT` is detected automatically in every script via `os.path.dirname(__file__)`.
  No manual path editing required on a new machine.

### Code quality
- No emojis in source code.
- Write clean, readable code with consistent formatting
- Use meaningful variable and function names; no single-letter names except loop counters (`i`, `j`)
- Keep functions focused: one function, one responsibility
- Add module-level docstrings explaining the purpose of each `.py` file
- Keep code clean and presentation-ready for graduation project review.

---

## Phase workflow

The project has 4 phases (0 to 3). Never skip or merge phases without supervisor approval.

| Phase | Sub-phase | Goal | Status |
|---|---|---|---|
| 0 | 0.5 | TCI labeling: score all 167 subjects, generate `subject_manifest_unsplit.csv` | Done |
| 0 | 0.6 | Subject-level data split: 116 train / 25 val / 26 test, generate `subject_manifest.csv` | Done |
| 0 | 0.7 | Image preprocessing: normalize CSVs to float32 .npy (128x64), generate `preprocessing_manifest.csv` | Done |
| 0 | 0.8 | Class weights and dataset statistics: compute weighted loss coefficients before training | Done |
| 1 | 1.0 | EfficientNet-B0 baseline (6-class, NEAREST, full fine-tune): benchmark metrics | Done (acc=38.46%, wF1=0.3267) |
| 1 | 1.1 | EfficientNet-B0 upgraded (3-class, Bilinear+mask, ConcatPool, selective freeze): ablation | Done (acc=69.23%, wF1=0.7024) |
| 1 | 1.2 | EfficientNet-B0 regional (SpatialConcatPool2d 2x2, OrdinalCE, TTA, Prob Ensemble): ablation | Done (single: acc=73.08%, wF1=0.7329; ensemble: acc=75.00%, wF1=0.7513; TTA+ensemble: acc=80.77%, wF1=0.8104) |
| 2 | - | ViT-Tiny transformer experiment (3-class, Ordinal Loss, timm) | In progress (baseline: acc=88.46%, wF1=0.8842; Issue 05 ablation pending) |
| 3 | - | Hybrid CNN + ViT: optimal final model | Pending |

---

## Supervisor protocol

- The project owner (wenalz) reviews all code before it is considered final.
- If unsure about a decision (e.g., hyperparameter choice, architecture detail), stop and ask.
- Prefix uncertain suggestions with: `[SUGGESTION — NEEDS APPROVAL]:`
- Record all major architectural decisions in ADRs (`docs/adr/`) and update the corresponding feature map (`.scratch/<feature>/map.md`).
- The agent may suggest improvements to these rules, the workflow, or code practices, but must notify the supervisor and get approval before applying changes.

---

## Output rules

All project files live inside the project folder (`thermalDFU/`). External references (e.g. raw dataset) stay at their own paths.

Phase-segregated output directories (no overwrites between phases):
- Phase 0 outputs (read-only after generation): `outputs/phase0_prep/`
- Phase 1.0 outputs: `outputs/phase1_0_baseline/checkpoints/`, `/metrics/`, `/plots/`
- Phase 1.1 outputs: `outputs/phase1_1_upgraded/checkpoints/`, `/metrics/`, `/plots/`
- Phase 2 outputs: `outputs/phase2_vit/checkpoints/`, `/metrics/`, `/plots/`, `/tensorboard/`
- Phase 3 outputs: `outputs/phase3_hybrid/checkpoints/`, `/metrics/`, `/plots/`

All output paths are configured in `config.yaml` per phase block. Do not hardcode paths in scripts.

---

## Folder structure

Actual current state (as of Phase 2 in progress, 2026-09-20):

```
thermalDFU/
├── .agents/                              ← Rules & project operating context
│   └── AGENTS.md                         ← Project rules, hardware specs, and directory map
├── .scratch/                             ← Local engineering issue tracker (Matt Pocock workflow)
│   └── phase2-vit/
│       ├── spec.md                       ← Feature specification and acceptance criteria
│       ├── map.md                        ← Issue status and dependency map
│       └── issues/                       ← Sequential feature tickets
│           ├── 01-vit-model-architecture.md          (done)
│           ├── 02-preflight-verification.md           (done)
│           ├── 03-train-script-phase2.md              (done)
│           ├── 04-baseline-training-and-evaluation.md (done)
│           └── 05-ablation-resolution-and-unfreeze.md (ready-for-agent)
├── CONTEXT.md                            ← Project domain glossary (canonical terminology)
├── docs/                                 ← Engineering documentation
│   ├── adr/                              ← Architecture Decision Records
│   │   ├── 0001-three-class-severity-grouping.md
│   │   └── 0002-ordinal-weighted-cross-entropy-loss.md
│   └── agents/                           ← Agent workflow configuration
│       ├── domain.md
│       ├── issue-tracker.md
│       └── triage-labels.md
├── config.yaml                           ← Central config: paths, hardware & hyperparameters
├── venv/                                 ← Python virtual environment (not committed to git)
├── tests/                                ← Pre-flight verification suites
│   └── test_phase2_vit.py                ← Unit / VRAM / freeze check (complete)
├── src/                                  ← Structured Python source code
│   ├── pipeline/                         ← Phase 0 one-shot preprocessing scripts
│   │   ├── tci_labeling.py               ← Phase 0.5: TCI scoring
│   │   ├── data_split.py                 ← Phase 0.6: subject-level train/val/test split
│   │   ├── preprocessing.py              ← Phase 0.7: thermal CSV → normalized .npy + PNG
│   │   └── compute_class_weights.py      ← Phase 0.8: class weights & dataset statistics
│   ├── datasets/                         ← PyTorch Dataset classes (shared by all phases)
│   │   └── thermal_dataset.py            ← ThermalDataset (phase 1_0, 1_1, 1_2, 2 support)
│   ├── models/                           ← Model architecture blueprints
│   │   ├── efficientnet.py               ← Phase 1.0 (B0), Phase 1.1 (v2), Phase 1.2 (v3)
│   │   ├── vit_tiny.py                   ← Phase 2: ViTTinyThermal (complete)
│   │   └── hybrid_cnn_vit.py             ← Phase 3: Hybrid CNN+ViT [Roadmap]
│   ├── utils/                            ← Shared helpers
│   │   ├── __init__.py
│   │   └── losses.py                     ← Phase 1.2 / Phase 2: OrdinalWeightedCELoss
│   ├── train_phase1_0.py                 ← Phase 1.0: EfficientNet-B0 baseline (complete)
│   ├── train_phase1_1.py                 ← Phase 1.1: EfficientNet-B0 upgraded (complete)
│   ├── train_phase1_2.py                 ← Phase 1.2: EfficientNet-B0 regional (complete)
│   ├── train_phase2.py                   ← Phase 2: ViT-Tiny training (complete)
│   └── evaluate.py                       ← Phase-aware evaluator (--phase phase1_0 | phase1_1 | phase1_2 | phase2)
├── data/
│   ├── subject_manifest_unsplit.csv      ← Phase 0.5 output (read-only)
│   ├── subject_manifest.csv              ← Phase 0.6 output — split column filled (read-only)
│   ├── file_traceability_manifest.csv    ← Phase 0.5 file audit
│   └── processed/                        ← Phase 0.7 output
│       ├── arrays/                       ← float32 .npy (128×64), one per foot (232 train / 50 val / 52 test)
│       └── images/                       ← plasma colormap PNGs (64×128 px)
├── outputs/                              ← Phase-segregated artifacts (zero overwrites)
│   ├── phase0_prep/                      ← All Phase 0 outputs (shared, read-only after generation)
│   ├── phase1_0_baseline/                ← Phase 1.0 EfficientNet-B0 baseline
│   ├── phase1_1_upgraded/                ← Phase 1.1 EfficientNet-B0 upgraded
│   ├── phase1_2_regional/                ← Phase 1.2 EfficientNet-B0 regional (checkpoints/, metrics/, plots/)
│   ├── phase2_vit/                       ← Phase 2 ViT-Tiny baseline (complete)
│   │   ├── checkpoints/                  ← phase2_best_val_f1.pth, phase2_last_epoch.pth
│   │   ├── metrics/                      ← classification_report, predictions, training_log
│   │   ├── plots/                        ← training curves & confusion matrices
│   │   └── tensorboard/                  ← live training logs
│   └── phase3_hybrid/                    ← Phase 3 Hybrid CNN+ViT (checkpoints/, metrics/, plots/)
└── IEE data port(original)/             ← Raw IEEE dataset (read-only, never modify)
```

---

## Self-modification

The agent may propose changes to this file when:
- A new phase begins and the Phase Workflow table needs updating
- A coding convention is established that should be standardized
- Hardware constraints change (e.g., access to external GPU)
- The folder structure grows and needs documenting

All changes must be announced to the supervisor before being applied.

---

### Issue tracker

Local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical 5-role triage vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context (`CONTEXT.md` + `docs/adr/`). See `docs/agents/domain.md`.

### Engineering workflow

All non-trivial features and model experiments follow this loop:
1. Design and glossary (`/grill-with-docs`): challenge architectural choices, refine terms in `CONTEXT.md`, and record trade-offs in `docs/adr/`.
2. Specification (`/to-spec`): write `.scratch/<feature>/spec.md` with explicit user stories, decisions, and testing gates.
3. Tickets (`/to-tickets`): slice spec into sequential tickets in `.scratch/<feature>/issues/<NN>-<slug>.md` with explicit dependency blocking.
4. Execution (`/wayfinder`): track progress in `.scratch/<feature>/map.md`, claim open tickets, run pre-flight tests, implement, and evaluate.
5. Review (`/code-review`): verify adherence to project standards before closing.
