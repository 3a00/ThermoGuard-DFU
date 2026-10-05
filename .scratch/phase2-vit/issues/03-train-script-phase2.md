# 03: Phase 2 Configuration and Training Pipeline

**What to build:** The training orchestration pipeline for Phase 2, including complete hyperparameter definitions in `config.yaml` under `phase2:`, and the training script `src/train_phase2.py`. Employs `ThermalDataset` with dynamic 3-class severity mapping, `OrdinalWeightedCELoss` (`under_penalty=2.0`), 2-stage Adam optimizer, CosineAnnealingLR, FP16 AMP, gradient clipping, checkpoint saving, CSV metrics logging, training curves plot generation, and real-time TensorBoard scalar logging to `outputs/phase2_vit/tensorboard/`.

**Implementation Reference Blueprint (CRITICAL FOR IMPLEMENTING AGENT):**
  The implementation design has been reviewed and finalized across two review rounds with external LLMs. The implementing agent must follow the vetted blueprint in:
  `docs/external llm reviews/phase2-vit reviews and pakages/ticket03_train_script_review.md`

**Controlled Comparison Note:**
  To maintain strict scientific comparability with Phase 1.2 (single model checkpoint), Phase 2 baseline locks optimizer parameters to Phase 1.2 (Adam, uniform `lr_stage_b = 1e-4`, single best validation checkpoint). Proposed transformer-specific enhancements (AdamW, differential learning rates `1e-5`/`1e-4`, checkpoint ensembling) are routed to Ticket 05 for systematic post-baseline ablation.

**Blocked by:** 02: Pre-flight Verification Test Suite

**Type:** task

**Status:** closed

**Dependency note:**
  TensorBoard is not yet installed in `venv/`. The agent implementing this ticket must prompt the user for explicit confirmation before running `pip install tensorboard` in `venv/`.

**Training loop contract (NON-NEGOTIABLE):**
  - Do NOT call `model.train()` anywhere in the training loop.
  - Call `restore_stage_modes(model, stage)` at the start of each epoch to safely re-establish train/eval modes after evaluation without triggering the recursive `model.train()` footgun.
  - Call `assert_stage_a_modes(model)` at the start of each Stage A epoch and `assert_stage_b_modes(model)` at the start of each Stage B epoch (tripwires from `src/models/vit_tiny.py`).
  - Class weights: Recompute balanced 3-class weights directly from aggregated train sample counts (`Healthy`: 62, `Low`: 60, `High`: 110) via sklearn balanced formula (do not sum raw 6-class weights).
  - Multi-worker seeding: Supply `_worker_init_fn` to `DataLoader` to ensure augmentation RNG entropy across worker threads.
  - Numerical stability: Assert `torch.isfinite(loss)` before backpropagation.

- [x] `config.yaml` populated with full `phase2` hyperparameters and `out_tensorboard_dir`
- [x] Confirm and install `tensorboard` package in `venv/`
- [x] `src/train_phase2.py` implemented with `if __name__ == "__main__"` entrypoint adhering to the blueprint in `ticket03_train_script_review.md`
- [x] Incorporates `OrdinalWeightedCELoss` using count-aggregated 3-class balanced weights
- [x] Implements Stage A warm-up (8 epochs) and Stage B fine-tuning (45 epochs) WITHOUT calling `model.train()`
- [x] Calls `restore_stage_modes(model, stage)` and `assert_stage_*_modes(model)` at top of each epoch
- [x] Computes and logs both `low_severity_recall` and `high_severity_recall` in validation evaluation
- [x] Adds `_worker_init_fn` to training DataLoader for RNG diversification
- [x] Initializes TensorBoard `SummaryWriter` at `outputs/phase2_vit/tensorboard/` and logs per-epoch scalars (`loss/train`, `loss/val`, `metrics/val_macro_f1`, `metrics/val_weighted_f1`, `metrics/val_accuracy`, `metrics/val_low_severity_recall`, `metrics/val_high_severity_recall`, `lr/backbone`, `lr/head`)
- [x] Saves `phase2_best_val_f1.pth` and `phase2_last_epoch.pth` in `outputs/phase2_vit/checkpoints/`
- [x] Writes CSV epoch metrics log to `outputs/phase2_vit/metrics/phase2_training_log.csv`
- [x] Generates loss and F1 training curves in `outputs/phase2_vit/plots/phase2_training_curves.png`
