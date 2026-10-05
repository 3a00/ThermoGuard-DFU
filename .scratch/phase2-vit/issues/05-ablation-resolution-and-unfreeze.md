# 05: Phase 2 Ablation Studies (Resolution, Token Interpolation, and Unfreezing)

**What to build:** Systematic ablation experiments to investigate Phase 2 design choices agreed during planning and external review:
1. Non-padded $224 \times 112$ inputs with bicubic interpolated positional embeddings ($14 \times 7 = 98$ tokens) versus $224 \times 224$ zero-padded inputs (196 tokens).
2. Deeper backbone unfreezing (blocks 8–11 vs 10–11).
3. Optimizer and LR dynamics: `AdamW` (decoupled weight decay $10^{-3}$) vs `Adam`, and differential learning rates (`lr_backbone = 1e-5`, `lr_head = 1e-4`) vs uniform `lr_stage_b = 1e-4`.
4. Multi-checkpoint probability ensembling / weight averaging across Stage B epochs (matching Phase 1.2 ensembling strategy) to reduce validation variance.
5. Clinical checkpoint selection: evaluate `high_severity_recall` tie-breaker or composite metric vs standard `val_weighted_f1` for model saving.
6. CutMix/Mixup regularizations on small-sample training sets.

**Blocked by:** 04: Phase 2 Baseline Training and Evaluation Benchmark

**Type:** research

**Status:** ready-for-agent

- [ ] Implement toggle for interpolated positional embeddings in `src/models/vit_tiny.py` for $224 \times 112$ resolution
- [ ] Run ablation experiment A: evaluate $224 \times 112$ (98 tokens) vs $224 \times 224$ padded (196 tokens) on test split
- [ ] Run ablation experiment B: evaluate deeper unfreezing (blocks 8–11)
- [ ] Run ablation experiment C: evaluate AdamW and differential LR (`1e-5` backbone / `1e-4` head)
- [ ] Run ablation experiment D1: evaluate Stage B multi-checkpoint ensembling / weight averaging
- [ ] Run ablation experiment D2: evaluate clinical checkpoint selection (high-severity recall tie-breaker)
- [ ] Record comparative findings and metrics in `outputs/phase2_vit/metrics/phase2_ablation_summary.csv`
