# Phase 2 — ViT-Tiny Wayfinder Map

**Effort:** Phase 2 Vision Transformer Experiment  
**Status:** In Progress  
**Spec:** `.scratch/phase2-vit/spec.md`

## Notes

- Evaluates whether Vision Transformer (ViT) self-attention can model bilateral angiosome asymmetry on plantar thermograms.
- Uses `vit_tiny_patch16_224` (`timm`) with symmetric zero-padding and 3-channel replication.
- Constrained to NVIDIA GTX 1650 (~4GB VRAM): batch size 16, AMP FP16, `num_workers=2`.

## Decisions So Far

- **2026-09-04**: Consolidated 6 TCI classes into 3 clinical Severity Grades (`Healthy`, `Low_Severity`, `High_Severity`) via [[docs/adr/0001-three-class-severity-grouping.md]].
- **2026-09-04**: Adopted `OrdinalWeightedCELoss` (`under_penalty=2.0`, `over_penalty=1.0`) with inverse-frequency weights via [[docs/adr/0002-ordinal-weighted-cross-entropy-loss.md]].
- **2026-09-04**: Adopted Matt Pocock's local engineering ticket workflow under `.scratch/phase2-vit/`.
- **2026-09-05**: Updated padding to patch-aligned 48/64 (avoiding mixed boundary patches) and added explicit submodule `.eval()` / `.train()` control per architecture review.
- **2026-09-15**: Integrated real-time TensorBoard scalar logging (`outputs/phase2_vit/tensorboard/`) into Ticket 03 for live training/validation tracking.

## Tickets & Dependencies

1. **`issues/01-vit-model-architecture.md`** (task, done) — Blocked by: None
2. **`issues/02-preflight-verification.md`** (task, done) — Blocked by: None (01 completed)
3. **`issues/03-train-script-phase2.md`** (task, done) — Blocked by: None (02 completed)
4. **`issues/04-baseline-training-and-evaluation.md`** (task, done) — Blocked by: None (03 completed)
5. **`issues/05-ablation-resolution-and-unfreeze.md`** (research, ready-for-agent) — Blocked by: None (04 completed)

## Frontier

- Currently unblocked: **Ticket 05 (`05-ablation-resolution-and-unfreeze.md`)**

## Fog

- ViT behavior under small sample sizes (~232 training feet): will attention heads overfit or extract coherent global angiosome representations?
- Impact of aspect ratio padding ($224 \times 224$ with 48/64 border zeros) versus token interpolation ($224 \times 112$ with 98 tokens).
