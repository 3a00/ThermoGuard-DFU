# 01: ViT-Tiny Thermal Model Architecture

**What to build:** A PyTorch model module wrapping `timm`'s `vit_tiny_patch16_224` that accepts plantar thermogram tensors of shape `[B, 1, 224, 112]` or `[B, 3, 224, 112]`, performs internal patch-aligned zero-padding (48 px left, 64 px right) to `[B, C, 224, 224]` avoiding mixed boundary patches, replicates single channels to 3 channels, and predicts logits across 3 clinical Severity Grades (`Healthy`, `Low_Severity`, `High_Severity`). Includes helper methods for two-stage transfer learning (freezing backbone with `eval()` mode, unfreezing top blocks with selective `train()` mode).

**Blocked by:** None (can start immediately)

**Type:** task

**Status:** done

- [x] Model architecture defined in `src/models/vit_tiny.py` with class `ViTTinyThermal`
- [x] Internal patch-aligned horizontal padding from 112 to 224 (48 px left, 64 px right; 3 zero patches, 7 content patches, 4 zero patches)
- [x] 1-channel to 3-channel broadcast before patch projection
- [x] Head replaced with `Dropout(0.3)` + `Linear(192, 3)`
- [x] Helper functions `freeze_backbone(model)` and `unfreeze_stage_b(model)` controlling both layer gradients and submodule `.eval()` / `.train()` states
- [x] Module importable and instantiateable with both pretrained and randomly initialized weights
