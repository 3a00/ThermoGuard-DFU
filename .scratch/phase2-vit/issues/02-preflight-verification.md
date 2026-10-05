# 02: Pre-flight Verification Test Suite

**What to build:** An automated test script (`tests/test_phase2_vit.py`) that acts as an engineering gate prior to model training. Verifies forward-pass tensor dimensions, absence of NaN/Inf values, correctness of Stage A and Stage B parameter gradient freeze boundaries, tripwire mode assertion helpers, and checks that forward + backward execution under FP16 mixed precision does not exceed GTX 1650 VRAM headroom (threshold 3.5GB).

**Blocked by:** 01: ViT-Tiny Thermal Model Architecture

**Type:** task

**Status:** done

**Training loop contract (MUST follow in Ticket 03):**
  Do NOT call `model.train()` in the training loop. Use `freeze_backbone(model)` / `unfreeze_stage_b(model)` as the sole mode-setting calls (once per stage transition). Call `assert_stage_a_modes(model)` / `assert_stage_b_modes(model)` at the top of every epoch as tripwires.

- [x] Test script created at `tests/test_phase2_vit.py` runnable via `python -m unittest` or `pytest`
- [x] Test 1: Asserts input `[16, 1, 224, 112]` produces output shape `[16, 3]`
- [x] Test 2: In Stage A mode, asserts gradients exist only for classification head (`head.weight`, `head.bias`), zero for backbone
- [x] Test 3: In Stage B mode, asserts gradients exist for `blocks[10:12]`, `norm`, and `head`, and are zero for `patch_embed` and `blocks[0:10]`
- [x] Test 4: Peak CUDA memory consumption verified on GPU under `torch.amp.autocast('cuda')`
- [x] Test 5: After `freeze_backbone()`, simulate `model.train()` footgun — assert that `assert_stage_a_modes()` raises `RuntimeError`
- [x] Test 6: After `unfreeze_stage_b()`, simulate `model.train()` footgun — assert that `assert_stage_b_modes()` raises `RuntimeError`
- [x] Script exits with code 0 on success

