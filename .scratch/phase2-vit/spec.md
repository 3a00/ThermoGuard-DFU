# Phase 2 — ViT-Tiny Vision Transformer for Plantar Thermal Risk Assessment

**Status:** ready-for-agent

## Problem Statement

Phase 1 established convolutional baselines (EfficientNet-B0) for automated diabetic foot ulcer risk prediction from plantar thermal images. While Phase 1.1 and Phase 1.2 achieved strong results (up to 80.77% with test-time augmentation and spatial angiosome pooling), convolutional backbones rely heavily on local spatial priors and inductive bias. It remains unknown whether self-attention mechanisms in Vision Transformers (ViT) can effectively model non-local, contralateral, and bilateral thermal discrepancies across distinct foot angiosomes, or whether data scarcity (~232 training feet) will cause catastrophic overfitting without convolutional inductive biases.

## Solution

Implement and benchmark a lightweight Vision Transformer baseline using `vit_tiny_patch16_224` (via `timm`, pretrained on ImageNet-1k). Adapt the input thermal pipeline to support transformer patch embeddings, apply asymmetric ordinal-weighted cross-entropy loss (`OrdinalWeightedCELoss`), execute a staged transfer-learning schedule tailored to GPU memory constraints (GTX 1650 ~4GB VRAM), and compare representations against the Phase 1 convolutional benchmarks.

## User Stories

1. As an ML researcher, I want a Vision Transformer backbone (`vit_tiny_patch16_224`) integrated into the model registry, so that I can evaluate transformer self-attention on plantar thermogram representations.
2. As an ML researcher, I want input thermal arrays to be symmetrically padded to $224 \times 224$ and replicated to 3 channels, so that the model can directly leverage ImageNet pretrained positional embeddings and early patch projection weights.
3. As a clinician, I want the model to classify subjects into 3 clinically grounded Severity Grades (`Healthy`, `Low_Severity`, `High_Severity`), so that high-risk patients are flagged without minority-class gradient collapse.
4. As a clinician, I want training guided by `OrdinalWeightedCELoss` with an under-estimation penalty factor of 2.0, so that dangerous under-diagnosis of ulceration risk is heavily penalized over conservative over-diagnosis.
5. As an ML engineer, I want a two-stage training schedule (Stage A: head only; Stage B: selective unfreeze of the top 2 transformer blocks), so that pretrained representations are stabilized before fine-tuning on a small medical dataset.
6. As an ML engineer, I want DataLoader and mixed-precision operations optimized for an NVIDIA GTX 1650 (batch size 16, AMP FP16, `num_workers=2`), so that training executes stably within 4GB VRAM without out-of-memory crashes.
7. As an ML engineer, I want an automated pre-flight test suite, so that tensor dimensions, gradient flow, freeze status, and peak memory usage are verified prior to launching multi-epoch training jobs.
8. As a project evaluator, I want standardized classification reports, confusion matrices, training curves, and real-time TensorBoard scalar logs recorded to segregated output paths (`outputs/phase2_vit/`), so that Phase 2 artifacts are fully observable and never overwrite previous phases.
9. As an ML researcher, I want test metrics directly comparable to Phase 1.1 and Phase 1.2 benchmarks, so that the academic graduation thesis has rigorous empirical evidence on CNN vs Transformer trade-offs.
10. As an ML researcher, I want an ablation path for input token interpolation and deeper unfreezing, so that we can systematically investigate whether preserving 2:1 aspect ratio without padding improves transformer attention maps.

## Implementation Decisions

- **Model Architecture**:
  - Backbone: `vit_tiny_patch16_224` initialized with pretrained ImageNet weights via `timm`.
  - Input adaptation: Patch-aligned zero-padding in width from 112 to 224 (48 pixels left, 64 pixels right; $48 = 3 \times 16$, $112 = 7 \times 16$, $64 = 4 \times 16$) to create a $224 \times 224$ array without mixed boundary patches, followed by 1-to-3 channel replication. Background remains 0.0.
  - Classification Head: Replaces the default 1000-class linear projection with a regularized classification head: `Dropout(p=0.3)` followed by `Linear(in_features=192, out_features=3)`.
- **Optimization & Loss**:
  - Loss function: `OrdinalWeightedCELoss` (`under_penalty=2.0`, `over_penalty=1.0`) combined with inverse-frequency class weights from Phase 0.8 (`outputs/phase0_prep/class_weights.json`).
  - Optimizer: Adam with stage-specific weight decays ($10^{-4}$ in Stage A, $10^{-3}$ in Stage B).
  - Scheduler: CosineAnnealingLR across Stage B epochs with minimum learning rate $\eta_{\min} = 10^{-6}$.
- **Transfer Learning Schedule**:
  - Stage A (8 epochs): All patch embeddings and transformer blocks frozen and set to `eval()` mode (eliminating stochastic dropout/drop-path noise); only the classification head trains at LR $10^{-3}$ in `train()` mode.
  - Stage B (45 epochs): Patch embeddings and blocks 0–9 remain frozen in `eval()` mode; blocks 10–11, final LayerNorm, and head train at LR $10^{-4}$ in `train()` mode. Gradient clipping `max_grad_norm=1.0`.
- **Data Augmentation**:
  - Train split only: On-the-fly random rotation ($\pm 10^\circ$), translation ($\pm 5\%$), scale ($0.95 - 1.05$), random erasing ($p=0.1$), Gaussian noise ($\sigma=0.01$). Flips remain permanently disabled.
  - Validation/Test splits: Zero augmentation applied (data leakage prevention).
- **Configuration & Directory Segregation**:
  - Central config: All hyperparameters declared under `phase2:` in `config.yaml`.
  - Artifacts: Checkpoints in `outputs/phase2_vit/checkpoints/`, logs/reports in `outputs/phase2_vit/metrics/`, plots in `outputs/phase2_vit/plots/`, and TensorBoard event logs in `outputs/phase2_vit/tensorboard/`.

## Testing Decisions

- Tests must assert external module contracts rather than internal variable naming.
- **Pre-flight contract test suite** (`tests/test_phase2_vit.py`):
  1. Forward pass shape assertion: Input tensor `[16, 1, 224, 112]` produces output logits `[16, 3]` without NaNs.
  2. Gradient boundary test: In Stage A, gradients are non-zero ONLY for the classification head; in Stage B, gradients flow through blocks 10, 11, LayerNorm, and head, while blocks 0–9 and patch projection have zero gradients.
  3. Peak memory test: Execution of forward + backward pass under `torch.cuda.amp.autocast()` uses $<3.5\text{GB}$ VRAM on GPU (or runs CPU fallback cleanly).
- **Evaluation test suite**:
  - Evaluates best validation checkpoint on the 26 held-out test subjects (52 feet).
  - Computes foot-level accuracy, macro F1, weighted F1, and subject-level aggregated predictions.

## Out of Scope

- Hybrid CNN + Transformer architectures (reserved for Phase 3).
- Test-Time Augmentation (TTA) and checkpoint ensembles for Phase 2 baseline (reserved for Phase 2 post-baseline ablation).
- Modifying Phase 0 preprocessed `.npy` arrays or data split manifests (Phase 0 files are strictly read-only).

## Further Notes

- Once baseline results are recorded, a follow-up ablation ticket will evaluate bicubic interpolation of positional embeddings to test 98 tokens ($14 \times 7$) without zero-padding, as well as CutMix/Mixup regularizations.
