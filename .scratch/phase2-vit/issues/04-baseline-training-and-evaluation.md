# 04: Phase 2 Baseline Training and Evaluation Benchmark

**What to build:** Execution of the full Phase 2 baseline training run, evaluation of the best checkpoint on the held-out test split (26 subjects, 52 feet), generation of test metrics (accuracy, macro F1, weighted F1, confusion matrix, subject-level predictions), and a structured comparison table against Phase 1.1 and Phase 1.2 baselines.

**Blocked by:** 03: Phase 2 Configuration and Training Pipeline

**Type:** task

**Status:** done

- [x] Complete execution of 53 epochs (8 Stage A + 45 Stage B) without out-of-memory or stability errors
- [x] Evaluation executed on the 52 test feet using best validation checkpoint (Epoch 35, val wF1=0.8803)
- [x] Classification report saved to `outputs/phase2_vit/metrics/phase2_test_classification_report.csv`
- [x] Predictions saved to `outputs/phase2_vit/metrics/phase2_test_predictions.csv`
- [x] Confusion matrix plotted to `outputs/phase2_vit/plots/phase2_confusion_matrix.png`
- [x] Subject-level aggregated report saved to `outputs/phase2_vit/metrics/phase2_test_subject_report.csv`
- [x] Comparison summary documented comparing Phase 2 ViT against Phase 1.1 (69.23% acc, 0.7024 wF1) and Phase 1.2 (73.08% acc, 0.7329 wF1)

## Benchmark Results (Executed 2026-09-20)

- **Best Checkpoint:** Epoch 35 (Stage B, fine-tuning epoch 27/45), Val Weighted-F1: 0.8803
- **Test Accuracy:** 88.46% (46 / 52)
- **Test Weighted-F1:** 0.8842
- **Test Macro-F1:** 0.8774
- **Per-Class Recall:** Healthy 100.0% (14/14), Low_Severity 78.6% (11/14), High_Severity 87.5% (21/24)
- **Clinical Safety / Anti-Mirror:**
  - Total errors: 6 (5 under-estimation, 1 over-estimation)
  - Under-estimation rate: 83.3% (passes `beats_phase1_1_baseline_gate` < 87.5%; `anti_mirror_balanced_gate` <= 50.0% not passed)
- **Subject-Level Accuracy (Bilateral Agreement):** 20 / 26 correct (76.92%), 0 fully wrong, 6 split
- **Paired McNemar Significance Test vs Phase 1.2:**
  - Discordant pairs: 14 (11 Phase 2 correct & Phase 1.2 wrong vs 3 Phase 2 wrong & Phase 1.2 correct)
  - Exact binomial p-value: 0.0574 ($p \ge 0.05$, difference is consistent with small-sample sampling variation on $n=52$)

### Cross-Phase Benchmark Comparison Table

| Architecture | Phase | Inference Mode | Test Acc | Weighted F1 | Macro F1 | High-Sev Recall | Under-Est Rate | McNemar $p$ vs P1.2 |
|---|---|---|---|---|---|---|---|---|
| EfficientNet-B0 (6-class) | Phase 1.0 | Single Best Ckpt | 38.46% (19/52) | 0.3267 | 0.2169 | N/A | N/A | — |
| EfficientNet-B0 v2 (3-class) | Phase 1.1 | Single Best Ckpt | 69.23% (36/52) | 0.7024 | 0.6971 | 83.3% | 87.5% (14/16) | — |
| EfficientNet-B0 v3 (Regional) | Phase 1.2 | Single Best Ckpt | 73.08% (38/52) | 0.7329 | 0.7258 | 87.5% | 35.7% (5/14) | Reference Baseline |
| EfficientNet-B0 v3 (Regional) | Phase 1.2 | TTA + Ensemble | 80.77% (42/52) | 0.8104 | 0.8038 | 91.7% | 30.0% (3/10) | — |
| **ViT-Tiny (`vit_tiny_patch16_224`)** | **Phase 2** | **Single Best Ckpt** | **88.46% (46/52)** | **0.8842** | **0.8774** | **87.5% (21/24)** | **83.3% (5/6)** | **$p = 0.0574$** |
