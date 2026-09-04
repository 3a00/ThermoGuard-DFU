# 0001. Three-Class Clinical Severity Grouping

Consolidate the original 6 TCI grades into 3 clinical severity classes (`Healthy`, `Low_Severity`, `High_Severity`) for all model training and evaluation.

## Context

The raw IEEE Plantar Thermogram dataset labels subjects across 6 categories: `Healthy` (Control group) and `DM_Grade0` through `DM_Grade4` (Diabetic group). In Phase 1.0, training an EfficientNet-B0 on all 6 classes yielded severe misclassifications (38.46% test accuracy) because minority classes like `DM_Grade0` contained as few as 7 training samples.

## Decision

Map the 6 classes dynamically at dataset loading time into 3 actionable clinical severity categories:
- `Healthy` (Class 0): Control subjects (Grade -1 / Healthy)
- `Low_Severity` (Class 1): Subclinical or mild temperature deviations (`DM_Grade0`, `DM_Grade1`)
- `High_Severity` (Class 2): Marked to critical angiosome asymmetry indicating high ulceration risk (`DM_Grade2`, `DM_Grade3`, `DM_Grade4`)

The underlying Phase 0 preprocessed arrays and manifests remain immutable and untouched.

## Consequences

- Prevents gradient starvation on rare grades while maintaining clinical distinction between mild and urgent diabetic foot complications.
- Validated in Phase 1.1 with an immediate leap from 38.46% to 69.23% test accuracy (0.7024 weighted F1).
- All subsequent model phases (Phase 1.2, Phase 2 ViT, Phase 3 Hybrid) will standardize on this 3-class target space.
