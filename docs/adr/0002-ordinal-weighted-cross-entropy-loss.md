# 0002. Ordinal-Weighted Cross-Entropy Loss for Clinical Severity

Use an asymmetric ordinal-weighted cross-entropy loss (`OrdinalWeightedCELoss`) for training all severity classification models.

## Context

In diabetic foot ulcer (DFU) screening, misclassifications are not equally costly. Predicting `Low_Severity` or `Healthy` when a patient has `High_Severity` (under-estimation / false negative) risks unmonitored ulceration, tissue necrosis, and potential amputation. Conversely, predicting `High_Severity` when a patient is `Low_Severity` (over-estimation / false positive) merely leads to precautionary clinical re-inspection. Standard Cross-Entropy treats all error directions equally.

## Decision

Train Phase 2 (and subsequent phases) with `OrdinalWeightedCELoss`:
- Scale the loss penalty by an asymmetric directional factor when prediction $\hat{y}$ deviates from true grade $y$.
- Set `under_penalty = 2.0` (penalizing under-estimation of severity $2\times$ harder).
- Set `over_penalty = 1.0` (standard penalty for conservative over-estimation).
- Combine with inverse-frequency class weights computed from the Phase 0.8 split to handle residual class imbalances.
- Restrict ordinal penalty to training only; evaluation and test metrics retain standard unpenalized Cross-Entropy and multiclass metrics for unbiased scientific reporting.

## Consequences

- Directs model gradients toward conservative clinical safety, significantly reducing dangerous high-severity misses.
- Validated in Phase 1.2, contributing to a boost in high-severity recall and overall weighted F1.
- Initial Phase 2 ViT-Tiny baseline will train under this objective to evaluate transformer feature representation under clinical safety constraints.
