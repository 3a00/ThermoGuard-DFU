"""Ordinal-aware loss functions for ThermoGuard-DFU severity classification.

Phase 1.2 introduces OrdinalWeightedCELoss to address the systematic
under-estimation bias observed in Phase 1.1 (14 of 16 test errors were
severity under-predictions).

Import from training/evaluation scripts:
    from utils.losses import OrdinalWeightedCELoss

Usage:
    criterion = OrdinalWeightedCELoss(
        class_weights=class_weights_tensor,   # 3-class weights, computed at runtime
        under_penalty=cfg['ordinal_under_penalty'],  # 2.0
        over_penalty=cfg['ordinal_over_penalty'],    # 1.0
    )
    loss = criterion(logits, targets)  # scalar, same API as nn.CrossEntropyLoss

Note on loss scales:
    OrdinalWeightedCELoss inflates train_loss by up to 5x vs plain CE val_loss.
    Both are logged separately (train_loss and train_loss_unweighted) in the
    training CSV for fair visual comparison. Overfitting diagnosis uses F1 gap,
    not the loss curves. Val/test always use plain CrossEntropyLoss.
"""

import torch
import torch.nn as nn


class OrdinalWeightedCELoss(nn.Module):
    """Cross-entropy loss with ordinal severity distance penalty.

    Applies a per-sample multiplier based on the gap between true and
    predicted class index. Under-prediction is penalised more aggressively
    than over-prediction, matching the clinical asymmetry of DFU screening
    (missing a high-severity case is far more dangerous than a false alarm).

    Severity order: Healthy=0 < Low_Severity=1 < High_Severity=2

    Ordinal factor table (under_penalty=2.0, over_penalty=1.0):
      Correct prediction            -> factor = 1.0
      1-tier under (High->Low)      -> factor = 3.0  (1 + 2.0 * 1)
      2-tier under (High->Healthy)  -> factor = 5.0  (1 + 2.0 * 2)
      1-tier over  (Healthy->Low)   -> factor = 2.0  (1 + 1.0 * 1)
      2-tier over  (Healthy->High)  -> factor = 3.0  (1 + 1.0 * 2)

    The worst-case factor is bounded at 5x (only 3 classes), preventing
    runaway gradient spikes even on a 232-image training set.
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        under_penalty: float = 2.0,
        over_penalty: float = 1.0,
    ) -> None:
        """Initialise with class weights and per-direction penalty scalars.

        Args:
            class_weights:  1-D tensor of per-class weights (e.g. balanced weights).
                            Passed directly to nn.CrossEntropyLoss(weight=...).
            under_penalty:  Multiplier per tier of under-prediction (true > pred).
            over_penalty:   Multiplier per tier of over-prediction (pred > true).
        """
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        self.under_penalty = under_penalty
        self.over_penalty = over_penalty

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ordinal-weighted loss. Returns scalar mean.

        Args:
            logits:  (B, num_classes) -- raw model outputs, not softmaxed.
            targets: (B,) -- integer class indices.

        Returns:
            Scalar loss tensor (mean over batch).
        """
        per_sample_ce = self.ce(logits, targets)             # (B,) -- weighted CE per sample
        preds = logits.argmax(dim=1)                          # (B,) -- argmax predicted class
        distance = targets.float() - preds.float()            # positive = under-prediction
        penalty = torch.where(
            distance > 0,
            self.under_penalty * distance.abs(),              # under-prediction: penalise more
            self.over_penalty  * distance.abs(),              # over-prediction:  penalise less
        )
        ordinal_factor = 1.0 + penalty                        # minimum = 1.0 (correct predictions)
        return (per_sample_ce * ordinal_factor).mean()


if __name__ == '__main__':
    # Unit test: verify ordinal factor magnitudes
    # Run: python src/utils/losses.py  (from project root with venv activated)
    dummy_weights = torch.ones(3)
    loss_fn = OrdinalWeightedCELoss(dummy_weights, under_penalty=2.0, over_penalty=1.0)

    # Craft logits to force specific predictions (high logit -> argmax = that class)
    high_logit = torch.tensor([[0.0, 0.0, 10.0]])   # predicts High_Severity (2)
    low_logit  = torch.tensor([[0.0, 10.0, 0.0]])   # predicts Low_Severity (1)
    heal_logit = torch.tensor([[10.0, 0.0, 0.0]])   # predicts Healthy (0)

    target_high = torch.tensor([2])   # true: High_Severity
    target_heal = torch.tensor([0])   # true: Healthy

    # Correct prediction -> factor = 1.0 (loss unchanged from CE)
    loss_correct = loss_fn(high_logit, target_high)
    # High->Low under-prediction (1 tier) -> factor = 3.0
    loss_high_to_low = loss_fn(low_logit, target_high)
    # High->Healthy under-prediction (2 tier) -> factor = 5.0
    loss_high_to_heal = loss_fn(heal_logit, target_high)
    # Healthy->Low over-prediction (1 tier) -> factor = 2.0
    loss_heal_to_low = loss_fn(low_logit, target_heal)

    print(f'Correct (factor=1.0):        {loss_correct.item():.4f}')
    print(f'High->Low (factor=3.0):      {loss_high_to_low.item():.4f}')
    print(f'High->Healthy (factor=5.0):  {loss_high_to_heal.item():.4f}')
    print(f'Healthy->Low (factor=2.0):   {loss_heal_to_low.item():.4f}')
    assert loss_high_to_heal.item() > loss_high_to_low.item() > loss_correct.item(), \
        'Ordinal factor ordering violated: High->Healthy should be > High->Low > Correct.'
    assert loss_high_to_low.item() > loss_heal_to_low.item(), \
        'Ordinal factor ordering violated: under-prediction should be > over-prediction.'
    print('All ordinal factor assertions passed.')
