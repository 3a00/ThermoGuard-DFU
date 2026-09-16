"""Unit tests for Phase 2 training pipeline components.

Tests:
  1. compute_3class_weights calculation against sklearn balanced formula.
  2. restore_stage_modes for Stage A and Stage B and tripwire compliance.
  3. _worker_init_fn execution.
  4. train_one_epoch and evaluate execution with dummy DataLoader and loss finite assertions.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.models.vit_tiny import (
    assert_stage_a_modes,
    assert_stage_b_modes,
    build_vit_tiny,
)
from src.train_phase2 import (
    _worker_init_fn,
    compute_3class_weights,
    evaluate,
    restore_stage_modes,
    train_one_epoch,
)
from src.utils.losses import OrdinalWeightedCELoss


class TestTrainPhase2Pipeline(unittest.TestCase):
    """Test suite for Phase 2 training routines and safety invariants."""

    def setUp(self) -> None:
        """Set up test environment and lightweight un-pretrained model."""
        with open(os.path.join(PROJECT_ROOT, 'config.yaml')) as config_file:
            self.cfg_all = yaml.safe_load(config_file)
        self.manifest_path = os.path.join(
            PROJECT_ROOT,
            self.cfg_all['phase0_outputs']['preprocessing_manifest'],
        )
        self.model = build_vit_tiny(num_classes=3, pretrained=False, dropout=0.3)

    def test_01_compute_3class_weights(self) -> None:
        """Verify 3-class weights computed from manifest match expected counts and formula."""
        weights = compute_3class_weights(self.manifest_path)
        self.assertIsInstance(weights, torch.Tensor)
        self.assertEqual(weights.shape, (3,))
        self.assertTrue(torch.isfinite(weights).all())

        # Expected counts: Healthy: 62, Low: 60, High: 110. Total: 232.
        # w_c = N_total / (n_classes * N_c)
        expected_w0 = 232.0 / (3.0 * 62.0)
        expected_w1 = 232.0 / (3.0 * 60.0)
        expected_w2 = 232.0 / (3.0 * 110.0)

        self.assertAlmostEqual(weights[0].item(), expected_w0, places=4)
        self.assertAlmostEqual(weights[1].item(), expected_w1, places=4)
        self.assertAlmostEqual(weights[2].item(), expected_w2, places=4)

    def test_02_restore_stage_modes_stage_a(self) -> None:
        """Verify restore_stage_modes correctly sets Stage A modes after model.eval()."""
        # First put entire model in eval (simulating post-validation state)
        self.model.eval()

        restore_stage_modes(self.model, stage='A')
        # Tripwire should pass without raising
        assert_stage_a_modes(self.model)

        self.assertFalse(self.model.backbone.training)
        self.assertTrue(self.model.head.training)

    def test_03_restore_stage_modes_stage_b(self) -> None:
        """Verify restore_stage_modes correctly sets Stage B modes after model.eval()."""
        # Put entire model in eval (simulating post-validation state)
        self.model.eval()

        restore_stage_modes(self.model, stage='B', unfreeze_blocks=2)
        # Tripwire should pass without raising
        assert_stage_b_modes(self.model, unfreeze_blocks=2)

        self.assertFalse(self.model.backbone.blocks[0].training)
        self.assertFalse(self.model.backbone.blocks[9].training)
        self.assertTrue(self.model.backbone.blocks[10].training)
        self.assertTrue(self.model.backbone.blocks[11].training)
        self.assertTrue(self.model.backbone.norm.training)
        self.assertTrue(self.model.head.training)

    def test_04_worker_init_fn(self) -> None:
        """Verify _worker_init_fn executes without error."""
        try:
            _worker_init_fn(0)
            _worker_init_fn(1)
        except Exception as error:
            self.fail(f"_worker_init_fn raised unexpected exception: {error}")

    def test_05_train_one_epoch_and_evaluate_smoke(self) -> None:
        """Smoke test for train_one_epoch and evaluate on synthetic DataLoader."""
        from src.train_phase2 import DEVICE, USE_AMP
        model = build_vit_tiny(num_classes=3, pretrained=False, dropout=0.0).to(DEVICE)

        # Synthetic dataset: 4 samples of shape (3, 224, 112)
        images_batch = torch.randn(4, 3, 224, 112)
        targets_batch = torch.tensor([0, 1, 2, 1], dtype=torch.long)
        dataset = TensorDataset(images_batch, targets_batch)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(DEVICE)
        criterion = OrdinalWeightedCELoss(class_weights=weights, under_penalty=2.0, over_penalty=1.0)
        criterion_plain = nn.CrossEntropyLoss()
        scaler = GradScaler(enabled=USE_AMP)

        # Test Stage A step
        restore_stage_modes(model, stage='A')
        assert_stage_a_modes(model)
        optimizer_a = torch.optim.Adam(
            filter(lambda param: param.requires_grad, model.parameters()),
            lr=1e-3,
        )
        ord_loss, plain_loss, acc = train_one_epoch(
            model, loader, criterion, criterion_plain, optimizer_a, scaler, stage='A'
        )
        self.assertTrue(ord_loss > 0)
        self.assertTrue(plain_loss > 0)
        self.assertTrue(0.0 <= acc <= 1.0)

        # Test evaluate
        val_loss, val_acc, val_wf1, val_mf1, low_rec, high_rec = evaluate(
            model, loader, criterion_plain
        )
        self.assertTrue(val_loss > 0)
        self.assertTrue(0.0 <= val_acc <= 1.0)
        self.assertTrue(0.0 <= val_wf1 <= 1.0)
        self.assertTrue(0.0 <= val_mf1 <= 1.0)
        self.assertTrue(0.0 <= low_rec <= 100.0)
        self.assertTrue(0.0 <= high_rec <= 100.0)

        # Test Stage B step
        restore_stage_modes(model, stage='B', unfreeze_blocks=2)
        assert_stage_b_modes(model, unfreeze_blocks=2)
        optimizer_b = torch.optim.Adam(
            filter(lambda param: param.requires_grad, model.parameters()),
            lr=1e-4,
        )
        ord_loss_b, plain_loss_b, acc_b = train_one_epoch(
            model, loader, criterion, criterion_plain, optimizer_b, scaler, stage='B'
        )
        self.assertTrue(ord_loss_b > 0)
        self.assertTrue(plain_loss_b > 0)
        self.assertTrue(0.0 <= acc_b <= 1.0)

    def test_06_non_finite_loss_raises_runtime_error(self) -> None:
        """Verify train_one_epoch raises RuntimeError on NaN or Inf loss."""
        from src.train_phase2 import DEVICE, USE_AMP
        model = build_vit_tiny(num_classes=3, pretrained=False, dropout=0.0).to(DEVICE)

        class NanLoss(nn.Module):
            def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                return torch.tensor(float('nan'), device=logits.device, requires_grad=True)

        images_batch = torch.randn(2, 3, 224, 112)
        targets_batch = torch.tensor([0, 1], dtype=torch.long)
        loader = DataLoader(TensorDataset(images_batch, targets_batch), batch_size=2)

        nan_criterion = NanLoss()
        criterion_plain = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
        scaler = GradScaler(enabled=USE_AMP)

        restore_stage_modes(model, stage='A')
        with self.assertRaises(RuntimeError) as context:
            train_one_epoch(
                model, loader, nan_criterion, criterion_plain, optimizer, scaler, stage='A'
            )
        self.assertIn("Non-finite loss encountered", str(context.exception))


if __name__ == '__main__':
    unittest.main()
