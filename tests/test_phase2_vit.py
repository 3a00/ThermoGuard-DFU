"""Pre-flight verification test suite for Phase 2 ViT-Tiny architecture.

Validates:
  1. Forward pass input/output tensor shapes and numerical stability (no NaN/Inf).
  2. Stage A parameter gradient isolation (head only).
  3. Stage B parameter gradient isolation (top-2 blocks, norm, head).
  4. Peak CUDA VRAM consumption under FP16 mixed precision (< 3.5 GB on GTX 1650).
  5. Stage A tripwire mode assertion against model.train() footgun.
  6. Stage B tripwire mode assertion against model.train() footgun.
"""

import logging
import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

from src.models.vit_tiny import (
    ViTTinyThermal,
    build_vit_tiny,
    freeze_backbone,
    unfreeze_stage_b,
    assert_stage_a_modes,
    assert_stage_b_modes,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)


class TestViTTinyThermal(unittest.TestCase):
    """Pre-flight verification suite for ViTTinyThermal architecture and contracts."""

    def setUp(self) -> None:
        """Initialize lightweight un-pretrained model and seed for deterministic testing."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.model = build_vit_tiny(num_classes=3, pretrained=False, dropout=0.3)

    def test_01_forward_shape_and_stability(self) -> None:
        """Assert input [16, 1, 224, 112] produces output shape [16, 3] with no NaN or Inf."""
        x = torch.randn(16, 1, 224, 112)
        logits = self.model(x)

        self.assertEqual(logits.shape, (16, 3), f"Expected shape (16, 3), got {logits.shape}")
        self.assertFalse(torch.isnan(logits).any(), "Logits contain NaN values")
        self.assertFalse(torch.isinf(logits).any(), "Logits contain Inf values")

        # Also verify 3-channel input compatibility
        x_3ch = torch.randn(2, 3, 224, 112)
        logits_3ch = self.model(x_3ch)
        self.assertEqual(logits_3ch.shape, (2, 3))

    def test_02_stage_a_gradient_isolation(self) -> None:
        """Assert Stage A mode gradients exist only for head, zero for backbone."""
        freeze_backbone(self.model)
        self.model.zero_grad()

        x = torch.randn(16, 1, 224, 112)
        logits = self.model(x)
        loss = logits.sum()
        loss.backward()

        # Backbone parameters must have no gradients (requires_grad is False)
        for name, param in self.model.backbone.named_parameters():
            self.assertIsNone(
                param.grad,
                f"Backbone parameter '{name}' received gradients in Stage A.",
            )

        # Head parameters must have valid non-zero gradients
        head_grad_found = False
        for name, param in self.model.head.named_parameters():
            self.assertIsNotNone(
                param.grad,
                f"Head parameter '{name}' did not receive gradients in Stage A.",
            )
            if torch.count_nonzero(param.grad) > 0:
                head_grad_found = True
        self.assertTrue(head_grad_found, "Head gradients are all zero in Stage A.")

    def test_03_stage_b_gradient_isolation(self) -> None:
        """Assert Stage B mode gradients exist for blocks[10:12], norm, head; zero for patch_embed, blocks[0:10]."""
        unfreeze_stage_b(self.model, unfreeze_blocks=2)
        self.model.zero_grad()

        x = torch.randn(16, 1, 224, 112)
        logits = self.model(x)
        loss = logits.sum()
        loss.backward()

        # Patch embed must have no gradients
        for name, param in self.model.backbone.patch_embed.named_parameters():
            self.assertIsNone(
                param.grad,
                f"patch_embed parameter '{name}' received gradients in Stage B.",
            )

        # Early blocks (0-9) must have no gradients
        for idx in range(10):
            block = self.model.backbone.blocks[idx]
            for name, param in block.named_parameters():
                self.assertIsNone(
                    param.grad,
                    f"Block {idx} parameter '{name}' received gradients in Stage B.",
                )

        # Unfrozen blocks (10-11) must have gradients
        for idx in range(10, 12):
            block = self.model.backbone.blocks[idx]
            block_grad_found = False
            for name, param in block.named_parameters():
                self.assertIsNotNone(
                    param.grad,
                    f"Block {idx} parameter '{name}' missing gradients in Stage B.",
                )
                if torch.count_nonzero(param.grad) > 0:
                    block_grad_found = True
            self.assertTrue(block_grad_found, f"Block {idx} gradients are all zero in Stage B.")

        # Norm must have gradients
        if hasattr(self.model.backbone, 'norm') and self.model.backbone.norm is not None:
            norm_grad_found = False
            for name, param in self.model.backbone.norm.named_parameters():
                self.assertIsNotNone(
                    param.grad,
                    f"Norm parameter '{name}' missing gradients in Stage B.",
                )
                if torch.count_nonzero(param.grad) > 0:
                    norm_grad_found = True
            self.assertTrue(norm_grad_found, "Norm gradients are all zero in Stage B.")

        # Head must have gradients
        head_grad_found = False
        for name, param in self.model.head.named_parameters():
            self.assertIsNotNone(
                param.grad,
                f"Head parameter '{name}' missing gradients in Stage B.",
            )
            if torch.count_nonzero(param.grad) > 0:
                head_grad_found = True
        self.assertTrue(head_grad_found, "Head gradients are all zero in Stage B.")

    def test_04_peak_cuda_memory(self) -> None:
        """Assert peak CUDA memory under FP16 autocast stays below 3.5GB threshold."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available on this host environment.")

        device = torch.device('cuda')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        cuda_model = build_vit_tiny(num_classes=3, pretrained=False, dropout=0.3).to(device)
        unfreeze_stage_b(cuda_model, unfreeze_blocks=2)

        optimizer = torch.optim.Adam(
            [p for p in cuda_model.parameters() if p.requires_grad],
            lr=1e-4,
        )
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler('cuda')

        x = torch.randn(16, 1, 224, 112, device=device)
        targets = torch.randint(0, 3, (16,), device=device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            out = cuda_model(x)
            loss = criterion(out, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        peak_bytes = torch.cuda.max_memory_allocated(device=device)
        peak_gb = peak_bytes / (1024 ** 3)
        logging.info("Peak CUDA memory allocated during forward + backward: %.3f GB", peak_gb)

        # Clean up GPU memory
        del cuda_model, x, targets, out, loss, optimizer, scaler
        torch.cuda.empty_cache()

        vram_threshold_gb = 3.5
        self.assertLess(
            peak_gb,
            vram_threshold_gb,
            f"Peak VRAM usage ({peak_gb:.3f} GB) exceeded safety threshold of {vram_threshold_gb} GB.",
        )

    def test_05_stage_a_train_footgun_tripwire(self) -> None:
        """Assert assert_stage_a_modes raises RuntimeError when model.train() overrides freeze."""
        freeze_backbone(self.model)
        # Should pass without error
        assert_stage_a_modes(self.model)

        # Simulate footgun: direct call to model.train()
        self.model.train()

        with self.assertRaises(RuntimeError) as ctx:
            assert_stage_a_modes(self.model)
        self.assertIn("Stage A mode violation", str(ctx.exception))

    def test_06_stage_b_train_footgun_tripwire(self) -> None:
        """Assert assert_stage_b_modes raises RuntimeError when model.train() overrides freeze."""
        unfreeze_stage_b(self.model, unfreeze_blocks=2)
        # Should pass without error
        assert_stage_b_modes(self.model, unfreeze_blocks=2)

        # Simulate footgun: direct call to model.train()
        self.model.train()

        with self.assertRaises(RuntimeError) as ctx:
            assert_stage_b_modes(self.model, unfreeze_blocks=2)
        self.assertIn("Stage B mode violation", str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
