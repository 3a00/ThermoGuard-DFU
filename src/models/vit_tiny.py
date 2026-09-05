"""ViT-Tiny model architecture and transfer learning helpers for Phase 2.

Implements ViTTinyThermal, wrapping timm's vit_tiny_patch16_224 for plantar
thermogram severity classification (3 classes: Healthy, Low_Severity, High_Severity).

Key architectural details:
  - Input: accepts [B, 1, 224, 112] or [B, 3, 224, 112] (2:1 aspect ratio).
  - Padding: internally applies patch-aligned zero-padding (48 px left, 64 px right),
    yielding [B, C, 224, 224]. Because 48 = 3x16, 112 = 7x16, and 64 = 4x16, all
    14 patch columns align exactly to patch boundaries with zero mixed-boundary patches.
  - Channel expansion: broadcasts single-channel input to 3 channels to leverage
    pretrained ImageNet patch projection weights.
  - Head: Dropout(0.3) -> Linear(192, 3).
  - Transfer learning:
      * Stage A: freeze_backbone() freezes patch embed and all 12 blocks, setting
        backbone to eval() mode. Note: timm's default vit_tiny_patch16_224 ships with
        drop_path_rate=0.0, pos_drop.p=0.0, and all attn/proj drops at 0.0, so the
        .eval() call is defensive insurance rather than an active fix — kept in place
        so a future drop_path_rate>0 variant works without code changes.
      * Stage B: unfreeze_stage_b() unfreezes blocks 10, 11, final norm, and head,
        setting only these active modules to train() mode while keeping early blocks
        in eval() mode.

CRITICAL TRAINING LOOP CONTRACT:
  Do NOT call model.train() in the training loop. PyTorch's model.train() recursively
  sets training=True on ALL submodules, silently undoing the selective eval() freeze
  on blocks 0-9 every single epoch with no error or warning.

  Correct pattern — call the stage helper once per stage transition, never call
  model.train() directly:

      # Stage A setup (once, before Stage A loop):
      freeze_backbone(model)
      for epoch in range(stage_a_epochs):
          # NO model.train() call here
          assert_stage_a_modes(model)   # tripwire: raises if invariant is broken
          for batch in train_loader: ...

      # Stage B setup (once, before Stage B loop):
      unfreeze_stage_b(model)
      for epoch in range(stage_b_epochs):
          # NO model.train() call here
          assert_stage_b_modes(model)   # tripwire: raises if invariant is broken
          for batch in train_loader: ...

  The assert_stage_*_modes() helpers are provided in this module.
"""


import logging
from typing import Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)


class ViTTinyThermal(nn.Module):
    """Vision Transformer Tiny adapted for plantar thermogram severity classification."""

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
        pad_width: Tuple[int, int] = (48, 64),
    ) -> None:
        """Initialize ViTTinyThermal.

        Args:
            num_classes: Number of target severity grades (default 3).
            pretrained: Whether to load ImageNet-1k pretrained weights from timm.
            dropout: Dropout probability preceding the final linear classifier.
            pad_width: (left_pad, right_pad) horizontal zero padding to reach 224 width.
                       Default (48, 64) aligns exactly to 16px patch boundaries.
        """
        super().__init__()
        self.num_classes = num_classes
        self.pad_left, self.pad_right = pad_width

        # Create timm ViT-Tiny backbone
        self.backbone = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=pretrained,
            num_classes=0,  # Remove original classifier head, outputs pooled 192-dim representation
        )

        embed_dim = self.backbone.num_features  # 192 for vit_tiny_patch16_224

        # Custom regularized classification head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, num_classes),
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(
            'ViTTinyThermal initialized (pretrained=%s, num_classes=%d, pad=%s). '
            'Total params: %d, Trainable: %d',
            pretrained, num_classes, pad_width, total_params, trainable_params,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Plantar thermogram tensor of shape [B, 1, 224, 112] or [B, 3, 224, 112].

        Returns:
            Logits of shape [B, num_classes].
        """
        # 1. Expand single-channel input to 3 channels if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"Expected input with 1 or 3 channels, got {x.shape[1]}")

        # 2. Patch-aligned horizontal padding from 112 to 224
        # F.pad format: (pad_left, pad_right, pad_top, pad_bottom)
        if self.pad_left > 0 or self.pad_right > 0:
            x = F.pad(x, (self.pad_left, self.pad_right, 0, 0), mode='constant', value=0.0)

        # 3. Backbone forward features (patch embedding, transformer blocks, norm, pooling)
        # timm's backbone with num_classes=0 returns the pooled embedding (B, 192)
        features = self.backbone(x)

        # 4. Classification head
        logits = self.head(features)
        return logits


def freeze_backbone(model: ViTTinyThermal) -> None:
    """Stage A: Freeze all backbone parameters and set backbone to eval mode.

    Setting backbone to eval() mode disables any internal dropout or stochastic
    depth modules, ensuring fixed representation for the trainable head.
    """
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.backbone.eval()

    for param in model.head.parameters():
        param.requires_grad = True
    model.head.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('Stage A: Backbone frozen in eval() mode. Trainable parameters: %d', trainable)


def unfreeze_stage_b(model: ViTTinyThermal, unfreeze_blocks: int = 2) -> None:
    """Stage B: Unfreeze the top N transformer blocks, final norm, and head.

    Keeps patch embeddings and blocks 0 to (total-N-1) frozen and in eval() mode.
    Sets unfrozen blocks, norm, and head to train() mode with requires_grad=True.

    Args:
        model: ViTTinyThermal instance.
        unfreeze_blocks: Number of final transformer blocks to unfreeze (default: 2, blocks 10 and 11).
    """
    # 1. Freeze all backbone weights and set entire backbone to eval() mode
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.backbone.eval()

    # 2. Unfreeze top N blocks and put them in train() mode
    total_blocks = len(model.backbone.blocks)  # 12 blocks (0 to 11)
    start_block = total_blocks - unfreeze_blocks
    for i in range(start_block, total_blocks):
        block = model.backbone.blocks[i]
        for param in block.parameters():
            param.requires_grad = True
        block.train()

    # 3. Unfreeze final LayerNorm if present and put in train() mode
    if hasattr(model.backbone, 'norm') and model.backbone.norm is not None:
        for param in model.backbone.norm.parameters():
            param.requires_grad = True
        model.backbone.norm.train()

    # 4. Ensure head remains trainable and in train() mode
    for param in model.head.parameters():
        param.requires_grad = True
    model.head.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(
        'Stage B: Unfroze top %d blocks (blocks %d-%d), norm, and head. '
        'Trainable parameters: %d',
        unfreeze_blocks, start_block, total_blocks - 1, trainable,
    )


def assert_stage_a_modes(model: ViTTinyThermal) -> None:
    """Tripwire: raise RuntimeError if Stage A training mode invariants are violated.

    Call at the top of every Stage A epoch. Guards against model.train() accidentally
    resetting backbone submodules back to train() mode, which silently defeats the
    eval() freeze without any PyTorch error or warning.

    Raises:
        RuntimeError: If the backbone is in train() mode or head is in eval() mode.
    """
    if model.backbone.training:
        raise RuntimeError(
            "Stage A mode violation: model.backbone is in train() mode. "
            "A call to model.train() has overridden the freeze set by freeze_backbone(). "
            "Do not call model.train() in the training loop — use freeze_backbone() exclusively."
        )
    if not model.head.training:
        raise RuntimeError(
            "Stage A mode violation: model.head is in eval() mode. "
            "Call freeze_backbone() to restore correct Stage A modes."
        )


def assert_stage_b_modes(model: ViTTinyThermal, unfreeze_blocks: int = 2) -> None:
    """Tripwire: raise RuntimeError if Stage B training mode invariants are violated.

    Call at the top of every Stage B epoch. Guards against model.train() accidentally
    resetting early frozen blocks back to train() mode.

    Args:
        model: ViTTinyThermal instance.
        unfreeze_blocks: Number of top blocks that should be in train() mode (default: 2).

    Raises:
        RuntimeError: If any frozen block is in train() mode, or any active block is in eval() mode.
    """
    total_blocks = len(model.backbone.blocks)
    start_block = total_blocks - unfreeze_blocks

    # Check early (frozen) blocks are all in eval mode
    for i in range(start_block):
        if model.backbone.blocks[i].training:
            raise RuntimeError(
                f"Stage B mode violation: backbone.blocks[{i}] is in train() mode. "
                f"A call to model.train() has overridden the selective freeze. "
                f"Do not call model.train() in the training loop — use unfreeze_stage_b() exclusively."
            )

    # Check active (unfrozen) blocks are in train mode
    for i in range(start_block, total_blocks):
        if not model.backbone.blocks[i].training:
            raise RuntimeError(
                f"Stage B mode violation: backbone.blocks[{i}] is in eval() mode. "
                f"Call unfreeze_stage_b() to restore correct Stage B modes."
            )

    if not model.head.training:
        raise RuntimeError(
            "Stage B mode violation: model.head is in eval() mode. "
            "Call unfreeze_stage_b() to restore correct Stage B modes."
        )


def build_vit_tiny(
    num_classes: int = 3,
    pretrained: bool = True,
    dropout: float = 0.3,
    pad_width: Tuple[int, int] = (48, 64),
) -> ViTTinyThermal:
    """Factory helper to build ViTTinyThermal."""
    return ViTTinyThermal(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        pad_width=pad_width,
    )


if __name__ == '__main__':
    logging.info('--- Running ViTTinyThermal Verification ---')

    # Test 1: Instantiation with pretrained=False
    model = build_vit_tiny(num_classes=3, pretrained=False, dropout=0.3, pad_width=(48, 64))
    assert isinstance(model, ViTTinyThermal)

    # Test 2: Forward pass with 1-channel tensor [4, 1, 224, 112]
    x_1ch = torch.randn(4, 1, 224, 112)
    out_1ch = model(x_1ch)
    assert out_1ch.shape == (4, 3), f"Expected shape (4, 3), got {out_1ch.shape}"
    assert not torch.isnan(out_1ch).any(), "Output contains NaN"

    # Test 3: Forward pass with 3-channel tensor [4, 3, 224, 112]
    x_3ch = torch.randn(4, 3, 224, 112)
    out_3ch = model(x_3ch)
    assert out_3ch.shape == (4, 3), f"Expected shape (4, 3), got {out_3ch.shape}"

    # Test 4: Stage A freeze verification
    freeze_backbone(model)
    assert model.backbone.training is False, "Backbone should be in eval() mode in Stage A"
    assert model.head.training is True, "Head should be in train() mode in Stage A"
    for name, param in model.backbone.named_parameters():
        assert not param.requires_grad, f"Backbone param {name} should be frozen in Stage A"
    for name, param in model.head.named_parameters():
        assert param.requires_grad, f"Head param {name} should be trainable in Stage A"

    # Forward + backward in Stage A
    out = model(x_1ch)
    loss = out.sum()
    loss.backward()
    for name, param in model.backbone.named_parameters():
        assert param.grad is None, f"Backbone param {name} should have None grad in Stage A"
    for name, param in model.head.named_parameters():
        assert param.grad is not None, f"Head param {name} should have grad in Stage A"

    # Test 5: Stage B unfreeze verification
    model.zero_grad()
    unfreeze_stage_b(model, unfreeze_blocks=2)
    assert model.backbone.blocks[0].training is False, "Block 0 should be in eval() mode in Stage B"
    assert model.backbone.blocks[9].training is False, "Block 9 should be in eval() mode in Stage B"
    assert model.backbone.blocks[10].training is True, "Block 10 should be in train() mode in Stage B"
    assert model.backbone.blocks[11].training is True, "Block 11 should be in train() mode in Stage B"
    assert model.backbone.norm.training is True, "Norm should be in train() mode in Stage B"
    assert model.head.training is True, "Head should be in train() mode in Stage B"

    for name, param in model.backbone.patch_embed.named_parameters():
        assert not param.requires_grad, f"patch_embed {name} should be frozen in Stage B"
    for idx in range(10):
        for name, param in model.backbone.blocks[idx].named_parameters():
            assert not param.requires_grad, f"block {idx} {name} should be frozen in Stage B"
    for idx in range(10, 12):
        for name, param in model.backbone.blocks[idx].named_parameters():
            assert param.requires_grad, f"block {idx} {name} should be trainable in Stage B"
    for name, param in model.backbone.norm.named_parameters():
        assert param.requires_grad, f"norm {name} should be trainable in Stage B"

    # Forward + backward in Stage B
    out = model(x_1ch)
    loss = out.sum()
    loss.backward()
    for name, param in model.backbone.blocks[0].named_parameters():
        assert param.grad is None, f"Block 0 {name} should have None grad in Stage B"
    for name, param in model.backbone.blocks[10].named_parameters():
        assert param.grad is not None, f"Block 10 {name} should have grad in Stage B"
    for name, param in model.head.named_parameters():
        assert param.grad is not None, f"Head {name} should have grad in Stage B"

    logging.info('All ViTTinyThermal verification checks passed successfully!')
