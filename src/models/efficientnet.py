"""EfficientNet-B0 model factories for Phase 1.0, Phase 1.1, and Phase 1.2.

Import from any script:
    from models.efficientnet import build_efficientnet_b0, freeze_backbone, unfreeze_all
    from models.efficientnet import build_efficientnet_b0_v2               # Phase 1.1
    from models.efficientnet import freeze_all_backbone, unfreeze_top_block_only  # Phase 1.1
    from models.efficientnet import build_efficientnet_b0_v3               # Phase 1.2
    from models.efficientnet import unfreeze_top_two_blocks                # Phase 1.2 ablation

Phase 1.0 factory  (build_efficientnet_b0):
  - Backbone: EfficientNet-B0 (5.3M params), ImageNet-pretrained.
  - Pool:   AdaptiveAvgPool2d(1) -> 1280-dim.
  - Head:   Dropout(0.2) -> Linear(1280, num_classes).
  - Freeze helpers: freeze_backbone(), unfreeze_all().

Phase 1.1 factory  (build_efficientnet_b0_v2):
  - Backbone: same EfficientNet-B0 features[0..8], ImageNet-pretrained.
  - Pool:   ConcatPool2d (GAP + GMP) -> 2560-dim.
    * Captures both mean temperature (GAP) and peak ulcer hotspot (GMP).
  - Head:   Dropout(p1) -> Linear(2560, bottleneck) -> BN1d -> ReLU
            -> Dropout(p2) -> Linear(bottleneck, num_classes).
    * All head params from config.yaml['phase1_1'].
  - Freeze helpers: freeze_all_backbone(), unfreeze_top_block_only().
    Stage A: freeze_all_backbone() + model.features.eval() in loop.
    Stage B: unfreeze_top_block_only() + model.features.eval() +
             model.features[7].train() in loop.
    Note: features[8] (Top Conv 320->1280) stays frozen in Stage B --
    isolating variables: this run already changes resize, pooling, and head;
    features[8] unfreeze is reserved for a dedicated ablation if needed.

Phase 1.2 factory  (build_efficientnet_b0_v3):
  - Backbone: same EfficientNet-B0 features[0..8], ImageNet-pretrained.
  - Pool:   SpatialConcatPool2d (2x2 quadrant GAP+GMP) -> 10240-dim.
    * Splits the 7x4 feature map into 4 quadrants matching the 4 angiosome
      regions (LCA/MCA forefoot, LPA/MPA rearfoot). Phase 0.7 horizontal flip
      ensures consistent anatomical alignment across all arrays.
    * A runtime assertion in forward() confirms the expected (7, 4) shape.
    * Dimension arithmetic: 1280 channels x 4 quadrant-values (from (2,2) pool)
      x 2 (GAP + GMP) = 10240.
  - Head:   Dropout(p1) -> Linear(10240, bottleneck) -> BN1d -> ReLU
            -> Dropout(p2) -> Linear(bottleneck, num_classes).
    * All head params from config.yaml['phase1_2'].
  - Freeze helpers: freeze_all_backbone() (shared), unfreeze_top_block_only() (shared).
    Optional ablation: unfreeze_top_two_blocks() unfreezes features[7] AND features[8].
    Stage A: freeze_all_backbone() + model.features.eval() in loop.
    Stage B (default): unfreeze_top_block_only() + model.features.eval() +
                       model.features[7].train() in loop.
    Stage B (ablation): unfreeze_top_two_blocks() + model.features.eval() +
                        model.features[7].train() + model.features[8].train() in loop.

Input for all: (3, 224, 112) -- channel-replicated, resized thermal image.
Final feature map before pool: (B, 1280, 7, 4) -- confirmed by SpatialConcatPool2d assertion.
SpatialConcatPool2d output: 1280 * (2*2) * 2 = 10240 dim  [channels * quadrant_vals * GAP+GMP].
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)


# ---------------------------------------------------------------------------
# Phase 1.0 factory and helpers (unchanged)
# ---------------------------------------------------------------------------

def build_efficientnet_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build EfficientNet-B0 with a single-layer classification head (Phase 1.0).

    Args:
        num_classes: From config.yaml['phase1_0']['num_classes'].
        pretrained:  From config.yaml['phase1_0']['pretrained'].
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, num_classes)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(
        'EfficientNet-B0 v1 (pretrained=%s). Head: Linear(%d, %d). '
        'Params total=%d, trainable=%d.',
        pretrained, in_features, num_classes, total, trainable,
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze backbone parameters for Stage A (Phase 1.0).

    Sets requires_grad=False on 'features'; leaves 'classifier' trainable.
    NOTE: also call model.features.eval() in the training loop to freeze
    BatchNorm running statistics (see train_one_epoch freeze_backbone_bn flag).
    """
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('freeze_backbone: %d trainable params (head only).', trainable)


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all model layers for Stage B end-to-end fine-tuning (Phase 1.0)."""
    for p in model.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('unfreeze_all: %d trainable params.', trainable)


# ---------------------------------------------------------------------------
# Phase 1.1 components
# ---------------------------------------------------------------------------

class ConcatPool2d(nn.Module):
    """Concatenate global average pool and global max pool outputs.

    Replaces EfficientNet-B0's AdaptiveAvgPool2d(1).

    Input:  (B, C, H, W)
    Output: (B, 2*C) -- flattened

    Captures both mean temperature distribution (GAP) and peak hotspot
    temperature (GMP), both clinically relevant for DFU detection.
    No learnable parameters.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate GAP and GMP feature vectors."""
        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)   # (B, C)
        mx  = F.adaptive_max_pool2d(x, 1).flatten(1)   # (B, C)
        return torch.cat([avg, mx], dim=1)              # (B, 2*C)


def build_efficientnet_b0_v2(
    num_classes: int,
    pretrained: bool = True,
    cfg: dict = None,
) -> nn.Module:
    """Build EfficientNet-B0 with ConcatPool2d and 2-layer bottleneck head (Phase 1.1).

    Architecture changes vs build_efficientnet_b0:
      - model.avgpool replaced with ConcatPool2d() -> 2560-dim feature vector.
      - model.classifier replaced with 2-layer bottleneck head.

    Args:
        num_classes: Number of output classes (3 for Phase 1.1).
        pretrained:  Load ImageNet weights for the backbone.
        cfg:         config.yaml['phase1_1'] dict. Reads head_dropout_1,
                     head_dropout_2, head_bottleneck_dim.
    """
    if cfg is None:
        raise ValueError('cfg (config.yaml[phase1_1]) must be provided to build_efficientnet_b0_v2.')

    bottleneck_dim = cfg['head_bottleneck_dim']   # 256
    dropout_1      = cfg['head_dropout_1']         # 0.4
    dropout_2      = cfg['head_dropout_2']         # 0.3

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Replace pooling: AdaptiveAvgPool2d(1) -> ConcatPool2d (1280 -> 2560).
    model.avgpool = ConcatPool2d()

    # Replace classifier: single linear -> 2-layer regularized bottleneck.
    # Input: 2560 (from ConcatPool2d on 1280-channel backbone output).
    in_features = 2560
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_1),
        nn.Linear(in_features, bottleneck_dim),
        nn.BatchNorm1d(bottleneck_dim),
        nn.ReLU(),
        nn.Dropout(p=dropout_2),
        nn.Linear(bottleneck_dim, num_classes),
    )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(
        'EfficientNet-B0 v2 (pretrained=%s). ConcatPool2d -> %d-dim. '
        'Head: Dropout(%.1f) -> Linear(%d, %d) -> BN1d -> ReLU -> Dropout(%.1f) -> Linear(%d, %d). '
        'Params total=%d, trainable=%d.',
        pretrained, in_features,
        dropout_1, in_features, bottleneck_dim,
        dropout_2, bottleneck_dim, num_classes,
        total, trainable,
    )
    return model


# ---------------------------------------------------------------------------
# Phase 1.1 freeze helpers
# ---------------------------------------------------------------------------

def freeze_all_backbone(model: nn.Module) -> None:
    """Stage A: freeze entire backbone (features[0..8]). Train head only.

    Call model.features.eval() in the training loop to also freeze
    BatchNorm running statistics (requires_grad=False alone does not stop
    running mean/variance from drifting in model.train() mode).
    ConcatPool2d has no learnable parameters -- no action needed for it.
    """
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('freeze_all_backbone: %d trainable params (head + ConcatPool2d).', trainable)


def unfreeze_top_block_only(model: nn.Module) -> None:
    """Stage B: unfreeze features[7] only. features[0..6] and features[8] stay frozen.

    features[7] is the final MBConv6 block (192->320 channels), the deepest
    block most sensitive to domain-specific patterns (thermal vs. ImageNet RGB).

    features[8] (Top Conv 320->1280) stays frozen to isolate variables --
    this run already introduces bilinear resize, ConcatPool2d, and a new head.
    If Stage B still shows a large train-val gap, unfreeze features[8] as a
    dedicated ablation in a follow-up run.

    Stage B BN pattern (must be called in train loop after this function):
        model.train()
        model.features.eval()       # freeze ALL backbone BN running stats
        model.features[7].train()   # re-enable only features[7] BN stat updates
    """
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.features[7].parameters():   # final MBConv block: 192->320 channels
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(
        'unfreeze_top_block_only: %d trainable params (features[7] + head).',
        trainable,
    )


# ---------------------------------------------------------------------------
# Phase 1.2 components
# ---------------------------------------------------------------------------

class SpatialConcatPool2d(nn.Module):
    """2x2 spatial quadrant pooling with GAP+GMP concatenation (Phase 1.2).

    Replaces EfficientNet-B0's AdaptiveAvgPool2d(1) and Phase 1.1's ConcatPool2d.
    Splits the backbone feature map into 4 anatomically meaningful quadrants
    (Forefoot-Lateral, Forefoot-Medial, Rearfoot-Lateral, Rearfoot-Medial)
    then concatenates global average and max pool from each quadrant.

    Phase 0.7 horizontally flips all left feet for canonical anatomical alignment,
    so column position consistently maps to medial/lateral and row position maps to
    toe/heel. The 2x2 split therefore mirrors the 4 angiosome regions (LCA/MCA
    forefoot, LPA/MPA rearfoot) that TCI labels are derived from.

    A runtime assertion on the first forward pass confirms the expected (7, 4)
    feature map shape. If EfficientNet-B0's internal padding resolves to width=3
    instead of 4, the assertion fires before training begins.

    Dimension arithmetic:
        adaptive_avg_pool2d(x, (2,2)) -> (B, C, 2, 2) -> flatten(1) -> (B, C*4)
        adaptive_max_pool2d(x, (2,2)) -> (B, C, 2, 2) -> flatten(1) -> (B, C*4)
        cat([avg, mx]) -> (B, C*8)
        With C=1280: output dim = 1280 * 4 * 2 = 10240.

    Input:  (B, C, H, W)   -- e.g. (B, 1280, 7, 4) for EfficientNet-B0 at 224x112
    Output: (B, C * 4 * 2) -- e.g. (B, 10240)
    No learnable parameters.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate 2x2 quadrant GAP and GMP feature vectors."""
        assert x.shape[-2:] == (7, 4), (
            f'SpatialConcatPool2d: expected feature map (7, 4), '
            f'got {tuple(x.shape[-2:])}. '
            f'Check input resolution (224x112) and EfficientNet-B0 backbone stride. '
            f'If width=3, adjust head_in_features and quadrant interpretation in the plan.'
        )
        avg = F.adaptive_avg_pool2d(x, (2, 2)).flatten(1)  # (B, C*4) = (B, 5120)
        mx  = F.adaptive_max_pool2d(x, (2, 2)).flatten(1)  # (B, C*4) = (B, 5120)
        return torch.cat([avg, mx], dim=1)                  # (B, C*8) = (B, 10240)


def build_efficientnet_b0_v3(
    num_classes: int,
    pretrained: bool = True,
    cfg: dict = None,
) -> nn.Module:
    """Build EfficientNet-B0 with SpatialConcatPool2d and 2-layer bottleneck head (Phase 1.2).

    Architecture changes vs build_efficientnet_b0_v2 (Phase 1.1):
      - model.avgpool replaced with SpatialConcatPool2d() -> 5120-dim feature vector.
        (vs ConcatPool2d -> 2560-dim in Phase 1.1)
      - model.classifier input adjusted from 2560 to 5120.

    Args:
        num_classes: Number of output classes (3 for Phase 1.2).
        pretrained:  Load ImageNet weights for the backbone.
        cfg:         config.yaml['phase1_2'] dict. Reads head_dropout_1,
                     head_dropout_2, head_bottleneck_dim.
    """
    if cfg is None:
        raise ValueError('cfg (config.yaml[phase1_2]) must be provided to build_efficientnet_b0_v3.')

    bottleneck_dim = cfg['head_bottleneck_dim']   # 256
    dropout_1      = cfg['head_dropout_1']         # 0.4
    dropout_2      = cfg['head_dropout_2']         # 0.3

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Replace pooling: AdaptiveAvgPool2d(1) -> SpatialConcatPool2d (1280 -> 10240).
    model.avgpool = SpatialConcatPool2d()

    # Replace classifier: single linear -> 2-layer regularised bottleneck.
    # Input: 10240 (from SpatialConcatPool2d: 1280 * 4 quadrant_vals * 2 (GAP+GMP)).
    in_features = 10240
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_1),
        nn.Linear(in_features, bottleneck_dim),
        nn.BatchNorm1d(bottleneck_dim),
        nn.ReLU(),
        nn.Dropout(p=dropout_2),
        nn.Linear(bottleneck_dim, num_classes),
    )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(
        'EfficientNet-B0 v3 (pretrained=%s). SpatialConcatPool2d -> %d-dim. '
        'Head: Dropout(%.1f) -> Linear(%d, %d) -> BN1d -> ReLU -> Dropout(%.1f) -> Linear(%d, %d). '
        'Params total=%d, trainable=%d. '
        '[Dim: 1280ch x 4quad_vals x 2(GAP+GMP) = 10240]',
        pretrained, in_features,
        dropout_1, in_features, bottleneck_dim,
        dropout_2, bottleneck_dim, num_classes,
        total, trainable,
    )
    return model


def unfreeze_top_two_blocks(model: nn.Module) -> None:
    """Stage B ablation: unfreeze features[7] AND features[8]. features[0..6] stay frozen.

    This is the optional Phase 1.2 ablation variant (config: unfreeze_features8: true).
    Default Phase 1.2 run uses unfreeze_top_block_only() to isolate the primary
    changes (SpatialConcatPool2d + OrdinalWeightedCELoss) first.

    Stage B BN pattern for this variant (in the training loop):
        model.train()
        model.features.eval()       # freeze ALL backbone BN running stats
        model.features[7].train()   # re-enable features[7] BN stat updates
        model.features[8].train()   # re-enable features[8] BN stat updates
    """
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.features[7].parameters():
        p.requires_grad = True
    for p in model.features[8].parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(
        'unfreeze_top_two_blocks: %d trainable params (features[7] + features[8] + head).',
        trainable,
    )
