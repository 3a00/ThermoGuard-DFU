"""ThermalDataset -- shared PyTorch Dataset for all training phases.

Import from any script:
    from datasets.thermal_dataset import ThermalDataset, CLASS_NAMES
    from datasets.thermal_dataset import CLASS_NAMES_3, SEVERITY_MAP   # Phase 1.1+
    from datasets.thermal_dataset import build_tta_transforms           # Phase 1.2+

Supports three phase pipelines via the `phase` parameter:

  phase='1_0'  (default):
    - Deterministic NEAREST resize to (resize_h, resize_w) from config['phase1_0'].
    - 6-class labels: Healthy, DM_Grade0, DM_Grade1, DM_Grade2, DM_Grade3, DM_Grade4.
    - On-the-fly augmentation (train split only) using Phase 1.0 config.

  phase='1_1':
    - Binary mask derived from original 128x64 array (arr > 0.0), NEAREST-resized to
      (resize_h, resize_w) -- stays hard/binary, no blending.
    - Bilinear resize of temperature values to (resize_h, resize_w).
    - Mask applied: tensor = tensor * mask_resized (background exactly 0.0).
    - 3-class severity labels via SEVERITY_MAP:
        Healthy -> 0, DM_Grade0/1/2 -> 1 (Low_Severity), DM_Grade3/4 -> 2 (High_Severity)
    - On-the-fly augmentation (train split only) using Phase 1.1 config.

  phase='1_2':
    - Identical preprocessing pipeline to phase='1_1' (bilinear + NEAREST mask).
    - Reads resize_h, resize_w and augmentation params from config['phase1_2'].
    - 3-class severity labels (same SEVERITY_MAP as Phase 1.1).
    - On-the-fly augmentation (train split only) using Phase 1.2 config.

Augmentation notes (all phases):
  - RandomHorizontalFlip: DISABLED. Left feet flipped canonically in Phase 0.7.
  - RandomVerticalFlip:   DISABLED. Toe<->heel inversion anatomically invalid
    for plantar thermography (angiosome regions fixed spatially). Supervisor-locked.
  - Val/test splits: NO augmentation.

Why mask from ORIGINAL array (Phase 1.1+)?
  Thresholding the bilinear-resized tensor at >0.0 is a no-op for halo pixels --
  a boundary pixel blended from foreground and background sources is strictly >0.0
  and passes unchanged. Deriving the mask from the original Phase 0.7 array
  (which already has background pinned to exactly 0.0) and NEAREST-resizing it
  inherits Phase 0.7's correct segmentation without introducing a new fallible mask.

Exports:
  CLASS_NAMES       -- 6-class list, single source of truth for Phase 1.0 scripts.
  CLASS_NAMES_3     -- 3-class list for Phase 1.1+ scripts.
  SEVERITY_MAP      -- dict mapping 6-class labels to 3-class severity indices.
  build_tta_transforms -- TTA transform factory for Phase 1.2+ inference.
"""

import logging
import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
import yaml
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)

# Project root = thermalDFU/ (two levels up from src/datasets/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def project_path(relative: str) -> str:
    """Resolve a project-relative path to an absolute path."""
    return os.path.join(PROJECT_ROOT, relative)


with open(project_path('config.yaml')) as _f:
    _CFG = yaml.safe_load(_f)

# ---------------------------------------------------------------------------
# Class label definitions
# ---------------------------------------------------------------------------

# 6-class: canonical class order matching class_index_map in config.yaml (Phase 1.0).
CLASS_NAMES = ['Healthy', 'DM_Grade0', 'DM_Grade1', 'DM_Grade2', 'DM_Grade3', 'DM_Grade4']
CLASS_INDEX_MAP = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}

# 3-class severity (Phase 1.1+): dynamic mapping applied at load time.
CLASS_NAMES_3 = ['Healthy', 'Low_Severity', 'High_Severity']

SEVERITY_MAP = {
    'Healthy':   0,
    'DM_Grade0': 1, 'DM_Grade1': 1, 'DM_Grade2': 1,
    'DM_Grade3': 2, 'DM_Grade4': 2,
}


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

class GaussianNoise:
    """Zero-mean Gaussian noise, clamped to [0, 1]. std from config.yaml."""

    def __init__(self, std: float) -> None:
        """Initialize with noise standard deviation."""
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise and clamp result to [0, 1]."""
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)


def _build_train_transforms(cfg: dict) -> T.Compose:
    """On-the-fly augmentation pipeline for the training split.

    All values from cfg (a phase config block from config.yaml).
    Applied AFTER the deterministic resize step.

    Excluded transforms (no config entries to prevent accidental re-enabling):
      - RandomHorizontalFlip: left feet already flipped canonically in Phase 0.7.
      - RandomVerticalFlip: toe<->heel inversion is anatomically invalid for plantar
        thermography (angiosome regions have fixed spatial meaning). Supervisor-locked.
    """
    return T.Compose([
        T.RandomRotation(
            degrees=cfg['augment_rotation_deg'],
            interpolation=InterpolationMode.NEAREST,
        ),
        T.RandomAffine(
            degrees=0,
            translate=(cfg['augment_translate'], cfg['augment_translate']),
            scale=(cfg['augment_scale_min'], cfg['augment_scale_max']),
            interpolation=InterpolationMode.NEAREST,
        ),
        T.RandomErasing(
            p=cfg['augment_erasing_p'],
            scale=(cfg['augment_erasing_scale_min'], cfg['augment_erasing_scale_max']),
            ratio=(cfg['augment_erasing_ratio_min'], cfg['augment_erasing_ratio_max']),
            value=cfg['augment_erasing_value'],
        ),
        GaussianNoise(std=cfg['augment_noise_std']),
    ])


def build_tta_transforms(cfg: dict) -> T.Compose:
    """Conservative augmentation pipeline for test-time augmentation (Phase 1.2+).

    Uses half the training augmentation range to stay within the anatomically
    feasible envelope for canonical toe-up foot images. Stochastic noise and
    random erasing are excluded (they destroy information at test time without
    helping estimate the decision boundary).

    Args:
        cfg: config.yaml['phase1_2'] (or any phase with tta_* keys).

    Returns:
        T.Compose pipeline applied once per augmented view.
    """
    return T.Compose([
        T.RandomRotation(
            degrees=cfg['tta_rotation_deg'],
            interpolation=InterpolationMode.NEAREST,
        ),
        T.RandomAffine(
            degrees=0,
            translate=(cfg['tta_translate'], cfg['tta_translate']),
            scale=(cfg['tta_scale_min'], cfg['tta_scale_max']),
            interpolation=InterpolationMode.NEAREST,
        ),
    ])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class ThermalDataset(Dataset):
    """PyTorch Dataset for preprocessed thermal foot images.

    Args:
        split:         One of 'train', 'val', 'test'.
        manifest_path: Path to preprocessing_manifest.csv (Phase 0.7, read-only).
        augment:       If True, apply on-the-fly augmentations. Forced False for val/test.
        phase:         '1_0' (default) or '1_1'. Controls resize mode and label mapping.
                       '1_0': NEAREST resize, 6-class labels.
                       '1_1': Bilinear resize + NEAREST mask from original, 3-class labels.
    """

    def __init__(
        self,
        split: str,
        manifest_path: str,
        augment: bool = False,
        phase: str = '1_0',
    ) -> None:
        """Load and filter manifest rows for the specified split."""
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'.")
        if phase not in ('1_0', '1_1', '1_2', '2'):
            raise ValueError(f"phase must be '1_0', '1_1', '1_2', or '2', got '{phase}'.")
        self.split = split
        self.phase = phase
        self.augment = augment and (split == 'train')
        df = pd.read_csv(manifest_path)
        self.df = df[df['split'] == split].reset_index(drop=True)
        cfg_key = {'1_0': 'phase1_0', '1_1': 'phase1_1', '1_2': 'phase1_2', '2': 'phase2'}[phase]
        cfg = _CFG[cfg_key]
        self.resize_h: int = cfg['resize_h']
        self.resize_w: int = cfg['resize_w']
        self.transforms = _build_train_transforms(cfg) if self.augment else None
        logging.info(
            'ThermalDataset [%s] phase=%s: %d samples, augment=%s, resize=(%d, %d)',
            split, phase, len(self.df), self.augment, self.resize_h, self.resize_w,
        )

    def __len__(self) -> int:
        """Return number of samples in this split."""
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load array, resize, replicate channels, optionally augment.

        Phase 1.0 pipeline:
            Load -> NEAREST resize -> 3-channel replicate -> augment -> 6-class label.

        Phase 1.1 pipeline:
            Load -> NEAREST mask from original -> Bilinear resize -> apply mask ->
            3-channel replicate -> augment -> 3-class severity label.

        Returns:
            tensor: (3, resize_h, resize_w) float32 in [0, 1].
            label:  Integer class index (0-5 for phase 1_0, 0-2 for phase 1_1).
        """
        row = self.df.iloc[idx]
        arr = np.load(project_path(row['array_output']))  # (128, 64) float32

        if self.phase == '1_0':
            tensor = torch.from_numpy(arr.copy()).unsqueeze(0)  # (1, 128, 64)
            tensor = TF.resize(
                tensor,
                size=[self.resize_h, self.resize_w],
                interpolation=InterpolationMode.NEAREST,
            )  # (1, resize_h, resize_w)
            label = CLASS_INDEX_MAP[row['model_class']]

        else:  # phase == '1_1' or '1_2' -- identical preprocessing pipeline
            # Step 1: Binary mask from ORIGINAL array (arr > 0.0), NEAREST-resized.
            # Phase 0.7 pinned background to exactly 0.0 -- inheriting that segmentation.
            # NEAREST keeps the mask hard/binary (no blending).
            mask_orig = torch.from_numpy(
                (arr > 0.0).astype(np.float32)
            ).unsqueeze(0)  # (1, 128, 64) -- binary {0.0, 1.0}
            mask_resized = TF.resize(
                mask_orig,
                size=[self.resize_h, self.resize_w],
                interpolation=InterpolationMode.NEAREST,
            )  # (1, resize_h, resize_w) -- hard binary, no blending

            # Step 2: Bilinear resize temperature values -- smooth interior gradients.
            tensor = torch.from_numpy(arr.copy()).unsqueeze(0)  # (1, 128, 64)
            tensor = TF.resize(
                tensor,
                size=[self.resize_h, self.resize_w],
                interpolation=InterpolationMode.BILINEAR,
            )  # (1, resize_h, resize_w)

            # Step 3: Apply hard mask -- eliminates halo bleed from bilinear interpolation.
            tensor = tensor * mask_resized  # background pixels exactly 0.0

            # Step 4: Dynamic 3-class severity label (Phase 0 files untouched).
            label = SEVERITY_MAP[row['model_class']]

        tensor = tensor.repeat(3, 1, 1)  # (3, resize_h, resize_w)

        if self.transforms is not None:
            tensor = self.transforms(tensor)

        return tensor, label


if __name__ == '__main__':
    manifest = project_path(_CFG['phase0_outputs']['preprocessing_manifest'])
    logging.info('--- Phase 1.0 pipeline ---')
    for split in ('train', 'val', 'test'):
        ds = ThermalDataset(split, manifest, augment=(split == 'train'), phase='1_0')
        img, lbl = ds[0]
        logging.info('  [%s] shape=%s  label=%d (%s)', split, img.shape, lbl, CLASS_NAMES[lbl])
    logging.info('--- Phase 1.1 pipeline ---')
    for split in ('train', 'val', 'test'):
        ds = ThermalDataset(split, manifest, augment=(split == 'train'), phase='1_1')
        img, lbl = ds[0]
        logging.info('  [%s] shape=%s  label=%d (%s)', split, img.shape, lbl, CLASS_NAMES_3[lbl])
    logging.info('--- Phase 1.2 pipeline ---')
    for split in ('train', 'val', 'test'):
        ds = ThermalDataset(split, manifest, augment=(split == 'train'), phase='1_2')
        img, lbl = ds[0]
        logging.info('  [%s] shape=%s  label=%d (%s)', split, img.shape, lbl, CLASS_NAMES_3[lbl])
    logging.info('--- TTA transforms ---')
    tta = build_tta_transforms(_CFG['phase1_2'])
    ds_test = ThermalDataset('test', manifest, augment=False, phase='1_2')
    img_orig, _ = ds_test[0]
    img_tta = tta(img_orig)
    logging.info('  TTA output shape: %s (same as input: %s)', img_tta.shape, img_orig.shape)
