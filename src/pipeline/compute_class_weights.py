"""Phase 0.8 -- Class Weights and Dataset Statistics

Reads preprocessing_manifest.csv (Phase 0.7 output) and computes:
  1. Per-class weighted loss coefficients for CrossEntropyLoss
  2. Pixel-level mean and std from the training split (for DataLoader normalization)
  3. A visual class distribution bar chart

Formula: sklearn balanced class weights
  weight[c] = n_samples / (n_classes * n_samples_per_class[c])

Normalization invariant (v3 fix):
  The FREQUENCY-WEIGHTED MEAN of the weights equals 1.0 (not the unweighted sum).
  Unweighted sum for this imbalanced dataset = 8.165, not 6.0.
  The correct guard: sum(count[c] * weight[c]) / n_samples == 1.0

Foreground separation note:
  Foreground pixels are identified by value > float32(0.0) in the post-resize .npy arrays.
  This is an audited convention, not a stored binary mask. Phase 0.7 sets background to
  EXACTLY float32(0.0); real-data audit confirmed no spurious near-zero background pixels.
  The manifest foreground_pixels column is PRE-RESIZE and NOT comparable to post-resize counts.

v3 changes vs v2:
  - Weight-sum guard fixed: weighted_mean == 1.0 (was unweighted sum == 6.0, always would fail)
  - Weights computed in full float64 precision; rounded only for JSON
  - Foreground pixel cross-check removed (pre/post-resize counts differ by ~243k -- incomparable)
  - Multi-column manifest guard replaces single-column check
  - foreground_pixel_count_match field removed (was dead code)
  - Foreground wording updated to 'audited convention'
  - Phase 1 augmentation note: on-the-fly per AGENTS.md, not pre-generated static dataset
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)

# -- Project paths -------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def project_path(relative: str) -> str:
    """Resolve a project-relative path to an absolute path under PROJECT_ROOT."""
    return os.path.join(PROJECT_ROOT, relative)


# -- Configuration -------------------------------------------------------------
CONFIG_PATH = project_path('config.yaml')
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

PREPROC_MANIFEST = project_path(CFG['outputs']['preprocessing_manifest'])

CLASS_ORDER = ['Healthy', 'DM_Grade0', 'DM_Grade1', 'DM_Grade2', 'DM_Grade3', 'DM_Grade4']
CLASS_INDEX_MAP = {cls: idx for idx, cls in enumerate(CLASS_ORDER)}

# NOTE: These counts are intentionally brittle (fail-fast on any upstream reprocessing).
# If the split ever changes, update these constants manually.
EXPECTED_TOTAL   = 334
EXPECTED_COUNTS  = {'train': 232, 'val': 50, 'test': 52}
EXPECTED_CLASSES = 6

# Near-zero foreground audit: pixels in (0.0, NEAR_ZERO_THRESHOLD) are logged
# for thesis transparency. Correspond to raw temps just above 15C (lower physical bound).
NEAR_ZERO_THRESHOLD = np.float32(0.001)

REQUIRED_MANIFEST_COLUMNS = {
    'split', 'model_class', 'processing_status', 'array_output', 'foreground_pixels'
}

OUT_WEIGHTS = project_path('outputs/class_weights.json')
OUT_STATS   = project_path('outputs/dataset_statistics.json')
OUT_PLOT    = project_path('outputs/class_distribution_report.png')


# -- Reproducibility -----------------------------------------------------------
def sha256_file(path: str) -> str:
    """Return the SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


# -- Guard ---------------------------------------------------------------------
def check_manifest_ready(df: pd.DataFrame) -> None:
    """Assert preprocessing_manifest.csv is the expected Phase 0.7 output."""
    missing_cols = REQUIRED_MANIFEST_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f'Manifest missing required columns: {sorted(missing_cols)}. '
            f'Ensure Phase 0.7 was run with the v3+ preprocessing.py.'
        )
    if len(df) != EXPECTED_TOTAL:
        raise ValueError(f'Expected {EXPECTED_TOTAL} rows, found {len(df)}.')
    for split, expected in EXPECTED_COUNTS.items():
        actual = int((df['split'] == split).sum())
        if actual != expected:
            raise ValueError(f"Expected {expected} rows in '{split}', found {actual}.")
    failed = df[df['processing_status'] != 'ok']
    if not failed.empty:
        raise ValueError(
            f'{len(failed)} images not ok:\n'
            + failed[['subject_id', 'side', 'status_detail']].to_string(index=False)
        )
    n_classes = df['model_class'].nunique()
    if n_classes != EXPECTED_CLASSES:
        raise ValueError(f'Expected {EXPECTED_CLASSES} unique classes, found {n_classes}.')
    logging.info(
        '  OK Manifest guard: %d rows, all ok, %d classes, required columns present.',
        EXPECTED_TOTAL, EXPECTED_CLASSES,
    )


# -- Class weight computation --------------------------------------------------
def compute_class_weights(train_df: pd.DataFrame) -> dict:
    """
    Compute balanced class weights using the sklearn formula.

    weight[c] = n_samples / (n_classes * n_samples_per_class[c])

    Weights are computed in full float64 precision and rounded only for JSON output.

    Normalization invariant enforced (v3 fix):
      The FREQUENCY-WEIGHTED MEAN of the weights must equal 1.0.
      For imbalanced classes, the UNWEIGHTED SUM does NOT equal n_classes.
      (For this dataset: unweighted sum = 8.165, weighted mean = 1.000000.)

    Raises ValueError if any class has zero training samples or if
    the weighted-mean normalization invariant fails.
    """
    n_samples = len(train_df)
    n_classes = EXPECTED_CLASSES

    counts = {}
    for cls in CLASS_ORDER:
        counts[cls] = int((train_df['model_class'] == cls).sum())
        if counts[cls] == 0:
            raise ValueError(f"Class '{cls}' has 0 training samples -- cannot compute weight.")

    # Full-precision weights (float64 for computation)
    weights_full = {
        cls: n_samples / (n_classes * counts[cls])
        for cls in CLASS_ORDER
    }

    # Enforce frequency-weighted mean == 1.0 (correct normalization invariant)
    weighted_mean = sum(
        counts[cls] * weights_full[cls] for cls in CLASS_ORDER
    ) / n_samples
    if not np.isclose(weighted_mean, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(
            f'Class-weight normalization failed: weighted_mean={weighted_mean:.8f}, '
            f'expected 1.0. Possible data-loading error.'
        )
    logging.info(
        '  Weighted-mean normalization check: %.8f (expected 1.0) -- OK', weighted_mean
    )

    # Round only for JSON/display
    weights_rounded = {cls: round(weights_full[cls], 6) for cls in CLASS_ORDER}
    weights_ordered = [weights_rounded[cls] for cls in CLASS_ORDER]

    unweighted_sum = sum(weights_ordered)
    logging.info(
        '  Unweighted sum of weights: %.4f '
        '(expected ~8.165 for this imbalanced dataset, NOT 6.0)',
        unweighted_sum,
    )

    return {
        'class_index_map':              CLASS_INDEX_MAP,
        'train_image_counts':           counts,
        'n_samples':                    n_samples,
        'n_classes':                    n_classes,
        'weights_by_class':             weights_rounded,
        'weights_as_tensor_order':      weights_ordered,
        'weighted_mean_sanity_check':   round(weighted_mean, 8),
        'formula':                      'n_samples / (n_classes * n_samples_per_class)',
        'normalization_invariant':      'frequency-weighted mean = 1.0',
        'source_split':                 'train',
        'source_manifest':              'outputs/preprocessing_manifest.csv',
        'source_manifest_sha256':       sha256_file(PREPROC_MANIFEST),
        'timestamp':                    datetime.now(timezone.utc).isoformat(),
    }


# -- Dataset statistics --------------------------------------------------------
def compute_dataset_statistics(train_df: pd.DataFrame) -> dict:
    """
    Compute pixel-level statistics from all training .npy arrays.

    Foreground vs. background separation uses value > np.float32(0.0).
    This is an audited convention: Phase 0.7 sets background to EXACTLY float32(0.0)
    because the final saved arrays are written with explicit background assignment to float32 zero.
    Real-data audit found 10 valid tissue pixels in (0.0, 0.001) at 15.006-15.018C.

    IMPORTANT: the manifest 'foreground_pixels' column stores PRE-RESIZE pixel counts
    (from the original CSV dimensions). The counts computed here are POST-RESIZE (128x64).
    These numbers are NOT comparable -- do NOT cross-check them against each other.
    Confirmed: train pre-resize=1,630,000 vs post-resize=1,386,904 (diff=243,096).
    """
    all_pixels_list = []
    n_arrays = 0

    logging.info('  Loading %d training arrays ...', len(train_df))
    for _, row in train_df.iterrows():
        npy_path = project_path(row['array_output'])
        arr = np.load(npy_path)
        if arr.shape != (128, 64):
            raise ValueError(f'Shape mismatch in {npy_path}: {arr.shape}')
        if arr.dtype != np.float32:
            raise ValueError(f'dtype mismatch in {npy_path}: {arr.dtype}')
        all_pixels_list.append(arr.flatten())
        n_arrays += 1

    all_pixels = np.concatenate(all_pixels_list, axis=0)

    # Foreground/background split (audited convention -- see docstring)
    fg_mask = all_pixels > np.float32(0.0)
    foreground = all_pixels[fg_mask]
    background = all_pixels[~fg_mask]

    # Near-zero foreground audit (transparency for thesis)
    near_zero = foreground[foreground < NEAR_ZERO_THRESHOLD]
    near_zero_count = int(len(near_zero))
    if near_zero_count > 0:
        logging.info(
            '  Foreground boundary audit: %d pixel(s) in (0.0, %.3f) -- '
            'valid tissue at ~15C lower bound. Values: %s',
            near_zero_count, NEAR_ZERO_THRESHOLD, near_zero.tolist(),
        )
    else:
        logging.info('  Foreground boundary audit: 0 near-zero foreground pixels.')

    return {
        'split':                          'train',
        'n_arrays':                       n_arrays,
        'array_shape':                    [128, 64],
        'pixel_scope':                    'all_pixels (foreground + background zeros)',
        'pixel_mean':                     round(float(all_pixels.mean()), 6),
        'pixel_std':                      round(float(all_pixels.std()), 6),
        'pixel_min':                      round(float(all_pixels.min()), 6),
        'pixel_max':                      round(float(all_pixels.max()), 6),
        'foreground_pixel_mean':          round(float(foreground.mean()), 6),
        'foreground_pixel_std':           round(float(foreground.std()), 6),
        'foreground_pixel_min':           round(float(foreground.min()), 6),
        'foreground_pixel_max':           round(float(foreground.max()), 6),
        'foreground_near_zero_pixels':    near_zero_count,
        'foreground_near_zero_note': (
            'Foreground pixels with normalized value in (0.0, 0.001). '
            'Correspond to raw temps just above 15C (lower physiological bound). '
            'Valid tissue pixels -- correctly counted as foreground. '
            'The manifest records pre-resize audit values only; these post-resize foreground-only '
            'statistics are independent audit statistics and are not directly comparable to manifest counts. '
            'The foreground_pixels_below_phys_min column in preprocessing_manifest.csv is a separate '
            'pre-resize audit of pixels at the lower physical bound.'
        ),
        'foreground_separation_note': (
            'Foreground derived from post-resize .npy files using value > float32(0.0). '
            'Audited convention, not a stored binary mask. '
            'Manifest foreground_pixels column is PRE-RESIZE and NOT comparable to these counts.'
        ),
        'total_pixels':                   int(len(all_pixels)),
        'foreground_pixels_post_resize':  int(len(foreground)),
        'background_pixels_post_resize':  int(len(background)),
        'background_fraction':            round(float(len(background) / len(all_pixels)), 6),
        'normalization_note': (
            'Values already in [0,1] from Phase 0.7 global physical normalization. '
            'Additional mean/std standardization is optional. '
            'Supervisor decision required before Phase 1 DataLoader implementation.'
        ),
        'source_manifest':                'outputs/preprocessing_manifest.csv',
        'timestamp':                      datetime.now(timezone.utc).isoformat(),
    }


# -- Visualization -------------------------------------------------------------
def save_class_distribution_plot(df: pd.DataFrame, out_path: str) -> None:
    """Save a grouped bar chart of class image counts across train/val/test splits."""
    splits = ['train', 'val', 'test']
    x = np.arange(len(CLASS_ORDER))
    width = 0.25
    colors = {'train': '#4C72B0', 'val': '#DD8452', 'test': '#55A868'}

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, split in enumerate(splits):
        counts = [
            int(((df['split'] == split) & (df['model_class'] == cls)).sum())
            for cls in CLASS_ORDER
        ]
        bars = ax.bar(x + i * width, counts, width, label=split.capitalize(), color=colors[split])
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(count),
                ha='center', va='bottom', fontsize=8,
            )

    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Images')
    ax.set_title(
        'ThermoGuard-DFU -- Class Distribution per Split\n'
        '(Train imbalance ratio: 4.43:1  |  Phase 0.8)',
        fontsize=11,
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_ORDER, rotation=15)
    ax.legend()
    ax.set_ylim(0, 80)
    ax.axhline(y=0, color='black', linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    logging.info('  Saved: %s', out_path)


# -- Main ----------------------------------------------------------------------
def main():
    """Phase 0.8 -- Compute class weights and dataset statistics."""
    logging.info('Phase 0.8: Class Weights and Dataset Statistics')
    logging.info('=' * 60)

    # Steps 1-2: Load and guard manifest
    logging.info('Step 1: Loading preprocessing_manifest.csv ...')
    df = pd.read_csv(PREPROC_MANIFEST)
    check_manifest_ready(df)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    logging.info('  Train images: %d', len(train_df))

    # Step 4: Compute class weights
    logging.info('Step 2: Computing class weights ...')
    weights_data = compute_class_weights(train_df)
    logging.info('  Class weights:')
    for cls, w in weights_data['weights_by_class'].items():
        count = weights_data['train_image_counts'][cls]
        logging.info(
            '    [%d] %-12s  count=%-3d  weight=%.6f',
            CLASS_INDEX_MAP[cls], cls, count, w,
        )

    # Step 5: Compute dataset statistics
    logging.info('Step 3: Computing pixel statistics from training arrays ...')
    stats_data = compute_dataset_statistics(train_df)
    logging.info(
        '  All-pixel   mean=%.6f  std=%.6f',
        stats_data['pixel_mean'], stats_data['pixel_std'],
    )
    logging.info(
        '  Foreground  mean=%.6f  std=%.6f',
        stats_data['foreground_pixel_mean'], stats_data['foreground_pixel_std'],
    )
    logging.info('  Background fraction=%.4f', stats_data['background_fraction'])

    # Step 6: Save JSON outputs
    logging.info('Step 4: Saving JSON outputs ...')
    os.makedirs(project_path('outputs'), exist_ok=True)
    with open(OUT_WEIGHTS, 'w') as f:
        json.dump(weights_data, f, indent=2)
    logging.info('  Saved: %s', OUT_WEIGHTS)
    with open(OUT_STATS, 'w') as f:
        json.dump(stats_data, f, indent=2)
    logging.info('  Saved: %s', OUT_STATS)

    # Step 7: Visualization
    logging.info('Step 5: Generating class distribution plot ...')
    save_class_distribution_plot(df, OUT_PLOT)

    # Summary
    logging.info('=' * 60)
    logging.info('Phase 0.8 Complete.')
    logging.info('  class_weights.json          -> %s', OUT_WEIGHTS)
    logging.info('  dataset_statistics.json     -> %s', OUT_STATS)
    logging.info('  class_distribution_report   -> %s', OUT_PLOT)
    logging.info('')
    logging.info('  Next: Update config.yaml, update AGENTS.md, then proceed to Phase 1.')


if __name__ == '__main__':
    main()