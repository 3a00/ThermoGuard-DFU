"""Phase 0.7 -- Thermal Image Preprocessing Pipeline

Reads subject_manifest.csv (Phase 0.6 output) and processes every
full-foot thermal CSV into:
  1. Normalized float32 .npy array (canonical format for model training)
  2. Colormap-rendered RGB PNG (thesis figures and manual QA)

Normalization: predefined fixed physical range [15, 35] degrees C.
  - Preserves absolute temperature differences between feet.
  - Critical because TCI grades depend on absolute deviation from a fixed reference vector.
  - Per-image min-max normalization is explicitly NOT used.

Resize: target 128 rows x 64 cols (H x W).
  - Thermal array: one method per image -- INTER_AREA if either source dim > target, else INTER_LINEAR.
  - Binary foreground mask: always INTER_NEAREST -- never blended; reapplied after thermal resize.
  - This prevents background bleed at foot boundaries from interpolation.

Other decisions (v4, all finalized):
  - Left feet flipped horizontally for anatomical alignment (not augmentation).
  - Mask also flipped together with thermal when doing L-foot flip.
  - Colormap: 'plasma'.
  - No data augmentation -- this is preprocessing only.

Reproducibility:
  SHA-256 of subject_manifest.csv saved in preprocessing_summary.json.
"""

import hashlib
import json
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import yaml
from PIL import Image as PILImage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)

# -- Project-root-relative paths -----------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def project_path(relative: str) -> str:
    """Resolve a project-relative path to an absolute path under PROJECT_ROOT."""
    return os.path.join(PROJECT_ROOT, relative)


# -- Configuration -------------------------------------------------------------
CONFIG_PATH = project_path('config.yaml')

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

MANIFEST_PATH    = project_path(CFG['data']['manifest_path'])
PREPROC_MANIFEST = project_path(CFG['outputs']['preprocessing_manifest'])
PREPROC_SUMMARY  = project_path(CFG['outputs']['preprocessing_summary'])
PREPROC_REPORT   = project_path(CFG['outputs']['preprocessing_report'])

TARGET_H     = CFG['preprocessing']['target_height']
TARGET_W     = CFG['preprocessing']['target_width']
COLORMAP     = CFG['preprocessing']['colormap']
PHYS_MIN     = CFG['preprocessing']['phys_min']
PHYS_MAX     = CFG['preprocessing']['phys_max']
BG_THRESHOLD = CFG['preprocessing']['background_threshold']

CLASS_ORDER = ['Healthy', 'DM_Grade0', 'DM_Grade1', 'DM_Grade2', 'DM_Grade3', 'DM_Grade4']
SPLITS      = ['train', 'val', 'test']

EXPECTED_SUBJECTS = 167
EXPECTED_IMAGES   = 334
EXPECTED_COUNTS   = {'train': 232, 'val': 50, 'test': 52}


# -- Reproducibility helpers ---------------------------------------------------
def sha256_file(path: str) -> str:
    """Return the SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


# -- Path helpers --------------------------------------------------------------
def find_subject_folder(extracted_root: str, subject_id: str, group: str) -> Path:
    """
    Locate the subject folder in the raw dataset.
    Uses startswith(subject_id + '_') to prevent substring prefix collisions
    (e.g., DM1 accidentally matching DM10_M or DM11_F).
    """
    group_dir = Path(extracted_root) / ('Control Group' if group == 'CG' else 'DM Group')
    matching = [d for d in group_dir.iterdir()
                if d.is_dir() and d.name.startswith(subject_id + '_')]
    if len(matching) != 1:
        raise FileNotFoundError(
            f"Expected one folder for {subject_id} in {group_dir}, "
            f"found: {[d.name for d in matching]}"
        )
    return matching[0]


def find_foot_csv(subject_folder: Path, subject_id: str, side: str) -> Path:
    """
    Locate full-foot CSV: {subject_folder}/{subject_id}_{gender}_{side}.csv
    Gender character is embedded in the folder name (e.g., CG001_M -> 'M').
    """
    gender_char = subject_folder.name.split('_')[-1]
    csv_path = subject_folder / f"{subject_id}_{gender_char}_{side}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Full-foot CSV not found: {csv_path}")
    return csv_path


# -- Pre-processing guard ------------------------------------------------------
def check_manifest_ready(df: pd.DataFrame) -> None:
    """Assert subject_manifest.csv is the expected Phase 0.6 output."""
    if len(df) != EXPECTED_SUBJECTS:
        raise ValueError(f"Expected {EXPECTED_SUBJECTS} subjects, found {len(df)}.")
    if not df['subject_id'].is_unique:
        raise ValueError("Duplicate subject IDs in manifest.")
    blank_splits = df['split'].isna() | (df['split'].astype(str).str.strip() == '')
    if blank_splits.any():
        raise ValueError(
            f"{blank_splits.sum()} subjects have no split assignment. Run Phase 0.6 first."
        )
    invalid = df[~df['split'].isin({'train', 'val', 'test'})]
    if not invalid.empty:
        raise ValueError(f"Invalid split values: {invalid['split'].unique().tolist()}")
    logging.info("  OK Manifest integrity check passed: 167 subjects, all splits assigned.")


# -- Normalization -------------------------------------------------------------
def normalize_thermal(arr, phys_min, phys_max, bg_threshold):
    """
    Apply predefined fixed physical range normalization.

    Background pixels (<= bg_threshold) are masked before normalization
    and explicitly set to 0.0 in the output. This preserves the absolute
    temperature signal between feet -- critical because TCI grades depend
    on absolute deviation from a fixed reference vector.

    Raw foreground stats and clipping audit counts are returned for manifest audit.

    Returns:
        arr_norm:   normalized float32 array, values in [0, 1]
        foreground: boolean mask, True for non-background pixels
        stats:      dict of raw foreground temperature statistics and clipping counts
    """
    foreground = arr > bg_threshold
    fg_count = int(foreground.sum())

    if fg_count == 0:
        raise ValueError("No foreground pixels found (all pixels are background).")

    fg_values = arr[foreground]

    stats = {
        'foreground_pixels':                fg_count,
        'temp_min':                         round(float(fg_values.min()), 4),
        'temp_max':                         round(float(fg_values.max()), 4),
        'temp_mean':                        round(float(fg_values.mean()), 4),
        'temp_std':                         round(float(fg_values.std()), 4),
        'foreground_pixels_below_phys_min': int((fg_values < phys_min).sum()),
        'foreground_pixels_above_phys_max': int((fg_values > phys_max).sum()),
    }

    arr_clipped = np.clip(arr, phys_min, phys_max)
    arr_norm = np.zeros_like(arr, dtype=np.float32)
    arr_norm[foreground] = (arr_clipped[foreground] - phys_min) / (phys_max - phys_min)
    arr_norm[~foreground] = 0.0

    return arr_norm, foreground, stats


# -- Mask-preserving resize ----------------------------------------------------
def resize_thermal(arr_norm, foreground, target_h, target_w):
    """
    Resize the normalized thermal array and foreground mask to (target_h, target_w).

    One interpolation method is chosen per image:
      INTER_AREA if either source dimension is larger than the target;
      INTER_LINEAR otherwise.

    The binary mask is always resized with INTER_NEAREST to keep it binary.
    After resizing, the mask is reapplied to restore clean background zeros,
    preventing interpolation bleed at the foreground/background boundary.

    Returns:
        arr_resized:  resized float32 array, values in [0, 1]
        interp_name:  interpolation method used for the thermal array
    """
    src_h, src_w = arr_norm.shape

    downsampling = (src_h > target_h) or (src_w > target_w)
    interp = cv2.INTER_AREA if downsampling else cv2.INTER_LINEAR
    interp_name = 'INTER_AREA' if downsampling else 'INTER_LINEAR'

    thermal_resized = cv2.resize(arr_norm, (target_w, target_h), interpolation=interp)

    mask_resized = cv2.resize(
        foreground.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    thermal_resized[~mask_resized] = 0.0
    thermal_resized = np.clip(thermal_resized, 0.0, 1.0).astype(np.float32)

    return thermal_resized, interp_name


# -- Core processing -----------------------------------------------------------
def process_foot(csv_path, subject_id, side, split,
                 target_h, target_w, colormap_name,
                 phys_min, phys_max, bg_threshold):
    """
    Process one full-foot thermal CSV into a normalized .npy and colormap PNG.
    Returns a metadata dict for the preprocessing manifest row.
    """
    arr = pd.read_csv(csv_path, header=None).values.astype(np.float32)
    original_shape = f"{arr.shape[0]}x{arr.shape[1]}"

    arr_norm, foreground, stats = normalize_thermal(arr, phys_min, phys_max, bg_threshold)

    l_flipped = (side == 'L')
    if l_flipped:
        arr_norm = np.fliplr(arr_norm)
        foreground = np.fliplr(foreground)

    arr_resized, interp_name = resize_thermal(arr_norm, foreground, target_h, target_w)
    target_shape = f"{target_h}x{target_w}"

    npy_rel = os.path.join('data', 'processed', 'arrays', split, f"{subject_id}_{side}.npy")
    npy_abs = project_path(npy_rel)
    os.makedirs(os.path.dirname(npy_abs), exist_ok=True)
    np.save(npy_abs, arr_resized)

    cmap = matplotlib.colormaps[colormap_name]
    rgb_arr = (cmap(arr_resized)[:, :, :3] * 255).astype(np.uint8)
    png_rel = os.path.join('data', 'processed', 'images', split, f"{subject_id}_{side}.png")
    png_abs = project_path(png_rel)
    os.makedirs(os.path.dirname(png_abs), exist_ok=True)
    PILImage.fromarray(rgb_arr, mode='RGB').save(png_abs)

    return {
        'original_shape':     original_shape,
        'target_shape':       target_shape,
        'interpolation_used': interp_name,
        'l_foot_flipped':     l_flipped,
        'array_output':       npy_rel,
        'png_output':         png_rel,
        'processing_status':  'ok',
        'status_detail':      '',
        **stats,
    }


# -- Validation ----------------------------------------------------------------
def validate_outputs(manifest_df: pd.DataFrame) -> None:
    """
    Post-processing assertions. Raises ValueError on any failure.
    Checks: total count, per-split counts, processing status,
    .npy shape/dtype/range/NaN/Inf, PNG size/mode.
    Also prints clipping audit summary.
    """
    failed = manifest_df[manifest_df['processing_status'] != 'ok']
    if not failed.empty:
        raise ValueError(
            f"{len(failed)} images failed processing:\n"
            + failed[['subject_id', 'side', 'status_detail']].to_string(index=False)
        )

    if len(manifest_df) != EXPECTED_IMAGES:
        raise ValueError(f"Expected {EXPECTED_IMAGES} images, found {len(manifest_df)}.")

    for split, expected in EXPECTED_COUNTS.items():
        actual = int((manifest_df['split'] == split).sum())
        if actual != expected:
            raise ValueError(f"Expected {expected} images in '{split}', found {actual}.")

    logging.info("  Validating .npy files (shape, dtype, range, NaN/Inf)...")
    for _, row in manifest_df.iterrows():
        npy_abs = project_path(row['array_output'])
        if not os.path.exists(npy_abs):
            raise ValueError(f"Missing .npy: {npy_abs}")
        arr = np.load(npy_abs)
        if arr.shape != (TARGET_H, TARGET_W):
            raise ValueError(
                f"Shape mismatch: expected ({TARGET_H},{TARGET_W}), "
                f"got {arr.shape} in {npy_abs}"
            )
        if arr.dtype != np.float32:
            raise ValueError(f"dtype mismatch: expected float32, got {arr.dtype} in {npy_abs}")
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError(f"NaN/Inf detected in {npy_abs}")
        if arr.min() < -1e-6 or arr.max() > 1.0 + 1e-6:
            raise ValueError(
                f"Values out of [0,1] range: "
                f"[{arr.min():.4f}, {arr.max():.4f}] in {npy_abs}"
            )

    logging.info("  Validating PNG files (size, mode)...")
    for _, row in manifest_df.iterrows():
        png_abs = project_path(row['png_output'])
        if not os.path.exists(png_abs):
            raise ValueError(f"Missing PNG: {png_abs}")
        with PILImage.open(png_abs) as img:
            if img.mode != 'RGB':
                raise ValueError(f"PNG not RGB mode: {png_abs}")
            if img.size != (TARGET_W, TARGET_H):
                raise ValueError(
                    f"PNG size mismatch: expected ({TARGET_W},{TARGET_H}), "
                    f"got {img.size} in {png_abs}"
                )

    logging.info("  OK All output validation assertions passed.")

    ok_rows = manifest_df[manifest_df['processing_status'] == 'ok']
    total_below = int(ok_rows['foreground_pixels_below_phys_min'].sum())
    total_above = int(ok_rows['foreground_pixels_above_phys_max'].sum())
    if total_below > 0 or total_above > 0:
        logging.warning(
            f"  Clipping audit: {total_below} foreground pixels below {PHYS_MIN}C, "
            f"{total_above} pixels above {PHYS_MAX}C. Note in thesis methodology."
        )
    else:
        logging.info(
            f"  Clipping audit: 0 foreground pixels outside [{PHYS_MIN},{PHYS_MAX}]C. "
            f"Physical range bounds are non-restrictive for this dataset."
        )


# -- Report generation ---------------------------------------------------------
def save_preprocessing_report(manifest_df: pd.DataFrame, out_path: str) -> None:
    """Save a 6x3 visual grid: one sample per class x split."""
    fig, axes = plt.subplots(
        len(CLASS_ORDER), len(SPLITS),
        figsize=(len(SPLITS) * 3, len(CLASS_ORDER) * 3.5)
    )

    for row_idx, cls in enumerate(CLASS_ORDER):
        for col_idx, split in enumerate(SPLITS):
            ax = axes[row_idx][col_idx]
            subset = manifest_df[
                (manifest_df['model_class'] == cls) &
                (manifest_df['split'] == split) &
                (manifest_df['processing_status'] == 'ok')
            ]
            if subset.empty:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue
            sample = subset.iloc[0]
            with PILImage.open(project_path(sample['png_output'])) as img:
                ax.imshow(np.array(img))
            ax.set_title(
                f"{sample['subject_id']}_{sample['side']}\n"
                f"({sample['original_shape']}->{sample['target_shape']})\n"
                f"T=[{sample['temp_min']},{sample['temp_max']}]C",
                fontsize=6
            )
            ax.axis('off')

    for col_idx, split in enumerate(SPLITS):
        ax = axes[0][col_idx]
        ax.set_title(f"[ {split.upper()} ]\n" + ax.get_title(), fontsize=7, fontweight='bold')
    for row_idx, cls in enumerate(CLASS_ORDER):
        axes[row_idx][0].set_ylabel(cls, fontsize=9, labelpad=10)

    fig.suptitle(
        f'Phase 0.7 -- Preprocessing Report (v4)\n'
        f'Normalization: fixed range [{PHYS_MIN},{PHYS_MAX}]C  |  '
        f'Target: {TARGET_H}x{TARGET_W}  |  Mask-preserving resize  |  Colormap: {COLORMAP}',
        fontsize=8
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info(f"  Saved: {out_path}")


def save_summary_json(manifest_df: pd.DataFrame, out_path: str) -> None:
    """Save machine-readable preprocessing summary with full reproducibility audit."""
    per_class = {}
    for cls in CLASS_ORDER:
        sub = manifest_df[manifest_df['model_class'] == cls]
        per_class[cls] = {
            'total_images': len(sub),
            'train': int((sub['split'] == 'train').sum()),
            'val':   int((sub['split'] == 'val').sum()),
            'test':  int((sub['split'] == 'test').sum()),
        }

    ok_rows = manifest_df[manifest_df['processing_status'] == 'ok']

    summary = {
        'total_images': len(manifest_df),
        'per_split': {
            split: {
                'total_images': int((manifest_df['split'] == split).sum()),
                'subjects': EXPECTED_COUNTS[split] // 2,
            }
            for split in SPLITS
        },
        'per_class': per_class,
        'preprocessing_params': {
            'normalization':               'predefined_fixed_physical_range',
            'phys_min_celsius':            PHYS_MIN,
            'phys_max_celsius':            PHYS_MAX,
            'normalization_note':          (
                'Range [15,35]C fixed as physiologically plausible bounds before training. '
                'Same constants applied to all splits. Not derived from data statistics.'
            ),
            'background_threshold':        BG_THRESHOLD,
            'background_note':             (
                'Background pixels explicitly set to 0 after masking. '
                'In this dataset, foreground temperatures are above the lower normalization bound, '
                'so 0.0 is expected to represent background in practice.'
            ),
            'target_height':               TARGET_H,
            'target_width':                TARGET_W,
            'resize_thermal':              (
                'One method per image: INTER_AREA if either source dim > target, else INTER_LINEAR'
            ),
            'resize_mask':                 'INTER_NEAREST always (binary mask, never blended)',
            'mask_reapplied_after_resize':  True,
            'colormap':                    COLORMAP,
            'l_foot_horizontal_flip':      True,
            'l_flip_is_augmentation':      False,
            'augmentation':                'none',
        },
        'clipping_audit': {
            'total_foreground_pixels_below_phys_min': int(
                ok_rows['foreground_pixels_below_phys_min'].sum()
            ),
            'total_foreground_pixels_above_phys_max': int(
                ok_rows['foreground_pixels_above_phys_max'].sum()
            ),
        },
        'input_manifest_sha256': sha256_file(MANIFEST_PATH),
        'python_version':        platform.python_version(),
        'numpy_version':         np.__version__,
        'cv2_version':           cv2.__version__,
        'pandas_version':        pd.__version__,
        'timestamp':             datetime.now(timezone.utc).isoformat(),
    }

    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"  Saved: {out_path}")


# -- Error record helper -------------------------------------------------------
def _error_record(subject_id, side, split, model_class, gender,
                  tci_subject, severity_grade, error_msg, source_csv='') -> dict:
    """Build a standardized error row for the preprocessing manifest."""
    return {
        'subject_id': subject_id, 'side': side, 'split': split,
        'model_class': model_class, 'gender': gender,
        'tci_subject': tci_subject, 'severity_grade': severity_grade,
        'source_csv': source_csv, 'source_png': '',
        'original_shape': '', 'target_shape': '', 'interpolation_used': '',
        'foreground_pixels': 0,
        'temp_min': None, 'temp_max': None, 'temp_mean': None, 'temp_std': None,
        'foreground_pixels_below_phys_min': None,
        'foreground_pixels_above_phys_max': None,
        'l_foot_flipped': None, 'array_output': '', 'png_output': '',
        'processing_status': 'error', 'status_detail': error_msg,
    }


# -- Main ----------------------------------------------------------------------
def main() -> None:
    """Run the Phase 0.7 thermal image preprocessing pipeline."""
    logging.info("=" * 60)
    logging.info("Phase 0.7 -- Thermal Image Preprocessing Pipeline (v4)")
    logging.info(f"  Normalization: fixed physical range [{PHYS_MIN}, {PHYS_MAX}]C")
    logging.info(f"  Target size:   {TARGET_H} x {TARGET_W} (H x W)")
    logging.info("  Resize:        mask-preserving (thermal=INTER_AREA/LINEAR, mask=INTER_NEAREST)")
    logging.info("=" * 60)

    logging.info(f"Loading: {MANIFEST_PATH}")
    manifest_df = pd.read_csv(MANIFEST_PATH)
    logging.info(f"  Loaded {len(manifest_df)} subjects.")

    logging.info("Checking manifest integrity:")
    check_manifest_ready(manifest_df)

    for split in SPLITS:
        os.makedirs(project_path(f'data/processed/arrays/{split}'), exist_ok=True)
        os.makedirs(project_path(f'data/processed/images/{split}'), exist_ok=True)
    os.makedirs(os.path.dirname(PREPROC_MANIFEST), exist_ok=True)

    extracted_root = CFG['data']['extracted_root']
    records = []
    total = len(manifest_df) * 2
    processed = 0
    errors = 0

    logging.info(f"Processing {total} images (2 feet x {len(manifest_df)} subjects)...")

    for _, row in manifest_df.iterrows():
        subject_id     = row['subject_id']
        group          = row['group']
        split          = row['split']
        model_class    = row['model_class']
        gender         = row['gender']
        tci_subject    = row['tci_subject']
        severity_grade = row.get('severity_grade', '')

        try:
            subject_folder = find_subject_folder(extracted_root, subject_id, group)
        except FileNotFoundError as e:
            logging.error(f"  [ERROR] {subject_id}: {e}")
            for side in ['L', 'R']:
                records.append(_error_record(
                    subject_id, side, split, model_class,
                    gender, tci_subject, severity_grade, str(e)
                ))
            errors += 2
            continue

        for side in ['L', 'R']:
            csv_path = None
            try:
                csv_path = find_foot_csv(subject_folder, subject_id, side)
                gender_char = subject_folder.name.split('_')[-1]
                source_png = str(subject_folder / f"{subject_id}_{gender_char}_{side}.png")

                result = process_foot(
                    csv_path=csv_path,
                    subject_id=subject_id,
                    side=side,
                    split=split,
                    target_h=TARGET_H,
                    target_w=TARGET_W,
                    colormap_name=COLORMAP,
                    phys_min=PHYS_MIN,
                    phys_max=PHYS_MAX,
                    bg_threshold=BG_THRESHOLD,
                )
                records.append({
                    'subject_id':     subject_id,
                    'side':           side,
                    'split':          split,
                    'model_class':    model_class,
                    'gender':         gender,
                    'tci_subject':    tci_subject,
                    'severity_grade': severity_grade,
                    'source_csv':     str(csv_path),
                    'source_png':     source_png,
                    **result,
                })
                processed += 1
                if processed % 50 == 0:
                    logging.info(f"  Progress: {processed}/{total}")

            except Exception as e:
                logging.error(f"  [ERROR] {subject_id}_{side}: {e}")
                records.append(_error_record(
                    subject_id, side, split, model_class,
                    gender, tci_subject, severity_grade, str(e),
                    source_csv=str(csv_path) if csv_path else '',
                ))
                errors += 1

    preproc_manifest = pd.DataFrame(records)

    logging.info("Validating outputs:")
    validate_outputs(preproc_manifest)

    logging.info("Saving outputs:")
    preproc_manifest.to_csv(PREPROC_MANIFEST, index=False)
    logging.info(f"  Saved: {PREPROC_MANIFEST}")
    save_summary_json(preproc_manifest, PREPROC_SUMMARY)
    save_preprocessing_report(preproc_manifest, PREPROC_REPORT)

    logging.info("")
    logging.info("-" * 60)
    logging.info("SUMMARY")
    logging.info("-" * 60)
    logging.info(f"  Total images processed : {processed}/{total}")
    logging.info(f"  Errors                 : {errors}")
    for split in SPLITS:
        n = int((preproc_manifest['split'] == split).sum())
        logging.info(f"  {split:<6} images         : {n}")
    logging.info("")
    logging.info(f"  STATUS: {'PASS' if errors == 0 else f'PARTIAL ({errors} errors)'}")
    logging.info("  -> Manifest:  outputs/preprocessing_manifest.csv")
    logging.info("  -> Summary:   outputs/preprocessing_summary.json")
    logging.info("  -> Report:    outputs/preprocessing_report.png")
    logging.info("  Next: Phase 0.8 -- Class weights and dataset statistics.")


if __name__ == '__main__':
    main()
