"""Phase 0.6 — Subject-Level Data Split (70 / 15 / 15)

Reads subject_manifest_unsplit.csv (Phase 0.5 output).
Assigns stratified train/val/test labels by model_class using
the sklearn two-step train_test_split approach.
Writes subject_manifest.csv with split column filled.

Reproducibility note:
  Results are reproducible with unchanged input manifest, identical code,
  seed=42, and compatible library versions (saved in split_summary.json).
  The SHA-256 checksum of the input manifest is also saved for audit.
"""

import hashlib
import json
import logging
import os
import platform
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)

# ── Project-root-relative paths ───────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def project_path(relative: str) -> str:
    return os.path.join(PROJECT_ROOT, relative)

# ── Configuration ──────────────────────────────────────────────────────────────
CONFIG_PATH = project_path('config.yaml')

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

SEED          = CFG['seed']
MANIFEST_IN   = project_path(CFG['data']['manifest_unsplit_path'])
MANIFEST_OUT  = project_path(CFG['data']['manifest_path'])
DIST_REPORT   = project_path(CFG['outputs']['split_distribution_report'])
SPLIT_SUMMARY = project_path(CFG['outputs']['split_summary'])
SPLIT_PLOT    = project_path(CFG['outputs']['split_distribution_plot'])

CLASS_ORDER = ['Healthy', 'DM_Grade0', 'DM_Grade1', 'DM_Grade2', 'DM_Grade3', 'DM_Grade4']

TEST_SIZE         = 0.15
VAL_FROM_TV       = 0.17647
RATIO_TOLERANCE   = 0.02
TARGET_RATIOS     = {'train': 0.70, 'val': 0.15, 'test': 0.15}
EXPECTED_SUBJECTS = 167


# ── Reproducibility helpers ────────────────────────────────────────────────────
def sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


# ── Pre-split manifest guard ───────────────────────────────────────────────────
def check_manifest_integrity(df: pd.DataFrame) -> None:
    if len(df) != EXPECTED_SUBJECTS:
        raise ValueError(f"Expected {EXPECTED_SUBJECTS} subjects, found {len(df)}.")

    if df['subject_id'].isna().any():
        raise ValueError("Manifest contains missing subject_id values.")

    if not df['subject_id'].is_unique:
        duplicates = df.loc[df['subject_id'].duplicated(keep=False), 'subject_id'].tolist()
        raise ValueError(f"Duplicate subject IDs found: {sorted(set(duplicates))}.")

    invalid_rows = df[df['calculation_status'] != 'valid']
    if not invalid_rows.empty:
        raise ValueError(
            "Cannot split: manifest contains non-valid TCI calculations:\n"
            + invalid_rows[['subject_id', 'calculation_status', 'status_detail']].to_string(index=False)
        )

    split_values = df['split'].fillna('').astype(str).str.strip()
    already_split = split_values != ''
    if already_split.any():
        raise ValueError(
            f"{already_split.sum()} rows already have a split value. "
            "This script must run on the unsplit manifest only."
        )

    logging.info(
        "  \u2713 Manifest integrity check passed: "
        "167 subjects, all unique IDs, all valid, no prior splits."
    )


# ── Core splitting logic ───────────────────────────────────────────────────────
def assign_splits(df: pd.DataFrame, seed: int):
    trainval_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df['model_class'], random_state=seed,
    )
    train_df, val_df = train_test_split(
        trainval_df, test_size=VAL_FROM_TV, stratify=trainval_df['model_class'], random_state=seed,
    )

    logging.info(f"  Raw split counts: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    df = df.copy()
    df['split'] = ''
    df.loc[train_df.index, 'split'] = 'train'
    df.loc[val_df.index,   'split'] = 'val'
    df.loc[test_df.index,  'split'] = 'test'

    return df, train_df, val_df, test_df


# ── Validation ─────────────────────────────────────────────────────────────────
def validate_splits(df: pd.DataFrame, train_df, val_df, test_df) -> None:
    n = len(df)

    split_values = df['split'].fillna('').astype(str).str.strip()
    blank = split_values == ''
    if blank.any():
        raise ValueError(f"{blank.sum()} subjects have no split label after assignment.")

    valid_splits = {'train', 'val', 'test'}
    invalid = df[~df['split'].isin(valid_splits)]
    if not invalid.empty:
        raise ValueError(f"Invalid split values found: {invalid['split'].unique().tolist()}")

    train_ids = set(train_df['subject_id'])
    val_ids   = set(val_df['subject_id'])
    test_ids  = set(test_df['subject_id'])

    overlaps = {
        'train\u2229val':  train_ids & val_ids,
        'train\u2229test': train_ids & test_ids,
        'val\u2229test':   val_ids   & test_ids,
    }
    for name, overlap in overlaps.items():
        if overlap:
            raise ValueError(f"Subject-level leakage: {name} shares {len(overlap)} subject(s): {sorted(overlap)}")

    total = len(train_df) + len(val_df) + len(test_df)
    if total != n:
        raise ValueError(f"Total after split ({total}) != total subjects ({n}).")

    split_map = {'train': train_df, 'val': val_df, 'test': test_df}
    for split_name, target_ratio in TARGET_RATIOS.items():
        actual_count = len(split_map[split_name])
        actual_ratio = actual_count / n
        if abs(actual_ratio - target_ratio) > RATIO_TOLERANCE:
            raise ValueError(
                f"'{split_name}' ratio {actual_ratio:.3f} deviates >{RATIO_TOLERANCE} from target {target_ratio:.2f}."
            )

    for split_name, split_df in {'train': train_df, 'val': val_df, 'test': test_df}.items():
        for cls in CLASS_ORDER:
            if (split_df['model_class'] == cls).sum() == 0:
                raise ValueError(f"Class '{cls}' is absent from the '{split_name}' split.")

    logging.info("  \u2713 All split validation assertions passed.")
    logging.info("  Per-class breakdown:")
    for cls in CLASS_ORDER:
        tr = (train_df['model_class'] == cls).sum()
        v  = (val_df['model_class']   == cls).sum()
        te = (test_df['model_class']  == cls).sum()
        logging.info(f"    {cls:<12} : train={tr}, val={v}, test={te}  (total={tr+v+te})")


# ── Output generation ──────────────────────────────────────────────────────────
def save_distribution_report(df: pd.DataFrame, out_path: str) -> None:
    rows = []
    for cls in CLASS_ORDER:
        sub = df[df['model_class'] == cls]
        n   = len(sub)
        tr  = (sub['split'] == 'train').sum()
        v   = (sub['split'] == 'val').sum()
        te  = (sub['split'] == 'test').sum()
        rows.append({
            'model_class': cls, 'total': n, 'train': tr, 'val': v, 'test': te,
            'train_pct': round(tr/n*100, 1), 'val_pct': round(v/n*100, 1), 'test_pct': round(te/n*100, 1),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logging.info(f"  Saved: {out_path}")


def save_summary_json(df: pd.DataFrame, out_path: str) -> None:
    per_class = {}
    for cls in CLASS_ORDER:
        sub = df[df['model_class'] == cls]
        per_class[cls] = {
            'total': len(sub),
            'train': int((sub['split'] == 'train').sum()),
            'val':   int((sub['split'] == 'val').sum()),
            'test':  int((sub['split'] == 'test').sum()),
        }

    summary = {
        'total_subjects': len(df),
        'split_totals': {
            'train': int((df['split'] == 'train').sum()),
            'val':   int((df['split'] == 'val').sum()),
            'test':  int((df['split'] == 'test').sum()),
        },
        'per_class':             per_class,
        'seed':                  SEED,
        'test_size_first_split': TEST_SIZE,
        'val_size_second_split': VAL_FROM_TV,
        'split_method':          'sklearn two-step train_test_split (stratified by model_class)',
        'input_manifest_sha256': sha256_file(MANIFEST_IN),
        'python_version':        platform.python_version(),
        'pandas_version':        pd.__version__,
        'sklearn_version':       sklearn.__version__,
        'numpy_version':         np.__version__,
        'timestamp':             datetime.now(timezone.utc).isoformat(),
    }

    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"  Saved: {out_path}")


def save_distribution_plot(df: pd.DataFrame, out_path: str) -> None:
    x      = np.arange(len(CLASS_ORDER))
    width  = 0.25
    colors = {'train': '#4C72B0', 'val': '#DD8452', 'test': '#55A868'}

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (split_name, color) in enumerate(colors.items()):
        counts = [
            df[(df['model_class'] == cls) & (df['split'] == split_name)].shape[0]
            for cls in CLASS_ORDER
        ]
        bars = ax.bar(x + (i - 1) * width, counts, width, label=split_name.capitalize(), color=color)
        ax.bar_label(bars, padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER, rotation=15, ha='right')
    ax.set_ylabel('Number of subjects')
    ax.set_title(
        'Phase 0.6 \u2014 Subject split distribution by model class\n'
        '(seed=42, two-step stratified train_test_split, ~70 / 15 / 15)'
    )
    ax.legend()
    max_train = df[df['split'] == 'train']['model_class'].value_counts().max()
    ax.set_ylim(0, max_train + 8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info(f"  Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    logging.info("=" * 60)
    logging.info("Phase 0.6 \u2014 Subject-Level Data Split")
    logging.info("=" * 60)

    logging.info(f"Loading: {MANIFEST_IN}")
    df = pd.read_csv(MANIFEST_IN)
    logging.info(f"  Loaded {len(df)} rows.")

    logging.info("Checking manifest integrity:")
    check_manifest_integrity(df)

    logging.info("Assigning stratified splits (seed=42, two-step train_test_split):")
    df, train_df, val_df, test_df = assign_splits(df, seed=SEED)

    logging.info("Validating splits:")
    validate_splits(df, train_df, val_df, test_df)

    for path in [MANIFEST_OUT, DIST_REPORT, SPLIT_SUMMARY, SPLIT_PLOT]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    logging.info("Saving outputs:")
    df.to_csv(MANIFEST_OUT, index=False)
    logging.info(f"  Saved: {MANIFEST_OUT}")
    save_distribution_report(df, DIST_REPORT)
    save_summary_json(df, SPLIT_SUMMARY)
    save_distribution_plot(df, SPLIT_PLOT)

    train_n = (df['split'] == 'train').sum()
    val_n   = (df['split'] == 'val').sum()
    test_n  = (df['split'] == 'test').sum()

    logging.info("")
    logging.info("\u2500" * 60)
    logging.info("SUMMARY")
    logging.info("\u2500" * 60)
    logging.info(f"  Total subjects : {len(df)}")
    logging.info(f"  Train          : {train_n}  ({train_n/len(df)*100:.1f}%)")
    logging.info(f"  Val            : {val_n}   ({val_n/len(df)*100:.1f}%)")
    logging.info(f"  Test           : {test_n}   ({test_n/len(df)*100:.1f}%)")
    logging.info("")
    logging.info("  STATUS: PASS \u2014 all assertions satisfied.")
    logging.info("  \u2192 Definitive counts: outputs/split_distribution_report.csv")
    logging.info("  \u2192 Reproducibility record: outputs/split_summary.json")
    logging.info("  Next: Phase 0.7 \u2014 Image preprocessing / resize pipeline.")


if __name__ == '__main__':
    main()
