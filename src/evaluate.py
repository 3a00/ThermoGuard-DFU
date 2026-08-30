"""Phase-aware Test-Set Evaluation Script.

Supports Phase 1.0, Phase 1.1, and Phase 1.2 via --phase flag.

Usage:
    cd <project_root> && source venv/bin/activate

    # Phase 1.0 (6-class, NEAREST resize)
    python src/evaluate.py --phase phase1_0

    # Phase 1.1 (3-class, Bilinear+mask)
    python src/evaluate.py --phase phase1_1

    # Phase 1.2 -- single best checkpoint
    python src/evaluate.py --phase phase1_2

    # Phase 1.2 -- with TTA (8 views, half training range)
    python src/evaluate.py --phase phase1_2 --use-tta

    # Phase 1.2 -- with probability ensemble across plateau checkpoints (primary)
    python src/evaluate.py --phase phase1_2 --use-prob-ensemble

    # Phase 1.2 -- TTA + probability ensemble (combined; commutative since both average probs)
    python src/evaluate.py --phase phase1_2 --use-tta --use-prob-ensemble

    # Phase 1.2 -- weight averaging + BN recalibration (secondary comparison)
    python src/evaluate.py --phase phase1_2 --use-weight-avg

Phase 1.2 outputs (all paths from config.yaml['phase1_2']):
  - Confusion matrix PNG (foot-level)
  - Classification report CSV (per-class precision/recall/F1)
  - Test predictions CSV (foot-level: single, TTA, ensemble columns)
  - Subject-level summary CSV (majority-vote across L/R feet per subject)
"""

import argparse
import logging
import os
import sys
from itertools import cycle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

# Project root = thermalDFU/ (one level up from src/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from datasets.thermal_dataset import (
    CLASS_NAMES, CLASS_NAMES_3, SEVERITY_MAP, ThermalDataset,
    build_tta_transforms,
)
from models.efficientnet import (
    build_efficientnet_b0,
    build_efficientnet_b0_v2,
    build_efficientnet_b0_v3,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type == 'cuda'


def project_path(relative: str) -> str:
    """Resolve a project-relative path to an absolute path."""
    return os.path.join(PROJECT_ROOT, relative)


# ---------------------------------------------------------------------------
# Shared output helpers (all phases)
# ---------------------------------------------------------------------------

def save_confusion_matrix(cm: np.ndarray, out_path: str, label_names: list) -> None:
    """Save a normalized confusion matrix heatmap PNG to disk."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(max(6, len(label_names) * 1.5), max(5, len(label_names) * 1.3)))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha='right')
    ax.set_yticklabels(label_names)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('ThermoGuard-DFU -- Test Confusion Matrix (Normalized)')
    for r in range(len(label_names)):
        for c in range(len(label_names)):
            ax.text(c, r, f'{cm_norm[r, c]:.2f}', ha='center', va='center',
                    color='white' if cm_norm[r, c] > 0.5 else 'black', fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close()
    logging.info('Confusion matrix saved: %s', out_path)


def save_classification_report_csv(
    all_labels: list, all_preds: list, label_names: list, out_path: str,
) -> None:
    """Save sklearn classification report as CSV (persistent artifact)."""
    report_dict = classification_report(
        all_labels, all_preds, target_names=label_names,
        output_dict=True, zero_division=0,
    )
    rows = []
    for cls in label_names:
        d = report_dict[cls]
        rows.append({'class': cls, 'precision': round(d['precision'], 6),
                     'recall': round(d['recall'], 6), 'f1': round(d['f1-score'], 6),
                     'support': int(d['support'])})
    for key in ('accuracy', 'macro avg', 'weighted avg'):
        if key == 'accuracy':
            rows.append({'class': key, 'precision': '', 'recall': '',
                         'f1': round(report_dict[key], 6),
                         'support': int(report_dict['weighted avg']['support'])})
        else:
            d = report_dict[key]
            rows.append({'class': key, 'precision': round(d['precision'], 6),
                         'recall': round(d['recall'], 6), 'f1': round(d['f1-score'], 6),
                         'support': int(d['support'])})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logging.info('Classification report CSV saved: %s', out_path)


def save_test_predictions_csv_1_0(
    test_df: pd.DataFrame, all_labels: list, all_preds: list, out_path: str,
) -> None:
    """Save per-sample predictions (Phase 1.0: 6-class)."""
    df = pd.DataFrame({
        'subject_id':      test_df['subject_id'].values,
        'side':            test_df['side'].values,
        'true_class':      [CLASS_NAMES[i] for i in all_labels],
        'predicted_class': [CLASS_NAMES[i] for i in all_preds],
    })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    logging.info('Test predictions CSV saved: %s (%d rows)', out_path, len(df))


def save_test_predictions_csv_1_1(
    test_df: pd.DataFrame, all_labels_3: list, all_preds_3: list,
    all_labels_6: list, out_path: str,
) -> None:
    """Save per-sample predictions (Phase 1.1: 3-class, plus original 6-class label).

    Preserves true_6class for cross-phase error analysis.
    """
    df = pd.DataFrame({
        'subject_id':       test_df['subject_id'].values,
        'side':             test_df['side'].values,
        'true_6class':      [CLASS_NAMES[i] for i in all_labels_6],
        'true_3class':      [CLASS_NAMES_3[i] for i in all_labels_3],
        'predicted_3class': [CLASS_NAMES_3[i] for i in all_preds_3],
    })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    logging.info('Test predictions CSV saved: %s (%d rows)', out_path, len(df))


# ---------------------------------------------------------------------------
# Phase 1.2 -- checkpoint ensemble and TTA helpers (Section 7.3/7.4/7.5)
# ---------------------------------------------------------------------------

def load_ensemble_state_dicts(
    checkpoint_paths: list,
    device: torch.device,
) -> list:
    """Load all plateau checkpoint state dicts into memory once.

    Call once before the evaluation loop; pass the returned list to
    predict_with_checkpoint_ensemble(). Avoids re-loading from disk
    for every test image (52 images x 5 checkpoints = 260 disk reads
    vs 5 with this approach).
    """
    return [
        torch.load(p, map_location=device, weights_only=False)['model_state_dict']
        for p in checkpoint_paths
    ]


def predict_with_checkpoint_ensemble(
    model: nn.Module,
    image: torch.Tensor,
    state_dicts: list,
    device: torch.device,
) -> torch.Tensor:
    """Average softmax predictions across multiple Stage B checkpoints (primary method).

    Each checkpoint uses its own correct BN running stats from its own forward pass.
    Averaging in probability space avoids the weight-averaging BN mismatch problem.
    Returns softmax probability vector (num_classes,).

    Accepts pre-loaded state_dicts (from load_ensemble_state_dicts()) so that
    disk reads happen once outside the per-image loop, not once per image.
    """
    probs_list = []
    for state_dict in state_dicts:
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            with autocast(enabled=USE_AMP):
                probs_list.append(
                    torch.softmax(model(image.unsqueeze(0)), dim=1).squeeze()
                )
    return torch.stack(probs_list).mean(dim=0)


def predict_with_tta(
    model: nn.Module,
    image: torch.Tensor,
    n_augments: int,
    tta_transforms,
) -> torch.Tensor:
    """Average softmax predictions over N augmented views of an image.

    Returns softmax probability vector (num_classes,) averaged across views.
    Original (un-augmented) view is always included as one of the N views.
    """
    model.eval()
    probs_list = []
    with torch.no_grad():
        with autocast(enabled=USE_AMP):
            probs_list.append(
                torch.softmax(model(image.unsqueeze(0)), dim=1).squeeze()
            )
        for _ in range(n_augments - 1):
            aug_img = tta_transforms(image)
            with autocast(enabled=USE_AMP):
                probs_list.append(
                    torch.softmax(model(aug_img.unsqueeze(0)), dim=1).squeeze()
                )
    return torch.stack(probs_list).mean(dim=0)


def load_averaged_checkpoint(
    model: nn.Module,
    checkpoint_paths: list,
    device: torch.device,
) -> nn.Module:
    """Average model weights across multiple Stage B checkpoints.

    Arithmetic mean of all parameter tensors (including BN running stats).
    After calling this, run recalibrate_bn_after_averaging() to resync the
    BN buffers to the new averaged weight distribution.

    Usage:
        model = load_averaged_checkpoint(model, plateau_paths, device)
        recalibrate_bn_after_averaging(model, train_loader, n_batches=20)
    """
    if not checkpoint_paths:
        raise ValueError('At least one checkpoint path required for averaging.')
    state_dicts = [
        torch.load(p, map_location=device, weights_only=False)['model_state_dict']
        for p in checkpoint_paths
    ]
    avg_state = {}
    for key in state_dicts[0]:
        avg_state[key] = torch.stack(
            [sd[key].float() for sd in state_dicts], dim=0
        ).mean(dim=0)
    model.load_state_dict(avg_state)
    logging.info('Weight averaging complete: %d checkpoints averaged.', len(checkpoint_paths))
    return model


def recalibrate_bn_after_averaging(
    model: nn.Module,
    train_loader: DataLoader,
    n_batches: int = 20,
) -> None:
    """Refresh BatchNorm running stats after weight averaging.

    After weight averaging, the BN running_mean/running_var buffers no longer
    match the activation distribution the averaged weights produce. This pass
    runs a brief forward-only phase in train mode to resync them.

    cycle(train_loader) is required: with drop_last=True and 232 train images
    at batch_size=16, the loader yields only 14 batches per epoch. Without
    cycle(), setting n_batches > 14 silently stops at 14 regardless.

    Reference: Izmailov et al. 2018, Stochastic Weight Averaging.
    """
    from itertools import cycle as itertools_cycle
    model.train()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(itertools_cycle(train_loader)):
            if batch_idx >= n_batches:
                break
            images = images.to(DEVICE, non_blocking=True)
            _ = model(images)   # BN running stats update automatically in train mode
    logging.info('BN recalibration complete: %d batches processed.', n_batches)


# ---------------------------------------------------------------------------
# Phase 1.2 -- subject-level metrics
# ---------------------------------------------------------------------------

def compute_subject_level_metrics(
    test_df: pd.DataFrame,
    all_preds: list,
    pred_col_label: str,
) -> pd.DataFrame:
    """Compute subject-level (majority vote) accuracy from foot-level predictions.

    For each subject: both feet correct -> subject correct; both wrong -> subject wrong;
    L/R disagree -> split case.

    Args:
        test_df:       The test manifest DataFrame (52 rows, L+R per subject).
        all_preds:     Foot-level integer predictions aligned with test_df rows.
        pred_col_label: Label for the prediction source (e.g. 'single', 'tta', 'ensemble').

    Returns:
        DataFrame with one row per subject.
    """
    test_df = test_df.copy()
    test_df['pred_idx'] = all_preds
    test_df['pred_name'] = [CLASS_NAMES_3[p] for p in all_preds]
    test_df['true_3class'] = test_df['model_class'].map(SEVERITY_MAP)
    test_df['correct'] = test_df['pred_idx'] == test_df['true_3class']

    rows = []
    for subj_id, grp in test_df.groupby('subject_id'):
        grp = grp.reset_index(drop=True)
        true_cls = grp['true_3class'].iloc[0]  # L/R labels are always identical
        preds = grp['pred_idx'].tolist()
        sides = grp['side'].tolist()
        pred_by_side = {s: p for s, p in zip(sides, preds)}
        pred_left  = pred_by_side.get('L', None)
        pred_right = pred_by_side.get('R', None)
        both_correct = all(p == true_cls for p in preds)
        both_wrong   = all(p != true_cls for p in preds)
        split_case   = not both_correct and not both_wrong
        rows.append({
            'subject_id': subj_id,
            'true_3class': CLASS_NAMES_3[true_cls],
            f'pred_left_{pred_col_label}':  CLASS_NAMES_3[pred_left]  if pred_left  is not None else '',
            f'pred_right_{pred_col_label}': CLASS_NAMES_3[pred_right] if pred_right is not None else '',
            f'subject_correct_{pred_col_label}': both_correct,
            'split_case': split_case,
        })
    return pd.DataFrame(rows)


def log_subject_level_summary(subject_df: pd.DataFrame, pred_col_label: str) -> None:
    """Log subject-level accuracy summary to console."""
    col = f'subject_correct_{pred_col_label}'
    n_subjects = len(subject_df)
    n_correct  = subject_df[col].sum()
    n_wrong    = (~subject_df[col] & ~subject_df['split_case']).sum()
    n_split    = subject_df['split_case'].sum()
    logging.info(
        'Subject-level [%s]: %d/%d correct (%.1f%%) | %d fully wrong | %d split',
        pred_col_label, n_correct, n_subjects,
        100.0 * n_correct / n_subjects if n_subjects > 0 else 0.0,
        n_wrong, n_split,
    )
    logging.info('Phase 1.1 reference: 16/26 correct, 6/26 wrong, 4/26 split')


# ---------------------------------------------------------------------------
# Phase 1.2 -- training curves plot with dual-loss caption
# ---------------------------------------------------------------------------

def plot_training_curves_1_2(log_csv_path: str, out_path: str) -> None:
    """Plot Phase 1.2 training curves with dual-loss formulation note.

    train_loss uses OrdinalWeightedCELoss (inflated up to 5x vs val_loss).
    train_loss_unweighted uses plain CE for apples-to-apples comparison with val_loss.
    Both are plotted on the same axis with a caption noting the formulation difference.
    Overfitting diagnosis uses the val_acc / val_wF1 gap, not the loss curves.
    """
    if not os.path.exists(log_csv_path):
        logging.warning('Training log not found, skipping training curves: %s', log_csv_path)
        return

    df = pd.read_csv(log_csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Phase 1.2 Training Curves -- EfficientNet-B0 Regional', fontsize=13)

    stage_b_start = df[df['stage'] == 'B']['epoch'].min() if 'B' in df['stage'].values else None

    # --- Loss ---
    ax = axes[0]
    ax.plot(df['epoch'], df['train_loss'],             label='train_loss (OrdinalCE)', color='tomato',      linewidth=1.5)
    ax.plot(df['epoch'], df['train_loss_unweighted'],   label='train_loss_unweighted (plain CE)', color='salmon',   linewidth=1.0, linestyle='--')
    ax.plot(df['epoch'], df['val_loss'],                label='val_loss (plain CE)',   color='steelblue',   linewidth=1.5)
    if stage_b_start is not None:
        ax.axvline(x=stage_b_start, color='gray', linestyle=':', label=f'Stage B start (ep {stage_b_start})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend(fontsize=7)
    ax.text(0.01, 0.01,
            'NOTE: train_loss (OrdinalCE) inflated up to 5x vs val_loss (plain CE).\n'
            'Use train_loss_unweighted for apples-to-apples comparison.\n'
            'Overfitting diagnosis: use accuracy/F1 gap, not loss curves.',
            transform=ax.transAxes, fontsize=6, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # --- Accuracy ---
    ax = axes[1]
    ax.plot(df['epoch'], df['train_acc'], label='train_acc', color='tomato',    linewidth=1.5)
    ax.plot(df['epoch'], df['val_acc'],   label='val_acc',   color='steelblue', linewidth=1.5)
    if stage_b_start is not None:
        ax.axvline(x=stage_b_start, color='gray', linestyle=':')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy')
    ax.legend()
    ax.set_ylim(0, 1)

    # --- F1 ---
    ax = axes[2]
    ax.plot(df['epoch'], df['val_weighted_f1'], label='val_weighted_F1', color='steelblue', linewidth=1.5)
    ax.plot(df['epoch'], df['val_macro_f1'],    label='val_macro_F1',    color='navy',      linewidth=1.0, linestyle='--')
    if stage_b_start is not None:
        ax.axvline(x=stage_b_start, color='gray', linestyle=':', label=f'Stage B start (ep {stage_b_start})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1')
    ax.set_title('Validation F1')
    ax.legend()
    ax.set_ylim(0, 1)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close()
    logging.info('Training curves saved: %s', out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def main() -> None:
    """Run test-set evaluation using the best checkpoint for the specified phase."""
    parser = argparse.ArgumentParser(description='ThermoGuard-DFU Phase-Aware Evaluation')
    parser.add_argument(
        '--phase', type=str, default='phase1_0',
        help="Config key to use (e.g. 'phase1_0', 'phase1_1', 'phase1_2'). Default: phase1_0",
    )
    parser.add_argument(
        '--use-tta', action='store_true',
        help='Phase 1.2: apply test-time augmentation (8 views, half training range).',
    )
    parser.add_argument(
        '--use-prob-ensemble', action='store_true',
        help='Phase 1.2: average softmax across plateau checkpoints (primary ensemble method).',
    )
    parser.add_argument(
        '--use-weight-avg', action='store_true',
        help='Phase 1.2: weight-average plateau checkpoints + BN recalibration (secondary).',
    )
    args = parser.parse_args()

    with open(project_path('config.yaml')) as f:
        cfg = yaml.safe_load(f)

    if args.phase not in cfg:
        raise KeyError(
            f"Phase key '{args.phase}' not found in config.yaml. "
            f"Available keys: {list(cfg.keys())}"
        )

    phase_cfg  = cfg[args.phase]
    phase0_out = cfg['phase0_outputs']
    is_phase11 = (args.phase == 'phase1_1')
    is_phase12 = (args.phase == 'phase1_2')

    manifest_path   = project_path(phase0_out['preprocessing_manifest'])
    checkpoint_path = project_path(phase_cfg['out_checkpoint_best'])
    out_cm_path     = project_path(phase_cfg['out_cm_plot'])
    out_report_csv  = project_path(phase_cfg['out_test_report'])
    out_preds_csv   = project_path(phase_cfg['out_test_preds'])

    logging.info('Phase: %s | Device: %s | AMP: %s', args.phase, DEVICE, USE_AMP)
    logging.info('=' * 60)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f'Checkpoint not found: {checkpoint_path}\n'
            f'Run the corresponding train script first.'
        )

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    # --- Build model ---
    if is_phase12:
        label_names = CLASS_NAMES_3
        model = build_efficientnet_b0_v3(
            num_classes=phase_cfg['num_classes'], pretrained=False, cfg=phase_cfg,
        )
    elif is_phase11:
        label_names = CLASS_NAMES_3
        model = build_efficientnet_b0_v2(
            num_classes=phase_cfg['num_classes'], pretrained=False, cfg=phase_cfg,
        )
    else:
        label_names = CLASS_NAMES
        model = build_efficientnet_b0(
            num_classes=phase_cfg['num_classes'], pretrained=False,
        )

    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    logging.info(
        'Checkpoint loaded: epoch=%d, val_weighted_f1=%.4f',
        ckpt.get('epoch', -1), ckpt.get('val_weighted_f1', ckpt.get('best_val_f1', float('nan'))),
    )

    # --- Dataset ---
    dataset_phase = '1_2' if is_phase12 else ('1_1' if is_phase11 else '1_0')
    test_ds = ThermalDataset('test', manifest_path, augment=False, phase=dataset_phase)
    test_loader = DataLoader(
        test_ds,
        batch_size=phase_cfg['batch_size'],
        shuffle=False,
        num_workers=phase_cfg['num_workers'],
        pin_memory=phase_cfg['pin_memory'],
        drop_last=False,
    )

    # ---------------------------------------------------------------------------
    # Standard single-checkpoint evaluation (all phases)
    # ---------------------------------------------------------------------------
    all_preds, all_labels = [], []
    for images, labels in test_loader:
        images = images.to(DEVICE, non_blocking=True)
        preds  = model(images).argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())

    logging.info(
        '\nClassification Report (single checkpoint):\n%s',
        classification_report(all_labels, all_preds, target_names=label_names, zero_division=0),
    )
    logging.info('Test Accuracy:     %.4f', accuracy_score(all_labels, all_preds))
    logging.info('Test Weighted-F1:  %.4f', f1_score(all_labels, all_preds, average='weighted', zero_division=0))
    logging.info('Test Macro-F1:     %.4f', f1_score(all_labels, all_preds, average='macro',    zero_division=0))

    save_confusion_matrix(confusion_matrix(all_labels, all_preds), out_cm_path, label_names)
    save_classification_report_csv(all_labels, all_preds, label_names, out_report_csv)

    # ---------------------------------------------------------------------------
    # Phase 1.0 / 1.1 specific outputs (no further steps)
    # ---------------------------------------------------------------------------
    if is_phase11:
        all_labels_6 = [
            CLASS_NAMES.index(row['model_class']) for _, row in test_ds.df.iterrows()
        ]
        save_test_predictions_csv_1_1(
            test_ds.df, all_labels, all_preds, all_labels_6, out_preds_csv,
        )
        return

    if not is_phase12:
        save_test_predictions_csv_1_0(test_ds.df, all_labels, all_preds, out_preds_csv)
        return

    # ---------------------------------------------------------------------------
    # Phase 1.2 -- TTA and/or probability ensemble
    # ---------------------------------------------------------------------------
    all_labels_6 = [
        CLASS_NAMES.index(row['model_class']) for _, row in test_ds.df.iterrows()
    ]

    # Initialise per-sample probability containers
    all_preds_tta      = []
    all_preds_ensemble = []

    tta_transforms = build_tta_transforms(phase_cfg)
    n_augments     = phase_cfg['tta_n_augments']

    # Find available plateau checkpoints
    ckpt_dir = project_path(os.path.dirname(phase_cfg['out_checkpoint_best']))
    plateau_epoch_keys = phase_cfg['stage_b_checkpoint_epochs']
    stage_b_start_abs  = phase_cfg['stage_a_epochs']
    plateau_paths = []
    for rel_ep in plateau_epoch_keys:
        abs_ep   = stage_b_start_abs + rel_ep
        ckpt_p   = os.path.join(ckpt_dir, f'phase1_2_stage_b_ep{abs_ep}.pth')
        if os.path.exists(ckpt_p):
            plateau_paths.append(ckpt_p)
    if plateau_paths:
        logging.info('Found %d plateau checkpoints for ensemble.', len(plateau_paths))
    else:
        logging.warning(
            'No plateau checkpoints found in %s -- '
            'probability ensemble and weight averaging will be skipped.', ckpt_dir,
        )

    # Load all plateau state dicts once (avoid 52 x N disk reads)
    ensemble_state_dicts = load_ensemble_state_dicts(plateau_paths, DEVICE) if plateau_paths else []

    # Per-image inference
    for i in range(len(test_ds)):
        image, _ = test_ds[i]
        image = image.to(DEVICE)

        # TTA prediction
        if args.use_tta:
            probs_tta = predict_with_tta(model, image, n_augments, tta_transforms)
            all_preds_tta.append(probs_tta.argmax().item())

        # Probability ensemble prediction (primary; also applies TTA if --use-tta is set)
        if args.use_prob_ensemble and ensemble_state_dicts:
            probs_ensemble = predict_with_checkpoint_ensemble(
                model, image, ensemble_state_dicts, DEVICE,
            )
            if args.use_tta:
                # Combined: average across checkpoints AND augmented views
                tta_probs_per_ckpt = []
                for sd in ensemble_state_dicts:
                    model.load_state_dict(sd)
                    model.eval()
                    probs_ckpt = predict_with_tta(model, image, n_augments, tta_transforms)
                    tta_probs_per_ckpt.append(probs_ckpt)
                probs_ensemble = torch.stack(tta_probs_per_ckpt).mean(dim=0)
            all_preds_ensemble.append(probs_ensemble.argmax().item())

    # Restore best checkpoint weights for any post-loop code
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # ---------------------------------------------------------------------------
    # Phase 1.2 -- weight averaging + BN recalibration (secondary comparison)
    # ---------------------------------------------------------------------------
    all_preds_weight_avg = []
    if args.use_weight_avg and plateau_paths:
        logging.info('Running weight averaging + BN recalibration (secondary method)...')
        train_ds = ThermalDataset('train', manifest_path, augment=True, phase='1_2')
        train_loader_recal = DataLoader(
            train_ds,
            batch_size=phase_cfg['batch_size'],
            shuffle=True,
            num_workers=phase_cfg['num_workers'],
            pin_memory=phase_cfg['pin_memory'],
            drop_last=True,
        )
        load_averaged_checkpoint(model, plateau_paths, DEVICE)
        recalibrate_bn_after_averaging(model, train_loader_recal, n_batches=phase_cfg['bn_recal_n_batches'])
        # Save averaged checkpoint
        avg_ckpt_path = project_path(phase_cfg['out_checkpoint_avg'])
        os.makedirs(os.path.dirname(avg_ckpt_path), exist_ok=True)
        torch.save({'model_state_dict': model.state_dict(), 'source': 'weight_average_bn_recal'}, avg_ckpt_path)
        logging.info('Weight-averaged + BN-recalibrated checkpoint saved: %s', avg_ckpt_path)

        model.eval()
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(DEVICE, non_blocking=True)
                preds  = model(images).argmax(dim=1)
                all_preds_weight_avg.extend(preds.cpu().tolist())
        logging.info(
            'Weight avg test Weighted-F1: %.4f',
            f1_score(all_labels, all_preds_weight_avg, average='weighted', zero_division=0),
        )

    # ---------------------------------------------------------------------------
    # Phase 1.2 -- foot-level predictions CSV (all inference modes)
    # ---------------------------------------------------------------------------
    preds_df = pd.DataFrame({
        'subject_id':        test_ds.df['subject_id'].values,
        'side':              test_ds.df['side'].values,
        'true_6class':       [CLASS_NAMES[i]   for i in all_labels_6],
        'true_3class':       [CLASS_NAMES_3[i] for i in all_labels],
        'predicted_3class':  [CLASS_NAMES_3[i] for i in all_preds],
    })
    if all_preds_tta:
        preds_df['predicted_3class_tta'] = [CLASS_NAMES_3[i] for i in all_preds_tta]
    if all_preds_ensemble:
        preds_df['predicted_3class_ensemble'] = [CLASS_NAMES_3[i] for i in all_preds_ensemble]
    if all_preds_weight_avg:
        preds_df['predicted_3class_weight_avg'] = [CLASS_NAMES_3[i] for i in all_preds_weight_avg]
    os.makedirs(os.path.dirname(project_path(out_preds_csv)), exist_ok=True)
    preds_df.to_csv(project_path(out_preds_csv), index=False)
    logging.info('Foot-level predictions CSV saved: %s', out_preds_csv)

    # ---------------------------------------------------------------------------
    # Phase 1.2 -- subject-level metrics
    # ---------------------------------------------------------------------------
    subj_dfs = []
    subj_df_single = compute_subject_level_metrics(test_ds.df, all_preds, 'single')
    log_subject_level_summary(subj_df_single, 'single')
    subj_dfs.append(subj_df_single)

    if all_preds_tta:
        subj_df_tta = compute_subject_level_metrics(test_ds.df, all_preds_tta, 'tta')
        log_subject_level_summary(subj_df_tta, 'tta')
        subj_dfs.append(subj_df_tta.drop(columns=['true_3class', 'split_case']))

    if all_preds_ensemble:
        subj_df_ens = compute_subject_level_metrics(test_ds.df, all_preds_ensemble, 'ensemble')
        log_subject_level_summary(subj_df_ens, 'ensemble')
        subj_dfs.append(subj_df_ens.drop(columns=['true_3class', 'split_case']))

    if all_preds_weight_avg:
        subj_df_wavg = compute_subject_level_metrics(test_ds.df, all_preds_weight_avg, 'weight_avg')
        log_subject_level_summary(subj_df_wavg, 'weight_avg')
        subj_dfs.append(subj_df_wavg.drop(columns=['true_3class', 'split_case']))

    import functools
    subject_report = functools.reduce(
        lambda l, r: l.merge(r, on='subject_id'), subj_dfs,
    )
    subject_report_path = project_path(phase_cfg['out_subject_report'])
    os.makedirs(os.path.dirname(subject_report_path), exist_ok=True)
    subject_report.to_csv(subject_report_path, index=False)
    logging.info('Subject-level report CSV saved: %s', subject_report_path)

    # ---------------------------------------------------------------------------
    # Phase 1.2 -- training curves
    # ---------------------------------------------------------------------------
    log_csv_path = project_path(phase_cfg['out_metrics_log'])
    plot_training_curves_1_2(log_csv_path, project_path(phase_cfg['out_curves_plot']))

    # ---------------------------------------------------------------------------
    # Phase 1.2 -- anti-mirror safety check (acceptance threshold)
    # ---------------------------------------------------------------------------
    logging.info('--- Anti-mirror safety check ---')
    under_count = sum(1 for t, p in zip(all_labels, all_preds) if t > p)
    over_count  = sum(1 for t, p in zip(all_labels, all_preds) if p > t)
    error_count = sum(1 for t, p in zip(all_labels, all_preds) if t != p)
    logging.info(
        'Errors: total=%d | under-estimation=%d | over-estimation=%d',
        error_count, under_count, over_count,
    )
    if over_count > under_count:
        logging.warning(
            'ANTI-MIRROR CHECK FAILED: over-estimation (%d) > under-estimation (%d). '
            'Ordinal loss may have flipped the bias. Consider reducing ordinal_under_penalty.',
            over_count, under_count,
        )
    elif error_count > 0 and under_count / error_count >= 0.875:
        logging.warning(
            'Under-estimation rate %.1f%% still >= 87.5%% threshold. '
            'Ordinal loss may not have reduced systematic under-estimation.',
            100.0 * under_count / error_count,
        )
    else:
        logging.info(
            'Anti-mirror check passed. Under-estimation rate: %.1f%%.',
            100.0 * under_count / error_count if error_count > 0 else 0.0,
        )

    logging.info('Phase 1.2 evaluation complete.')


if __name__ == '__main__':
    main()
