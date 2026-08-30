"""Phase 1.2 Training Script — EfficientNet-B0 Regional (Spatial Angiosome Pooling).

Upgrades over Phase 1.1 (train_phase1_1.py):
  - SpatialConcatPool2d (2x2 quadrant GAP+GMP) -> 5120-dim feature vector
    (vs ConcatPool2d -> 2560-dim in Phase 1.1)
  - OrdinalWeightedCELoss (under_penalty=2.0, over_penalty=1.0) targeting
    the 14/16 under-estimation error pattern from Phase 1.1
  - Plateau checkpoints saved at configurable epochs for ensemble averaging
  - train_loss_unweighted logged alongside train_loss for fair visual comparison
    (OrdinalCE inflates train_loss up to 5x vs plain CE val_loss)
  - Optional features[8] ablation via config: unfreeze_features8 (default: false)

Identical to Phase 1.1:
  - Two-stage transfer learning: Stage A (head warmup) + Stage B (selective fine-tune)
  - Stage B BN pattern: model.features.eval() + model.features[7].train()
  - Bilinear resize + NEAREST mask preprocessing (phase='1_2' in ThermalDataset)
  - 3-class severity formulation (Healthy / Low_Severity / High_Severity)
  - drop_last=True on train DataLoader (BN1d safety)
  - CosineAnnealingLR per stage, Adam with per-stage weight decay
  - Seed=42 for reproducibility

Usage (from project root with venv activated):
    source venv/bin/activate
    python src/train_phase1_2.py

Pre-flight check:
    python -c "
    import json, pandas as pd
    d = json.load(open('outputs/phase0_prep/class_weights.json'))
    m = pd.read_csv('outputs/phase0_prep/preprocessing_manifest.csv')
    counts = {s: len(m[m.split==s]) for s in ['train','val','test']}
    assert counts == {'train':232,'val':50,'test':52}, f'Split mismatch: {counts}'
    print('Pre-flight OK:', counts)
    "
"""

import csv
import logging
import os
import sys

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from datasets.thermal_dataset import CLASS_NAMES_3, ThermalDataset
from models.efficientnet import (
    build_efficientnet_b0_v3,
    freeze_all_backbone,
    unfreeze_top_block_only,
    unfreeze_top_two_blocks,
)
from utils.losses import OrdinalWeightedCELoss

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)

# ---------------------------------------------------------------------------
# Config and constants
# ---------------------------------------------------------------------------

with open(os.path.join(PROJECT_ROOT, 'config.yaml')) as _f:
    CFG_ALL = yaml.safe_load(_f)

CFG = CFG_ALL['phase1_2']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type == 'cuda'
SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def project_path(relative: str) -> str:
    """Resolve a project-relative path to an absolute path."""
    return os.path.join(PROJECT_ROOT, relative)


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    """Create all output directories declared in config if they do not exist."""
    paths = [
        CFG['out_checkpoint_best'],
        CFG['out_checkpoint_last'],
        CFG['out_metrics_log'],
        CFG['out_test_report'],
        CFG['out_test_preds'],
        CFG['out_subject_report'],
        CFG['out_curves_plot'],
        CFG['out_cm_plot'],
        CFG['out_checkpoint_avg'],
    ]
    for path in paths:
        os.makedirs(project_path(os.path.dirname(path)), exist_ok=True)


def compute_class_weights(train_dataset: ThermalDataset) -> torch.Tensor:
    """Compute balanced 3-class weights at runtime from the training split.

    Formula: weight[c] = n_samples / (n_classes * n_samples_per_class[c])
    This mirrors sklearn's 'balanced' strategy and is computed from the CURRENT
    split -- no dependency on the locked Phase 0 class_weights.json (which has
    6-class weights).
    """
    labels = [train_dataset.df.iloc[i]['model_class'] for i in range(len(train_dataset))]
    from datasets.thermal_dataset import SEVERITY_MAP
    severity_labels = [SEVERITY_MAP[lbl] for lbl in labels]
    n_total = len(severity_labels)
    n_classes = CFG['num_classes']
    weights = []
    for cls in range(n_classes):
        n_cls = severity_labels.count(cls)
        weights.append(n_total / (n_classes * n_cls) if n_cls > 0 else 1.0)
    w_tensor = torch.tensor(weights, dtype=torch.float32)
    for cls_idx, cls_name in enumerate(CLASS_NAMES_3):
        logging.info('Class weight: %s (idx=%d) = %.4f', cls_name, cls_idx, w_tensor[cls_idx])
    return w_tensor


# ---------------------------------------------------------------------------
# Per-epoch training and validation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: OrdinalWeightedCELoss,
    criterion_plain: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    stage: str,
    unfreeze_f8: bool,
) -> tuple[float, float, float]:
    """Run one training epoch. Returns (train_loss, train_loss_unweighted, train_acc)."""
    model.train()
    # BN freeze pattern (Stage B): freeze all backbone BN, then selectively re-enable.
    if stage == 'B':
        model.features.eval()
        model.features[7].train()
        if unfreeze_f8:
            model.features[8].train()
    # Stage A: features.eval() handled by freeze_all_backbone() gradient freeze +
    # the model.train() call above triggers BN accumulation -- but requires eval():
    if stage == 'A':
        model.features.eval()

    total_loss = 0.0
    total_loss_plain = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=USE_AMP):
            logits = model(images)
            loss = criterion(logits, labels)
            loss_plain = criterion_plain(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            max_norm=CFG['max_grad_norm'],
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        total_loss_plain += loss_plain.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    avg_loss_plain = total_loss_plain / total
    acc = correct / total
    return avg_loss, avg_loss_plain, acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion_plain: nn.CrossEntropyLoss,
) -> tuple[float, float, float, float]:
    """Run evaluation. Returns (val_loss, val_acc, val_weighted_f1, val_macro_f1)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast(enabled=USE_AMP):
            logits = model(images)
            loss = criterion_plain(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    w_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    m_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, w_f1, m_f1


def compute_low_severity_recall(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[float, int, int]:
    """Compute Low_Severity (class 1) recall. Returns (recall_pct, correct_n, total_n)."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            with autocast(enabled=USE_AMP):
                logits = model(images)
            preds = logits.argmax(dim=1)
            mask = labels == 1  # Low_Severity = class index 1
            correct += ((preds == labels) & mask).sum().item()
            total += mask.sum().item()
    recall = correct / total if total > 0 else 0.0
    return recall * 100.0, correct, total


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_wf1: float,
    path: str,
) -> None:
    """Save a training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_weighted_f1': val_wf1,
    }, project_path(path))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Phase 1.2 training: Stage A (head warmup) + Stage B (selective fine-tune)."""
    set_seed(SEED)
    ensure_dirs()

    logging.info('Phase 1.2 training started. Device: %s. AMP: %s.', DEVICE, USE_AMP)
    logging.info('Config: %s epochs Stage A + %s epochs Stage B. unfreeze_features8=%s.',
                 CFG['stage_a_epochs'], CFG['stage_b_epochs'], CFG['unfreeze_features8'])

    # --- Pre-flight check ---
    manifest_path = project_path(CFG_ALL['phase0_outputs']['preprocessing_manifest'])
    import pandas as pd
    df_manifest = pd.read_csv(manifest_path)
    split_counts = {s: len(df_manifest[df_manifest['split'] == s]) for s in ['train', 'val', 'test']}
    assert split_counts == {'train': 232, 'val': 50, 'test': 52}, \
        f'Pre-flight FAILED -- unexpected split counts: {split_counts}'
    logging.info('Pre-flight OK: train=%d, val=%d, test=%d.', 232, 50, 52)

    # --- Datasets ---
    train_ds = ThermalDataset(
        'train', manifest_path, augment=True, phase='1_2',
    )
    val_ds = ThermalDataset(
        'val', manifest_path, augment=False, phase='1_2',
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG['batch_size'],
        shuffle=True,
        num_workers=CFG['num_workers'],
        pin_memory=CFG['pin_memory'],
        drop_last=True,           # BN1d crash prevention: never send batch of size 1
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG['batch_size'],
        shuffle=False,
        num_workers=CFG['num_workers'],
        pin_memory=CFG['pin_memory'],
    )

    # --- Class weights (computed at runtime from training split) ---
    class_weights = compute_class_weights(train_ds).to(DEVICE)

    # --- Model ---
    model = build_efficientnet_b0_v3(
        num_classes=CFG['num_classes'],
        pretrained=CFG['pretrained'],
        cfg=CFG,
    ).to(DEVICE)

    # --- Loss functions ---
    # Training: ordinal-aware asymmetric loss (inflates train_loss up to 5x vs val_loss)
    criterion = OrdinalWeightedCELoss(
        class_weights=class_weights,
        under_penalty=CFG['ordinal_under_penalty'],
        over_penalty=CFG['ordinal_over_penalty'],
    )
    # Unweighted plain CE for: (a) val/test loss (apples-to-apples), (b) train_loss_unweighted column
    criterion_plain = nn.CrossEntropyLoss()
    # Weighted plain CE for val/test loss reporting (same as Phase 1.1 for fair comparison)
    criterion_val = nn.CrossEntropyLoss()

    # --- AMP scaler ---
    scaler = GradScaler(enabled=USE_AMP)

    # --- Training log CSV header ---
    log_path = project_path(CFG['out_metrics_log'])
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'stage', 'lr',
            'train_loss', 'train_loss_unweighted', 'val_loss',
            'train_acc', 'val_acc', 'val_weighted_f1', 'val_macro_f1',
            'low_severity_recall_pct', 'low_severity_recall_n', 'low_severity_recall_denom',
        ])

    best_val_wf1 = 0.0
    stage_b_checkpoint_epochs = set(CFG['stage_b_checkpoint_epochs'])
    total_epochs = CFG['stage_a_epochs'] + CFG['stage_b_epochs']
    # Absolute epoch numbers for Stage B plateau checkpoints (1-indexed from overall training)
    # Stage B starts at epoch = stage_a_epochs + 1 (1-indexed).
    # Config values [30, 35, 40, 45, 50] are relative to Stage B start (epoch 1 of Stage B).
    # Map to absolute epoch numbers:
    stage_b_start_abs = CFG['stage_a_epochs']  # after completing Stage A epochs
    plateau_ckpt_abs_epochs = {
        stage_b_start_abs + rel_ep for rel_ep in stage_b_checkpoint_epochs
    }

    # ---------------------------------------------------------------------------
    # Stage A: Freeze backbone, warm up head
    # ---------------------------------------------------------------------------
    logging.info('--- Stage A: head warmup (%d epochs) ---', CFG['stage_a_epochs'])
    freeze_all_backbone(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG['lr_stage_a'],
        weight_decay=CFG['weight_decay_stage_a'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CFG['stage_a_epochs'],
        eta_min=CFG['eta_min'],
    )

    for epoch in range(1, CFG['stage_a_epochs'] + 1):
        train_loss, train_loss_plain, train_acc = train_one_epoch(
            model, train_loader, criterion, criterion_plain,
            optimizer, scaler, stage='A', unfreeze_f8=False,
        )
        val_loss, val_acc, val_wf1, val_mf1 = evaluate(model, val_loader, criterion_val)
        low_rec, low_n, low_denom = compute_low_severity_recall(model, val_loader)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        logging.info(
            'Stage A  Epoch %2d/%d  lr=%.2e  train_loss=%.4f (raw=%.4f)  '
            'val_loss=%.4f  train_acc=%.4f  val_acc=%.4f  val_wF1=%.4f  val_mF1=%.4f  '
            'LowRec=%.1f%% (%d/%d)',
            epoch, CFG['stage_a_epochs'], current_lr, train_loss, train_loss_plain,
            val_loss, train_acc, val_acc, val_wf1, val_mf1, low_rec, low_n, low_denom,
        )

        # Save best checkpoint
        if val_wf1 > best_val_wf1:
            best_val_wf1 = val_wf1
            save_checkpoint(model, optimizer, epoch, val_wf1, CFG['out_checkpoint_best'])
            logging.info('  -> New best val wF1: %.4f  (checkpoint saved)', best_val_wf1)

        # Save last checkpoint (overwrite each epoch)
        save_checkpoint(model, optimizer, epoch, val_wf1, CFG['out_checkpoint_last'])

        # Log to CSV
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, 'A', f'{current_lr:.6e}',
                f'{train_loss:.6f}', f'{train_loss_plain:.6f}', f'{val_loss:.6f}',
                f'{train_acc:.6f}', f'{val_acc:.6f}', f'{val_wf1:.6f}', f'{val_mf1:.6f}',
                f'{low_rec:.2f}', low_n, low_denom,
            ])

    # ---------------------------------------------------------------------------
    # Stage B: Selective fine-tune
    # ---------------------------------------------------------------------------
    unfreeze_f8 = CFG['unfreeze_features8']
    logging.info(
        '--- Stage B: selective fine-tune (%d epochs, unfreeze_features8=%s) ---',
        CFG['stage_b_epochs'], unfreeze_f8,
    )
    if unfreeze_f8:
        unfreeze_top_two_blocks(model)
    else:
        unfreeze_top_block_only(model)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG['lr_stage_b'],
        weight_decay=CFG['weight_decay_stage_b'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CFG['stage_b_epochs'],
        eta_min=CFG['eta_min'],
    )

    for epoch_b in range(1, CFG['stage_b_epochs'] + 1):
        epoch_abs = CFG['stage_a_epochs'] + epoch_b  # absolute epoch number
        train_loss, train_loss_plain, train_acc = train_one_epoch(
            model, train_loader, criterion, criterion_plain,
            optimizer, scaler, stage='B', unfreeze_f8=unfreeze_f8,
        )
        val_loss, val_acc, val_wf1, val_mf1 = evaluate(model, val_loader, criterion_val)
        low_rec, low_n, low_denom = compute_low_severity_recall(model, val_loader)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        logging.info(
            'Stage B  Epoch %2d/%d (abs %2d)  lr=%.2e  train_loss=%.4f (raw=%.4f)  '
            'val_loss=%.4f  train_acc=%.4f  val_acc=%.4f  val_wF1=%.4f  val_mF1=%.4f  '
            'LowRec=%.1f%% (%d/%d)',
            epoch_b, CFG['stage_b_epochs'], epoch_abs, current_lr,
            train_loss, train_loss_plain, val_loss, train_acc, val_acc,
            val_wf1, val_mf1, low_rec, low_n, low_denom,
        )

        # Save best checkpoint
        if val_wf1 > best_val_wf1:
            best_val_wf1 = val_wf1
            save_checkpoint(model, optimizer, epoch_abs, val_wf1, CFG['out_checkpoint_best'])
            logging.info('  -> New best val wF1: %.4f  (checkpoint saved)', best_val_wf1)

        # Save last checkpoint (overwrite each epoch)
        save_checkpoint(model, optimizer, epoch_abs, val_wf1, CFG['out_checkpoint_last'])

        # Save plateau checkpoint if this absolute epoch is in the configured list
        if epoch_abs in plateau_ckpt_abs_epochs:
            ckpt_dir = os.path.dirname(project_path(CFG['out_checkpoint_best']))
            ckpt_path = os.path.join(ckpt_dir, f'phase1_2_stage_b_ep{epoch_abs}.pth')
            save_checkpoint(model, optimizer, epoch_abs, val_wf1, os.path.relpath(ckpt_path, PROJECT_ROOT))
            logging.info('  -> Plateau checkpoint saved: %s', os.path.basename(ckpt_path))

        # Log to CSV
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch_abs, 'B', f'{current_lr:.6e}',
                f'{train_loss:.6f}', f'{train_loss_plain:.6f}', f'{val_loss:.6f}',
                f'{train_acc:.6f}', f'{val_acc:.6f}', f'{val_wf1:.6f}', f'{val_mf1:.6f}',
                f'{low_rec:.2f}', low_n, low_denom,
            ])

    logging.info(
        'Training complete. Best val wF1: %.4f. '
        'Training log: %s',
        best_val_wf1, log_path,
    )
    logging.info(
        'Next step: python src/evaluate.py --phase 1_2  '
        '(optionally with --use-tta and/or --use-prob-ensemble)',
    )


if __name__ == '__main__':
    main()
