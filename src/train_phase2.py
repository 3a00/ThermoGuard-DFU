"""Phase 2 Training Script — ViT-Tiny Vision Transformer Baseline.

Trains ViTTinyThermal on plantar thermogram severity classification (3 classes:
Healthy, Low_Severity, High_Severity) using a two-stage transfer learning schedule.
Matches Phase 1.2 optimization parameters for a controlled head-to-head comparison.

Key architectural & training loop rules:
  - Input: [B, 3, 224, 112] from ThermalDataset (phase='2'), padded internally to 224x224.
  - Stage A (8 epochs): Head warmup only; backbone frozen in eval() mode.
  - Stage B (45 epochs): Blocks 10-11, norm, and head fine-tuned; blocks 0-9 frozen in eval().
  - DO NOT CALL model.train() in the training loop. Use restore_stage_modes() to
    re-establish selective train/eval modes after evaluation without triggering the
    recursive submodule train() footgun.
  - Tripwires: assert_stage_a_modes(model) and assert_stage_b_modes(model) called at
    the start of each epoch.
  - Loss: OrdinalWeightedCELoss (under_penalty=2.0, over_penalty=1.0) on train; plain CE on val.
  - Observability: High & Low Severity recall tracked, worker RNG diversified, TensorBoard scalars.

Usage:
    source venv/bin/activate
    python src/train_phase2.py
"""

import csv
import logging
import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from datasets.thermal_dataset import CLASS_NAMES_3, SEVERITY_MAP, ThermalDataset
from models.vit_tiny import (
    assert_stage_a_modes,
    assert_stage_b_modes,
    build_vit_tiny,
    freeze_backbone,
    unfreeze_stage_b,
)
from utils.losses import OrdinalWeightedCELoss

# ---------------------------------------------------------------------------
# Class Index Constants
# ---------------------------------------------------------------------------
CLASS_HEALTHY_IDX = 0
CLASS_LOW_SEVERITY_IDX = 1
CLASS_HIGH_SEVERITY_IDX = 2

# ---------------------------------------------------------------------------
# Logging & Config
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)

with open(os.path.join(PROJECT_ROOT, 'config.yaml')) as config_file:
    CFG_ALL = yaml.safe_load(config_file)

CFG = CFG_ALL['phase2']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type == 'cuda'
SEED = CFG_ALL.get('seed', 42)


def project_path(relative: str) -> str:
    """Resolve a project-relative path to an absolute path."""
    return os.path.join(PROJECT_ROOT, relative)


def set_seed(seed: int) -> None:
    """Set all random seeds for deterministic execution."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker process uniquely to ensure augmentation diversity."""
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


def ensure_dirs() -> None:
    """Create all output directories declared in config."""
    paths = [
        CFG['out_checkpoint_best'],
        CFG['out_checkpoint_last'],
        CFG['out_metrics_log'],
        CFG['out_curves_plot'],
    ]
    for target_path in paths:
        os.makedirs(project_path(os.path.dirname(target_path)), exist_ok=True)
    os.makedirs(project_path(CFG['out_tensorboard_dir']), exist_ok=True)


def compute_3class_weights(manifest_path: str) -> torch.Tensor:
    """Compute balanced 3-class weights from train split manifest counts."""
    manifest_df = pd.read_csv(manifest_path)
    train_df = manifest_df[manifest_df['split'] == 'train']
    counts = {0: 0, 1: 0, 2: 0}
    for _, row in train_df.iterrows():
        counts[SEVERITY_MAP[row['model_class']]] += 1

    n_total = sum(counts.values())
    n_classes = CFG['num_classes']
    weights = [n_total / (n_classes * counts[cls_idx]) for cls_idx in range(n_classes)]
    logging.info('Train sample counts (Healthy/Low/High): %s', dict(counts))
    for cls_idx, class_name in enumerate(CLASS_NAMES_3):
        logging.info('Class weight: %s (idx=%d) = %.4f', class_name, cls_idx, weights[cls_idx])
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Mode restoration helper (Resolves post-evaluation model.eval() reset)
# ---------------------------------------------------------------------------
def restore_stage_modes(model: nn.Module, stage: str, unfreeze_blocks: int = 2) -> None:
    """Re-establish stage-specific train/eval module modes after validation.

    Never call model.train() directly. This safely restores active submodules
    to train() mode while keeping frozen submodules strictly in eval() mode.
    """
    if stage == 'A':
        freeze_backbone(model)
    elif stage == 'B':
        unfreeze_stage_b(model, unfreeze_blocks=unfreeze_blocks)
    else:
        raise ValueError(f"Unknown stage: {stage}")


# ---------------------------------------------------------------------------
# Training & Evaluation routines
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: OrdinalWeightedCELoss,
    criterion_plain: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    stage: str,
) -> Tuple[float, float, float]:
    """Execute one training epoch. Returns (avg_ordinal_loss, avg_plain_loss, train_acc)."""
    total_ordinal_loss = 0.0
    total_plain_loss = 0.0
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

        # Numerical stability guard against FP16 overflow under weighted loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss encountered in Stage {stage}: {loss.item()}")

        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if stage == 'B':
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda param: param.requires_grad, model.parameters()),
                    max_norm=CFG['max_grad_norm'],
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if stage == 'B':
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda param: param.requires_grad, model.parameters()),
                    max_norm=CFG['max_grad_norm'],
                )
            optimizer.step()

        total_ordinal_loss += loss.item() * images.size(0)
        total_plain_loss += loss_plain.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_ordinal = total_ordinal_loss / total
    avg_plain = total_plain_loss / total
    acc = correct / total
    return avg_ordinal, avg_plain, acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion_plain: nn.CrossEntropyLoss,
) -> Tuple[float, float, float, float, float, float]:
    """Evaluate model on validation split.

    Returns:
        (val_loss, val_acc, val_wf1, val_mf1, low_rec_pct, high_rec_pct)
    """
    model.eval()  # Puts full model in eval mode for inference
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
    acc = sum(pred == label for pred, label in zip(all_preds, all_labels)) / len(all_labels)
    wf1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    mf1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Low_Severity recall (Class 1)
    low_corr = sum(
        pred == label and label == CLASS_LOW_SEVERITY_IDX
        for pred, label in zip(all_preds, all_labels)
    )
    low_tot = sum(label == CLASS_LOW_SEVERITY_IDX for label in all_labels)
    low_rec = (low_corr / low_tot * 100.0) if low_tot > 0 else 0.0

    # High_Severity recall (Class 2) — Critical for clinical under-diagnosis monitoring
    high_corr = sum(
        pred == label and label == CLASS_HIGH_SEVERITY_IDX
        for pred, label in zip(all_preds, all_labels)
    )
    high_tot = sum(label == CLASS_HIGH_SEVERITY_IDX for label in all_labels)
    high_rec = (high_corr / high_tot * 100.0) if high_tot > 0 else 0.0

    return avg_loss, acc, wf1, mf1, low_rec, high_rec


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_wf1: float,
    path: str,
) -> None:
    """Save model and optimizer checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_weighted_f1': val_wf1,
    }, project_path(path))


def plot_training_curves(log_path: str, output_path: str) -> None:
    """Generate and save loss and F1 training curves from CSV log."""
    df = pd.read_csv(log_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curves
    axes[0].plot(df['epoch'], df['train_loss'], label='Train Ordinal Loss', color='tab:blue', alpha=0.7)
    axes[0].plot(df['epoch'], df['train_loss_unweighted'], label='Train Unweighted Loss', color='tab:blue', linestyle='--')
    axes[0].plot(df['epoch'], df['val_loss'], label='Val Plain CE Loss', color='tab:red')
    axes[0].axvline(x=CFG['stage_a_epochs'], color='grey', linestyle=':', label='Stage B Transition')
    axes[0].set_title('Loss Curves')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1 and Accuracy Curves
    axes[1].plot(df['epoch'], df['val_weighted_f1'], label='Val Weighted F1', color='tab:green')
    axes[1].plot(df['epoch'], df['val_macro_f1'], label='Val Macro F1', color='tab:orange')
    axes[1].plot(df['epoch'], df['val_acc'], label='Val Accuracy', color='tab:purple', linestyle='--')
    axes[1].axvline(x=CFG['stage_a_epochs'], color='grey', linestyle=':', label='Stage B Transition')
    axes[1].set_title('Validation Metrics')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Main Orchestration Loop
# ---------------------------------------------------------------------------
def main() -> None:
    """Run full Phase 2 training pipeline."""
    set_seed(SEED)
    ensure_dirs()

    logging.info('Phase 2 ViT-Tiny training starting on %s (AMP=%s)', DEVICE, USE_AMP)
    tb_dir = project_path(CFG['out_tensorboard_dir'])
    writer = SummaryWriter(log_dir=tb_dir)
    logging.info('TensorBoard SummaryWriter initialized at: %s', tb_dir)

    manifest_path = project_path(CFG_ALL['phase0_outputs']['preprocessing_manifest'])

    # Pre-flight dataset sanity check
    df_manifest = pd.read_csv(manifest_path)
    split_counts = {
        split_name: len(df_manifest[df_manifest['split'] == split_name])
        for split_name in ['train', 'val', 'test']
    }
    assert split_counts == {'train': 232, 'val': 50, 'test': 52}, f"Unexpected split counts: {split_counts}"
    logging.info('Dataset verified: train=232, val=50, test=52')

    train_ds = ThermalDataset('train', manifest_path, augment=True, phase='2')
    val_ds = ThermalDataset('val', manifest_path, augment=False, phase='2')

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG['batch_size'],
        shuffle=True,
        num_workers=CFG['num_workers'],
        pin_memory=CFG['pin_memory'],
        worker_init_fn=_worker_init_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG['batch_size'],
        shuffle=False,
        num_workers=CFG['num_workers'],
        pin_memory=CFG['pin_memory'],
    )

    class_weights = compute_3class_weights(manifest_path).to(DEVICE)

    # Instantiate model
    model = build_vit_tiny(
        num_classes=CFG['num_classes'],
        pretrained=CFG['pretrained'],
        dropout=CFG['dropout'],
        pad_width=(CFG['pad_left'], CFG['pad_right']),
    ).to(DEVICE)

    criterion = OrdinalWeightedCELoss(
        class_weights=class_weights,
        under_penalty=CFG['ordinal_under_penalty'],
        over_penalty=CFG['ordinal_over_penalty'],
    )
    criterion_plain = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=USE_AMP)

    # Initialize CSV log file with both High and Low severity recall
    log_path = project_path(CFG['out_metrics_log'])
    with open(log_path, 'w', newline='') as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow([
            'epoch', 'stage', 'lr_backbone', 'lr_head',
            'train_loss', 'train_loss_unweighted', 'val_loss',
            'train_acc', 'val_acc', 'val_weighted_f1', 'val_macro_f1',
            'low_severity_recall_pct', 'high_severity_recall_pct',
        ])

    best_val_wf1 = 0.0

    # -----------------------------------------------------------------------
    # Stage A: Head Warm-up (8 epochs)
    # -----------------------------------------------------------------------
    logging.info('=== Starting Stage A: Head Warm-up (%d epochs) ===', CFG['stage_a_epochs'])
    freeze_backbone(model)

    # Controlled Phase 1.2 comparison: Adam optimizer
    optimizer_a = torch.optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=CFG['lr_stage_a'],
        weight_decay=CFG['weight_decay_stage_a'],
    )
    scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_a,
        T_max=CFG['stage_a_epochs'],
        eta_min=CFG['eta_min'],
    )

    for epoch in range(1, CFG['stage_a_epochs'] + 1):
        restore_stage_modes(model, stage='A')
        assert_stage_a_modes(model)  # Tripwire validation

        train_loss, train_loss_raw, train_acc = train_one_epoch(
            model, train_loader, criterion, criterion_plain,
            optimizer_a, scaler, stage='A',
        )

        val_loss, val_acc, val_wf1, val_mf1, low_rec, high_rec = evaluate(
            model, val_loader, criterion_plain,
        )

        scheduler_a.step()
        lr_head = scheduler_a.get_last_lr()[0]
        lr_backbone = 0.0

        # Logging & Checkpointing
        if val_wf1 > best_val_wf1:
            best_val_wf1 = val_wf1
            save_checkpoint(model, optimizer_a, epoch, val_wf1, CFG['out_checkpoint_best'])
            logging.info('  -> Best val wF1 updated: %.4f', best_val_wf1)
        save_checkpoint(model, optimizer_a, epoch, val_wf1, CFG['out_checkpoint_last'])

        # TensorBoard
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/train_unweighted', train_loss_raw, epoch)
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('metrics/val_accuracy', val_acc, epoch)
        writer.add_scalar('metrics/val_weighted_f1', val_wf1, epoch)
        writer.add_scalar('metrics/val_macro_f1', val_mf1, epoch)
        writer.add_scalar('metrics/val_low_severity_recall', low_rec, epoch)
        writer.add_scalar('metrics/val_high_severity_recall', high_rec, epoch)
        writer.add_scalar('lr/head', lr_head, epoch)
        writer.add_scalar('lr/backbone', lr_backbone, epoch)

        # CSV Log
        with open(log_path, 'a', newline='') as log_file:
            csv.writer(log_file).writerow([
                epoch, 'A', f'{lr_backbone:.6e}', f'{lr_head:.6e}',
                f'{train_loss:.6f}', f'{train_loss_raw:.6f}', f'{val_loss:.6f}',
                f'{train_acc:.6f}', f'{val_acc:.6f}', f'{val_wf1:.6f}', f'{val_mf1:.6f}',
                f'{low_rec:.2f}', f'{high_rec:.2f}',
            ])

        logging.info(
            'Stage A  Epoch %2d/%d  lr=%.2e  train_loss=%.4f (raw=%.4f)  '
            'val_loss=%.4f  val_acc=%.4f  val_wF1=%.4f  val_mF1=%.4f  LowRec=%.1f%%  HighRec=%.1f%%',
            epoch, CFG['stage_a_epochs'], lr_head, train_loss, train_loss_raw,
            val_loss, val_acc, val_wf1, val_mf1, low_rec, high_rec,
        )

    # -----------------------------------------------------------------------
    # Stage B: Fine-Tuning Top-2 Blocks + LayerNorm + Head (45 epochs)
    # -----------------------------------------------------------------------
    logging.info('=== Starting Stage B: Fine-Tuning (%d epochs) ===', CFG['stage_b_epochs'])
    unfreeze_stage_b(model, unfreeze_blocks=CFG['unfreeze_blocks'])

    # Baseline: Uniform Adam matching Phase 1.2 for controlled head-to-head comparison
    # (Note: Differential LR and AdamW are recorded for Ticket 05 ablation)
    optimizer_b = torch.optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=CFG['lr_stage_b'],
        weight_decay=CFG['weight_decay_stage_b'],
    )
    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_b,
        T_max=CFG['stage_b_epochs'],
        eta_min=CFG['eta_min'],
    )

    for epoch_b in range(1, CFG['stage_b_epochs'] + 1):
        global_epoch = CFG['stage_a_epochs'] + epoch_b

        restore_stage_modes(model, stage='B', unfreeze_blocks=CFG['unfreeze_blocks'])
        assert_stage_b_modes(model, unfreeze_blocks=CFG['unfreeze_blocks'])  # Tripwire

        train_loss, train_loss_raw, train_acc = train_one_epoch(
            model, train_loader, criterion, criterion_plain,
            optimizer_b, scaler, stage='B',
        )

        val_loss, val_acc, val_wf1, val_mf1, low_rec, high_rec = evaluate(
            model, val_loader, criterion_plain,
        )

        scheduler_b.step()
        current_lr = scheduler_b.get_last_lr()[0]
        lr_head = current_lr
        lr_backbone = current_lr

        # Logging & Checkpointing (single best val weighted-F1 checkpoint)
        if val_wf1 > best_val_wf1:
            best_val_wf1 = val_wf1
            save_checkpoint(model, optimizer_b, global_epoch, val_wf1, CFG['out_checkpoint_best'])
            logging.info('  -> Best val wF1 updated: %.4f', best_val_wf1)
        save_checkpoint(model, optimizer_b, global_epoch, val_wf1, CFG['out_checkpoint_last'])

        # TensorBoard
        writer.add_scalar('loss/train', train_loss, global_epoch)
        writer.add_scalar('loss/train_unweighted', train_loss_raw, global_epoch)
        writer.add_scalar('loss/val', val_loss, global_epoch)
        writer.add_scalar('metrics/val_accuracy', val_acc, global_epoch)
        writer.add_scalar('metrics/val_weighted_f1', val_wf1, global_epoch)
        writer.add_scalar('metrics/val_macro_f1', val_mf1, global_epoch)
        writer.add_scalar('metrics/val_low_severity_recall', low_rec, global_epoch)
        writer.add_scalar('metrics/val_high_severity_recall', high_rec, global_epoch)
        writer.add_scalar('lr/head', lr_head, global_epoch)
        writer.add_scalar('lr/backbone', lr_backbone, global_epoch)

        # CSV Log
        with open(log_path, 'a', newline='') as log_file:
            csv.writer(log_file).writerow([
                global_epoch, 'B', f'{lr_backbone:.6e}', f'{lr_head:.6e}',
                f'{train_loss:.6f}', f'{train_loss_raw:.6f}', f'{val_loss:.6f}',
                f'{train_acc:.6f}', f'{val_acc:.6f}', f'{val_wf1:.6f}', f'{val_mf1:.6f}',
                f'{low_rec:.2f}', f'{high_rec:.2f}',
            ])

        logging.info(
            'Stage B  Epoch %2d/%d (Tot %2d)  lr=%.2e  train_loss=%.4f (raw=%.4f)  '
            'val_loss=%.4f  val_acc=%.4f  val_wF1=%.4f  val_mF1=%.4f  LowRec=%.1f%%  HighRec=%.1f%%',
            epoch_b, CFG['stage_b_epochs'], global_epoch, current_lr, train_loss,
            train_loss_raw, val_loss, val_acc, val_wf1, val_mf1, low_rec, high_rec,
        )

    writer.close()
    plot_training_curves(log_path, project_path(CFG['out_curves_plot']))
    logging.info('Phase 2 training complete. Best val weighted-F1: %.4f', best_val_wf1)


if __name__ == '__main__':
    main()
