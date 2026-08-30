"""Phase 1.0 -- EfficientNet-B0 Training Script.

New location: src/train_phase1_0.py
Usage:  cd <project_root> && source venv/bin/activate && python src/train_phase1_0.py

All paths resolved from PROJECT_ROOT (thermalDFU/ directory) via config.yaml.
No hardcoded paths or numbers in this script (AGENTS.md rule).

Two-stage transfer learning:
  Stage A (epochs 1-5):  Backbone frozen + eval mode (true BN freeze),
                         head-only training, lr=1e-3, no grad clipping.
  Stage B (epochs 6-50): Full fine-tuning, model.train() (BN adapts),
                         lr=1e-4 with CosineAnnealingLR, gradient clipping.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import yaml
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

# Project root = thermalDFU/ (one level up from src/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from datasets.thermal_dataset import CLASS_NAMES, ThermalDataset
from models.efficientnet import build_efficientnet_b0, freeze_backbone, unfreeze_all

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)


def project_path(relative: str) -> str:
    """Resolve a project-relative path to an absolute path."""
    return os.path.join(PROJECT_ROOT, relative)


with open(project_path('config.yaml')) as _f:
    CFG = yaml.safe_load(_f)

PHASE_CFG  = CFG['phase1_0']
PHASE0_OUT = CFG['phase0_outputs']

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type == 'cuda'
SEED    = CFG['seed']

MANIFEST_PATH   = project_path(PHASE0_OUT['preprocessing_manifest'])
WEIGHTS_PATH    = project_path(PHASE0_OUT['class_weights'])

CHECKPOINT_BEST = project_path(PHASE_CFG['out_checkpoint_best'])
CHECKPOINT_LAST = project_path(PHASE_CFG['out_checkpoint_last'])
METRICS_CSV     = project_path(PHASE_CFG['out_metrics_log'])
CURVES_PLOT     = project_path(PHASE_CFG['out_curves_plot'])


def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker for reproducibility."""
    np.random.seed(SEED + worker_id)


def load_class_weights() -> torch.Tensor:
    """Load class weights from outputs/phase0_prep/class_weights.json (Phase 0.8)."""
    with open(WEIGHTS_PATH) as f:
        w = json.load(f)
    weights = torch.tensor(w['weights_as_tensor_order'], dtype=torch.float32)
    logging.info('Class weights loaded: %s', weights.tolist())
    return weights


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_weighted: nn.Module,
    scaler: torch.amp.GradScaler,
    clip_gradients: bool,
    max_grad_norm: float,
    freeze_backbone_bn: bool,
) -> tuple[float, float]:
    """Run one training epoch.

    Args:
        clip_gradients:    Apply gradient clipping (Stage B only).
        max_grad_norm:     Max grad norm from config.yaml.
        freeze_backbone_bn: If True, call model.features.eval() after model.train()
                           to freeze BN running stats (Stage A true freeze).
    """
    model.train()
    if freeze_backbone_bn:
        model.features.eval()

    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            logits = model(images)
            loss   = criterion_weighted(logits, labels)

        scaler.scale(loss).backward()

        if clip_gradients:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion_unweighted: nn.Module,
) -> tuple[float, float, float, float, list, list]:
    """Evaluate on val or test split.

    Returns: avg_loss, accuracy, weighted_f1, macro_f1, all_preds, all_labels.
    Val loss computed UNWEIGHTED for unbiased generalization signal.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        logits = model(images)
        loss   = criterion_unweighted(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    wf1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    mf1 = f1_score(all_labels, all_preds, average='macro',    zero_division=0)
    return total_loss / total, correct / total, wf1, mf1, all_preds, all_labels


def save_checkpoint(state: dict, path: str) -> None:
    """Save a training checkpoint dict to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    logging.info('  Checkpoint saved: %s', path)


def _save_training_curves(log_df: pd.DataFrame) -> None:
    """Save 4-panel training curves: loss, accuracy, weighted F1, macro F1."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    axes[0].plot(log_df['epoch'], log_df['train_loss'], label='Train', color='#4C72B0')
    axes[0].plot(log_df['epoch'], log_df['val_loss'],   label='Val',   color='#DD8452')
    axes[0].set_title('Loss per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(log_df['epoch'], log_df['train_acc'], label='Train', color='#4C72B0')
    axes[1].plot(log_df['epoch'], log_df['val_acc'],   label='Val',   color='#DD8452')
    axes[1].set_title('Accuracy per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    axes[2].plot(log_df['epoch'], log_df['val_weighted_f1'], color='#55A868')
    axes[2].set_title('Val Weighted F1')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Weighted F1')

    axes[3].plot(log_df['epoch'], log_df['val_macro_f1'], color='#C44E52')
    axes[3].set_title('Val Macro F1')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Macro F1')

    stage_b = log_df[log_df['stage'] == 'B']
    if not stage_b.empty:
        for ax in axes:
            ax.axvline(x=stage_b['epoch'].min() - 0.5, color='gray', linestyle='--', lw=0.8)

    fig.suptitle('ThermoGuard-DFU Phase 1.0 -- EfficientNet-B0 Training Curves', fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(CURVES_PLOT), exist_ok=True)
    fig.savefig(CURVES_PLOT, dpi=150)
    plt.close()
    logging.info('Training curves saved: %s', CURVES_PLOT)


def main() -> None:
    """Phase 1.0 -- EfficientNet-B0 training entry point."""
    logging.info('Phase 1.0: EfficientNet-B0 Baseline Training')
    logging.info('=' * 60)
    logging.info('Device: %s | AMP: %s | Seed: %d', DEVICE, USE_AMP, SEED)
    set_seed(SEED)

    os.makedirs(os.path.dirname(CHECKPOINT_BEST), exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(CURVES_PLOT), exist_ok=True)

    batch_size  = PHASE_CFG['batch_size']
    num_workers = PHASE_CFG['num_workers']
    pin_memory  = PHASE_CFG['pin_memory']

    train_ds = ThermalDataset('train', MANIFEST_PATH, augment=True)
    val_ds   = ThermalDataset('val',   MANIFEST_PATH, augment=False)

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    model = build_efficientnet_b0(
        num_classes=PHASE_CFG['num_classes'],
        pretrained=PHASE_CFG['pretrained'],
    ).to(DEVICE)

    class_weights = load_class_weights().to(DEVICE)
    criterion_w   = nn.CrossEntropyLoss(weight=class_weights)
    criterion_uw  = nn.CrossEntropyLoss()

    stage_a_epochs = PHASE_CFG['stage_a_epochs']
    stage_b_epochs = PHASE_CFG['stage_b_epochs']
    lr_a           = PHASE_CFG['lr_stage_a']
    lr_b           = PHASE_CFG['lr_stage_b']
    wd             = PHASE_CFG['weight_decay']
    max_grad_norm  = PHASE_CFG['max_grad_norm']
    eta_min        = PHASE_CFG['eta_min']

    freeze_backbone(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_a, weight_decay=wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=stage_a_epochs, eta_min=eta_min,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    metrics_rows, best_val_f1, in_stage_b = [], 0.0, False
    total_epochs = stage_a_epochs + stage_b_epochs

    for epoch in range(1, total_epochs + 1):
        if epoch == stage_a_epochs + 1:
            logging.info('--- Stage B: Full fine-tuning ---')
            unfreeze_all(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_b, weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=stage_b_epochs, eta_min=eta_min,
            )
            scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
            in_stage_b = True

        stage_label = 'B' if in_stage_b else 'A'
        current_lr  = optimizer.param_groups[0]['lr']
        logging.info(
            'Epoch %d/%d  [Stage %s]  lr=%.2e', epoch, total_epochs, stage_label, current_lr,
        )

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion_w, scaler,
            clip_gradients=in_stage_b,
            max_grad_norm=max_grad_norm,
            freeze_backbone_bn=(not in_stage_b),
        )
        val_loss, val_acc, val_f1, val_macro, val_preds, val_labels = evaluate(
            model, val_loader, criterion_uw,
        )
        scheduler.step()

        logging.info(
            '  train_loss=%.4f  train_acc=%.4f  val_loss=%.4f  val_acc=%.4f  '
            'val_wf1=%.4f  val_mf1=%.4f',
            train_loss, train_acc, val_loss, val_acc, val_f1, val_macro,
        )

        report_dict  = classification_report(
            val_labels, val_preds, target_names=CLASS_NAMES,
            output_dict=True, zero_division=0,
        )
        dm0_recall = report_dict.get('DM_Grade0', {}).get('recall', 0.0)
        logging.info('  DM_Grade0 val recall = %.4f', dm0_recall)

        new_best = val_f1 > best_val_f1
        if new_best:
            best_val_f1 = val_f1

        ckpt = {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_weighted_f1':      val_f1,
            'val_macro_f1':         val_macro,
            'best_val_f1':          best_val_f1,
            'config':               PHASE_CFG,
            'timestamp':            datetime.now(timezone.utc).isoformat(),
            'seed':                 SEED,
            'device':               str(DEVICE),
            'class_names':          CLASS_NAMES,
            'input_resolution':     [PHASE_CFG['resize_h'], PHASE_CFG['resize_w']],
            'torch_version':        torch.__version__,
            'torchvision_version':  torchvision.__version__,
        }
        save_checkpoint(ckpt, CHECKPOINT_LAST)
        if new_best:
            save_checkpoint(ckpt, CHECKPOINT_BEST)
            logging.info('  ** New best val weighted-F1: %.4f **', best_val_f1)

        metrics_rows.append({
            'epoch': epoch, 'stage': stage_label, 'lr': current_lr,
            'train_loss': round(train_loss, 6), 'val_loss': round(val_loss, 6),
            'train_acc':  round(train_acc, 6),  'val_acc':  round(val_acc, 6),
            'val_weighted_f1': round(val_f1, 6), 'val_macro_f1': round(val_macro, 6),
            'dm_grade0_recall': round(dm0_recall, 6),
        })
        pd.DataFrame(metrics_rows).to_csv(METRICS_CSV, index=False)

    _save_training_curves(pd.DataFrame(metrics_rows))
    logging.info('=' * 60)
    logging.info('Phase 1.0 Training Complete. Best val weighted-F1: %.4f', best_val_f1)
    logging.info('Checkpoint (best): %s', CHECKPOINT_BEST)
    logging.info('Checkpoint (last): %s', CHECKPOINT_LAST)
    logging.info('Metrics CSV:       %s', METRICS_CSV)
    logging.info('Next: python src/evaluate.py --phase phase1_0')


if __name__ == '__main__':
    main()
