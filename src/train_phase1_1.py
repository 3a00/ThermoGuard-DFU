"""Phase 1.1 -- EfficientNet-B0 Upgraded Training Script.

Upgrades over Phase 1.0:
  - Bilinear resize + hard NEAREST mask from original 128x64 array (in ThermalDataset).
  - ConcatPool2d (GAP+GMP) -> 2560-dim feature vector.
  - 2-layer regularized bottleneck head: Dropout(0.4)->Linear(2560,256)->BN1d->ReLU
    ->Dropout(0.3)->Linear(256,3).
  - 3-class severity labels: Healthy=0, Low_Severity=1, High_Severity=2.
  - Stage-split weight decay: 1e-4 (Stage A warm-up), 1e-3 (Stage B fine-tune).
  - Stage A: 8 epochs (extended for BN1d stabilization).
  - Stage B: features[7] only -- selective unfreeze (not full backbone).
  - Stage B BN fix: model.features.eval() then model.features[7].train() per epoch.
  - drop_last=True on train DataLoader (BN1d crash prevention).
  - 3-class weights computed at runtime from manifest (Phase 0 files read-only).

All hyperparameters from config.yaml['phase1_1'] -- no hardcoded numbers.
Run from project root:
    cd thermalDFU && source venv/bin/activate && python src/train_phase1_1.py
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

# Project root detection -- works regardless of invocation directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from datasets.thermal_dataset import CLASS_NAMES_3, SEVERITY_MAP, ThermalDataset
from models.efficientnet import (
    build_efficientnet_b0_v2,
    freeze_all_backbone,
    unfreeze_top_block_only,
)

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

PHASE_CFG = CFG['phase1_1']
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP    = DEVICE.type == 'cuda'
SEED       = CFG['seed']

MANIFEST_PATH = project_path(CFG['phase0_outputs']['preprocessing_manifest'])

CHECKPOINT_BEST = project_path(PHASE_CFG['out_checkpoint_best'])
CHECKPOINT_LAST = project_path(PHASE_CFG['out_checkpoint_last'])
METRICS_CSV     = project_path(PHASE_CFG['out_metrics_log'])
CURVES_PLOT     = project_path(PHASE_CFG['out_curves_plot'])

# Val Low_Severity denominator -- verified from manifest (4xDM_Grade0 + 4xDM_Grade1 + 6xDM_Grade2).
VAL_LOW_SEVERITY_N = 14


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker for reproducibility."""
    np.random.seed(SEED + worker_id)


def compute_3class_weights(manifest_path: str, severity_map: dict) -> torch.Tensor:
    """Compute class weights from 3-class severity distribution (train split only).

    Uses sklearn balanced formula: weight[c] = n_samples / (n_classes * n_per_class[c]).
    Does NOT use class_weights.json -- that file stores 6-class weights (Phase 0.8,
    read-only). Computes fresh 3-class weights from the manifest at runtime.
    """
    df = pd.read_csv(manifest_path)
    train_df = df[df['split'] == 'train']
    counts = {0: 0, 1: 0, 2: 0}
    for _, row in train_df.iterrows():
        counts[severity_map[row['model_class']]] += 1
    n_total   = sum(counts.values())
    n_classes = 3
    weights   = [n_total / (n_classes * counts[c]) for c in range(n_classes)]
    logging.info('3-class train counts (Healthy/Low/High): %s', dict(counts))
    logging.info('3-class weights (Healthy/Low/High): %s', [round(w, 6) for w in weights])
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_weighted: nn.Module,
    scaler: torch.amp.GradScaler,
    clip_gradients: bool,
    max_grad_norm: float,
    stage: str,
) -> tuple[float, float]:
    """Run one training epoch with correct BN freeze pattern per stage.

    Stage A (stage='A'):
        model.train() then model.features.eval() -- entire backbone BN frozen.
        Only the classifier head (including BN1d) is in train mode.

    Stage B (stage='B'):
        model.train() then model.features.eval() -- entire backbone BN frozen.
        Then model.features[7].train() -- re-enable ONLY features[7] BN stat updates.
        This is the critical fix: requires_grad=False alone does NOT prevent BN
        running stats from drifting when model.train() is called. The explicit
        per-block override is required for selective BN freeze.

    Args:
        clip_gradients: Apply clip_grad_norm_ (Stage B only).
        max_grad_norm:  From config.yaml['phase1_1']['max_grad_norm'].
        stage:          'A' or 'B' -- controls BN freeze pattern.
    """
    model.train()               # baseline: all submodules to train mode
    model.features.eval()       # freeze backbone BN running stats (both stages)
    if stage == 'B':
        model.features[7].train()   # re-enable BN stat updates for selectively unfrozen block

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
            scaler.unscale_(optimizer)   # required before clip_grad_norm_ with AMP
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
    """Evaluate on val or test split (no grad, no augmentation).

    Val/test loss computed UNWEIGHTED for unbiased generalization signal.
    Returns: avg_loss, accuracy, weighted_f1, macro_f1, all_preds, all_labels.
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
    """Save a training checkpoint with full metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    logging.info('  Checkpoint saved: %s', path)


def _save_training_curves(log_df: pd.DataFrame) -> None:
    """Save 4-panel training curve figure: loss, accuracy, weighted F1, macro F1."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    axes[0].plot(log_df['epoch'], log_df['train_loss'], label='Train', color='#4C72B0')
    axes[0].plot(log_df['epoch'], log_df['val_loss'],   label='Val',   color='#DD8452')
    axes[0].set_title('Loss per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(log_df['epoch'], log_df['train_acc'], label='Train', color='#4C72B0')
    axes[1].plot(log_df['epoch'], log_df['val_acc'],   label='Val',   color='#DD8452')
    axes[1].set_title('Accuracy per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    axes[2].plot(log_df['epoch'], log_df['val_weighted_f1'], color='#55A868')
    axes[2].set_title('Val Weighted F1')
    axes[2].set_xlabel('Epoch')

    axes[3].plot(log_df['epoch'], log_df['val_macro_f1'], color='#C44E52')
    axes[3].set_title('Val Macro F1')
    axes[3].set_xlabel('Epoch')

    # Stage A / B boundary line
    stage_b = log_df[log_df['stage'] == 'B']
    if not stage_b.empty:
        for ax in axes:
            ax.axvline(x=stage_b['epoch'].min() - 0.5, color='gray', linestyle='--', lw=0.8)

    fig.suptitle('ThermoGuard-DFU Phase 1.1 -- EfficientNet-B0 Upgraded Training Curves', fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(CURVES_PLOT), exist_ok=True)
    fig.savefig(CURVES_PLOT, dpi=150)
    plt.close()
    logging.info('Training curves saved: %s', CURVES_PLOT)


def main() -> None:
    """Phase 1.1 main training loop."""
    logging.info('Phase 1.1: EfficientNet-B0 Upgraded (3-class severity, selective freeze)')
    logging.info('=' * 70)
    logging.info('Device: %s | AMP: %s | Seed: %d', DEVICE, USE_AMP, SEED)
    set_seed(SEED)

    for d in (
        os.path.dirname(CHECKPOINT_BEST),
        os.path.dirname(METRICS_CSV),
        os.path.dirname(CURVES_PLOT),
    ):
        os.makedirs(d, exist_ok=True)

    # -- DataLoaders -----------------------------------------------------------
    batch_size  = PHASE_CFG['batch_size']
    num_workers = PHASE_CFG['num_workers']
    pin_memory  = PHASE_CFG['pin_memory']

    train_ds = ThermalDataset('train', MANIFEST_PATH, augment=True,  phase='1_1')
    val_ds   = ThermalDataset('val',   MANIFEST_PATH, augment=False, phase='1_1')

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,          # BN1d safety: prevent batch-of-1 crash
        worker_init_fn=_worker_init_fn,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # -- Model -----------------------------------------------------------------
    model = build_efficientnet_b0_v2(
        num_classes=PHASE_CFG['num_classes'],
        pretrained=PHASE_CFG['pretrained'],
        cfg=PHASE_CFG,
    ).to(DEVICE)

    # -- Loss functions --------------------------------------------------------
    class_weights = compute_3class_weights(MANIFEST_PATH, SEVERITY_MAP).to(DEVICE)
    criterion_w   = nn.CrossEntropyLoss(weight=class_weights)
    criterion_uw  = nn.CrossEntropyLoss()

    # -- Stage config ----------------------------------------------------------
    stage_a_epochs = PHASE_CFG['stage_a_epochs']   # 8
    stage_b_epochs = PHASE_CFG['stage_b_epochs']   # 45
    lr_a           = PHASE_CFG['lr_stage_a']
    lr_b           = PHASE_CFG['lr_stage_b']
    wd_a           = PHASE_CFG['weight_decay_stage_a']   # 1e-4 mild during warm-up
    wd_b           = PHASE_CFG['weight_decay_stage_b']   # 1e-3 strong during fine-tune
    max_grad_norm  = PHASE_CFG['max_grad_norm']
    eta_min        = PHASE_CFG['eta_min']
    total_epochs   = stage_a_epochs + stage_b_epochs    # 53 (not in config)

    # -- Stage A setup ---------------------------------------------------------
    freeze_all_backbone(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_a,
        weight_decay=wd_a,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=stage_a_epochs, eta_min=eta_min,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    metrics_rows, best_val_f1, in_stage_b = [], 0.0, False

    # -- Training loop ---------------------------------------------------------
    for epoch in range(1, total_epochs + 1):

        # Transition to Stage B
        if epoch == stage_a_epochs + 1:
            logging.info('--- Stage B: Selective fine-tuning (features[7] + head) ---')
            unfreeze_top_block_only(model)
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr_b,
                weight_decay=wd_b,
            )
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
            stage=stage_label,
        )
        val_loss, val_acc, val_wf1, val_mf1, val_preds, val_labels = evaluate(
            model, val_loader, criterion_uw,
        )
        scheduler.step()

        logging.info(
            '  train_loss=%.4f  train_acc=%.4f  val_loss=%.4f  val_acc=%.4f  '
            'val_wf1=%.4f  val_mf1=%.4f',
            train_loss, train_acc, val_loss, val_acc, val_wf1, val_mf1,
        )

        # Low_Severity recall -- reported as pct + raw count/denominator.
        report_dict = classification_report(
            val_labels, val_preds,
            target_names=CLASS_NAMES_3,
            output_dict=True,
            zero_division=0,
        )
        low_sev_recall_pct = report_dict.get('Low_Severity', {}).get('recall', 0.0)
        low_sev_n          = round(low_sev_recall_pct * VAL_LOW_SEVERITY_N)
        logging.info(
            '  Low_Severity val recall = %.1f%% (%d/%d val samples)',
            low_sev_recall_pct * 100, low_sev_n, VAL_LOW_SEVERITY_N,
        )

        # Warning flag: swing > 20pp between consecutive epochs (= ~3 samples).
        if metrics_rows:
            prev_recall = metrics_rows[-1]['low_severity_recall_pct']
            if abs(low_sev_recall_pct - prev_recall) > 0.20:
                logging.warning(
                    '  [MONITOR] Low_Severity recall swung >20pp from epoch %d '
                    '(%.1f%% -> %.1f%%) -- ~%d sample(s) changed. Normal noise given '
                    'only %d val samples.',
                    epoch - 1, prev_recall * 100, low_sev_recall_pct * 100,
                    abs(low_sev_n - metrics_rows[-1]['low_severity_recall_n']),
                    VAL_LOW_SEVERITY_N,
                )

        # Checkpoint (no off-by-one: update best_val_f1 BEFORE building state dict).
        new_best = val_wf1 > best_val_f1
        if new_best:
            best_val_f1 = val_wf1

        ckpt = {
            'phase':                '1_1',
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_weighted_f1':      val_wf1,
            'val_macro_f1':         val_mf1,
            'best_val_f1':          best_val_f1,
            'config':               PHASE_CFG,
            'timestamp':            datetime.now(timezone.utc).isoformat(),
            'seed':                 SEED,
            'device':               str(DEVICE),
            'class_names':          CLASS_NAMES_3,
            'input_resolution':     [PHASE_CFG['resize_h'], PHASE_CFG['resize_w']],
            'torch_version':        torch.__version__,
            'torchvision_version':  torchvision.__version__,
        }
        save_checkpoint(ckpt, CHECKPOINT_LAST)
        if new_best:
            save_checkpoint(ckpt, CHECKPOINT_BEST)
            logging.info('  ** New best val weighted-F1: %.4f **', best_val_f1)

        metrics_rows.append({
            'epoch':                      epoch,
            'stage':                      stage_label,
            'lr':                         current_lr,
            'train_loss':                 round(train_loss, 6),
            'val_loss':                   round(val_loss, 6),
            'train_acc':                  round(train_acc, 6),
            'val_acc':                    round(val_acc, 6),
            'val_weighted_f1':            round(val_wf1, 6),
            'val_macro_f1':               round(val_mf1, 6),
            'low_severity_recall_pct':    round(low_sev_recall_pct, 6),
            'low_severity_recall_n':      low_sev_n,
            'low_severity_recall_denom':  VAL_LOW_SEVERITY_N,
        })
        pd.DataFrame(metrics_rows).to_csv(METRICS_CSV, index=False)

    _save_training_curves(pd.DataFrame(metrics_rows))
    logging.info('Phase 1.1 Training Complete. Best val weighted-F1: %.4f', best_val_f1)
    logging.info('Next: python src/evaluate.py --phase 1_1')


if __name__ == '__main__':
    main()
