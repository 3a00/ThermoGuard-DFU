# ThermoGuard-DFU

**Deep Learning-Driven Diabetic Foot Ulcer (DFU) Risk Stratification from Plantar Thermograms**  
*Graduation Capstone Project*

---

## Overview

The pipeline is benchmarked on the **IEEE Plantar Thermogram Database** (167 subjects: 122 diabetic, 45 control; 334 feet total) using deterministic subject-level splitting (116 train / 25 val / 26 test) and fixed physical temperature normalization ($[15^\circ\text{C}, 35^\circ\text{C}]$).

---

## Clinical Formulation & Class Hierarchy

Following Architecture Decision Record [0001](docs/adr/0001-three-class-severity-grouping.md), the system stratifies patients into three ordinal risk categories:

| Class Index | Clinical Category | Included TCI Grades | Clinical Meaning |
| :---: | :--- | :--- | :--- |
| **0** | **Healthy** | Control Group (CG) | Normal autonomic regulation, symmetric thermal pattern. |
| **1** | **Low Severity** | DM Grade 0 & Grade 1 | Early-stage neurovascular dysfunction, mild hyperthermia/hypothermia. |
| **2** | **High Severity** | DM Grade 2, 3 & 4 | Advanced neuropathy/ischemia, critical ulceration or amputation risk. |

### Loss Function: Ordinal Weighted Cross-Entropy
To penalize clinically catastrophic classification errors (e.g., predicting *Healthy* for a *High Severity* patient) more heavily than adjacent mistakes, models are trained with **Ordinal Weighted Cross-Entropy Loss** ([ADR 0002](docs/adr/0002-ordinal-weighted-cross-entropy-loss.md)):
$$\mathcal{L} = \text{weight}[y] \times \left(1.0 + \text{penalty}(y, \hat{y})\right) \times \mathcal{L}_{\text{CE}}$$
where underestimating severity carries double the penalty of overestimating ($\text{under\_penalty}=2.0$, $\text{over\_penalty}=1.0$).

---

## Architectural Evolution & Performance

| Phase | Architecture | Key Novelties | Test Accuracy | Test Weighted F1 |
| :--- | :--- | :--- | :---: | :---: |
| **Phase 1.0** | EfficientNet-B0 (6-class) | Nearest-neighbor resize, standard CE | 38.46% | 0.3267 |
| **Phase 1.1** | EfficientNet-B0 (3-class) | Bilinear+mask, ConcatPool, selective unfreeze | 69.23% | 0.7024 |
| **Phase 1.2** | **Regional EfficientNet-B0** | `SpatialConcatPool2d` (2×2), OrdinalCE, TTA + Ensemble | **80.77%** | **0.8104** |
| **Phase 2.0** | **ViT-Tiny Transformer** | Patch self-attention, aspect-ratio padding, OrdinalCE | **88.46%** | **0.8842** |

---

## Phase 1.2 Deep Dive: Regional CNN Architecture

[`src/models/efficientnet.py`](src/models/efficientnet.py) (`EfficientNetThermalV3`)

Standard Global Average Pooling (GAP) collapses 2D feature maps into a 1D vector, discarding spatial localization. In plantar thermography, pathological hotspots are localized to specific vascular angiosomes (LCA, LPA, MCA, MPA).

```text
+---------------------------------------------------------------------------------+
|                   PHASE 1.2: REGIONAL CNN ARCHITECTURE                          |
+---------------------------------------------------------------------------------+

                               [ Raw Thermogram (CSV) ]
                                          |
                                          v
+---------------------------------------------------------------------------------+
|  Phase 0 Preprocessing Pipeline:                                                |
|   - Fixed Physical Normalization: [15°C, 35°C] (Preserves absolute gradients)   |
|   - Anatomical Alignment: Horizontal flip for Left feet (Canonical orientation) |
|   - Background Zero-Masking: Explicitly set non-foot pixels to 0.0              |
+-------------------------------------+-------------------------------------------+
                                      |
                                      v
                     [ Canonical Array: 128 x 64 x 1 ]
                                      |
                                      v
                  +---------------------------------------+
                  |   EfficientNet-B0 Backbone (Stage B)  |
                  |   - Pretrained on ImageNet-1k         |
                  |   - Lower CNN layers:  LR = 1e-5      |
                  |   - Classification:    LR = 1e-4      |
                  +-------------------+-------------------+
                                      |
                                      v
                     [ Spatial Feature Map: 7 x 4 x 1280 ]
                                      |
                                      v
                  +---------------------------------------+
                  |      SpatialConcatPool2d (2 x 2)      |
                  |   Splits into 4 Anatomical Quadrants: |
                  |    * Forefoot Medial / Lateral (MCA)  |
                  |    * Heel Medial / Lateral     (LPA)  |
                  +-------------------+-------------------+
                                      |
                 +--------------------+--------------------+
                 |                                         |
                 v                                         v
    +-------------------------+               +-------------------------+
    |  Regional Avg Pooling   |               |   Regional Max Pooling  |
    |  (4 regions x 1280-d)   |               |   (4 regions x 1280-d)  |
    +------------+------------+               +------------+------------+
                 |                                         |
                 +--------------------+--------------------+
                                      |
                                      v
                  [ Concatenated Regional Vector: 10,240-d ]
                                      |
                                      v
                  +---------------------------------------+
                  |       Linear Classification Head      |
                  |       - BatchNorm1d + Dropout (0.4)   |
                  |       - Linear Projection (10k -> 3)  |
                  +-------------------+-------------------+
                                      |
    - - - - - - - - - - - - - - - - - + - - - - - - - - - - - - - - - - -
    | Training Objective              | Inference Pipeline
    v                                 v
  +---------------------------+     +-----------------------------------------+
  |  Ordinal Weighted CELoss  |     |  Test-Time Augmentation (TTA)           |
  |   - Class-balanced weights|     |   + Probability Checkpoint Ensemble     |
  |   - Under-penalty: 2.0x   |     +--------------------+--------------------+
  |   - Over-penalty:  1.0x   |                          |
  +---------------------------+                          v
                                    +-----------------------------------------+
                                    |       3-Class Risk Stratification       |
                                    |  [0] Healthy        ->  100.0% Recall   |
                                    |  [1] Low Severity   ->   78.6% Prec/Rec |
                                    |  [2] High Severity  ->   95.5% Precision|
                                    |  Overall Test: 80.77% Acc | 0.8104 wF1  |
                                    +-----------------------------------------+
```

### Core Components
1. **`SpatialConcatPool2d` (2×2 Quadrant Pooling):**
   * Divides the final $7 \times 4$ convolutional feature map into a $2 \times 2$ grid (quadrants corresponding to Forefoot Left/Right and Heel Left/Right).
   * Computes both Average Pooling and Max Pooling per quadrant, yielding $4 \text{ quadrants} \times 2 \text{ modes} = 8$ regional feature vectors.
   * Concatenates them into a 10,240-dimensional localized descriptor ($1280 \times 8$).
2. **Two-Stage Fine-Tuning:**
   * **Stage A (Warmup):** Backbone frozen, training only the regional projection head and BatchNorm.
   * **Stage B (Discriminative Unfreeze):** Lower CNN layers trained with $\text{LR}=10^{-5}$, classification head trained with $\text{LR}=10^{-4}$.
3. **Inference Optimization:**
   * **Test-Time Augmentation (TTA):** Multi-view evaluation combining identity, subtle vertical flips, and contrast perturbations.
   * **Probability Ensembling:** Checkpoint ensembling across top validation plateau epochs.
   * **Result:** Reached **80.77% Test Accuracy** and **0.8104 Weighted F1**.

---

## Phase 2 Deep Dive: Vision Transformer (ViT-Tiny)

[`src/models/vit_tiny.py`](src/models/vit_tiny.py) (`ViTTinyThermal`)

Phase 2 investigates whether multi-head self-attention can capture long-range bilateral thermal correlations across non-adjacent angiosomes without inductive spatial bias.

```text
+---------------------------------------------------------------------------------+
|                   PHASE 2: ViT-TINY VISION TRANSFORMER                          |
+---------------------------------------------------------------------------------+

                         [ Canonical Array: 128 x 64 x 1 ]
                                          |
                                          v
+---------------------------------------------------------------------------------+
|  Aspect-Ratio Preserving Adaptation:                                            |
|   1. Bilinear Interpolation: 128 x 64  ----->  224 x 112 (Preserves 2:1 anatomy)|
|   2. Symmetric Zero-Padding: 224 x 112 ----->  224 x 224 (+56 px border zeros)  |
|   3. 3-Channel Replication:  1 x 224^2 ----->  3 x 224 x 224 (RGB-equivalent)   |
+-------------------------------------+-------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------------+
|  Patch Embedding Layer (timm: vit_tiny_patch16_224):                            |
|   - 16 x 16 Non-overlapping Patches    ----->  196 Patch Tokens (14 x 14 grid)  |
|   - Prepend Learnable [CLS] Token      ----->  197 Total Tokens                 |
|   - Add 1D Learnable Positional Embeds ----->  Embedding Dimension D = 192      |
+-------------------------------------+-------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------------+
|  12 Transformer Encoder Blocks (Selective Fine-Tuning):                         |
|   - Multi-Head Self-Attention (MHSA: 3 heads per block, head dimension = 64)    |
|   - Models global bilateral asymmetry across distant angiosome vascular beds    |
|   - LayerNorm + MLP Blocks with Residual Skip Connections                       |
+-------------------------------------+-------------------------------------------+
                                      |
                                      v
                  [ Extract [CLS] Token Representation (192-d) ]
                                      |
                                      v
                  +---------------------------------------+
                  |       Linear Classification Head      |
                  |       - LayerNorm + Linear (192 -> 3) |
                  |       - Evaluated under FP16 AMP      |
                  +-------------------+-------------------+
                                      |
    - - - - - - - - - - - - - - - - - + - - - - - - - - - - - - - - - - -
    | Loss Optimization               | Benchmark Evaluation
    v                                 v
  +---------------------------+     +-----------------------------------------+
  |  Ordinal Weighted CELoss  |     |  Test Benchmark Performance (52 feet):  |
  |   - High-severity penalty |     |   * Accuracy:          88.46% (46 / 52) |
  |   - Symmetric class weights|    |   * Weighted F1:       0.8842           |
  |   - Asymmetric penalties  |     |   * Healthy Recall:   100.0%  (14 / 14) |
  +---------------------------+     |   * High-Sev Precision: 95.5% (21 / 22) |
                                    +-----------------------------------------+
```

### Core Components
1. **Backbone (`vit_tiny_patch16_224` via `timm`):**
   * 12 transformer encoder blocks, 3 attention heads per block, embedding dimension $D=192$ (~5.7M parameters).
   * Pretrained on ImageNet-1k with selective backbone fine-tuning.
2. **Aspect-Ratio Preserving Symmetric Zero-Padding:**
   * The canonical thermal array is $128 \times 64$ (2:1 aspect ratio). Directly stretching it to $224 \times 224$ introduces severe non-affine anatomical distortion.
   * ViTTinyThermal applies bilinear interpolation to $224 \times 112$, followed by patch-aligned symmetric horizontal zero-padding (+56 pixels left/right) to form a standard $224 \times 224$ input ($14 \times 14 = 196$ patches of $16 \times 16$).
   * Non-foot background padding is explicitly masked to $0.0$, preventing patch corruption.
3. **Input Channel Replication:**
   * The single-channel float32 temperature matrix is replicated across 3 RGB channels to align with pretrained transformer patch projection weights.
4. **Benchmark Results:**
   * **Test Accuracy:** **88.46%** (46 / 52 feet correct).
   * **Weighted F1 Score:** **0.8842**.
   * **Healthy Recall:** **100.0%** (14 / 14 controls correctly identified).
   * **High Severity Precision:** **95.5%** (21 / 22 high-risk feet accurately detected).

---

## Hardware Optimization (GTX 1650 Compliant)

All pipelines are engineered to run within a **4 GB VRAM budget** (NVIDIA GeForce GTX 1650, 16 GB RAM):
* **Automatic Mixed Precision (AMP FP16):** `torch.amp.autocast('cuda')` used across all forward passes.
* **DataLoader:** `num_workers=2`, `pin_memory=True`.
* **Batch Size:** 16 (effective batch size 32 via 2-step gradient accumulation where needed).
* **On-the-fly Data Augmentation:** Prevents RAM exhaustion.

---

## Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/3a00/ThermoGuard-DFU.git
cd ThermoGuard-DFU

# Create and activate a Python virtual environment (Python 3.10+)
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Dataset Download & Configuration

1. Download the **Plantar Thermogram Database** from [IEEE Dataport](https://ieee-dataport.org/open-access/plantar-thermogram-database).
2. Extract the archive into `IEE data port(original)/ThermoDataBase(IEE data port)/ThermoDataBase/` (or your preferred local directory).
3. If using a custom path, update `config.yaml`:

```yaml
data:
  extracted_root: "path/to/ThermoDataBase"
  excel_path: "path/to/Plantar Thermogram Database.xlsx"
```

> **Note on Model Checkpoints:** Large `.pth` model weights are excluded from Git history via `.gitignore` to keep the repository lightweight (~3 MB). Running the training commands below will produce and save fresh checkpoints to `outputs/phase1_2_regional/checkpoints/` and `outputs/phase2_vit/checkpoints/`.

### 3. Running Preprocessing Pipeline (Phase 0)

```bash
# Compute TCI ground-truth labels
python src/pipeline/tci_labeling.py

# Generate deterministic subject-level split
python src/pipeline/data_split.py

# Process raw CSVs into normalized .npy arrays (128x64)
python src/pipeline/preprocessing.py

# Calculate class distribution and loss weights
python src/pipeline/compute_class_weights.py
```

### 4. Training & Evaluating Models

```bash
# Train Phase 1.2 (Regional EfficientNet-B0)
python src/train_phase1_2.py

# Evaluate Phase 1.2 on test split
python src/evaluate.py --phase phase1_2

# Train Phase 2 (ViT-Tiny Transformer)
python src/train_phase2.py

# Evaluate Phase 2 on test split
python src/evaluate.py --phase phase2
```

---

## Repository Structure

```text
thermalDFU/
├── config.yaml                    # Central experiment & hardware configuration
├── CONTEXT.md                     # Domain glossary & terminology standards
├── data/                          # Data manifests (subject-level splits & metadata)
├── docs/                          # Architecture Decision Records (ADRs) & guidelines
├── outputs/                       # Metric reports, training curves, confusion matrices
│   ├── phase1_2_regional/         # Phase 1.2 evaluation artifacts
│   └── phase2_vit/                # Phase 2 evaluation artifacts & TensorBoard logs
├── src/
│   ├── datasets/                  # PyTorch Dataset loaders with augmentation
│   ├── models/                    # EfficientNet-B0 and ViT-Tiny blueprints
│   ├── pipeline/                  # Phase 0 data processing & manifest scripts
│   ├── utils/                     # Custom losses (OrdinalWeightedCELoss)
│   ├── train_phase1_2.py          # Phase 1.2 two-stage training script
│   ├── train_phase2.py            # Phase 2 ViT training script
│   └── evaluate.py                # Unified multi-phase test evaluator
└── tests/                         # Pre-flight verification suites
```

---

## Authors & Citation

* **Project Owner:** wenalz
* **Project:** Senior Graduation Capstone Project — ThermoGuard-DFU
* **Dataset Reference:** Hernandez-Contreras et al., *Plantar Thermogram Database*, IEEE Dataport.
