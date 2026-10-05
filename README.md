ThermoGuard-DFU
===============

ThermoGuard-DFU stratifies diabetic foot ulcer risk from plantar thermograms.
It reads raw thermal matrices, applies fixed physical temperature normalization,
and predicts a 3-class neurovascular severity grade.


Quick start
-----------

* Environment:
    python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

* Dataset paths in config.yaml:
    data:
      extracted_root: "path/to/ThermoDataBase"
      excel_path: "path/to/Plantar Thermogram Database.xlsx"

* Run Phase 0 data pipeline:
    python src/pipeline/tci_labeling.py
    python src/pipeline/data_split.py
    python src/pipeline/preprocessing.py
    python src/pipeline/compute_class_weights.py

* Train and evaluate regional CNN (Phase 1.2):
    python src/train_phase1_2.py
    python src/evaluate.py --phase phase1_2

* Train and evaluate Vision Transformer (Phase 2):
    python src/train_phase2.py
    python src/evaluate.py --phase phase2

* Training telemetry:
    tensorboard --logdir outputs/phase2_vit/tensorboard/

Documentation
-------------

* Clinical and technical glossary: CONTEXT.md
* 3-Class clinical severity grouping: docs/adr/0001-three-class-severity-grouping.md
* Ordinal loss penalty formulation: docs/adr/0002-ordinal-weighted-cross-entropy-loss.md
* Project rules and hardware constraints: .agents/AGENTS.md
* Experiment configuration: config.yaml


Completed work
==============

Phase 0: Data preparation
-------------------------

* Phase 0.5 (TCI labeling):
  Calculated Thermal Change Index (TCI) ground-truth labels across 167 subjects
  (122 diabetic, 45 control; 334 feet total) and 4 angiosomes (LCA, LPA, MCA, MPA).
  - Script: src/pipeline/tci_labeling.py
  - Artifacts: data/subject_manifest_unsplit.csv, data/file_traceability_manifest.csv

* Phase 0.6 (Subject-level split):
  Split subjects into 116 train, 25 val, and 26 test (232 train, 50 val, 52 test feet)
  with seed 42. Bilateral feet from the same subject stay in the same split.
  - Script: src/pipeline/data_split.py
  - Artifacts: data/subject_manifest.csv, outputs/phase0_prep/split_summary.json

* Phase 0.7 (Array preprocessing):
  Normalized temperature matrices into float32 .npy arrays at 128x64 (H x W).
  Used a fixed physical range [15.0°C, 35.0°C] across all images. Flipped left feet
  horizontally for anatomical alignment, and zero-masked backgrounds.
  - Script: src/pipeline/preprocessing.py
  - Artifacts: data/processed/arrays/{train,val,test}/*.npy, data/processed/images/

* Phase 0.8 (Class weights and statistics):
  Computed inverse class frequency weights (6-class) and channel statistics.
  (3-class weights are computed dynamically in training scripts).
  - Script: src/pipeline/compute_class_weights.py
  - Artifacts: outputs/phase0_prep/class_weights.json, dataset_statistics.json

Phase 1: Convolutional models
-----------------------------

* Phase 1.0 (Baseline EfficientNet-B0):
  Standard EfficientNet-B0 trained on 6 TCI classes with nearest-neighbor resize
  and unweighted cross-entropy.
  - Benchmark: 38.46% test accuracy, 0.3267 weighted F1
  - Script: src/train_phase1_0.py

* Phase 1.1 (Upgraded EfficientNet-B0):
  Switched to 3-class severity grouping (ADR 0001). Replaced resize with bilinear
  interpolation and boundary masking, added ConcatPool (GAP + GMP), and unfroze
  the top block.
  - Benchmark: 69.23% test accuracy, 0.7024 weighted F1
  - Script: src/train_phase1_1.py

* Phase 1.2 (Regional EfficientNet):
  Replaced global pooling with SpatialConcatPool2d. The 7x4 feature map is split into
  a 2x2 grid covering four anatomical quadrants (Forefoot Medial/Lateral, Heel Medial/Lateral).
  Each quadrant gets average and max pooling, producing a 10,240-d vector.
  Trained with OrdinalWeightedCELoss (ADR 0002) using two-stage transfer learning
  (Stage A head warm-up at 1e-3, Stage B fine-tune at 1e-4), test-time augmentation (TTA),
  and checkpoint ensembling.
  - Benchmark: 80.77% test accuracy, 0.8104 weighted F1
  - Clinical metrics: 100.0% Healthy recall, 100.0% High-Severity precision
  - Model: src/models/efficientnet.py
  - Script: src/train_phase1_2.py

Regional CNN architecture (Phase 1.2):

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
                  |   - features[0..6]:    Frozen         |
                  |   - features[7] + Head: LR = 1e-4     |
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
                  |      2-Layer Bottleneck Head          |
                  |   - Linear(10240 -> 256) + BN1d + ReLU|
                  |   - Dropout(0.3) -> Linear(256 -> 3)  |
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
                                    |  [1] Low Severity   ->  76.9% P / 71.4% R|
                                    |  [2] High Severity  ->  100.0% Precision|
                                    |  Overall Test: 80.77% Acc | 0.8104 wF1  |
                                    +-----------------------------------------+

Phase 2: Vision transformer (ViT-Tiny)
--------------------------------------

* Phase 2.0 (ViTTinyThermal):
  Evaluated vision transformers for bilateral thermal asymmetry. Used timm's
  vit_tiny_patch16_224 (~5.7M base parameters, 12 blocks, 3 heads/block, 192 embedding dim).
  Adapted input resolution by resizing 128x64 to 224x112 with bilinear interpolation,
  padding horizontally to 224x224 (48 px left, 64 px right zeros) to align with 16 px
  patches, and repeating to 3 channels.
  Trained with FP16 AMP and OrdinalWeightedCELoss.
  - Benchmark: 88.46% test accuracy, 0.8842 weighted F1 (46/52 feet correct)
  - Clinical metrics: 100.0% Healthy recall, 95.5% High-Severity precision
  - Model: src/models/vit_tiny.py
  - Script: src/train_phase2.py
  - Status: Baseline verified. Patch resolution and unfreeze depth ablations pending.

Vision Transformer architecture (Phase 2):

+---------------------------------------------------------------------------------+
|                   PHASE 2: ViT-TINY VISION TRANSFORMER                          |
+---------------------------------------------------------------------------------+

                         [ Canonical Array: 128 x 64 x 1 ]
                                          |
                                          v
+---------------------------------------------------------------------------------+
|  Aspect-Ratio Preserving Adaptation:                                            |
|   1. Bilinear Interpolation: 128 x 64  ----->  224 x 112 (Preserves 2:1 anatomy)|
|   2. Asymmetric Zero-Padding: 224 x 112 ---->  224 x 224 (+48L / +64R zeros)    |
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
|   - Models global bilateral thermal relationships across angiosomes             |
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

Phase 3: Hybrid CNN + ViT (roadmap)
-----------------------------------

* Phase 3.0 (Hybrid Thermal Architecture):
  Combines localized spatial representations from regional convolutional stages
  with global contextual self-attention tokens from transformer blocks.
  Blueprint: src/models/hybrid_cnn_vit.py


Benchmark summary
=================

+---------------+-----------------------+--------------------------------------+---------------+-------------+
| Phase         | Architecture          | Key changes                          | Test Accuracy | Weighted F1 |
+---------------+-----------------------+--------------------------------------+---------------+-------------+
| Phase 1.0     | EfficientNet-B0 (6-cl)| Nearest-neighbor resize, standard CE | 38.46%        | 0.3267      |
| Phase 1.1     | EfficientNet-B0 (3-cl)| Bilinear+mask, ConcatPool, freeze    | 69.23%        | 0.7024      |
| Phase 1.2     | Regional EfficientNet | SpatialConcatPool2d (2x2), TTA + Ens | 80.77%        | 0.8104      |
| Phase 2.0     | ViT-Tiny Transformer  | Patch self-attention, aspect padding | 88.46%        | 0.8842      |
+---------------+-----------------------+--------------------------------------+---------------+-------------+


Project reference
=================

Clinical setup
--------------
* Severity classes (ADR 0001):
  - Healthy (Class 0): Control group subjects with normal thermoregulation.
  - Low Severity (Class 1): DM Grade 0, Grade 1, and Grade 2 (early/mild neurovascular changes).
  - High Severity (Class 2): DM Grade 3 and Grade 4 (advanced neuropathy and ulcer risk).
* TCI scoring: src/pipeline/tci_labeling.py
* Physical normalization: [15°C, 35°C] in src/pipeline/preprocessing.py
* Domain terminology: CONTEXT.md

Components and scripts
----------------------
* Dataset loader with dynamic training augmentations: src/datasets/thermal_dataset.py
* Regional CNN model: src/models/efficientnet.py (EfficientNetThermalV3)
* Vision transformer model: src/models/vit_tiny.py (ViTTinyThermal)
* Ordinal loss function: src/utils/losses.py (OrdinalWeightedCELoss)
* Training scripts: src/train_phase1_2.py (Phase 1.2) and src/train_phase2.py (Phase 2)
* Evaluation script: src/evaluate.py
* Telemetry: outputs/phase2_vit/tensorboard/

Hardware and training settings
------------------------------
* Platform: NVIDIA GTX 1650 (~4 GB VRAM), 16 GB RAM, Intel Core i5.
* Precision: FP16 AMP via torch.amp.autocast('cuda').
* DataLoader: num_workers=2, pin_memory=True.
* Batch size: 16 per step without accumulation.
* Checkpoints: saved in outputs/*/checkpoints/.
* Tests: tests/test_phase2_vit.py.

Validation and integrity
------------------------
* Architecture decisions: docs/adr/0001-three-class-severity-grouping.md, docs/adr/0002-ordinal-weighted-cross-entropy-loss.md.
* Split integrity: Subject-level split in data/subject_manifest.csv (seed 42) prevents bilateral foot leakage.
* Traceability: Raw-to-processed mapping in data/file_traceability_manifest.csv.
* Plots and metrics: Confusion matrices and per-class reports in outputs/phase2_vit/.


Repository layout
=================

thermalDFU/
|-- config.yaml                     # Hyperparameters, hardware settings, and paths
|-- CONTEXT.md                      # Clinical and technical glossary
|-- README.md                       # Project documentation
|-- readmeB.txt                     # Technical documentation
|-- requirements.txt                # Python dependencies
|-- .agents/
|   `-- AGENTS.md                   # Project directives and hardware bounds
|-- .scratch/
|   `-- phase2-vit/                 # Phase 2 planning and tickets
|-- data/
|   |-- subject_manifest.csv        # Split manifest (116 train / 25 val / 26 test)
|   |-- file_traceability_manifest.csv
|   `-- processed/
|       |-- arrays/                 # 128x64 float32 canonical .npy arrays
|       `-- images/                 # Plasma colormap PNG previews
|-- docs/
|   `-- adr/                        # Architecture Decision Records (0001, 0002)
|-- outputs/                        # Experiment artifacts by phase
|   |-- phase0_prep/                # Manifests, distribution plots, class weights
|   |-- phase1_0_baseline/          # Phase 1.0 metrics, predictions, logs
|   |-- phase1_1_upgraded/          # Phase 1.1 metrics, predictions, logs
|   |-- phase1_2_regional/          # Phase 1.2 metrics, checkpoints
|   `-- phase2_vit/                 # Phase 2 metrics, checkpoints, TensorBoard
|-- src/
|   |-- datasets/                   # PyTorch dataset loaders with dynamic augmentation
|   |-- models/                     # EfficientNetThermalV3, ViTTinyThermal, Hybrid
|   |-- pipeline/                   # Phase 0 extraction, split, and prep scripts
|   |-- utils/                      # OrdinalWeightedCELoss and evaluation helpers
|   |-- train_phase1_0.py           # Phase 1.0 baseline training script
|   |-- train_phase1_1.py           # Phase 1.1 upgraded training script
|   |-- train_phase1_2.py           # Phase 1.2 regional CNN training script
|   |-- train_phase2.py             # Phase 2 Vision Transformer training script
|   `-- evaluate.py                 # Evaluation script
`-- tests/
    `-- test_phase2_vit.py          # Pre-flight architecture and freeze tests


Project information
===================

* Project Lead: wenalz (Graduation Capstone Project)
* Primary Dataset: IEEE Plantar Thermogram Database (Hernandez-Contreras et al.)
* Architecture Decisions: By abed (see AGENTS.md for more context)
