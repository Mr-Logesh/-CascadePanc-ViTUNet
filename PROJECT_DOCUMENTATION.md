# CascadePanc-ViTUNet — Complete Project Documentation

> **Cascaded Vision Transformer U-Net for Pancreatic Cancer Detection, Tissue Characterization, and Clinical Staging from CT Scans**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Model Architecture](#3-model-architecture)
4. [Data Processing Pipeline](#4-data-processing-pipeline)
5. [Training Strategy](#5-training-strategy)
6. [Inference Pipeline (Cascade)](#6-inference-pipeline-cascade)
7. [Post-Processing & Validation](#7-post-processing--validation)
8. [Clinical Analysis Module](#8-clinical-analysis-module)
9. [Explainable AI (XAI) Module](#9-explainable-ai-xai-module)
10. [Web Application](#10-web-application)
11. [Model Performance & Metrics](#11-model-performance--metrics)
12. [Project Structure](#12-project-structure)
13. [Dependencies & Setup](#13-dependencies--setup)
14. [Limitations & Disclaimers](#14-limitations--disclaimers)

---

## 1. Project Overview

**CascadePanc-ViTUNet** is an end-to-end deep learning system for detecting and characterizing pancreatic tumors from 3D CT volumes. It combines:

- A **two-stage cascaded segmentation pipeline** (Pancreas Localization → Tumor Segmentation)
- A **Vision Transformer (ViT) enhanced 3D U-Net** architecture
- **HU-based clinical analysis** (radiomics, pathology classification, T-staging)
- **Explainable AI** (Attention Rollout + Grad-CAM)
- A **Flask web application** with a clinical dashboard

### Problem Statement

Pancreatic cancer is one of the deadliest cancers with a 5-year survival rate of approximately 10%. Early and accurate detection from CT scans is critical but challenging because:

- The pancreas is small (~100 cm³) relative to the abdomen
- Pancreatic tumors can be hypoattenuating and hard to distinguish from normal tissue
- The organ has high anatomical variability across patients

### Solution

A cascaded approach that first localizes the pancreas in the full CT volume, then zooms into the pancreatic region for fine-grained tumor segmentation, followed by automated clinical analysis.

```
Input: 3D CT Volume (.nii.gz / .dcm)
  │
  ├─→ Stage 1: Pancreas Localization (Binary: background vs pancreas)
  │     └─→ ROI extraction (bounding box around pancreas)
  │
  ├─→ Stage 2: Tumor Segmentation (3-class: background, pancreas, tumor)
  │     └─→ Post-processed segmentation mask
  │
  ├─→ Clinical Analysis
  │     ├─→ Radiomics feature extraction (HU-based)
  │     ├─→ Pathology classification (PDAC, cystic, pancreatitis, etc.)
  │     ├─→ T-staging (AJCC 8th Edition)
  │     └─→ Pancreatitis severity scoring (Modified CTSI)
  │
  ├─→ Explainable AI
  │     ├─→ ViT Attention Rollout
  │     └─→ Grad-CAM heatmaps
  │
  └─→ Clinical Dashboard (Flask Web App)
        ├─→ CT slice viewer with overlays
        ├─→ Risk assessment & recommendations
        └─→ Interactive XAI visualizations
```

---

## 2. Dataset

### Source

**Task07_Pancreas** from the **Medical Segmentation Decathlon (MSD)**

- **Website**: http://medicaldecathlon.com
- **Modality**: Portal venous phase contrast-enhanced CT
- **Format**: NIfTI (`.nii.gz`)

### Dataset Composition

| Split      | Count | Description                         |
|------------|-------|-------------------------------------|
| Training   | 281   | CT volumes with ground truth labels |
| Test       | 139   | CT volumes (no public labels)       |
| **Total**  | **420** | 3D abdominal CT scans            |

### Label Schema

| Class | Value | Description                    |
|-------|-------|--------------------------------|
| 0     | Background | Everything outside the pancreas |
| 1     | Pancreas   | Normal pancreatic parenchyma    |
| 2     | Tumor      | Pancreatic tumor (PDAC)         |

### Data Characteristics

- **Spatial resolution**: Variable across patients (typically 0.6–0.8 mm in-plane, 1.0–5.0 mm slice thickness)
- **Volume dimensions**: Vary per patient (typically ~100–200 slices of 512×512)
- **HU range**: Full Hounsfield Unit range from soft tissue CT
- **Tumor prevalence**: Not all training cases contain tumors; some are normal pancreas only

### Additional Data (Normal Pancreas)

The project also includes a `tcianormalpancreas/` directory with approximately 19,000 files from the TCIA (The Cancer Imaging Archive) normal pancreas dataset, used for additional training/validation reference.

---

## 3. Model Architecture

### ViTUNet (Vision Transformer + 3D U-Net)

The core model is a **3D U-Net with a Vision Transformer bottleneck**. It inherits the spatial hierarchy of U-Net while leveraging the global attention mechanism of ViT at the deepest feature level.

```
Architecture: ViTUNet
├── Encoder (4 levels)
│   ├── Level 1: ConvBlock(1 → 24)             [full resolution]
│   ├── Level 2: DownBlock(24 → 48)            [1/2 resolution]
│   ├── Level 3: DownBlock(48 → 96, dropout=0.1)  [1/4 resolution]
│   └── Level 4: DownBlock(96 → 192, dropout=0.1) [1/8 resolution]
│
├── ViT Bottleneck                              [1/16 resolution]
│   ├── Strided Conv3D(192 → 192, stride=2)
│   ├── PatchEmbedding3D(192 → 384, patch=2)   → tokens
│   ├── Positional Embedding (27 tokens)
│   ├── 3× TransformerBlock(dim=384, heads=6, mlp_ratio=4)
│   ├── LayerNorm → Linear(384 → 192)
│   └── Reshape back to 3D feature map
│
├── Decoder (4 levels, with skip connections)
│   ├── Level 4: UpBlock(192 + 192 → 192)
│   ├── Level 3: UpBlock(192 + 96 → 96)
│   ├── Level 2: UpBlock(96 + 48 → 48)
│   └── Level 1: UpBlock(48 + 24 → 24)
│
├── Final: Conv3D(24 → num_classes, kernel=1)
│
└── Deep Supervision (training only)
    ├── ds3: Conv3D(96 → num_classes, 1)   at Level 3
    └── ds2: Conv3D(48 → num_classes, 1)   at Level 2
```

### Key Components

| Component | Description |
|-----------|-------------|
| **ConvBlock** | Two 3×3×3 Conv3D layers with InstanceNorm3d, LeakyReLU (0.01), optional Dropout3D, and residual connection (1×1 conv if channel mismatch) |
| **DownBlock** | Strided 2×2×2 Conv3D for downsampling + ConvBlock |
| **UpBlock** | ConvTranspose3d (stride 2) for upsampling + concatenation with skip + ConvBlock |
| **PatchEmbedding3D** | Conv3D with `kernel=patch_size, stride=patch_size` to create tokens, followed by LayerNorm |
| **TransformerBlock** | Pre-norm architecture with Multi-Head Self-Attention (6 heads) + MLP (4× expansion with GELU), both with residual connections and 10% dropout |
| **ViTBottleneck** | Patches the deepest feature map into tokens, adds learned positional embeddings, processes through 3 transformer layers, then projects back to 3D |

### Model Configurations

| Parameter | Stage 1 (Pancreas) | Stage 2 (Tumor) |
|-----------|-------------------|------------------|
| Input channels | 1 | 1 |
| Output classes | 2 (bg, pancreas) | 3 (bg, pancreas, tumor) |
| Base channels | 24 | 24 |
| ViT embedding dim | 384 | 384 |
| ViT depth | 3 layers | 3 layers |
| ViT attention heads | 6 | 6 |
| Patch size (input) | 96×96×96 | 64×64×64 |
| Deep supervision | Yes | Yes |

### Weight Initialization

- **Conv3D / ConvTranspose3D**: Kaiming Normal (fan_out, leaky_relu)
- **Linear layers**: Truncated Normal (std=0.02), biases initialized to 0

### Model Checkpoints

| File | Size | Description |
|------|------|-------------|
| `stage1_best.pth` | ~164 MB | Best pancreas localization model |
| `stage2_v2_best.pth` | ~164 MB | Best tumor segmentation model (v2) |

Each checkpoint stores `model_state_dict` and `best_dice` metric.

---

## 4. Data Processing Pipeline

### 4.1 Preprocessing (at Inference Time)

The CT volume undergoes the following preprocessing before being fed to the model:

#### Step 1: Loading
- **NIfTI files**: Loaded via `nibabel`, extracting voxel data and spacing from the header
- **DICOM files**: Loaded via `pydicom`, with HU conversion using RescaleSlope and RescaleIntercept, supporting single-frame, multi-frame, and directory-based series

#### Step 2: Resampling to Isotropic Spacing

```
Target Spacing: 1.5 × 1.5 × 2.5 mm (depth × height × width)
Method: scipy.ndimage.zoom with order=1 (bilinear interpolation)
Scale factors = original_spacing / target_spacing
```

This normalizes all CT volumes to a consistent physical scale regardless of the scanner's acquisition parameters.

#### Step 3: HU Windowing

```
Window: [-125, 275] HU (total width = 400 HU)
Operation: np.clip(ct, -125, 275)
```

This window is optimized for abdominal soft tissue, pancreas, and tumor visualization:
- **-125 HU**: Captures fat and low-density tissue
- **+275 HU**: Captures calcifications and high-density structures

#### Step 4: Intensity Normalization

```python
ct_normalized = (ct_clipped - ct_clipped.mean()) / (ct_clipped.std() + 1e-8)
```

Global z-score normalization is used (zero mean, unit variance) since no foreground ground truth is available at inference time.

**Important**: Raw HU values are preserved separately (before normalization) for the clinical analysis module.

### 4.2 DICOM-Specific Processing

The `dicom_utils.py` module handles:

1. **Single-frame DICOM**: Loaded as a 1-slice volume with spacing from PixelSpacing and SliceThickness
2. **Multi-frame DICOM**: Extracted from NumberOfFrames > 1, with per-frame HU conversion
3. **DICOM directory (series)**: All `.dcm` files sorted by ImagePositionPatient[2] (Z-position) or InstanceNumber, then stacked into a 3D volume
4. **HU conversion**: `pixel_value * RescaleSlope + RescaleIntercept`
5. **Spacing extraction**: From PixelSpacing and computed inter-slice spacing

---

## 5. Training Strategy

Training was performed on **Kaggle notebooks** with GPU acceleration (see `notebooks/` directory).

### Two-Stage Training

#### Stage 1: Pancreas Localization
- **Notebook**: `stage1final.ipynb`
- **Task**: Binary segmentation (background vs. pancreas)
- **Patch size**: 96×96×96 voxels
- **Output classes**: 2
- **Goal**: Learn to find the pancreas in the full abdominal CT

#### Stage 2: Tumor Segmentation
- **Notebooks**: `stage2v1.ipynb` (v1), `stage2v2 (1).ipynb` (v2 — final)
- **Task**: 3-class segmentation within the pancreas ROI
- **Patch size**: 64×64×64 voxels (smaller, zoomed into ROI)
- **Output classes**: 3 (background, pancreas, tumor)
- **Input**: Cropped CT volume around the pancreas (from Stage 1 prediction)

### Training Techniques (from setup guide)

| Technique | Description |
|-----------|-------------|
| **Loss Function** | Boundary-aware loss + Online Hard Example Mining (OHEM) |
| **Data Augmentation** | CutMix augmentation |
| **Optimizer** | Adam with learning rate scheduling |
| **Mixed Precision** | AMP (Automatic Mixed Precision) for GPU efficiency |
| **Deep Supervision** | Auxiliary losses at decoder levels 2 and 3 during training |
| **Regularization** | Dropout3D (10%) in deeper encoder/decoder blocks |

### Evaluation / Testing
- **Notebook**: `finaltesting.ipynb` — End-to-end cascade evaluation
- **Stored results**: `docs/results/cascade_eval_results.npz`

---

## 6. Inference Pipeline (Cascade)

The full cascade inference runs in the following stages (implemented in `inference.py`):

### Stage-by-Stage Flow

```
┌──────────────────────────────────────────────────────────┐
│  STEP 1: PREPROCESSING                                  │
│  • Load NIfTI/DICOM → ct_raw (float32)                  │
│  • Resample to 1.5×1.5×2.5 mm                           │
│  • Clip to [-125, 275] HU                                │
│  • Save raw HU copy for clinical analysis                │
│  • Z-score normalize                                     │
├──────────────────────────────────────────────────────────┤
│  STEP 2: STAGE 1 — PANCREAS LOCALIZATION                │
│  • Sliding window: 96³ patches, stride ratio 0.5         │
│  • 2-class softmax → argmax → binary mask                │
│  • Post-process: largest connected component + fill holes│
│  • Output: binary pancreas mask                          │
├──────────────────────────────────────────────────────────┤
│  STEP 3: ROI EXTRACTION                                 │
│  • Bounding box around Stage 1 mask ± 15 voxel margin    │
│  • Crop CT volume to pancreas region                     │
│  • Fallback: center 50% crop if no pancreas detected     │
├──────────────────────────────────────────────────────────┤
│  STEP 4: STAGE 2 — TUMOR SEGMENTATION                  │
│  • Sliding window: 64³ patches, stride ratio 0.5         │
│  • 3-class softmax → post-process                        │
│  • Anatomical constraint: tumor must be inside pancreas  │
│  • Place ROI prediction back into full volume             │
├──────────────────────────────────────────────────────────┤
│  STEP 5: CLINICAL ANALYSIS                              │
│  • Extract radiomics from raw HU                         │
│  • Classify pathology (PDAC, cystic, pancreatitis, etc.) │
│  • Compute T-staging (AJCC 8th Edition)                  │
│  • Compute pancreatitis severity (Modified CTSI)         │
│  • Generate clinical report                              │
├──────────────────────────────────────────────────────────┤
│  STEP 6: EXPLAINABLE AI                                 │
│  • Attention Rollout from ViT bottleneck                 │
│  • Grad-CAM from deepest encoder layer (enc4)            │
│  • Upsample maps to full resolution                      │
└──────────────────────────────────────────────────────────┘
```

### Sliding Window Inference

Both stages use overlapping sliding window inference:

```
Algorithm:
1. Define grid of start positions with stride = patch_size × stride_ratio
2. For each patch position:
   a. Extract patch (pad with zeros if at volume boundary)
   b. Forward pass through model (with AMP if CUDA)
   c. Softmax to get class probabilities
   d. Accumulate probabilities into prediction map
   e. Increment overlap count map
3. Final prediction = accumulated_probs / count_map
```

| Parameter | Stage 1 | Stage 2 |
|-----------|---------|---------|
| Patch size | 96×96×96 | 64×64×64 |
| Stride ratio | 0.5 (50% overlap) | 0.5 (50% overlap) |
| Num classes | 2 | 3 |
| AMP | Yes (CUDA) | Yes (CUDA) |

---

## 7. Post-Processing & Validation

### Post-Processing Steps

#### Stage 1 Post-Processing
1. **Argmax**: Class with highest probability per voxel
2. **Largest connected component**: Keep only the largest 3D connected region (removes false positives)
3. **Binary hole filling**: `scipy.ndimage.binary_fill_holes()` to fill internal cavities

#### Stage 2 Post-Processing
1. **Generate combined pancreas mask**: All voxels with class ≥ 1
2. **Largest connected component**: On the combined pancreas+tumor mask
3. **Hole filling**: On the combined mask
4. **Tumor extraction**: Voxels with class == 2
5. **Anatomical constraint**: Tumor mask is AND-ed with pancreas mask → `tumor = tumor * pancreas_mask` (tumors outside the pancreas are removed)
6. **Tumor connected component**: Keep only the largest tumor region

### Metrics Computed After Segmentation

| Metric | Description |
|--------|-------------|
| `pancreas_volume_cm3` | Volume of segmented pancreas in cm³ |
| `tumor_detected` | Boolean: whether any tumor voxels exist |
| `tumor_volume_cm3` | Volume of tumor in cm³ |
| `tumor_voxels` | Raw count of tumor voxels |
| `tumor_center_mm` | 3D center of mass of tumor (in mm) |
| `tumor_extent_mm` | Bounding box dimensions of tumor (D × H × W in mm) |
| `max_dimension_mm` | Largest dimension of the tumor bounding box |

### Volume Calculation

```
Volume (cm³) = voxel_count × voxel_volume_mm³ / 1000
voxel_volume_mm³ = 1.5 × 1.5 × 2.5 = 5.625 mm³
```

---

## 8. Clinical Analysis Module

The `clinical_analysis.py` module performs HU-based tissue characterization **after** the segmentation model has identified regions of interest. This is a **rule-based** system using known HU thresholds from radiology literature.

### 8.1 Radiomics Feature Extraction

Features are extracted from raw HU values within three regions:

#### Pancreas Region (class 1)
- Voxel count, volume (cm³)
- HU statistics: mean, std, median, min, max, skewness, kurtosis, 25th/75th percentiles

#### Tumor Region (class 2)
All pancreas features plus:
- 3D extent in mm, max dimension, center coordinates
- Boundary vs. interior HU analysis (using morphological erosion)
- Heterogeneity coefficient: `std / |mean|`
- Enhancement ratio: `tumor_hu_mean / pancreas_hu_mean`
- HU distribution percentages across clinical ranges:

| Range | HU Values | Clinical Meaning |
|-------|-----------|------------------|
| Very Low | < 20 HU | Cystic / fluid density |
| Low | 20–60 HU | Necrotic / edematous tissue |
| Medium | 60–100 HU | Solid tumor |
| High | 100–150 HU | Normal pancreas / inflamed tissue |
| Very High | > 150 HU | Calcification |

#### Peripancreatic Region
- Obtained by dilating the pancreas+tumor mask by 5 voxels
- Fat stranding score: Normalized measure of how elevated fat HU values are above normal (-80 HU baseline)

### 8.2 Pathology Classification

A **scoring-based rule system** evaluates six possible diagnoses:

| Diagnosis | Key HU Indicators |
|-----------|-------------------|
| **PDAC (Solid Tumor)** | Mean HU 30–90, enhancement ratio < 0.8, moderate heterogeneity |
| **Cystic Neoplasm** | > 50% voxels < 20 HU, mean HU < 30, low variance |
| **Acute Pancreatitis** | Fat stranding > 0.5, high heterogeneity, HU 80–130 with high std |
| **Chronic Pancreatitis** | > 10% voxels > 150 HU (calcification), max HU > 200 |
| **Edema** | Mean HU 15–50, std < 20, > 50% voxels in 20–60 HU range |
| **Necrosis** | Mean HU < 25 with fat stranding > 0.3, large very-low-density component |

Each diagnosis receives a score (0–10+), and the highest-scoring diagnosis becomes the primary finding. Confidence is derived from: `min(0.95, score/10 + 0.3)`.

Risk levels and clinical recommendations are assigned per diagnosis type (e.g., HIGH risk for PDAC with urgent referral recommendations).

### 8.3 T-Staging (AJCC 8th Edition)

Based on the **maximum tumor dimension** in millimeters:

| Stage | T-subclass | Size Criteria | Description |
|-------|-----------|---------------|-------------|
| T1 | T1a | ≤ 5 mm | Tumor confined to pancreas |
| T1 | T1b | > 5 mm, ≤ 10 mm | Tumor confined to pancreas |
| T1 | T1c | > 10 mm, ≤ 20 mm | Tumor confined to pancreas |
| T2 | T2 | > 20 mm, ≤ 40 mm | Tumor confined to pancreas |
| T3 | T3 | > 40 mm | May extend beyond pancreas |

**Note**: N-staging (lymph nodes) and M-staging (metastases) cannot be determined from segmentation alone and are reported as `Nx` and `Mx`.

Resectability assessment:
- **≤ 40 mm**: "Potentially Resectable" (pending vascular involvement assessment)
- **> 40 mm**: "Borderline / Locally Advanced"

### 8.4 Pancreatitis Severity Scoring

Applied only when pancreatitis is the primary or differential diagnosis. Uses an adapted **Modified CT Severity Index (CTSI)**:

| Component | Score Range | Assessment Method |
|-----------|-------------|-------------------|
| Inflammation | 0–4 | Pancreas HU std + fat stranding score |
| Necrosis | 0–4 | Percentage of very low + low HU voxels |
| Extrapancreatic | 0–2 | Peripancreatic fluid detection |
| **Total** | **0–10** | Sum of components |

Severity classification:
- **Mild** (0–2): Expected recovery within 1 week
- **Moderate** (3–6): Possible transient organ failure, close monitoring
- **Severe** (7–10): Persistent organ failure likely, ICU admission

Maps to the **Revised Atlanta Classification** for acute pancreatitis.

---

## 9. Explainable AI (XAI) Module

The `xai_module.py` provides two complementary interpretability methods:

### 9.1 Attention Rollout (ViT)

**Purpose**: Shows WHERE the Vision Transformer is focusing globally in the image.

**Method** (Abnar & Zuidema, 2020):
1. Hook into all 3 transformer blocks to capture attention weight matrices
2. For each layer's attention matrix A (size N×N where N=27 tokens):
   - Add identity matrix for residual connection: `A' = 0.5×A + 0.5×I`
   - Normalize rows to sum to 1
3. Multiply across layers: `Rollout = A'₁ × A'₂ × A'₃`
4. Take column-wise mean → 27-element attention vector
5. Reshape to 3×3×3 spatial grid
6. Upsample to full CT volume resolution using cubic interpolation

**Output**: 3D heatmap showing which spatial regions most influenced the ViT's features.

### 9.2 Grad-CAM (CNN)

**Purpose**: Shows WHICH CNN features were most important for the tumor class prediction.

**Method** (Selvaraju et al., 2017, adapted for 3D):
1. Register hooks on the deepest encoder layer (`enc4.conv.conv[4]` — InstanceNorm3d)
2. Forward pass: Get model output
3. Backward pass: Compute gradients of tumor class (class 2) logits w.r.t. hooked layer
4. Global average pool gradients → channel-wise weights
5. Weighted sum of activation maps → `CAM = Σ(weight_c × activation_c)`
6. Apply ReLU (only positive contributions)
7. Upsample to full resolution using bilinear interpolation

**Output**: 3D heatmap highlighting regions the CNN considers most important for tumor detection.

### 9.3 XAI Visualizations Generated

The module produces:

1. **Combined XAI panels** (per slice): CT | Segmentation | Attention Rollout | Grad-CAM | Combined overlay
2. **HU distribution analysis**: Histogram of pancreas vs. lesion HU, pie chart of HU composition, box plot comparison
3. **Tumor boundary**: Yellow boundary overlay from morphological erosion

---

## 10. Web Application

### Architecture

- **Backend**: Flask (Python)
- **Frontend**: HTML/CSS/JS with dark medical theme
- **Communication**: RESTful JSON API with background processing

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main clinical dashboard page |
| `/upload` | POST | Upload CT file → get immediate preview (MIP views, montage, center slices) |
| `/analyze` | POST | Start full cascade analysis (runs in background thread) |
| `/status/<session_id>` | GET | Poll analysis progress and retrieve results |
| `/health` | GET | Health check: model status, GPU availability, DICOM support |

### Upload Flow

```
1. User uploads .nii, .nii.gz, or .dcm file (≤500 MB)
2. Server saves to static/uploads/<session_id>/
3. Immediately returns:
   - MIP projections (axial, coronal, sagittal) as base64
   - Center orthogonal slices as base64
   - Montage grid of up to 20 representative slices
   - File metadata (shape, spacing, HU range)
```

### Analysis Flow

```
1. User clicks "Analyze" → POST /analyze
2. Background thread starts cascade inference
3. Frontend polls /status/<session_id> every 500ms
4. Progress stages: preprocessing (5%) → stage1 (15-45%) →
   roi_extraction (50%) → stage2 (55-80%) →
   clinical_analysis (82%) → xai (85%) →
   generating_images (92%) → complete (100%)
5. On completion, returns full results JSON with:
   - Segmentation metrics
   - Base64 result slice images (CT + Stage1 + Segmentation)
   - HU analysis
   - Clinical report with risk level
   - T-staging data
   - Pancreatitis severity (if applicable)
   - XAI image paths
```

### Supported File Formats

| Format | Extension | Library |
|--------|-----------|---------|
| NIfTI | `.nii`, `.nii.gz` | nibabel |
| DICOM | `.dcm` | pydicom |

---

## 11. Model Performance & Metrics

### Reported Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Full Cascade Tumor Dice** | ~0.787 | End-to-end (Stage 1 → ROI → Stage 2) |
| **Stage 1 Best Dice** | Stored in checkpoint (`best_dice` key) | Pancreas localization |

### IEEE Paper Comparison Context

The project includes SOTA comparison figures and tables in `docs/figures/`:

| Figure | Content |
|--------|---------|
| `fig1_architecture.png` | ViTUNet architecture diagram |
| `fig2_sota_comparison.png` | Comparison with other methods |
| `fig3_ablation_study.png` | Ablation study results |
| `fig4_segmentation_samples.png` | Qualitative segmentation examples |
| `fig5_tstaging_distribution.png` | T-staging distribution |
| `fig6_clinical_pipeline.png` | Clinical pipeline diagram |
| `fig7_hu_characterization.png` | HU characterization results |
| `fig8_dataset_statistics.png` | Dataset statistics |
| `table1_results.png` | Quantitative results table |

### Evaluation Results

Stored in `docs/results/cascade_eval_results.npz` — contains numpy arrays with per-case metrics from cascade evaluation.

### Key Contributions (for IEEE Paper)

1. **ViT + U-Net Cascade**: Novel combination of Vision Transformer bottleneck with cascade architecture
2. **Improved Loss Functions**: Boundary-aware + OHEM + CutMix achieving competitive Dice scores
3. **HU-Based Tissue Characterization**: Post-segmentation radiomics analysis for pathology classification
4. **Dual Explainability**: Attention Rollout + Grad-CAM for clinical interpretability
5. **Integrated Clinical Pipeline**: End-to-end from CT upload to T-staging and clinical recommendations

---

## 12. Project Structure

```
THEFINALPROJ/
│
├── README.md                        # Quick-start guide
├── PROJECT_DOCUMENTATION.md         # This file (comprehensive documentation)
├── requirements.txt                 # Python dependencies
│
├── app/                             # Flask web application
│   ├── app.py                       # Flask server & routes (556 lines)
│   ├── model.py                     # ViTUNet architecture definition (190 lines)
│   ├── inference.py                 # Cascade inference pipeline (649 lines)
│   ├── clinical_analysis.py         # Radiomics, pathology, T-staging (776 lines)
│   ├── xai_module.py                # Attention Rollout + Grad-CAM (636 lines)
│   ├── dicom_utils.py               # DICOM file loading utilities (234 lines)
│   ├── templates/
│   │   └── index.html               # Clinical dashboard UI
│   └── static/
│       ├── css/                     # Stylesheets
│       ├── js/                      # Frontend JavaScript
│       ├── uploads/                 # Temporary uploads (per session)
│       └── results/                 # Generated visualizations (per session)
│
├── models/                          # Trained model checkpoints
│   ├── stage1_best.pth              # Stage 1: Pancreas localization (~164 MB)
│   └── stage2_v2_best.pth           # Stage 2: Tumor segmentation (~164 MB)
│
├── data/                            # MSD Task07_Pancreas dataset
│   ├── imagesTr/                    # 281 training CT volumes (.nii.gz)
│   ├── imagesTs/                    # 139 test CT volumes (.nii.gz)
│   └── labelsTr/                    # 281 training label masks (.nii.gz)
│
├── notebooks/                       # Training & evaluation notebooks (Kaggle)
│   ├── stage1final.ipynb            # Stage 1 training
│   ├── stage2v1.ipynb               # Stage 2 v1 training
│   ├── stage2v2 (1).ipynb           # Stage 2 v2 training (final)
│   └── finaltesting.ipynb           # End-to-end cascade evaluation
│
├── docs/                            # Documentation & IEEE paper assets
│   ├── setup_guide.md               # Detailed setup instructions
│   ├── figures/                     # IEEE paper figures (PNG + PDF)
│   │   ├── fig1_architecture.*      # Architecture diagram
│   │   ├── fig2_sota_comparison.*   # SOTA comparison
│   │   ├── fig3_ablation_study.*    # Ablation study
│   │   ├── fig4_segmentation_samples.* # Qualitative results
│   │   ├── fig5_tstaging_distribution.* # T-staging
│   │   ├── fig6_clinical_pipeline.* # Clinical pipeline
│   │   ├── fig7_hu_characterization.* # HU analysis
│   │   ├── fig8_dataset_statistics.* # Dataset stats
│   │   └── table1_results.*         # Results table
│   ├── paper/                       # IEEE paper files
│   │   ├── CascadePanc_ViTUNet_IEEE_Paper.tex
│   │   ├── CascadePanc_ViTUNet_IEEE_Paper.docx
│   │   └── *.pdf                    # Submitted papers & reports
│   └── results/                     # Evaluation outputs
│       ├── cascade_eval_results.npz
│       ├── cascade_analysis.png
│       ├── comparison_chart.png
│       └── stage2_visual_verification.png
│
├── tcianormalpancreas/              # TCIA Normal Pancreas dataset (~19K files)
│
└── archive/                         # Zip backups
```

---

## 13. Dependencies & Setup

### Python Dependencies

```
flask>=2.3.0          # Web framework
numpy>=1.24.0         # Numerical computing
torch>=2.0.0          # Deep learning framework (PyTorch)
nibabel>=5.0.0        # NIfTI file I/O
scipy>=1.10.0         # Scientific computing (zoom, morphology, stats)
einops>=0.6.0         # Tensor reshaping for ViT
Pillow>=9.0.0         # Image processing
matplotlib>=3.7.0     # Visualization / plot generation
pydicom>=2.3.0        # DICOM file support
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | CPU-only (slow) | NVIDIA GPU with ≥6 GB VRAM |
| RAM | 8 GB | 16+ GB |
| Disk | 1 GB (app only) | 5+ GB (with dataset) |

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure model checkpoints are in models/
#    stage1_best.pth and stage2_v2_best.pth

# 3. Run the web app
cd app
python app.py

# 4. Open http://localhost:5000 in browser
```

### Processing Time

| Device | Approximate Time |
|--------|-----------------|
| GPU (CUDA) | ~4 seconds per CT volume |
| CPU | ~60 seconds per CT volume |

---

## 14. Limitations & Disclaimers

### Research Limitations

1. **HU-based pathology classification is rule-based**, not a trained classifier. It has NOT been validated against pathology-confirmed diagnoses and uses thresholds from radiology literature.

2. **T-staging is approximate**, based on tumor maximum dimension only. N-staging (lymph nodes) and M-staging (distant metastases) require additional data not available from this segmentation.

3. **Pancreatitis severity scoring** is adapted from the Modified CT Severity Index but uses automated HU analysis rather than radiologist assessment. Clinical validation is required.

4. **The MSD Task07 dataset contains primarily tumor cases**. The pathology classification module has NOT been validated on pathology-confirmed pancreatitis, edema, or cystic neoplasm cases.

5. **Deep supervision outputs are only used during training** — at inference time, only the final decoder output is used for prediction.

6. **ViT positional embedding** is fixed at 27 tokens. Different input sizes are handled via interpolation, but this may affect attention quality for non-standard volumes.

### Clinical Disclaimer

> ⚠️ **This system is a research prototype and is NOT approved for clinical diagnosis or treatment decisions.** All findings must be verified by a qualified radiologist. This software is intended for research and educational purposes only.

---

*Document generated on 2026-02-21. For the latest code, refer to the source files in the repository.*
