# CascadePanc-ViTUNet: Enhanced Clinical Project
# Complete Step-by-Step Implementation Guide

## WHAT THIS PROJECT NOW DOES

```
Input: Abdominal CT scan (.nii.gz)
  │
  ├─→ Stage 1: Pancreas Localization (ViTU-Net, 96³ patches)
  │     └─→ ROI extraction with bounding box
  │
  ├─→ Stage 2: Lesion Segmentation (ViTU-Net, 64³ patches)
  │     └─→ Post-processed 3-class mask
  │
  ├─→ HU-Based Tissue Characterization (NEW)
  │     ├─→ Extract raw HU from segmented region
  │     ├─→ Compute radiomics features (mean, std, skew, heterogeneity)
  │     ├─→ Classify pathology:
  │     │     ├─ PDAC (solid tumor): HU 40-80, hypoattenuating
  │     │     ├─ Cystic neoplasm: HU 0-20, near water
  │     │     ├─ Acute pancreatitis: heterogeneous, fat stranding
  │     │     ├─ Chronic pancreatitis: calcifications >150 HU
  │     │     ├─ Edema: HU 20-40, homogeneous
  │     │     └─ Necrosis: very low HU with peripancreatic changes
  │     └─→ Confidence score + evidence trail
  │
  ├─→ T-Staging (if tumor detected) (NEW)
  │     ├─ T1a: ≤0.5cm  │ T1b: 0.5-1cm  │ T1c: 1-2cm
  │     ├─ T2: 2-4cm    │ T3: >4cm
  │     └─→ Resectability assessment
  │
  ├─→ Pancreatitis Severity (if pancreatitis detected) (NEW)
  │     ├─→ Modified CT Severity Index (0-10)
  │     ├─→ Revised Atlanta Classification
  │     └─→ MILD / MODERATE / SEVERE
  │
  ├─→ Explainable AI (NEW)
  │     ├─→ ViT Attention Rollout (global reasoning map)
  │     ├─→ Grad-CAM (local feature importance)
  │     └─→ HU distribution analysis plots
  │
  └─→ Clinical Dashboard
        ├─→ Risk level (LOW / MODERATE / HIGH)
        ├─→ Primary diagnosis + differential diagnoses
        ├─→ Evidence trail (why this classification)
        ├─→ Clinical recommendations
        ├─→ Interactive slice viewer with XAI overlays
        └─→ Exportable clinical report
```

## PROJECT STRUCTURE

```
CascadePanc_App/
├── app.py                  ← Flask web server
├── model.py                ← ViTU-Net architecture
├── inference.py            ← Cascade inference pipeline
├── clinical_analysis.py    ← HU analysis, pathology classification, T-staging
├── xai_module.py           ← Attention rollout + Grad-CAM
├── requirements.txt        ← Python dependencies
├── SETUP_GUIDE.md          ← This file
│
├── models/                 ← Your trained checkpoints
│   ├── stage1_best.pth
│   └── stage2_v2_best.pth
│
├── templates/
│   └── index.html          ← Clinical dashboard UI
│
└── static/
    ├── css/style.css       ← Dark medical theme
    ├── js/app.js           ← Frontend logic
    ├── uploads/            ← Temporary upload storage
    └── results/            ← Generated visualizations
```

## STEP-BY-STEP SETUP

### Step 1: Download Checkpoints from Kaggle
```
From your training notebooks, download:
  - stage1_best.pth (from Stage 1 training)
  - stage2_v2_best.pth (from Stage 2 v2 training)
```

### Step 2: Extract App & Place Checkpoints
```bash
unzip CascadePanc_Flask_App.zip
cd CascadePanc_App
mkdir -p models
cp /path/to/stage1_best.pth models/
cp /path/to/stage2_v2_best.pth models/
```

### Step 3: Create Virtual Environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Run
```bash
python app.py
```

### Step 6: Open Browser
```
http://localhost:5000
```

### Step 7: Upload a CT Scan & Analyze
```
1. Drag & drop a .nii.gz file (from MSD Task07 test cases)
2. Click "Analyze with CascadePanc-ViTUNet"
3. Wait for processing (4 sec GPU / 60 sec CPU)
4. View results:
   - Tumor status + T-staging
   - Pathology classification + confidence
   - Attention rollout + Grad-CAM heatmaps
   - HU distribution analysis
   - Clinical recommendations
```

## HOW TO INTEGRATE INTO inference.py

Replace the run_cascade() function in inference.py with:

```python
from clinical_analysis import (
    extract_radiomics,
    classify_pathology,
    compute_t_stage,
    compute_pancreatitis_severity,
    generate_clinical_report
)
from xai_module import run_xai_pipeline, generate_xai_visualizations

def run_cascade(nifti_path, progress_callback=None):
    # ... existing preprocessing code ...
    
    # IMPORTANT: Save raw HU before normalization
    ct_raw_hu = np.clip(ct, HU_WINDOW[0], HU_WINDOW[1])
    
    # ... existing Stage 1 + ROI extraction + Stage 2 code ...
    
    # === NEW: HU-Based Clinical Analysis ===
    # Get raw HU values within the ROI
    hu_roi = ct_raw_hu[bbox]
    
    # Extract radiomics features
    features = extract_radiomics(hu_roi, s2_pred, TARGET_SPACING)
    
    # Classify pathology
    classification = classify_pathology(features)
    
    # T-staging (if tumor detected)
    t_stage = compute_t_stage(features)
    
    # Pancreatitis severity (if applicable)
    severity = compute_pancreatitis_severity(features, classification)
    
    # Generate clinical report
    clinical_report = generate_clinical_report(
        features, classification, t_stage, severity
    )
    
    results['clinical_report'] = clinical_report
    results['radiomics'] = features
    results['t_staging'] = t_stage
    results['pancreatitis_severity'] = severity
    
    # === NEW: Explainable AI ===
    attention_map, gradcam_map = run_xai_pipeline(
        _models['stage2'], ct_roi, s2_pred, device, ct_roi.shape
    )
    
    results['attention_map'] = attention_map
    results['gradcam_map'] = gradcam_map
    results['hu_roi'] = hu_roi
    
    return results
```

## HOW TO UPDATE app.py /analyze ENDPOINT

Add clinical data to the JSON response:

```python
# After existing response building...

# Add clinical analysis
if 'clinical_report' in results:
    report = results['clinical_report']
    response['clinical'] = {
        'summary': report['summary'],
        'risk_level': report['risk_level'],
        'primary_finding': report['primary_finding'],
        'confidence': report['confidence'],
        'evidence': report['evidence'],
        'differential': report['differential'],
        'recommendations': report['recommendations'],
    }

# Add T-staging
if 't_staging' in results:
    ts = results['t_staging']
    response['t_staging'] = {
        'stage': ts.get('stage', 'N/A'),
        'substage': ts.get('substage', 'N/A'),
        'description': ts.get('description', ''),
        'max_dimension_mm': ts.get('max_dimension_mm', 0),
        'tnm_string': ts.get('tnm_string', 'N/A'),
        'resectability': ts.get('resectability', 'N/A'),
        'notes': ts.get('notes', []),
    }

# Add pancreatitis severity
if 'pancreatitis_severity' in results:
    sev = results['pancreatitis_severity']
    if sev.get('applicable'):
        response['pancreatitis'] = {
            'severity': sev['severity'],
            'score': sev['score'],
            'max_score': sev['max_score'],
            'revised_atlanta': sev['revised_atlanta'],
            'components': sev['components'],
            'details': sev['details'],
        }

# Generate XAI visualizations
if results.get('attention_map') is not None:
    xai_images = generate_xai_visualizations(
        results['original_ct'][results['roi_bbox']],
        results['segmentation'][results['roi_bbox']],
        results['attention_map'],
        results['gradcam_map'],
        result_dir,
        num_slices=5,
        hu_volume=results.get('hu_roi'),
        features=results.get('radiomics'),
    )
    response['xai_images'] = [
        f'/static/results/{session_id}/{p}' for p in xai_images
    ]
```

## IEEE PAPER TITLE (Updated)

"CascadePanc-ViTUNet: An Explainable Vision Transformer Cascade Framework
 for Pancreatic Tumor Segmentation, Tissue Characterization, and T-Staging
 from CT Scans"

## 5 KEY CONTRIBUTIONS FOR IEEE PAPER

1. **ViT + U-Net Cascade**: Novel combination of Vision Transformer bottleneck
   with cascade architecture for pancreatic segmentation

2. **Improved Loss Functions**: Boundary-aware + OHEM + CutMix achieving 0.787
   tumor Dice (full cascade) — competitive with nnU-Net

3. **HU-Based Tissue Characterization**: Post-segmentation radiomics analysis
   to classify pathology type (PDAC, cystic, pancreatitis, edema, necrosis)

4. **Dual Explainability**: Attention Rollout (ViT global reasoning) + Grad-CAM
   (CNN local features) for clinical interpretability

5. **Integrated Clinical Pipeline**: End-to-end system from CT upload to
   T-staging, risk assessment, and clinical recommendations

## IMPORTANT HONEST DISCLAIMERS FOR IEEE PAPER

Write these clearly in your paper:

1. "HU-based pathology classification uses rule-based thresholds from
    radiology literature. It is NOT a trained classifier and has NOT been
    validated against pathology-confirmed diagnoses."

2. "T-staging is approximate, based on tumor dimensions only. N and M
    staging require additional data (lymph node assessment, distant
    metastasis screening) not available from this segmentation."

3. "Pancreatitis severity scoring is adapted from Modified CTSI but
    uses automated HU analysis rather than radiologist assessment.
    Clinical validation is required."

4. "The MSD Task07 dataset contains only tumor cases. The pathology
    classification module has NOT been validated on confirmed
    pancreatitis or edema cases."

5. "This system is a research prototype and is NOT approved for
    clinical use. All findings require verification by a qualified
    radiologist."

Being honest about limitations is a STRENGTH in IEEE papers, not a weakness.
Reviewers respect transparency.
