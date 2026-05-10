# CascadePanc-ViTUNet

**Cascaded Pancreatic Cancer Detection & Clinical Analysis**

A two-stage ViT-UNet deep learning pipeline for pancreatic tumor segmentation from 3D CT volumes, with automated T-staging, clinical diagnosis, pancreatitis severity scoring, and XAI interpretability.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r app/requirements.txt

# 2. Ensure model checkpoints are in models/
#    stage1_best.pth and stage2_v2_best.pth

# 3. Run the web app
cd app
python app.py

# 4. Open http://localhost:5000
```

## Project Structure

```
CascadePanc-ViTUNet/
│
├── README.md                           # This file
├── PROJECT_DOCUMENTATION.md            # Comprehensive technical documentation
├── .gitignore                          # Git ignore rules
│
├── app/                                # Flask web application
│   ├── app.py                          # Flask server & routes
│   ├── model.py                        # ViTUNet architecture definition
│   ├── inference.py                    # Cascade inference pipeline
│   ├── clinical_analysis.py            # Radiomics, pathology, T-staging
│   ├── xai_module.py                   # Attention Rollout + Grad-CAM
│   ├── dicom_utils.py                  # DICOM file loading utilities
│   ├── validate_model.py               # Model validation script
│   ├── requirements.txt                # Python dependencies
│   ├── templates/                      # HTML templates
│   │   └── index.html                  # Clinical dashboard UI
│   └── static/                         # CSS, JS, uploads, results
│       ├── css/                        # Stylesheets
│       ├── js/                         # Frontend JavaScript
│       ├── uploads/                    # Temporary uploads (per session)
│       └── results/                    # Generated visualizations
│
├── models/                             # Trained model checkpoints (download separately)
│   ├── stage1_best.pth                 # Stage 1: Pancreas localization (~164 MB)
│   └── stage2_v2_best.pth              # Stage 2: Tumor segmentation (~164 MB)
│
├── data/                               # MSD Task07_Pancreas dataset (download separately)
│   ├── imagesTr/                       # 281 training CT volumes (.nii.gz)
│   ├── imagesTs/                       # 139 test CT volumes (.nii.gz)
│   ├── labelsTr/                       # 281 training label masks (.nii.gz)
│   └── tcianormalpancreas/             # TCIA Normal Pancreas reference data
│
├── notebooks/                          # Training & evaluation notebooks (Kaggle)
│   ├── stage1final.ipynb               # Stage 1 training
│   ├── stage2v1.ipynb                  # Stage 2 v1 training
│   ├── stage2v2 (1).ipynb              # Stage 2 v2 training (final)
│   └── finaltesting.ipynb              # End-to-end cascade evaluation
│
└── docs/                               # Technical documentation & assets
    ├── CascadePanc_Technical_Document.docx                              # Technical reference document
    ├── Technical_Document - Software used, Instruction to execute the code.pdf
    ├── diagrams/                       # Architecture & system diagrams (PNG)
    │   ├── PROJECT_DIAGRAMS.md         # All project diagrams (Mermaid source)
    │   ├── arch_diagram.png            # System architecture
    │   ├── dfd_diagram.png             # Data flow diagram
    │   ├── uml_class_diagram.png       # UML class diagram
    │   ├── use_case_diagram.png        # Use case diagram
    │   ├── sequence_diagram.png        # Sequence diagram
    │   ├── er_diagram.png              # Entity-relationship diagram
    │   ├── activity_diagram.png        # Activity / system logic
    │   ├── component_diagram.png       # Component diagram
    │   └── deployment_diagram.png      # Deployment topology
    ├── guides/                         # Setup & explanation guides
    │   ├── setup_guide.md              # Detailed setup instructions
    │   ├── CascadePanc_FullExplanation.md
    │   └── CascadePanc_Master_Guide.md
    └── paper/
        └── generate_figures.py         # Script for generating paper figures
```

> **Note:** Model checkpoints (`models/*.pth`) and dataset (`data/`) are excluded from this repo due to size. See the *Models* and *Dataset* sections below for download instructions.

## Features

- **Two-stage cascade**: Stage 1 (pancreas localization) → Stage 2 (tumor segmentation)
- **DICOM + NIfTI support**: Upload `.nii`, `.nii.gz`, or `.dcm` files
- **Full CT preview**: MIP projections, orthogonal center slices, montage grid
- **Clinical analysis**: HU characterization, T-staging, pancreatitis severity
- **XAI**: Attention Rollout & Grad-CAM visualizations
- **Background processing**: Upload → Preview → Analyze with live progress

## Models

| Checkpoint | Description |
|---|---|
| `stage1_best.pth` | Pancreas localization (binary) |
| `stage2_v2_best.pth` | Tumor segmentation (3-class) |

## Dataset

Task07_Pancreas from the Medical Segmentation Decathlon.

## Diagrams

All project diagrams are available at [`docs/diagrams/PROJECT_DIAGRAMS.md`](docs/diagrams/PROJECT_DIAGRAMS.md):

| Diagram | Description |
|---|---|
| System Architecture | High-level system overview |
| Dataflow (DFD) | Data flow through the pipeline |
| UML Class | All Python classes with attributes & methods |
| Use Case | Actors and system interactions |
| Sequence | Full request lifecycle |
| Database / ER | Logical data model |
| System Logic | Decision flowchart & activity diagram |
| Component | Module dependencies |
| Deployment | System deployment topology |

## License

Research use only. Not for clinical diagnosis.
