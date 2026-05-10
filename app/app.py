"""
CascadePanc-ViTUNet Clinical Demo Application
Flow: Upload (NIfTI / DICOM) → View CT Slices → Run Analysis → See Results

Usage:
  1. Place stage1_best.pth and stage2_v2_best.pth in models/ folder
  2. pip install flask nibabel torch scipy einops pydicom
  3. python app.py
  4. Open http://localhost:5000
"""

import os
import time
import gc
import json
import uuid
import zipfile
import shutil
import threading
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from inference import (
    load_models, run_cascade, generate_slice_images,
    slice_to_base64, generate_preview_slices,
    generate_result_slices_b64, generate_mip_b64,
    HU_WINDOW, TARGET_SPACING, _models
)
from xai_module import generate_xai_visualizations
from dicom_utils import load_dicom_series, is_dicom_file, PYDICOM_AVAILABLE

# ============================================================
# App Configuration
# ============================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join('static', 'results')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.jinja_env.auto_reload = True

@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

ALLOWED_EXTENSIONS = {'.nii', '.nii.gz', '.dcm', '.zip'}

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)


# ============================================================
# Auto-Cleanup: Remove old uploads/results (>1 hour)
# ============================================================
def auto_cleanup():
    """Delete session folders older than 1 hour to save disk space."""
    cutoff = time.time() - 3600  # 1 hour
    for folder_key in ['UPLOAD_FOLDER', 'RESULTS_FOLDER']:
        base = app.config[folder_key]
        if not os.path.isdir(base):
            continue
        for item in os.listdir(base):
            item_path = os.path.join(base, item)
            if os.path.isdir(item_path):
                try:
                    mtime = os.path.getmtime(item_path)
                    if mtime < cutoff:
                        import shutil
                        shutil.rmtree(item_path, ignore_errors=True)
                except Exception:
                    pass


# Model paths (checkpoints live in ../models/ relative to app/)
STAGE1_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'stage1_best.pth')
STAGE2_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'stage2_v2_best.pth')

# Global state
MODELS_LOADED = False
_analysis_status = {}  # session_id -> status dict

# ============================================================
# Load models at startup
# ============================================================
print("\n" + "=" * 60)
print("  CascadePanc-ViTUNet Clinical Demo")
print("  Loading AI models...")
print("=" * 60 + "\n")

if os.path.exists(STAGE1_PATH) and os.path.exists(STAGE2_PATH):
    load_models(STAGE1_PATH, STAGE2_PATH)
    MODELS_LOADED = True
    print("\n[OK] Models loaded successfully!\n")
else:
    print(f"\n[WARNING] Model files not found!")
    print(f"  Expected: {STAGE1_PATH}")
    print(f"  Expected: {STAGE2_PATH}")
    print(f"  Place checkpoint files in the 'models/' directory.\n")

if PYDICOM_AVAILABLE:
    print("[INFO] DICOM support: ENABLED (pydicom installed)")
else:
    print("[INFO] DICOM support: DISABLED (install pydicom for .dcm files)")


def allowed_file(filename):
    fname = filename.lower()
    return fname.endswith('.nii') or fname.endswith('.nii.gz') or fname.endswith('.dcm') or fname.endswith('.zip')


def is_dicom_path(filename):
    """Check if the filename is a DICOM file."""
    return filename.lower().endswith('.dcm')


def is_nifti_path(filename):
    """Check if the filename is a NIfTI file."""
    fname = filename.lower()
    return fname.endswith('.nii') or fname.endswith('.nii.gz')


# ============================================================
# Routes
# ============================================================

@app.route('/')
def index():
    return render_template('index.html',
                           models_loaded=MODELS_LOADED,
                           dicom_available=PYDICOM_AVAILABLE)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload NIfTI or DICOM → return CT slice previews immediately (NO analysis yet)."""
    import nibabel as nib
    auto_cleanup()  # Clean up old sessions

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    fname = secure_filename(f.filename)

    if not allowed_file(fname):
        return jsonify({
            'error': 'Invalid file format',
            'message': 'Supported formats: .nii, .nii.gz (NIfTI) and .dcm (DICOM). '
                       'This model requires 3D CT volumes for accurate predictions.'
        }), 400

    # Save file
    session_id = str(uuid.uuid4())[:8]
    upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(upload_dir, exist_ok=True)
    upload_path = os.path.join(upload_dir, fname)
    f.save(upload_path)

    try:
        if fname.lower().endswith('.zip'):
            # ZIP file — extract and look for DICOM or NIfTI files inside
            if not zipfile.is_zipfile(upload_path):
                return jsonify({'error': 'Invalid ZIP file'}), 400

            extract_dir = os.path.join(upload_dir, 'extracted')
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(upload_path, 'r') as zf:
                zf.extractall(extract_dir)

            # Search for NIfTI files first
            nifti_files = []
            dicom_files = []
            for root, dirs, files in os.walk(extract_dir):
                for ef in files:
                    efl = ef.lower()
                    if efl.endswith('.nii') or efl.endswith('.nii.gz'):
                        nifti_files.append(os.path.join(root, ef))
                    elif efl.endswith('.dcm') or not os.path.splitext(ef)[1]:
                        dicom_files.append(os.path.join(root, ef))

            if nifti_files:
                # Use first NIfTI file found
                nii = nib.load(nifti_files[0])
                ct_raw = nii.get_fdata().astype(np.float32)
                spacing = np.array(nii.header.get_zooms()[:3])
                file_type = 'NIfTI (from ZIP)'
                upload_path = nifti_files[0]
            elif dicom_files:
                if not PYDICOM_AVAILABLE:
                    return jsonify({
                        'error': 'DICOM support not available',
                        'message': 'Install pydicom: pip install pydicom'
                    }), 400
                # Load the directory containing the DICOM files
                dicom_dir = os.path.dirname(dicom_files[0])
                ct_raw, spacing = load_dicom_series(dicom_dir)
                file_type = f'DICOM Series (from ZIP, {len(dicom_files)} files)'
                upload_path = dicom_dir
            else:
                return jsonify({
                    'error': 'No medical imaging files found in ZIP',
                    'message': 'ZIP should contain .nii/.nii.gz or .dcm DICOM files'
                }), 400

        elif is_dicom_path(fname):
            # DICOM file
            if not PYDICOM_AVAILABLE:
                return jsonify({
                    'error': 'DICOM support not available',
                    'message': 'Install pydicom: pip install pydicom'
                }), 400

            ct_raw, spacing = load_dicom_series(upload_path)
            file_type = 'DICOM'
        else:
            # NIfTI file
            nii = nib.load(upload_path)
            ct_raw = nii.get_fdata().astype(np.float32)
            spacing = np.array(nii.header.get_zooms()[:3])
            file_type = 'NIfTI'

        # Apply HU windowing for display
        ct_display = np.clip(ct_raw, HU_WINDOW[0], HU_WINDOW[1])
        ct_display = (ct_display - ct_display.min()) / (ct_display.max() - ct_display.min() + 1e-8)

        D, H, W = ct_raw.shape

        def safe_b64(arr_2d):
            """Safely convert a 2D array to base64, handling degenerate shapes."""
            if arr_2d.ndim < 2 or arr_2d.shape[0] < 2 or arr_2d.shape[1] < 2:
                # Fallback: use center axial slice
                return slice_to_base64(ct_display[D // 2])
            try:
                return slice_to_base64(arr_2d)
            except Exception:
                return slice_to_base64(ct_display[D // 2])

        # --- Full-volume preview: MIP projections ---
        mip_axial = safe_b64(ct_display.max(axis=0))
        mip_coronal = safe_b64(ct_display.max(axis=1))
        mip_sagittal = safe_b64(ct_display.max(axis=2))

        # --- Montage grid of key slices ---
        montage_count = min(D, 20)
        montage_indices = np.linspace(0, D - 1, montage_count).astype(int)
        montage_b64 = []
        for si in montage_indices:
            montage_b64.append(safe_b64(ct_display[si]))

        # --- Center slices for 3 orthogonal views ---
        center_axial = safe_b64(ct_display[D // 2])
        center_coronal = safe_b64(ct_display[:, H // 2, :])
        center_sagittal = safe_b64(ct_display[:, :, W // 2])

        info = {
            'session_id': session_id,
            'filename': fname,
            'file_type': file_type,
            'shape': [int(D), int(H), int(W)],
            'spacing': [round(float(s), 3) for s in spacing],
            'hu_range': [float(ct_raw.min()), float(ct_raw.max())],
            'num_slices': D,
            # Full-volume views
            'mip_axial': mip_axial,
            'mip_coronal': mip_coronal,
            'mip_sagittal': mip_sagittal,
            'center_axial': center_axial,
            'center_coronal': center_coronal,
            'center_sagittal': center_sagittal,
            # Montage grid
            'montage': montage_b64,
            'montage_indices': [int(x) for x in montage_indices],
            'file_path': upload_path,
        }

        return jsonify(info)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to read file: {str(e)}'}), 400


@app.route('/analyze', methods=['POST'])
def analyze():
    """Run the full cascade analysis on a previously uploaded file."""
    data = request.json
    session_id = data.get('session_id')
    file_path = data.get('file_path')

    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'File not found. Please re-upload.'}), 400

    if not MODELS_LOADED:
        return jsonify({'error': 'Models not loaded. Place .pth files in models/ and restart.'}), 400

    # Initialize status
    _analysis_status[session_id] = {'stage': 'preprocessing', 'progress': 0}

    def run_analysis():
        try:
            import nibabel as nib
            from scipy.ndimage import zoom
            from inference import (
                sliding_window_predict, postprocess_stage1,
                postprocess_stage2, keep_largest
            )
            from clinical_analysis import (
                extract_radiomics, classify_pathology,
                compute_t_stage, compute_pancreatitis_severity,
                generate_clinical_report
            )
            from xai_module import run_xai_pipeline

            status = _analysis_status[session_id]
            device = _models['device']

            # ===== PREPROCESS =====
            status['stage'] = 'preprocessing'
            status['progress'] = 5

            if os.path.isdir(file_path) or is_dicom_path(file_path):
                ct_raw, spacing = load_dicom_series(file_path)
            else:
                nii = nib.load(file_path)
                ct_raw = nii.get_fdata().astype(np.float32)
                spacing = np.array(nii.header.get_zooms()[:3])

            original_shape = ct_raw.shape
            scale = spacing / np.array(TARGET_SPACING)
            ct = zoom(ct_raw, scale, order=1)

            # Keep raw HU for analysis
            ct_hu = np.clip(ct.copy(), HU_WINDOW[0], HU_WINDOW[1])

            # Normalize
            ct_clipped = np.clip(ct, HU_WINDOW[0], HU_WINDOW[1])
            ct_norm = (ct_clipped - ct_clipped.mean()) / (ct_clipped.std() + 1e-8)

            status['stage'] = 'stage1'
            status['progress'] = 15

            # ===== STAGE 1 =====
            s1_probs = sliding_window_predict(_models['stage1'], ct_norm, (96, 96, 96), 0.5, 2)
            s1_mask = postprocess_stage1(s1_probs)
            status['progress'] = 45

            # ===== ROI EXTRACTION =====
            status['stage'] = 'roi_extraction'
            status['progress'] = 50

            s1_coords = np.argwhere(s1_mask > 0)
            ROI_MARGIN = 15

            if len(s1_coords) == 0:
                D, H, W = ct_norm.shape
                cd, ch, cw = D // 4, H // 4, W // 4
                bbox = (slice(cd, cd + D // 2), slice(ch, ch + H // 2), slice(cw, cw + W // 2))
                s1_failed = True
            else:
                mins = np.maximum(s1_coords.min(0) - ROI_MARGIN, 0)
                maxs = np.minimum(s1_coords.max(0) + 1 + ROI_MARGIN, np.array(ct_norm.shape))
                bbox = tuple(slice(int(mn), int(mx)) for mn, mx in zip(mins, maxs))
                s1_failed = False

            ct_roi = ct_norm[bbox]
            ct_hu_roi = ct_hu[bbox]

            # ===== STAGE 2 =====
            status['stage'] = 'stage2'
            status['progress'] = 55
            s2_probs = sliding_window_predict(_models['stage2'], ct_roi, (64, 64, 64), 0.5, 3)
            s2_pred = postprocess_stage2(s2_probs)
            status['progress'] = 80

            # Place back into full volume
            full_seg = np.zeros(ct_norm.shape, dtype=np.uint8)
            full_seg[bbox] = s2_pred

            # ===== CLINICAL ANALYSIS =====
            status['stage'] = 'clinical_analysis'
            status['progress'] = 82

            tumor_voxels = int((full_seg == 2).sum())
            panc_voxels = int((full_seg >= 1).sum())
            tumor_vol_cm3 = tumor_voxels * np.prod(TARGET_SPACING) / 1000
            panc_vol_cm3 = panc_voxels * np.prod(TARGET_SPACING) / 1000

            # Clinical modules
            clinical_data = {}
            t_staging_data = {}
            severity_data = {}
            radiomics_data = {}
            try:
                radiomics = extract_radiomics(ct_hu_roi, s2_pred, TARGET_SPACING)
                classification = classify_pathology(radiomics)
                t_stage = compute_t_stage(radiomics)
                severity = compute_pancreatitis_severity(radiomics, classification)
                report = generate_clinical_report(radiomics, classification, t_stage, severity)

                clinical_data = {
                    'summary': report.get('summary', ''),
                    'risk_level': report.get('risk_level', 'LOW'),
                    'primary_finding': report.get('primary_finding', 'Normal'),
                    'confidence': report.get('confidence', 0),
                    'evidence': report.get('evidence', []),
                    'differential': report.get('differential', []),
                    'recommendations': report.get('recommendations', []),
                }

                if t_stage.get('stage') != 'N/A':
                    t_staging_data = {
                        'stage': t_stage.get('stage', 'N/A'),
                        'substage': t_stage.get('substage', 'N/A'),
                        'description': t_stage.get('description', ''),
                        'max_dimension_mm': t_stage.get('max_dimension_mm', 0),
                        'volume_cm3': t_stage.get('volume_cm3', 0),
                        'extent_mm': t_stage.get('extent_mm', [0, 0, 0]),
                        'tnm_string': t_stage.get('tnm_string', 'N/A'),
                        'resectability': t_stage.get('resectability', 'N/A'),
                        'resectability_detail': t_stage.get('resectability_detail', ''),
                        'notes': t_stage.get('notes', []),
                    }

                if severity.get('applicable'):
                    severity_data = {
                        'severity': severity['severity'],
                        'score': severity['score'],
                        'max_score': severity['max_score'],
                        'revised_atlanta': severity['revised_atlanta'],
                        'components': severity['components'],
                        'details': severity['details'],
                    }

                radiomics_data = {
                    'pancreas': radiomics.get('pancreas', {}),
                    'tumor': radiomics.get('tumor', {}),
                    'peripancreatic': radiomics.get('peripancreatic', {}),
                }
            except Exception as e:
                print(f"[WARNING] Clinical analysis failed: {e}")

            # ===== XAI =====
            status['stage'] = 'xai'
            status['progress'] = 85
            xai_images_b64 = []
            try:
                attention_map, gradcam_map = run_xai_pipeline(
                    _models['stage2'], ct_roi, s2_pred, device, ct_roi.shape
                )
                # Generate XAI result images for the session
                result_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id)
                os.makedirs(result_dir, exist_ok=True)
                if attention_map is not None or gradcam_map is not None:
                    xai_paths = generate_xai_visualizations(
                        ct_roi, s2_pred, attention_map, gradcam_map,
                        result_dir, num_slices=5,
                        hu_volume=ct_hu_roi,
                        features=radiomics_data if radiomics_data else None,
                    )
                    xai_images_b64 = [
                        f'/static/results/{session_id}/{p}' for p in xai_paths
                    ]
            except Exception as e:
                print(f"[WARNING] XAI pipeline failed: {e}")

            # ===== GENERATE RESULT IMAGES =====
            status['stage'] = 'generating_images'
            status['progress'] = 92

            result_slices = generate_result_slices_b64(ct_norm, full_seg, s1_mask)
            mip_data = generate_mip_b64(ct_norm)

            # Also generate file-based images for backward compatibility
            result_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id)
            os.makedirs(result_dir, exist_ok=True)

            # Tumor metrics
            if tumor_voxels > 0:
                t_coords = np.argwhere(full_seg == 2)
                t_extent = (t_coords.max(0) - t_coords.min(0) + 1) * np.array(TARGET_SPACING)
                max_dim = float(t_extent.max())
                t_center = t_coords.mean(axis=0) * np.array(TARGET_SPACING)
            else:
                max_dim = 0
                t_extent = [0, 0, 0]
                t_center = [0, 0, 0]

            # HU analysis
            tumor_hu_mean = 0
            tumor_hu_std = 0
            panc_hu_mean = 0
            hu_confidence = 'N/A'
            hu_class = 'No tumor'
            enhancement_ratio = 0

            if tumor_voxels > 0:
                tumor_hu_vals = ct_hu[full_seg == 2]
                tumor_hu_mean = float(tumor_hu_vals.mean())
                tumor_hu_std = float(tumor_hu_vals.std())
                panc_hu_vals = ct_hu[full_seg == 1]
                panc_hu_mean = float(panc_hu_vals.mean()) if len(panc_hu_vals) > 0 else 0
                enhancement_ratio = round(tumor_hu_mean / max(panc_hu_mean, 1), 3)

                if 40 <= tumor_hu_mean <= 80:
                    hu_confidence = "HIGH"
                    hu_class = "PDAC (typical hypoattenuating mass)"
                elif 0 <= tumor_hu_mean < 20:
                    hu_confidence = "MEDIUM"
                    hu_class = "Cystic Neoplasm (fluid-density)"
                elif 80 < tumor_hu_mean <= 120:
                    hu_confidence = "MEDIUM"
                    hu_class = "Pancreatitis/Inflamed Tissue"
                elif tumor_hu_mean > 150:
                    hu_confidence = "LOW"
                    hu_class = "Calcification (review recommended)"
                else:
                    hu_confidence = "LOW"
                    hu_class = f"Atypical HU ({tumor_hu_mean:.0f})"
            else:
                panc_hu_vals = ct_hu[full_seg == 1]
                panc_hu_mean = float(panc_hu_vals.mean()) if len(panc_hu_vals) > 0 else 0

            # ===== COMPILE RESULTS =====
            status['stage'] = 'complete'
            status['progress'] = 100
            status['results'] = {
                # Core metrics
                'tumor_detected': tumor_voxels > 0,
                'tumor_voxels': tumor_voxels,
                'tumor_volume_cm3': round(tumor_vol_cm3, 2),
                'pancreas_volume_cm3': round(panc_vol_cm3, 1),
                'max_dimension_mm': round(max_dim, 1),
                'tumor_center_mm': [round(float(c), 1) for c in t_center],
                'tumor_extent_mm': [round(float(e), 1) for e in t_extent],
                'original_shape': list(original_shape),
                'resampled_shape': [int(x) for x in ct_norm.shape],
                'original_spacing': [round(float(s), 3) for s in spacing],
                'stage1_failed': s1_failed,

                # HU Analysis
                'tumor_hu_mean': round(tumor_hu_mean, 1),
                'tumor_hu_std': round(tumor_hu_std, 1),
                'panc_hu_mean': round(panc_hu_mean, 1),
                'hu_confidence': hu_confidence,
                'hu_classification': hu_class,
                'enhancement_ratio': enhancement_ratio,

                # Inline images (base64)
                'result_slices': result_slices,
                'mip_axial': mip_data.get('axial', ''),
                'mip_coronal': mip_data.get('coronal', ''),
                'mip_sagittal': mip_data.get('sagittal', ''),

                # XAI images (file paths)
                'xai_images': xai_images_b64,

                # Clinical
                'clinical': clinical_data if clinical_data else None,
                't_staging': t_staging_data if t_staging_data else None,
                'pancreatitis': severity_data if severity_data else None,
                'radiomics': radiomics_data if radiomics_data else None,

                # Timing (approximate from status timestamps)
                'timing': {
                    'total': round(time.time() - status.get('_start_time', time.time()), 1)
                },
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            _analysis_status[session_id] = {
                'stage': 'error', 'progress': 0,
                'error': str(e)
            }
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Record start time
    _analysis_status[session_id]['_start_time'] = time.time()

    # Run in background thread
    thread = threading.Thread(target=run_analysis, daemon=True)
    thread.start()

    return jsonify({'status': 'started', 'session_id': session_id})


@app.route('/status/<session_id>')
def get_status(session_id):
    """Poll analysis progress."""
    status = _analysis_status.get(session_id, {'stage': 'unknown', 'progress': 0})
    # Don't expose internal keys
    response = {
        'stage': status.get('stage', 'unknown'),
        'progress': status.get('progress', 0),
    }
    if 'results' in status:
        response['results'] = status['results']
    if 'error' in status:
        response['error'] = status['error']
    return jsonify(response)


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'running',
        'models_loaded': MODELS_LOADED,
        'dicom_support': PYDICOM_AVAILABLE,
        'gpu_available': torch.cuda.is_available()
    })


# ============================================================
# Run
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Starting CascadePanc-ViTUNet Server")
    print("  Open: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
