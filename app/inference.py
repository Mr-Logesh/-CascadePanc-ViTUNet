"""
CascadePanc-ViTUNet Inference Pipeline
Full cascade: Preprocessing → Stage 1 → ROI → Stage 2 → Post-processing
             → Clinical Analysis → XAI
"""

import os
import time
import gc
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom, label as scipy_label, binary_fill_holes
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import ViTUNet
from clinical_analysis import (
    extract_radiomics,
    classify_pathology,
    compute_t_stage,
    compute_pancreatitis_severity,
    generate_clinical_report
)
from xai_module import run_xai_pipeline, generate_xai_visualizations

# ============================================================
# Configuration
# ============================================================
TARGET_SPACING = (1.5, 1.5, 2.5)
HU_WINDOW = (-125, 275)
ROI_MARGIN = 15

# ============================================================
# Global model storage
# ============================================================
_models = {
    'stage1': None,
    'stage2': None,
    'device': None
}


def get_device():
    """Auto-detect best available device."""
    if os.environ.get('FORCE_CPU', '0') == '1':
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_models(stage1_path, stage2_path):
    """Load both stage models into memory."""
    device = get_device()
    _models['device'] = device
    print(f"[INFO] Using device: {device}")

    # Stage 1: Pancreas localization (2 classes)
    print("[INFO] Loading Stage 1 model...")
    model_s1 = ViTUNet(1, 2, 24, 384, 3, 6).to(device)
    ck1 = torch.load(stage1_path, map_location=device, weights_only=False)
    model_s1.load_state_dict(ck1['model_state_dict'])
    model_s1.eval()
    _models['stage1'] = model_s1
    print(f"[INFO] Stage 1 loaded (Best Dice: {ck1.get('best_dice', 'N/A'):.4f})")

    # Stage 2: Tumor segmentation (3 classes)
    print("[INFO] Loading Stage 2 model...")
    model_s2 = ViTUNet(1, 3, 24, 384, 3, 6).to(device)
    ck2 = torch.load(stage2_path, map_location=device, weights_only=False)
    model_s2.load_state_dict(ck2['model_state_dict'])
    model_s2.eval()
    _models['stage2'] = model_s2
    print("[INFO] Stage 2 loaded")

    # GPU pre-warm: dummy forward pass on each model so the first real upload
    # is not slowed down by CUDA kernel JIT compilation.
    if os.environ.get('SKIP_WARMUP', '0') != '1':
        try:
            with torch.inference_mode():
                ctx = (
                    torch.amp.autocast('cuda')
                    if device.type == 'cuda'
                    else _NullCtx()
                )
                dummy_s1 = torch.zeros(1, 1, 96, 96, 96, device=device)
                with ctx:
                    _ = model_s1(dummy_s1)
                dummy_s2 = torch.zeros(1, 1, 64, 64, 64, device=device)
                with ctx:
                    _ = model_s2(dummy_s2)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            print("[INFO] GPU pre-warm complete (first real inference will be fast)")
        except Exception as e:
            print(f"[WARNING] GPU pre-warm failed (not critical): {e}")

    return True


class _NullCtx:
    """No-op context manager for CPU pre-warm path."""
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


# ============================================================
# Core inference functions
# ============================================================

def sliding_window_predict(model, volume, patch_size, stride_ratio=0.5, num_classes=2, batch_size=1):
    """Sliding window inference with overlap averaging.

    batch_size: number of patches per forward pass (default 1 = original behavior).
                Higher = faster on GPU but uses more VRAM.
    Set env var USE_TTA=1 to enable Test-Time Augmentation (4-way axis-flip averaging).
    """
    use_tta = os.environ.get('USE_TTA', '0') == '1'
    if not use_tta:
        return _sliding_window_core(model, volume, patch_size, stride_ratio, num_classes, batch_size)

    # 4-way TTA: identity + flip on each spatial axis
    flip_axes_list = [None, (0,), (1,), (2,)]
    pred_accum = None
    for fa in flip_axes_list:
        vol_in = volume if fa is None else np.flip(volume, axis=fa).copy()
        pm = _sliding_window_core(model, vol_in, patch_size, stride_ratio, num_classes, batch_size)
        if fa is not None:
            pm = np.flip(pm, axis=tuple(a + 1 for a in fa)).copy()
        pred_accum = pm if pred_accum is None else (pred_accum + pm)
    pred_accum /= len(flip_axes_list)
    return pred_accum


def _sliding_window_core(model, volume, patch_size, stride_ratio, num_classes, batch_size):
    """Core sliding-window pass — batched, fp16 autocast on CUDA."""
    device = _models['device']
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = int(pd * stride_ratio), int(ph * stride_ratio), int(pw * stride_ratio)

    pred_map = np.zeros((num_classes, D, H, W), dtype=np.float32)
    count_map = np.zeros((D, H, W), dtype=np.float32)

    d_starts = list(range(0, max(1, D - pd + 1), sd))
    if d_starts[-1] + pd < D:
        d_starts.append(D - pd)
    h_starts = list(range(0, max(1, H - ph + 1), sh))
    if h_starts[-1] + ph < H:
        h_starts.append(H - ph)
    w_starts = list(range(0, max(1, W - pw + 1), sw))
    if w_starts[-1] + pw < W:
        w_starts.append(W - pw)

    # Collect all patches and their target coords up front
    coords = []
    patches_np = []
    for di in d_starts:
        for hi in h_starts:
            for wi in w_starts:
                patch = volume[di:di+pd, hi:hi+ph, wi:wi+pw]
                actual = patch.shape
                if actual != (pd, ph, pw):
                    p = np.zeros((pd, ph, pw), dtype=np.float32)
                    p[:actual[0], :actual[1], :actual[2]] = patch
                    patch = p
                coords.append((di, hi, wi, actual))
                patches_np.append(patch)

    with torch.inference_mode():
        for bs in range(0, len(patches_np), batch_size):
            bp = patches_np[bs:bs+batch_size]
            bc = coords[bs:bs+batch_size]
            inp = torch.from_numpy(np.stack(bp, axis=0)).unsqueeze(1).float().to(device, non_blocking=True)

            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    out = model(inp)
                    if isinstance(out, tuple):
                        out = out[0]
            else:
                out = model(inp)
                if isinstance(out, tuple):
                    out = out[0]

            prob = F.softmax(out.float(), dim=1).cpu().numpy()  # (B, C, pd, ph, pw)
            for i, (di, hi, wi, actual) in enumerate(bc):
                p = prob[i]
                for c in range(num_classes):
                    pred_map[c, di:di+actual[0], hi:hi+actual[1], wi:wi+actual[2]] += \
                        p[c, :actual[0], :actual[1], :actual[2]]
                count_map[di:di+actual[0], hi:hi+actual[1], wi:wi+actual[2]] += 1

    count_map = np.maximum(count_map, 1e-8)
    pred_map /= count_map[np.newaxis]
    return pred_map


def keep_largest(mask):
    """Keep only the largest connected component."""
    if mask.sum() == 0:
        return mask
    labeled, num = scipy_label(mask)
    if num <= 1:
        return mask
    sizes = [(labeled == i).sum() for i in range(1, num + 1)]
    return (labeled == (np.argmax(sizes) + 1)).astype(mask.dtype)


def postprocess_stage1(probs):
    """Post-process Stage 1: binary pancreas mask."""
    mask = (probs.argmax(0) == 1).astype(np.uint8)
    mask = keep_largest(mask)
    mask = binary_fill_holes(mask).astype(np.uint8)
    return mask


def extract_roi(mask, volume, margin=15):
    """Extract bounding box ROI from Stage 1 prediction."""
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None, None
    mins = np.maximum(coords.min(0) - margin, 0)
    maxs = np.minimum(coords.max(0) + 1 + margin, volume.shape)
    bbox = tuple(slice(mn, mx) for mn, mx in zip(mins, maxs))
    return volume[bbox], bbox


def postprocess_stage2(probs, min_tumor_volume_cm3=0.0, voxel_volume_cm3=None):
    """Post-process Stage 2: 3-class with anatomical constraints.

    min_tumor_volume_cm3: drop tumor clusters smaller than this size (default 0.0 = no-op).
    voxel_volume_cm3:     volume of a single voxel in cm³ (required if min_tumor_volume_cm3 > 0).
    """
    result = np.zeros(probs.shape[1:], dtype=np.uint8)
    panc = (probs.argmax(0) >= 1).astype(np.uint8)
    panc = keep_largest(panc)
    panc = binary_fill_holes(panc).astype(np.uint8)
    tumor = (probs.argmax(0) == 2).astype(np.uint8)
    tumor = tumor * panc  # Tumor must be inside pancreas

    if tumor.sum() > 0 and min_tumor_volume_cm3 > 0 and voxel_volume_cm3 is not None:
        # Filter ALL clusters by min volume (not just the largest)
        min_voxels = max(1, int(min_tumor_volume_cm3 / voxel_volume_cm3))
        labeled, num = scipy_label(tumor)
        for k in range(1, num + 1):
            cluster = (labeled == k)
            if cluster.sum() < min_voxels:
                tumor[cluster] = 0

    if tumor.sum() > 0:
        tumor = keep_largest(tumor)

    result[panc > 0] = 1
    result[tumor > 0] = 2
    return result


def filter_tumor_by_hu(s2_pred, hu_volume, max_mean_hu=110.0):
    """Demote predicted tumor clusters whose mean HU is above PDAC range to pancreas.

    Pancreatic ductal adenocarcinoma is hypoattenuating (40-80 HU portal phase).
    A cluster with mean HU >> 100 is most likely normal parenchyma falsely flagged
    as tumor — demote it from class 2 (tumor) to class 1 (pancreas).
    Set max_mean_hu=None to disable.
    """
    if max_mean_hu is None:
        return s2_pred
    tumor_mask = (s2_pred == 2)
    if tumor_mask.sum() == 0:
        return s2_pred
    labeled, num = scipy_label(tumor_mask)
    cleaned = s2_pred.copy()
    for k in range(1, num + 1):
        cluster = (labeled == k)
        if cluster.sum() == 0:
            continue
        cluster_hu_mean = float(hu_volume[cluster].mean())
        if cluster_hu_mean > max_mean_hu:
            cleaned[cluster] = 1  # demote to pancreas
    return cleaned


# ============================================================
# Main inference pipeline
# ============================================================

def run_cascade(nifti_path, progress_callback=None):
    """
    Run full cascade inference on a NIfTI CT scan.
    
    Returns dict with:
        - segmentation: 3D numpy array (0=bg, 1=pancreas, 2=tumor)
        - original_ct: preprocessed CT volume
        - metrics: dict with measurements
        - slices: dict with pre-rendered slice images
        - timing: dict with per-stage timing
    """
    results = {
        'success': False,
        'error': None,
        'segmentation': None,
        'original_ct': None,
        'original_shape': None,
        'resampled_shape': None,
        'metrics': {},
        'timing': {},
        'stage1_mask': None,
        'roi_bbox': None,
    }

    try:
        # ==========================================
        # STEP 1: Load and Preprocess
        # ==========================================
        if progress_callback:
            progress_callback('preprocessing', 0)

        t0 = time.time()

        # Detect file type: DICOM directory, DICOM file, or NIfTI
        input_path = nifti_path
        is_dicom = False

        if os.path.isdir(input_path):
            # Directory path — treat as DICOM series
            is_dicom = True
        elif os.path.isfile(input_path):
            fname_lower = input_path.lower()
            if fname_lower.endswith('.dcm') or (
                not fname_lower.endswith('.nii') and
                not fname_lower.endswith('.nii.gz') and
                not os.path.splitext(input_path)[1]
            ):
                is_dicom = True

        if is_dicom:
            try:
                from dicom_utils import load_dicom_series
                ct_raw, spacing = load_dicom_series(input_path)
            except ImportError:
                raise ValueError(
                    "DICOM path detected but pydicom is not available. "
                    "Install with: pip install pydicom"
                )
        else:
            nii = nib.load(input_path)
            ct_raw = nii.get_fdata().astype(np.float32)
            spacing = np.array(nii.header.get_zooms()[:3])

        results['original_shape'] = ct_raw.shape
        results['metrics']['original_spacing'] = spacing.tolist()

        # Resample
        scale = spacing / np.array(TARGET_SPACING)
        ct = zoom(ct_raw, scale, order=1)

        # HU windowing
        ct = np.clip(ct, HU_WINDOW[0], HU_WINDOW[1])

        # IMPORTANT: Save raw HU before normalization for clinical analysis
        ct_raw_hu = ct.copy()

        # Normalize (global since no GT available)
        ct = (ct - ct.mean()) / (ct.std() + 1e-8)

        results['resampled_shape'] = ct.shape
        results['original_ct'] = ct
        results['ct_raw_hu'] = ct_raw_hu
        results['timing']['preprocessing'] = time.time() - t0

        if progress_callback:
            progress_callback('stage1', 20)

        # ==========================================
        # STEP 2: Stage 1 - Pancreas Localization
        # ==========================================
        t1 = time.time()
        s1_batch = int(os.environ.get('BATCH_SIZE', '4'))
        s1_probs = sliding_window_predict(
            _models['stage1'], ct, (96, 96, 96), 0.5, 2, batch_size=s1_batch
        )
        s1_mask = postprocess_stage1(s1_probs)
        results['stage1_mask'] = s1_mask
        results['timing']['stage1'] = time.time() - t1

        panc_volume = s1_mask.sum() * np.prod(TARGET_SPACING) / 1000  # in cm³
        results['metrics']['pancreas_volume_cm3'] = float(f"{panc_volume:.1f}")
        results['metrics']['stage1_pancreas_voxels'] = int(s1_mask.sum())

        if progress_callback:
            progress_callback('roi_extraction', 50)

        # ==========================================
        # STEP 3: Extract ROI
        # ==========================================
        t2 = time.time()
        ct_roi, bbox = extract_roi(s1_mask, ct, ROI_MARGIN)

        if ct_roi is None:
            # Fallback: center crop
            D, H, W = ct.shape
            cd, ch, cw = D // 4, H // 4, W // 4
            ct_roi = ct[cd:cd+D//2, ch:ch+H//2, cw:cw+W//2]
            bbox = (slice(cd, cd+D//2), slice(ch, ch+H//2), slice(cw, cw+W//2))
            results['metrics']['roi_fallback'] = True
        else:
            results['metrics']['roi_fallback'] = False

        results['roi_bbox'] = bbox
        results['metrics']['roi_shape'] = ct_roi.shape
        results['timing']['roi_extraction'] = time.time() - t2

        if progress_callback:
            progress_callback('stage2', 60)

        # ==========================================
        # STEP 4: Stage 2 - Tumor Segmentation
        # ==========================================
        t3 = time.time()
        s2_batch = int(os.environ.get('BATCH_SIZE', '4'))
        s2_probs = sliding_window_predict(
            _models['stage2'], ct_roi, (64, 64, 64), 0.5, 3, batch_size=s2_batch
        )
        # Stage 2 post-processing with improvements:
        # (a) Min-volume tumor filter — drop FP specks below 0.3 cm³
        # (b) HU sanity check — demote tumor clusters with mean HU > 110 (likely parenchyma)
        voxel_vol_cm3 = float(np.prod(TARGET_SPACING)) / 1000.0
        min_tumor_cm3 = float(os.environ.get('TUMOR_MIN_CM3', '0.3'))
        max_tumor_hu = os.environ.get('TUMOR_HU_MAX', '110')
        max_tumor_hu = None if max_tumor_hu.lower() in ('', 'none', 'off') else float(max_tumor_hu)

        s2_pred = postprocess_stage2(
            s2_probs,
            min_tumor_volume_cm3=min_tumor_cm3,
            voxel_volume_cm3=voxel_vol_cm3,
        )
        # Apply HU sanity check using the raw HU values inside the ROI
        hu_roi_for_filter = ct_raw_hu[bbox]
        s2_pred = filter_tumor_by_hu(s2_pred, hu_roi_for_filter, max_mean_hu=max_tumor_hu)
        results['timing']['stage2'] = time.time() - t3

        # Place ROI prediction back into full volume
        full_seg = np.zeros(ct.shape, dtype=np.uint8)
        full_seg[bbox] = s2_pred
        results['segmentation'] = full_seg

        # Compute metrics
        tumor_voxels = (full_seg == 2).sum()
        panc_voxels = (full_seg >= 1).sum()
        tumor_volume = tumor_voxels * np.prod(TARGET_SPACING) / 1000  # cm³
        panc_volume_final = panc_voxels * np.prod(TARGET_SPACING) / 1000

        results['metrics']['tumor_detected'] = bool(tumor_voxels > 0)
        results['metrics']['tumor_volume_cm3'] = float(f"{tumor_volume:.2f}")
        results['metrics']['pancreas_volume_final_cm3'] = float(f"{panc_volume_final:.1f}")
        results['metrics']['tumor_voxels'] = int(tumor_voxels)
        results['metrics']['pancreas_voxels'] = int(panc_voxels)

        # Tumor location
        if tumor_voxels > 0:
            tumor_coords = np.argwhere(full_seg == 2)
            tumor_center = tumor_coords.mean(axis=0) * np.array(TARGET_SPACING)
            results['metrics']['tumor_center_mm'] = tumor_center.tolist()
            tumor_extent = (tumor_coords.max(0) - tumor_coords.min(0)) * np.array(TARGET_SPACING)
            results['metrics']['tumor_extent_mm'] = tumor_extent.tolist()

        # ==========================================
        # STEP 5: Clinical Analysis (HU-Based)
        # ==========================================
        if progress_callback:
            progress_callback('clinical_analysis', 85)

        t4 = time.time()
        try:
            # Get raw HU values within the ROI
            hu_roi = ct_raw_hu[bbox]

            # Extract radiomics features
            radiomics = extract_radiomics(hu_roi, s2_pred, TARGET_SPACING)

            # Classify pathology
            classification = classify_pathology(radiomics)

            # T-staging (if tumor detected)
            t_stage = compute_t_stage(radiomics)

            # Pancreatitis severity (if applicable)
            severity = compute_pancreatitis_severity(radiomics, classification)

            # Generate clinical report
            clinical_report = generate_clinical_report(
                radiomics, classification, t_stage, severity
            )

            results['clinical_report'] = clinical_report
            results['radiomics'] = radiomics
            results['t_staging'] = t_stage
            results['pancreatitis_severity'] = severity
            results['hu_roi'] = hu_roi

            print(f"[INFO] Clinical analysis complete: {classification.get('primary_diagnosis', 'N/A')}")
        except Exception as e:
            print(f"[WARNING] Clinical analysis failed: {e}")
            import traceback
            traceback.print_exc()

        results['timing']['clinical_analysis'] = time.time() - t4

        # ==========================================
        # STEP 6: Explainable AI (XAI)
        # ==========================================
        if progress_callback:
            progress_callback('xai', 90)

        t5 = time.time()
        try:
            device = _models['device']
            attention_map, gradcam_map = run_xai_pipeline(
                _models['stage2'], ct_roi, s2_pred, device, ct_roi.shape
            )
            results['attention_map'] = attention_map
            results['gradcam_map'] = gradcam_map
            print("[INFO] XAI pipeline complete")
        except Exception as e:
            print(f"[WARNING] XAI pipeline failed: {e}")
            results['attention_map'] = None
            results['gradcam_map'] = None

        results['timing']['xai'] = time.time() - t5

        results['timing']['total'] = time.time() - t0
        results['success'] = True

        if progress_callback:
            progress_callback('complete', 100)

    except Exception as e:
        results['error'] = str(e)
        import traceback
        traceback.print_exc()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def generate_slice_images(results, output_dir, num_slices=5):
    """Generate visualization images for the web interface."""
    ct = results['original_ct']
    seg = results['segmentation']
    s1_mask = results['stage1_mask']
    
    if ct is None or seg is None:
        return []

    D = ct.shape[0]
    
    # Find slices with tumor content
    tumor_per_slice = [(seg[s] == 2).sum() for s in range(D)]
    
    if max(tumor_per_slice) > 0:
        # Get slices around the tumor
        best_slice = np.argmax(tumor_per_slice)
        spread = max(D // 10, 3)
        slice_indices = np.linspace(
            max(0, best_slice - spread),
            min(D - 1, best_slice + spread),
            num_slices
        ).astype(int)
    else:
        # Spread across pancreas region
        panc_slices = [s for s in range(D) if (seg[s] >= 1).sum() > 0]
        if panc_slices:
            slice_indices = np.linspace(panc_slices[0], panc_slices[-1], num_slices).astype(int)
        else:
            slice_indices = np.linspace(D // 4, 3 * D // 4, num_slices).astype(int)

    image_paths = []

    for idx, s in enumerate(slice_indices):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.patch.set_facecolor('#0a0a0a')

        ct_slice = ct[s]
        seg_slice = seg[s]
        s1_slice = s1_mask[s] if s1_mask is not None else np.zeros_like(ct_slice)

        # Col 1: Original CT
        axes[0].imshow(ct_slice, cmap='gray')
        axes[0].set_title('CT Scan', color='white', fontsize=12, fontweight='bold')

        # Col 2: Stage 1 - Pancreas
        axes[1].imshow(ct_slice, cmap='gray')
        s1_overlay = np.zeros((*s1_slice.shape, 4))
        s1_overlay[s1_slice == 1] = [0, 0.85, 1, 0.35]
        axes[1].imshow(s1_overlay)
        axes[1].set_title('Stage 1: Pancreas', color='#00d4ff', fontsize=12, fontweight='bold')

        # Col 3: Stage 2 - Full segmentation
        axes[2].imshow(ct_slice, cmap='gray')
        seg_overlay = np.zeros((*seg_slice.shape, 4))
        seg_overlay[seg_slice == 1] = [0, 0.85, 0.3, 0.4]
        seg_overlay[seg_slice == 2] = [1, 0.2, 0.2, 0.6]
        axes[2].imshow(seg_overlay)
        axes[2].set_title('Stage 2: Segmentation', color='#00ff4c', fontsize=12, fontweight='bold')

        # Col 4: Tumor only (highlighted)
        axes[3].imshow(ct_slice, cmap='gray')
        tumor_overlay = np.zeros((*seg_slice.shape, 4))
        tumor_overlay[seg_slice == 2] = [1, 0.1, 0.1, 0.7]
        axes[3].imshow(tumor_overlay)
        has_tumor = (seg_slice == 2).sum() > 0
        title_color = '#ff3333' if has_tumor else '#666666'
        axes[3].set_title(
            f'Tumor {"DETECTED" if has_tumor else "Not in slice"}',
            color=title_color, fontsize=12, fontweight='bold'
        )

        for ax in axes:
            ax.axis('off')
            ax.set_facecolor('#0a0a0a')

        plt.suptitle(
            f'Slice {s}/{D} (Axial)',
            color='white', fontsize=14, fontweight='bold', y=0.02
        )
        plt.tight_layout()

        fname = f'slice_{idx}.png'
        fpath = os.path.join(output_dir, fname)
        plt.savefig(fpath, dpi=120, bbox_inches='tight',
                    facecolor='#0a0a0a', edgecolor='none')
        plt.close(fig)
        image_paths.append(fname)

    # Generate 3D volume rendering (simple maximum intensity projection)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0a0a0a')

    # Axial MIP
    axes[0].imshow(ct.max(axis=0), cmap='gray', aspect='auto')
    panc_proj = (seg >= 1).max(axis=0).astype(float)
    tumor_proj = (seg == 2).max(axis=0).astype(float)
    overlay_ax = np.zeros((*panc_proj.shape, 4))
    overlay_ax[panc_proj > 0] = [0, 0.8, 0.3, 0.3]
    overlay_ax[tumor_proj > 0] = [1, 0.1, 0.1, 0.6]
    axes[0].imshow(overlay_ax)
    axes[0].set_title('Axial MIP', color='white', fontsize=13, fontweight='bold')

    # Coronal MIP
    axes[1].imshow(ct.max(axis=1), cmap='gray', aspect='auto')
    panc_proj_c = (seg >= 1).max(axis=1).astype(float)
    tumor_proj_c = (seg == 2).max(axis=1).astype(float)
    overlay_co = np.zeros((*panc_proj_c.shape, 4))
    overlay_co[panc_proj_c > 0] = [0, 0.8, 0.3, 0.3]
    overlay_co[tumor_proj_c > 0] = [1, 0.1, 0.1, 0.6]
    axes[1].imshow(overlay_co)
    axes[1].set_title('Coronal MIP', color='white', fontsize=13, fontweight='bold')

    # Sagittal MIP
    axes[2].imshow(ct.max(axis=2), cmap='gray', aspect='auto')
    panc_proj_s = (seg >= 1).max(axis=2).astype(float)
    tumor_proj_s = (seg == 2).max(axis=2).astype(float)
    overlay_sa = np.zeros((*panc_proj_s.shape, 4))
    overlay_sa[panc_proj_s > 0] = [0, 0.8, 0.3, 0.3]
    overlay_sa[tumor_proj_s > 0] = [1, 0.1, 0.1, 0.6]
    axes[2].imshow(overlay_sa)
    axes[2].set_title('Sagittal MIP', color='white', fontsize=13, fontweight='bold')

    for ax in axes:
        ax.axis('off')
        ax.set_facecolor('#0a0a0a')

    plt.tight_layout()
    fpath = os.path.join(output_dir, 'mip_views.png')
    plt.savefig(fpath, dpi=120, bbox_inches='tight',
                facecolor='#0a0a0a', edgecolor='none')
    plt.close(fig)
    image_paths.append('mip_views.png')

    return image_paths


# ============================================================
# Base64 inline image helpers (for new two-step flow)
# ============================================================

def slice_to_base64(arr, cmap='gray', overlay=None):
    """Convert a 2D numpy array to base64 PNG string."""
    import io
    import base64

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=80)
    fig.patch.set_facecolor('black')
    ax.imshow(arr, cmap=cmap, aspect='equal')
    if overlay is not None:
        ax.imshow(overlay, aspect='equal')
    ax.axis('off')
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_preview_slices(ct_volume, num_previews=50):
    """Generate base64 preview slices from a CT volume for the upload preview."""
    ct_display = np.clip(ct_volume, HU_WINDOW[0], HU_WINDOW[1])
    ct_display = (ct_display - ct_display.min()) / (ct_display.max() - ct_display.min() + 1e-8)

    D = ct_volume.shape[0]
    num_previews = min(D, num_previews)
    preview_indices = np.linspace(0, D - 1, num_previews).astype(int)

    slices_b64 = []
    for si in preview_indices:
        slices_b64.append(slice_to_base64(ct_display[si]))

    return slices_b64, preview_indices.tolist()


def generate_result_slices_b64(ct_norm, full_seg, s1_mask, num_result_slices=12):
    """
    Generate annotated result slices as base64 with CT, Stage1, and Segmentation views.

    Returns a list of dicts, each with keys:
        index, ct, stage1, segmentation, has_tumor, has_pancreas
    """
    D = ct_norm.shape[0]
    ct_display = (ct_norm - ct_norm.min()) / (ct_norm.max() - ct_norm.min() + 1e-8)

    tumor_per_slice = [(full_seg[s] == 2).sum() for s in range(D)]
    panc_per_slice = [(full_seg[s] >= 1).sum() for s in range(D)]

    if max(tumor_per_slice) > 0:
        best = np.argmax(tumor_per_slice)
        spread = max(D // 8, 5)
        indices = np.linspace(max(0, best - spread), min(D - 1, best + spread), num_result_slices).astype(int)
    else:
        panc_s = [s for s in range(D) if panc_per_slice[s] > 0]
        if panc_s:
            indices = np.linspace(panc_s[0], panc_s[-1], num_result_slices).astype(int)
        else:
            indices = np.linspace(D // 4, 3 * D // 4, num_result_slices).astype(int)

    result_slices = []
    for si in np.unique(indices):
        seg_s = full_seg[si]
        s1_s = s1_mask[si] if s1_mask is not None else np.zeros_like(ct_display[si])

        # CT only
        ct_b64 = slice_to_base64(ct_display[si])

        # Stage 1 overlay (cyan pancreas)
        ov1 = np.zeros((*s1_s.shape, 4))
        ov1[s1_s == 1] = [0, 0.85, 1, 0.35]
        s1_b64 = slice_to_base64(ct_display[si], overlay=ov1)

        # Full segmentation overlay (green pancreas + red tumor)
        ov2 = np.zeros((*seg_s.shape, 4))
        ov2[seg_s == 1] = [0, 0.85, 0.3, 0.4]
        ov2[seg_s == 2] = [1, 0.15, 0.15, 0.65]
        seg_b64 = slice_to_base64(ct_display[si], overlay=ov2)

        result_slices.append({
            'index': int(si),
            'ct': ct_b64,
            'stage1': s1_b64,
            'segmentation': seg_b64,
            'has_tumor': bool((seg_s == 2).sum() > 0),
            'has_pancreas': bool((seg_s >= 1).sum() > 0),
        })

    return result_slices


def generate_mip_b64(ct_norm):
    """Generate Maximum Intensity Projection views as base64."""
    ct_display = (ct_norm - ct_norm.min()) / (ct_norm.max() - ct_norm.min() + 1e-8)

    return {
        'axial': slice_to_base64(ct_display.max(axis=0)),
        'coronal': slice_to_base64(ct_display.max(axis=1)),
        'sagittal': slice_to_base64(ct_display.max(axis=2)),
    }
