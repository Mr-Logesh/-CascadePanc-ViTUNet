"""
CascadePanc-ViTUNet — Model Validation Script
==============================================
Runs the cascade inference on training cases (where ground truth exists),
computes Dice scores, and generates visual comparison images.

Usage:
    python validate_model.py                     # validate 5 random cases
    python validate_model.py --num_cases 10      # validate 10 random cases
    python validate_model.py --case pancreas_001 # validate a specific case
"""

import os
import sys
import argparse
import random
import time
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---- Project imports ----
from model import ViTUNet
from inference import (
    load_models, sliding_window_predict,
    postprocess_stage1, postprocess_stage2, filter_tumor_by_hu,
    extract_roi, HU_WINDOW, TARGET_SPACING, ROI_MARGIN,
)

# Improvement params (match run_cascade defaults; override via env vars)
_S_BATCH = int(os.environ.get('BATCH_SIZE', '4'))
_MIN_TUMOR_CM3 = float(os.environ.get('TUMOR_MIN_CM3', '0.3'))
_max_hu = os.environ.get('TUMOR_HU_MAX', '110')
_MAX_TUMOR_HU = None if _max_hu.lower() in ('', 'none', 'off') else float(_max_hu)
_VOXEL_VOL_CM3 = float(np.prod(TARGET_SPACING)) / 1000.0

# ============================================================
# Dice score computation
# ============================================================

def dice_score(pred, gt, class_id):
    """Compute Dice coefficient for a specific class."""
    p = (pred == class_id).astype(np.float32)
    g = (gt == class_id).astype(np.float32)
    intersection = (p * g).sum()
    total = p.sum() + g.sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0  # both empty = perfect
    return (2.0 * intersection) / total


def iou_score(pred, gt, class_id):
    """Compute IoU (Jaccard) for a specific class."""
    p = (pred == class_id).astype(np.float32)
    g = (gt == class_id).astype(np.float32)
    intersection = (p * g).sum()
    union = p.sum() + g.sum() - intersection
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def precision_recall(pred, gt, class_id):
    """Compute precision and recall for a specific class."""
    p = (pred == class_id).astype(np.float32)
    g = (gt == class_id).astype(np.float32)
    tp = (p * g).sum()
    precision = tp / (p.sum() + 1e-8) if p.sum() > 0 else 0.0
    recall = tp / (g.sum() + 1e-8) if g.sum() > 0 else 0.0
    return precision, recall

# ============================================================
# Run cascade on a single case
# ============================================================

def run_validation_case(ct_path, label_path, models, device):
    """Run cascade inference and compare with ground truth."""
    import torch
    from torch.cuda.amp import autocast

    # Load CT and label
    nii_ct = nib.load(ct_path)
    ct_raw = nii_ct.get_fdata().astype(np.float32)
    spacing = np.array(nii_ct.header.get_zooms()[:3])

    nii_lbl = nib.load(label_path)
    gt_raw = nii_lbl.get_fdata().astype(np.uint8)

    original_shape = ct_raw.shape

    # Resample CT
    scale = spacing / np.array(TARGET_SPACING)
    ct = zoom(ct_raw, scale, order=1)

    # Resample GT (nearest neighbor to preserve labels)
    gt = zoom(gt_raw.astype(np.float32), scale, order=0).astype(np.uint8)

    # Preprocess CT
    ct_clipped = np.clip(ct, HU_WINDOW[0], HU_WINDOW[1])
    ct_norm = (ct_clipped - ct_clipped.mean()) / (ct_clipped.std() + 1e-8)

    # --- Stage 1: Pancreas Localization ---
    s1_probs = sliding_window_predict(models['stage1'], ct_norm, (96, 96, 96), 0.5, 2,
                                       batch_size=_S_BATCH)
    s1_mask = postprocess_stage1(s1_probs)

    # --- ROI Extraction ---
    ct_roi, bbox = extract_roi(s1_mask, ct_norm, ROI_MARGIN)

    if ct_roi is None:
        D, H, W = ct_norm.shape
        cd, ch, cw = D // 4, H // 4, W // 4
        ct_roi = ct_norm[cd:cd+D//2, ch:ch+H//2, cw:cw+W//2]
        bbox = (slice(cd, cd+D//2), slice(ch, ch+H//2), slice(cw, cw+W//2))

    # --- Stage 2: Tumor Segmentation ---
    s2_probs = sliding_window_predict(models['stage2'], ct_roi, (64, 64, 64), 0.5, 3,
                                       batch_size=_S_BATCH)
    s2_pred = postprocess_stage2(
        s2_probs,
        min_tumor_volume_cm3=_MIN_TUMOR_CM3,
        voxel_volume_cm3=_VOXEL_VOL_CM3,
    )
    # HU sanity check using raw-HU values inside the ROI
    s2_pred = filter_tumor_by_hu(s2_pred, ct_clipped[bbox], max_mean_hu=_MAX_TUMOR_HU)

    # Place ROI prediction back into full volume
    full_pred = np.zeros(ct_norm.shape, dtype=np.uint8)
    full_pred[bbox] = s2_pred

    # --- Compute Metrics ---
    results = {
        'original_shape': original_shape,
        'resampled_shape': ct_norm.shape,
        'prediction': full_pred,
        'ground_truth': gt,
        'ct_display': ct_clipped,
        's1_mask': s1_mask,
    }

    # Stage 1 Dice (pancreas: gt >= 1 vs s1_mask)
    gt_panc_binary = (gt >= 1).astype(np.uint8)
    s1_dice = dice_score(s1_mask, gt_panc_binary, 1)
    results['s1_pancreas_dice'] = s1_dice

    # Full cascade — Pancreas Dice
    pred_panc_binary = (full_pred >= 1).astype(np.uint8)
    cascade_panc_dice = dice_score(pred_panc_binary, gt_panc_binary, 1)
    results['cascade_pancreas_dice'] = cascade_panc_dice

    # Full cascade — Tumor Dice
    gt_has_tumor = (gt == 2).sum() > 0
    results['gt_has_tumor'] = gt_has_tumor

    if gt_has_tumor:
        tumor_dice = dice_score(full_pred, gt, 2)
        tumor_iou = iou_score(full_pred, gt, 2)
        tumor_prec, tumor_rec = precision_recall(full_pred, gt, 2)
        results['tumor_dice'] = tumor_dice
        results['tumor_iou'] = tumor_iou
        results['tumor_precision'] = tumor_prec
        results['tumor_recall'] = tumor_rec
    else:
        # No tumor in GT — check if model also didn't detect
        pred_has_tumor = (full_pred == 2).sum() > 0
        results['tumor_dice'] = None
        results['tumor_false_positive'] = pred_has_tumor

    # ROI recall — fraction of GT pancreas captured by the ROI
    gt_roi = gt[bbox]
    roi_recall = (gt_roi >= 1).sum() / max((gt >= 1).sum(), 1)
    results['roi_recall'] = roi_recall

    # Volume metrics
    voxel_vol = np.prod(TARGET_SPACING) / 1000  # cm³
    results['pred_pancreas_vol_cm3'] = float((full_pred >= 1).sum() * voxel_vol)
    results['gt_pancreas_vol_cm3'] = float((gt >= 1).sum() * voxel_vol)
    results['pred_tumor_vol_cm3'] = float((full_pred == 2).sum() * voxel_vol)
    results['gt_tumor_vol_cm3'] = float((gt == 2).sum() * voxel_vol)

    return results


# ============================================================
# Visualization
# ============================================================

def generate_comparison_image(results, case_name, output_dir):
    """Generate side-by-side visual comparison: CT | Ground Truth | Prediction | Overlay."""

    ct = results['ct_display']
    gt = results['ground_truth']
    pred = results['prediction']

    D = ct.shape[0]
    # Pick slices where ground truth has content
    gt_slices = np.where(gt.max(axis=(1, 2)) > 0)[0]

    if len(gt_slices) == 0:
        slice_indices = [D // 2]
    else:
        # Pick 5 evenly spaced slices from the GT region
        n_slices = min(5, len(gt_slices))
        selected = np.linspace(0, len(gt_slices) - 1, n_slices).astype(int)
        slice_indices = gt_slices[selected]

    # Normalize CT for display
    ct_disp = (ct - ct.min()) / (ct.max() - ct.min() + 1e-8)

    # Create color overlays
    def make_overlay(ct_slice, mask_slice):
        """Create RGB overlay: green=pancreas, red=tumor."""
        rgb = np.stack([ct_slice] * 3, axis=-1)
        # Pancreas = green overlay
        panc_mask = mask_slice == 1
        rgb[panc_mask, 0] *= 0.5
        rgb[panc_mask, 1] = np.clip(rgb[panc_mask, 1] * 0.5 + 0.4, 0, 1)
        rgb[panc_mask, 2] *= 0.5
        # Tumor = red overlay
        tumor_mask = mask_slice == 2
        rgb[tumor_mask, 0] = np.clip(rgb[tumor_mask, 0] * 0.5 + 0.5, 0, 1)
        rgb[tumor_mask, 1] *= 0.3
        rgb[tumor_mask, 2] *= 0.3
        return rgb

    n_slices = len(slice_indices)
    fig, axes = plt.subplots(n_slices, 4, figsize=(20, 5 * n_slices))

    if n_slices == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['CT Scan', 'Ground Truth', 'Model Prediction', 'Overlay Comparison']
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=14, fontweight='bold', pad=10)

    for i, s_idx in enumerate(slice_indices):
        ct_s = ct_disp[s_idx]
        gt_s = gt[s_idx]
        pred_s = pred[s_idx]

        # Column 0: CT
        axes[i, 0].imshow(ct_s, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_ylabel(f'Slice {s_idx}', fontsize=11, fontweight='bold')

        # Column 1: Ground Truth overlay
        gt_overlay = make_overlay(ct_s, gt_s)
        axes[i, 1].imshow(gt_overlay)

        # Column 2: Prediction overlay
        pred_overlay = make_overlay(ct_s, pred_s)
        axes[i, 2].imshow(pred_overlay)

        # Column 3: Side-by-side overlay (GT contour + pred fill)
        combined = np.stack([ct_s] * 3, axis=-1)
        # Pred pancreas = green fill
        p_panc = pred_s == 1
        combined[p_panc, 1] = np.clip(combined[p_panc, 1] * 0.5 + 0.3, 0, 1)
        # Pred tumor = red fill
        p_tumor = pred_s == 2
        combined[p_tumor, 0] = np.clip(combined[p_tumor, 0] * 0.5 + 0.4, 0, 1)
        # GT contour = cyan outline
        from scipy.ndimage import binary_erosion
        gt_panc_boundary = (gt_s >= 1) & ~binary_erosion(gt_s >= 1, iterations=1)
        combined[gt_panc_boundary] = [0, 1, 1]  # cyan for GT boundary
        gt_tumor_boundary = (gt_s == 2) & ~binary_erosion(gt_s == 2, iterations=1)
        combined[gt_tumor_boundary] = [1, 1, 0]  # yellow for GT tumor boundary

        axes[i, 3].imshow(combined)

        for j in range(4):
            axes[i, j].axis('off')

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='lime', alpha=0.5, label='Pancreas (Pred)'),
        mpatches.Patch(facecolor='red', alpha=0.5, label='Tumor (Pred)'),
        mpatches.Patch(facecolor='cyan', alpha=0.8, label='Pancreas (GT boundary)'),
        mpatches.Patch(facecolor='yellow', alpha=0.8, label='Tumor (GT boundary)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11,
               frameon=True, facecolor='#1a1a2e', edgecolor='#333',
               labelcolor='white')

    # Metrics text
    metrics_text = f"CASE: {case_name}\n"
    metrics_text += f"Stage 1 Pancreas Dice:     {results['s1_pancreas_dice']:.4f}\n"
    metrics_text += f"Cascade Pancreas Dice:     {results['cascade_pancreas_dice']:.4f}\n"
    if results['gt_has_tumor']:
        metrics_text += f"Cascade Tumor Dice:        {results['tumor_dice']:.4f}\n"
        metrics_text += f"Tumor IoU:                 {results['tumor_iou']:.4f}\n"
        metrics_text += f"Tumor Precision:           {results['tumor_precision']:.4f}\n"
        metrics_text += f"Tumor Recall:              {results['tumor_recall']:.4f}\n"
    else:
        fp = results.get('tumor_false_positive', False)
        metrics_text += f"Tumor in GT:               NO\n"
        metrics_text += f"False Positive Tumor:      {'YES ⚠' if fp else 'NO ✓'}\n"
    metrics_text += f"ROI Recall:                {results['roi_recall']:.4f}\n"
    metrics_text += f"Pred Pancreas Vol:         {results['pred_pancreas_vol_cm3']:.1f} cm³\n"
    metrics_text += f"GT   Pancreas Vol:         {results['gt_pancreas_vol_cm3']:.1f} cm³\n"
    metrics_text += f"Pred Tumor Vol:            {results['pred_tumor_vol_cm3']:.2f} cm³\n"
    metrics_text += f"GT   Tumor Vol:            {results['gt_tumor_vol_cm3']:.2f} cm³\n"

    fig.text(0.02, 0.01, metrics_text, fontsize=10, fontfamily='monospace',
             color='white', backgroundcolor='#1a1a2e',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e', edgecolor='#444'))

    plt.tight_layout(rect=[0, 0.08, 1, 0.98])

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'validation_{case_name}.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0f0f1a', edgecolor='none')
    plt.close(fig)
    print(f"  [SAVED] {output_path}")
    return output_path


def generate_summary_report(all_results, output_dir):
    """Generate a summary chart + text report for all validated cases."""

    case_names = [r['case_name'] for r in all_results]
    s1_dices = [r['s1_pancreas_dice'] for r in all_results]
    panc_dices = [r['cascade_pancreas_dice'] for r in all_results]
    tumor_dices = [r['tumor_dice'] for r in all_results if r['tumor_dice'] is not None]
    tumor_cases = [r['case_name'] for r in all_results if r['tumor_dice'] is not None]

    # --- Summary Bar Chart ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0f0f1a')

    # Plot 1: Pancreas Dice per case
    x = np.arange(len(case_names))
    bars1 = axes[0].bar(x - 0.2, s1_dices, 0.4, label='Stage 1 Pancreas', color='#00d2ff', alpha=0.8)
    bars2 = axes[0].bar(x + 0.2, panc_dices, 0.4, label='Cascade Pancreas', color='#7b61ff', alpha=0.8)
    axes[0].set_xlabel('Case', color='white', fontsize=11)
    axes[0].set_ylabel('Dice Score', color='white', fontsize=11)
    axes[0].set_title('Pancreas Segmentation Dice', color='white', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c.replace('pancreas_', 'P') for c in case_names],
                             rotation=45, ha='right', fontsize=8, color='white')
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=np.mean(panc_dices), color='#7b61ff', linestyle='--', alpha=0.5)
    axes[0].set_facecolor('#1a1a2e')
    axes[0].tick_params(colors='white')
    axes[0].spines['bottom'].set_color('#333')
    axes[0].spines['left'].set_color('#333')

    # Plot 2: Tumor Dice (only cases with tumor)
    if tumor_dices:
        x2 = np.arange(len(tumor_cases))
        bars3 = axes[1].bar(x2, tumor_dices, 0.6, color='#ff4757', alpha=0.8)
        axes[1].set_xlabel('Case', color='white', fontsize=11)
        axes[1].set_ylabel('Dice Score', color='white', fontsize=11)
        axes[1].set_title('Tumor Segmentation Dice', color='white', fontsize=13, fontweight='bold')
        axes[1].set_xticks(x2)
        axes[1].set_xticklabels([c.replace('pancreas_', 'P') for c in tumor_cases],
                                 rotation=45, ha='right', fontsize=8, color='white')
        axes[1].set_ylim(0, 1.05)
        axes[1].axhline(y=np.mean(tumor_dices), color='#ff4757', linestyle='--', alpha=0.5,
                         label=f'Mean: {np.mean(tumor_dices):.3f}')
        axes[1].legend(fontsize=9)
    else:
        axes[1].text(0.5, 0.5, 'No tumor cases in selection', ha='center', va='center',
                     fontsize=14, color='white', transform=axes[1].transAxes)
    axes[1].set_facecolor('#1a1a2e')
    axes[1].tick_params(colors='white')
    axes[1].spines['bottom'].set_color('#333')
    axes[1].spines['left'].set_color('#333')

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'validation_summary_chart.png')
    fig.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close(fig)
    print(f"\n[SAVED] Summary chart: {chart_path}")

    # --- Text Report ---
    report = []
    report.append("=" * 70)
    report.append("  CascadePanc-ViTUNet — Validation Report")
    report.append("=" * 70)
    report.append(f"  Cases validated: {len(all_results)}")
    report.append(f"  Cases with tumor GT: {len(tumor_dices)}")
    report.append("")
    report.append("  AGGREGATE METRICS:")
    report.append(f"    Stage 1 Pancreas Dice:   {np.mean(s1_dices):.4f} ± {np.std(s1_dices):.4f}")
    report.append(f"    Cascade Pancreas Dice:   {np.mean(panc_dices):.4f} ± {np.std(panc_dices):.4f}")
    if tumor_dices:
        report.append(f"    Cascade Tumor Dice:      {np.mean(tumor_dices):.4f} ± {np.std(tumor_dices):.4f}")
        tumor_ious = [r['tumor_iou'] for r in all_results if r.get('tumor_iou') is not None]
        if tumor_ious:
            report.append(f"    Cascade Tumor IoU:       {np.mean(tumor_ious):.4f} ± {np.std(tumor_ious):.4f}")
    roi_recalls = [r['roi_recall'] for r in all_results]
    report.append(f"    ROI Recall:              {np.mean(roi_recalls):.4f} ± {np.std(roi_recalls):.4f}")
    report.append("")
    report.append("  PER-CASE RESULTS:")
    report.append("  " + "-" * 66)
    report.append(f"  {'Case':<18} {'S1 Panc':>9} {'Cas Panc':>9} {'Tumor':>9} {'ROI Rec':>9} {'Tumor?':>8}")
    report.append("  " + "-" * 66)
    for r in all_results:
        td = r['tumor_dice']
        td_str = f"{td:.4f}" if td is not None else "  N/A "
        gt_tumor = "YES" if r['gt_has_tumor'] else "no"
        report.append(f"  {r['case_name']:<18} {r['s1_pancreas_dice']:>9.4f} "
                      f"{r['cascade_pancreas_dice']:>9.4f} {td_str:>9} "
                      f"{r['roi_recall']:>9.4f} {gt_tumor:>8}")
    report.append("  " + "-" * 66)
    report.append("")

    report_text = "\n".join(report)
    report_path = os.path.join(output_dir, 'validation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n[SAVED] Report: {report_path}")
    return report_path


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Validate CascadePanc-ViTUNet model')
    parser.add_argument('--num_cases', type=int, default=5,
                        help='Number of random cases to validate (default: 5)')
    parser.add_argument('--case', type=str, default=None,
                        help='Specific case name to validate (e.g. pancreas_001)')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data'),
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'validation_results'),
                        help='Output directory for results')
    args = parser.parse_args()

    images_dir = os.path.join(args.data_dir, 'imagesTr')
    labels_dir = os.path.join(args.data_dir, 'labelsTr')

    if not os.path.isdir(images_dir):
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)
    if not os.path.isdir(labels_dir):
        print(f"ERROR: Labels directory not found: {labels_dir}")
        sys.exit(1)

    # Find matching image-label pairs
    image_files = sorted([f for f in os.listdir(images_dir)
                          if f.endswith('.nii.gz') and not f.startswith('.')])
    label_files = sorted([f for f in os.listdir(labels_dir)
                          if f.endswith('.nii.gz') and not f.startswith('.')])

    # Match by name
    label_set = set(label_files)
    pairs = [(f, f) for f in image_files if f in label_set]

    if not pairs:
        print("ERROR: No matching image-label pairs found.")
        sys.exit(1)

    print(f"\nFound {len(pairs)} image-label pairs in dataset.")

    # Select cases
    if args.case:
        case_name = args.case if args.case.endswith('.nii.gz') else args.case + '.nii.gz'
        pairs = [(img, lbl) for img, lbl in pairs if img == case_name]
        if not pairs:
            print(f"ERROR: Case '{args.case}' not found. Available: {[p[0] for p in pairs[:5]]}")
            sys.exit(1)
    else:
        random.seed(42)
        n = min(args.num_cases, len(pairs))
        pairs = random.sample(pairs, n)

    print(f"Validating {len(pairs)} case(s): {[p[0].replace('.nii.gz', '') for p in pairs]}")

    # Load models
    print("\nLoading models...")
    from inference import _models as model_store
    load_models(
        os.path.join(os.path.dirname(__file__), '..', 'models', 'stage1_best.pth'),
        os.path.join(os.path.dirname(__file__), '..', 'models', 'stage2_v2_best.pth'),
    )
    device = model_store['device']
    print(f"Models loaded on: {device}\n")

    # Run validation
    all_results = []
    for i, (img_file, lbl_file) in enumerate(pairs):
        case_name = img_file.replace('.nii.gz', '')
        print(f"\n[{i+1}/{len(pairs)}] Validating: {case_name}")
        print("  Loading and preprocessing...")

        t0 = time.time()
        results = run_validation_case(
            os.path.join(images_dir, img_file),
            os.path.join(labels_dir, lbl_file),
            model_store, device,
        )
        elapsed = time.time() - t0

        results['case_name'] = case_name
        results['elapsed_sec'] = elapsed

        print(f"  Stage 1 Pancreas Dice:  {results['s1_pancreas_dice']:.4f}")
        print(f"  Cascade Pancreas Dice:  {results['cascade_pancreas_dice']:.4f}")
        if results['gt_has_tumor']:
            print(f"  Cascade Tumor Dice:     {results['tumor_dice']:.4f}")
            print(f"  Tumor IoU:              {results['tumor_iou']:.4f}")
            print(f"  Tumor Precision:        {results['tumor_precision']:.4f}")
            print(f"  Tumor Recall:           {results['tumor_recall']:.4f}")
        else:
            fp = results.get('tumor_false_positive', False)
            print(f"  Tumor in GT:            NO")
            print(f"  False Positive:         {'YES ⚠' if fp else 'NO ✓'}")
        print(f"  ROI Recall:             {results['roi_recall']:.4f}")
        print(f"  Time:                   {elapsed:.1f}s")

        # Generate comparison image
        print("  Generating comparison image...")
        generate_comparison_image(results, case_name, args.output_dir)

        all_results.append(results)

    # Generate summary
    print("\n\n" + "=" * 60)
    generate_summary_report(all_results, args.output_dir)


if __name__ == '__main__':
    main()
