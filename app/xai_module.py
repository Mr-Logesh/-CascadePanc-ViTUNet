"""
CascadePanc-ViTUNet: Explainable AI Module
Attention Rollout (ViT) + Grad-CAM (CNN) for clinical interpretability

Two complementary XAI methods:
1. Attention Rollout: Shows WHERE the Vision Transformer focuses globally
2. Grad-CAM: Shows WHICH CNN features are most important for prediction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


# ============================================================
# 1. ATTENTION ROLLOUT
# ============================================================

class AttentionRolloutExtractor:
    """
    Extract and compute attention rollout from ViT bottleneck.
    
    Attention Rollout (Abnar & Zuidema, 2020):
    - Multiplies attention matrices across all transformer layers
    - Accounts for residual connections (identity matrix addition)
    - Result: spatial map showing which regions influence final prediction
    """
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = []
        self._hooks = []
    
    def register_hooks(self):
        """Register forward hooks on all transformer blocks to capture attention."""
        self.attention_weights = []
        self._hooks = []
        
        for block in self.model.vit.blocks:
            hook = block.attn.register_forward_hook(self._attn_hook)
            self._hooks.append(hook)
    
    def _attn_hook(self, module, input, output):
        """Capture attention weights from MultiheadAttention."""
        # output is (attn_output, attn_weights)
        # attn_weights shape: (batch, num_tokens, num_tokens)
        if isinstance(output, tuple) and len(output) > 1:
            self.attention_weights.append(output[1].detach().cpu())
    
    def remove_hooks(self):
        """Clean up hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def compute_rollout(self, discard_ratio=0.0):
        """
        Compute attention rollout from captured weights.
        
        Args:
            discard_ratio: Fraction of lowest attention values to zero out
        
        Returns:
            rollout_3d: (3, 3, 3) attention map in bottleneck space
        """
        if len(self.attention_weights) == 0:
            return None
        
        # Stack attention matrices: list of (B, N, N) → process
        result = None
        
        for attn in self.attention_weights:
            # attn shape: (B, N, N) — already averaged across heads by PyTorch
            attn_matrix = attn[0].numpy()  # Take first batch item
            
            # Optionally discard low-attention values
            if discard_ratio > 0:
                flat = attn_matrix.flatten()
                threshold = np.percentile(flat, discard_ratio * 100)
                attn_matrix[attn_matrix < threshold] = 0
                # Re-normalize rows
                row_sums = attn_matrix.sum(axis=-1, keepdims=True)
                attn_matrix = attn_matrix / (row_sums + 1e-8)
            
            # Add identity for residual connection
            identity = np.eye(attn_matrix.shape[0])
            attn_with_residual = 0.5 * attn_matrix + 0.5 * identity
            
            # Normalize rows to sum to 1
            row_sums = attn_with_residual.sum(axis=-1, keepdims=True)
            attn_with_residual = attn_with_residual / (row_sums + 1e-8)
            
            # Multiply across layers
            if result is None:
                result = attn_with_residual
            else:
                result = result @ attn_with_residual
        
        # Result: (N, N) matrix
        # Take mean attention TO each token (column mean)
        rollout_map = result.mean(axis=0)  # (N,)
        
        # Normalize to [0, 1]
        rollout_map = rollout_map - rollout_map.min()
        rollout_map = rollout_map / (rollout_map.max() + 1e-8)
        
        # Reshape to 3D grid
        # N = 27 = 3×3×3 (from 6×6×6 feature map with 2×2×2 patch embedding)
        N = len(rollout_map)
        grid_size = round(N ** (1/3))
        
        if grid_size ** 3 == N:
            rollout_3d = rollout_map.reshape(grid_size, grid_size, grid_size)
        else:
            # Fallback: just reshape to closest 3D
            rollout_3d = rollout_map.reshape(-1, 1, 1)
        
        return rollout_3d
    
    def get_full_resolution_map(self, rollout_3d, target_shape):
        """
        Upsample rollout from (3,3,3) to full volume resolution.
        
        Args:
            rollout_3d: (3,3,3) attention map
            target_shape: (D, H, W) target volume shape
        
        Returns:
            attention_map: (D, H, W) float32 in [0, 1]
        """
        if rollout_3d is None:
            return np.zeros(target_shape, dtype=np.float32)
        
        scale = np.array(target_shape) / np.array(rollout_3d.shape)
        attention_map = zoom(rollout_3d, scale, order=3)  # Cubic interpolation
        
        # Normalize
        attention_map = attention_map - attention_map.min()
        attention_map = attention_map / (attention_map.max() + 1e-8)
        
        return attention_map.astype(np.float32)


# ============================================================
# 2. GRAD-CAM for 3D CNN
# ============================================================

class GradCAM3D:
    """
    Gradient-weighted Class Activation Mapping for 3D medical images.
    
    Grad-CAM (Selvaraju et al., 2017) adapted for 3D volumes:
    - Computes gradients of target class w.r.t. feature maps
    - Weights feature maps by global-average-pooled gradients
    - Produces class-specific spatial heatmap
    """
    
    def __init__(self, model, target_layer=None):
        """
        Args:
            model: ViTUNet model
            target_layer: Which layer to compute Grad-CAM for.
                         Default: last encoder block (enc4)
        """
        self.model = model
        self.activations = None
        self.gradients = None
        self._hooks = []
        
        # Default target: encoder 4 (deepest CNN features)
        if target_layer is None:
            self.target_layer = model.enc4.conv.conv[4]  # Last InstanceNorm in enc4
        else:
            self.target_layer = target_layer
    
    def register_hooks(self):
        """Register forward and backward hooks."""
        self._hooks = []
        
        def fwd_hook(module, input, output):
            self.activations = output.detach()
        
        def bwd_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self._hooks.append(
            self.target_layer.register_forward_hook(fwd_hook)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(bwd_hook)
        )
    
    def remove_hooks(self):
        """Clean up hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def compute(self, input_tensor, target_class=2, device='cuda'):
        """
        Compute Grad-CAM heatmap.
        
        Args:
            input_tensor: (1, 1, D, H, W) input to model
            target_class: Class index for gradient computation (2=tumor)
            device: torch device
        
        Returns:
            cam: (D_feat, H_feat, W_feat) Grad-CAM heatmap normalized to [0, 1]
        """
        self.model.zero_grad()
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        
        # Target: sum of target class logits
        target = output[:, target_class].sum()
        
        # Backward pass
        target.backward(retain_graph=False)
        
        if self.activations is None or self.gradients is None:
            return None
        
        # Grad-CAM computation
        # Global average pooling of gradients → channel weights
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, D, H, W)
        
        # ReLU — only positive contributions
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.cpu().numpy()[0, 0]  # (D, H, W)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def get_full_resolution_map(self, cam, target_shape):
        """Upsample Grad-CAM to full volume resolution."""
        if cam is None:
            return np.zeros(target_shape, dtype=np.float32)
        
        scale = np.array(target_shape) / np.array(cam.shape)
        cam_full = zoom(cam, scale, order=1)  # Bilinear
        
        # Normalize
        cam_full = cam_full - cam_full.min()
        cam_full = cam_full / (cam_full.max() + 1e-8)
        
        return cam_full.astype(np.float32)


# ============================================================
# 3. XAI VISUALIZATION
# ============================================================

def generate_xai_visualizations(ct_volume, segmentation, attention_map, gradcam_map,
                                 output_dir, num_slices=5, hu_volume=None, features=None):
    """
    Generate publication-quality XAI visualization images.
    
    Creates:
    1. Attention rollout overlay on CT
    2. Grad-CAM overlay on CT
    3. Combined XAI panel (CT + Segmentation + Attention + Grad-CAM)
    4. HU distribution plots
    """
    image_paths = []
    D = ct_volume.shape[0]
    seg = segmentation
    
    # Find slices with tumor
    tumor_per_slice = [(seg[s] == 2).sum() for s in range(D)]
    best_slice = np.argmax(tumor_per_slice) if max(tumor_per_slice) > 0 else D // 2
    
    # Select slices around tumor
    spread = max(D // 12, 2)
    slice_indices = np.linspace(
        max(0, best_slice - spread),
        min(D - 1, best_slice + spread),
        num_slices
    ).astype(int)
    
    # Custom colormap for attention
    attention_cmap = plt.cm.inferno
    gradcam_cmap = plt.cm.jet
    
    # ---- Panel 1: Combined XAI slices ----
    for idx, s in enumerate(slice_indices):
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.patch.set_facecolor('#0a0a0a')
        
        ct_slice = ct_volume[s]
        seg_slice = seg[s]
        attn_slice = attention_map[s] if attention_map is not None else np.zeros_like(ct_slice)
        gcam_slice = gradcam_map[s] if gradcam_map is not None else np.zeros_like(ct_slice)
        
        # Col 1: Original CT
        axes[0].imshow(ct_slice, cmap='gray')
        axes[0].set_title('CT Scan', color='white', fontsize=11, fontweight='bold')
        
        # Col 2: Segmentation
        axes[1].imshow(ct_slice, cmap='gray')
        seg_ov = np.zeros((*seg_slice.shape, 4))
        seg_ov[seg_slice == 1] = [0, 0.85, 0.3, 0.4]
        seg_ov[seg_slice == 2] = [1, 0.15, 0.15, 0.6]
        axes[1].imshow(seg_ov)
        axes[1].set_title('Segmentation', color='#00e676', fontsize=11, fontweight='bold')
        
        # Col 3: Attention Rollout
        axes[2].imshow(ct_slice, cmap='gray')
        attn_ov = np.zeros((*attn_slice.shape, 4))
        # Map attention values to colormap
        for threshold in [0.3, 0.5, 0.7, 0.9]:
            mask = attn_slice >= threshold
            color = attention_cmap(threshold)
            attn_ov[mask] = [color[0], color[1], color[2], threshold * 0.6]
        axes[2].imshow(attn_ov)
        axes[2].set_title('ViT Attention Rollout', color='#ff9800', fontsize=11, fontweight='bold')
        
        # Col 4: Grad-CAM
        axes[3].imshow(ct_slice, cmap='gray')
        gcam_ov = np.zeros((*gcam_slice.shape, 4))
        for threshold in [0.3, 0.5, 0.7, 0.9]:
            mask = gcam_slice >= threshold
            color = gradcam_cmap(threshold)
            gcam_ov[mask] = [color[0], color[1], color[2], threshold * 0.5]
        axes[3].imshow(gcam_ov)
        axes[3].set_title('Grad-CAM (Tumor)', color='#ff4444', fontsize=11, fontweight='bold')
        
        # Col 5: Combined overlay
        axes[4].imshow(ct_slice, cmap='gray')
        # Segmentation boundary
        from scipy.ndimage import binary_erosion
        tumor_slice = (seg_slice == 2).astype(np.uint8)
        if tumor_slice.sum() > 0:
            boundary = tumor_slice - binary_erosion(tumor_slice, iterations=1).astype(np.uint8)
            boundary = np.clip(boundary, 0, 1)
            boundary_ov = np.zeros((*boundary.shape, 4))
            boundary_ov[boundary > 0] = [1, 1, 0, 1.0]  # Yellow boundary
            axes[4].imshow(boundary_ov)
        # Attention heatmap
        combined = np.maximum(attn_slice * 0.5, gcam_slice * 0.5)
        combined_ov = np.zeros((*combined.shape, 4))
        for threshold in [0.2, 0.4, 0.6, 0.8]:
            mask = combined >= threshold
            combined_ov[mask] = [1, threshold, 0, threshold * 0.4]
        axes[4].imshow(combined_ov)
        axes[4].set_title('Combined XAI', color='#ffeb3b', fontsize=11, fontweight='bold')
        
        for ax in axes:
            ax.axis('off')
            ax.set_facecolor('#0a0a0a')
        
        plt.suptitle(f'Explainable AI Analysis — Slice {s}/{D}',
                     color='white', fontsize=13, fontweight='bold', y=0.02)
        plt.tight_layout()
        
        fname = f'xai_slice_{idx}.png'
        fpath = os.path.join(output_dir, fname)
        plt.savefig(fpath, dpi=120, bbox_inches='tight',
                    facecolor='#0a0a0a', edgecolor='none')
        plt.close(fig)
        image_paths.append(fname)
    
    # ---- Panel 2: HU Distribution Analysis ----
    if hu_volume is not None and features is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.patch.set_facecolor('#0a0a0a')
        
        panc_mask = (seg == 1)
        tumor_mask = (seg == 2)
        
        panc_hu = hu_volume[panc_mask] if panc_mask.sum() > 0 else np.array([])
        tumor_hu = hu_volume[tumor_mask] if tumor_mask.sum() > 0 else np.array([])
        
        # Plot 1: HU histograms
        if len(panc_hu) > 0:
            axes[0].hist(panc_hu, bins=50, alpha=0.7, color='#00e676',
                        label=f'Pancreas (μ={panc_hu.mean():.0f} HU)', density=True)
        if len(tumor_hu) > 0:
            axes[0].hist(tumor_hu, bins=50, alpha=0.7, color='#ff4444',
                        label=f'Lesion (μ={tumor_hu.mean():.0f} HU)', density=True)
        axes[0].set_xlabel('Hounsfield Units', color='white', fontsize=11)
        axes[0].set_ylabel('Density', color='white', fontsize=11)
        axes[0].set_title('HU Distribution Analysis', color='white', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10, facecolor='#1a1a1a', edgecolor='#333',
                      labelcolor='white')
        axes[0].set_facecolor('#111')
        axes[0].tick_params(colors='white')
        axes[0].grid(alpha=0.2)
        
        # Add reference lines
        axes[0].axvline(x=40, color='#ff9800', linestyle='--', alpha=0.5, label='PDAC range')
        axes[0].axvline(x=80, color='#ff9800', linestyle='--', alpha=0.5)
        axes[0].axvline(x=100, color='#00b4d8', linestyle='--', alpha=0.5, label='Normal pancreas')
        axes[0].axvline(x=150, color='#00b4d8', linestyle='--', alpha=0.5)
        
        # Plot 2: HU composition pie chart
        if features.get('tumor', {}).get('voxel_count', 0) > 0:
            tumor_f = features['tumor']
            sizes = [
                tumor_f.get('pct_very_low_hu', 0),
                tumor_f.get('pct_low_hu', 0),
                tumor_f.get('pct_medium_hu', 0),
                tumor_f.get('pct_high_hu', 0),
                tumor_f.get('pct_very_high_hu', 0),
            ]
            labels_pie = ['Cystic\n(<20 HU)', 'Edema/Necrotic\n(20-60 HU)',
                         'Solid Tumor\n(60-100 HU)', 'Normal/Inflamed\n(100-150 HU)',
                         'Calcified\n(>150 HU)']
            colors_pie = ['#2196F3', '#FF9800', '#f44336', '#4CAF50', '#9E9E9E']
            
            # Filter out zero values
            nonzero = [(s, l, c) for s, l, c in zip(sizes, labels_pie, colors_pie) if s > 1]
            if nonzero:
                sizes_nz, labels_nz, colors_nz = zip(*nonzero)
                wedges, texts, autotexts = axes[1].pie(
                    sizes_nz, labels=labels_nz, colors=colors_nz,
                    autopct='%1.0f%%', startangle=90, textprops={'color': 'white', 'fontsize': 9}
                )
                for t in autotexts:
                    t.set_fontsize(9)
                    t.set_fontweight('bold')
        
        axes[1].set_title('Lesion HU Composition', color='white', fontsize=12, fontweight='bold')
        axes[1].set_facecolor('#0a0a0a')
        
        # Plot 3: Box plot comparison
        box_data = []
        box_labels = []
        box_colors = []
        if len(panc_hu) > 0:
            box_data.append(panc_hu[::max(1, len(panc_hu)//500)])  # Subsample for speed
            box_labels.append('Pancreas')
            box_colors.append('#00e676')
        if len(tumor_hu) > 0:
            box_data.append(tumor_hu[::max(1, len(tumor_hu)//500)])
            box_labels.append('Lesion')
            box_colors.append('#ff4444')
        
        if box_data:
            bp = axes[2].boxplot(box_data, labels=box_labels, patch_artist=True,
                                medianprops=dict(color='white', linewidth=2))
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        axes[2].set_ylabel('Hounsfield Units', color='white', fontsize=11)
        axes[2].set_title('HU Range Comparison', color='white', fontsize=12, fontweight='bold')
        axes[2].set_facecolor('#111')
        axes[2].tick_params(colors='white')
        axes[2].grid(alpha=0.2, axis='y')
        
        plt.tight_layout()
        fname = 'hu_analysis.png'
        fpath = os.path.join(output_dir, fname)
        plt.savefig(fpath, dpi=120, bbox_inches='tight',
                    facecolor='#0a0a0a', edgecolor='none')
        plt.close(fig)
        image_paths.append(fname)
    
    return image_paths


def run_xai_pipeline(model_s2, ct_roi, seg_roi, device, target_shape=None):
    """
    Complete XAI pipeline: Attention Rollout + Grad-CAM.
    
    Args:
        model_s2: Stage 2 ViTUNet model
        ct_roi: Preprocessed CT ROI volume (numpy)
        seg_roi: Segmentation of ROI (numpy)
        device: torch device
        target_shape: Output shape for upsampled maps
    
    Returns:
        attention_map: Full resolution attention rollout (numpy)
        gradcam_map: Full resolution Grad-CAM (numpy)
    """
    if target_shape is None:
        target_shape = ct_roi.shape
    
    # Find the patch with most tumor for XAI analysis
    tumor_per_slice = [(seg_roi[s] == 2).sum() for s in range(seg_roi.shape[0])]
    best_d = np.argmax(tumor_per_slice)
    
    # Extract a single patch centered on tumor
    D, H, W = ct_roi.shape
    pd, ph, pw = 64, 64, 64
    
    d_start = max(0, min(best_d - pd//2, D - pd))
    h_start = max(0, (H - ph) // 2)
    w_start = max(0, (W - pw) // 2)
    
    patch = ct_roi[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
    
    # Pad if needed
    actual_shape = patch.shape
    if patch.shape != (pd, ph, pw):
        padded = np.zeros((pd, ph, pw), dtype=np.float32)
        padded[:actual_shape[0], :actual_shape[1], :actual_shape[2]] = patch
        patch = padded
    
    input_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # ---- Attention Rollout ----
    attention_map = np.zeros(target_shape, dtype=np.float32)
    try:
        # We need to modify the forward to get attention weights
        # Store original forward methods
        original_forwards = []
        stored_attns = []
        
        for block in model_s2.vit.blocks:
            orig_fn = block.attn.forward
            original_forwards.append(orig_fn)
            
            def make_hook(attn_module, stored_list):
                orig = attn_module.forward
                def hooked_forward(query, key, value, **kwargs):
                    kwargs['need_weights'] = True
                    kwargs['average_attn_weights'] = True
                    out, weights = orig(query, key, value, **kwargs)
                    stored_list.append(weights.detach().cpu().numpy())
                    return out, weights
                return hooked_forward
            
            block.attn.forward = make_hook(block.attn, stored_attns)
        
        # Forward pass
        with torch.no_grad():
            model_s2(input_tensor)
        
        # Restore original forwards
        for block, orig_fn in zip(model_s2.vit.blocks, original_forwards):
            block.attn.forward = orig_fn
        
        # Compute rollout from stored attentions
        if stored_attns:
            result = None
            for attn in stored_attns:
                attn_matrix = attn[0]  # (N, N)
                identity = np.eye(attn_matrix.shape[0])
                attn_res = 0.5 * attn_matrix + 0.5 * identity
                row_sums = attn_res.sum(axis=-1, keepdims=True)
                attn_res = attn_res / (row_sums + 1e-8)
                result = attn_res if result is None else result @ attn_res
            
            rollout = result.mean(axis=0)
            rollout = (rollout - rollout.min()) / (rollout.max() - rollout.min() + 1e-8)
            
            N = len(rollout)
            grid = round(N ** (1/3))
            if grid ** 3 == N:
                rollout_3d = rollout.reshape(grid, grid, grid)
                scale = np.array(target_shape) / np.array(rollout_3d.shape)
                attention_map = zoom(rollout_3d, scale, order=3)
                attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    except Exception as e:
        print(f"[XAI] Attention rollout failed: {e}")
        attention_map = np.zeros(target_shape, dtype=np.float32)
    
    # ---- Grad-CAM ----
    gradcam_map = np.zeros(target_shape, dtype=np.float32)
    try:
        activations_store = {}
        gradients_store = {}
        
        # Hook on enc4's last normalization layer
        target_module = model_s2.enc4.conv.conv[4]  # InstanceNorm3d
        
        def fwd_hook(mod, inp, out):
            activations_store['value'] = out.detach()
        
        def bwd_hook(mod, grad_in, grad_out):
            gradients_store['value'] = grad_out[0].detach()
        
        h1 = target_module.register_forward_hook(fwd_hook)
        h2 = target_module.register_full_backward_hook(bwd_hook)
        
        # Forward + backward
        input_tensor.requires_grad_(True)
        model_s2.zero_grad()
        output = model_s2(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        
        # Target: tumor class
        target = output[:, 2].sum()  # Class 2 = tumor
        target.backward()
        
        h1.remove()
        h2.remove()
        
        if 'value' in activations_store and 'value' in gradients_store:
            grads = gradients_store['value']  # (B, C, D, H, W)
            acts = activations_store['value']
            
            weights = grads.mean(dim=(2, 3, 4), keepdim=True)
            cam = (weights * acts).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam.cpu().numpy()[0, 0]
            
            if cam.max() > 0:
                cam = cam / cam.max()
            
            # Upsample to target shape
            scale = np.array(target_shape) / np.array(cam.shape)
            gradcam_map = zoom(cam, scale, order=1)
            gradcam_map = np.clip(gradcam_map, 0, 1)
        
        input_tensor.requires_grad_(False)
        model_s2.zero_grad()
    
    except Exception as e:
        print(f"[XAI] Grad-CAM failed: {e}")
        gradcam_map = np.zeros(target_shape, dtype=np.float32)
    
    return attention_map.astype(np.float32), gradcam_map.astype(np.float32)
