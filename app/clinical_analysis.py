"""
CascadePanc-ViTUNet: Radiomics & Clinical Analysis Module
HU-based tissue characterization, T-staging, and pathology classification

Clinical Reference (Portal Venous Phase CT):
  Normal pancreas parenchyma:  100-150 HU
  PDAC (solid tumor):          40-80 HU (hypoattenuating)
  Cystic neoplasm:             0-20 HU (fluid density)
  Necrotic tissue:             10-30 HU
  Acute pancreatitis:          Heterogeneous, peripancreatic changes
  Edema:                       20-40 HU
  Calcification:               >150 HU
  Hemorrhage:                  50-70 HU (acute)
"""

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, label as scipy_label
from scipy.stats import skew, kurtosis


# ============================================================
# 1. RADIOMICS FEATURE EXTRACTION
# ============================================================

def extract_radiomics(raw_hu_volume, segmentation_mask, spacing=(1.5, 1.5, 2.5)):
    """
    Extract radiomics features from raw HU values within segmented regions.
    
    Args:
        raw_hu_volume: CT volume in original HU (clipped to window, NOT normalized)
        segmentation_mask: 0=bg, 1=pancreas, 2=tumor
        spacing: voxel spacing in mm (for volume calculation)
    
    Returns:
        dict with comprehensive radiomics features
    """
    features = {}
    voxel_vol_mm3 = np.prod(spacing)  # Volume of single voxel in mm³
    
    # ---- Pancreas Region Analysis ----
    panc_mask = (segmentation_mask == 1)
    panc_hu = raw_hu_volume[panc_mask] if panc_mask.sum() > 0 else np.array([])
    
    features['pancreas'] = {
        'voxel_count': int(panc_mask.sum()),
        'volume_cm3': float(panc_mask.sum() * voxel_vol_mm3 / 1000),
        'hu_mean': float(panc_hu.mean()) if len(panc_hu) > 0 else 0,
        'hu_std': float(panc_hu.std()) if len(panc_hu) > 0 else 0,
        'hu_median': float(np.median(panc_hu)) if len(panc_hu) > 0 else 0,
        'hu_min': float(panc_hu.min()) if len(panc_hu) > 0 else 0,
        'hu_max': float(panc_hu.max()) if len(panc_hu) > 0 else 0,
        'hu_skewness': float(skew(panc_hu)) if len(panc_hu) > 10 else 0,
        'hu_kurtosis': float(kurtosis(panc_hu)) if len(panc_hu) > 10 else 0,
        'hu_25pct': float(np.percentile(panc_hu, 25)) if len(panc_hu) > 0 else 0,
        'hu_75pct': float(np.percentile(panc_hu, 75)) if len(panc_hu) > 0 else 0,
    }
    
    # ---- Tumor Region Analysis ----
    tumor_mask = (segmentation_mask == 2)
    tumor_hu = raw_hu_volume[tumor_mask] if tumor_mask.sum() > 0 else np.array([])
    
    if tumor_mask.sum() > 0:
        # Tumor dimensions
        tumor_coords = np.argwhere(tumor_mask)
        tumor_min = tumor_coords.min(axis=0)
        tumor_max = tumor_coords.max(axis=0)
        tumor_extent_voxels = tumor_max - tumor_min + 1
        tumor_extent_mm = tumor_extent_voxels * np.array(spacing)
        tumor_center_mm = ((tumor_min + tumor_max) / 2) * np.array(spacing)
        max_dimension_mm = float(tumor_extent_mm.max())
        
        # Tumor boundary analysis
        tumor_interior = binary_erosion(tumor_mask, iterations=2)
        tumor_boundary = tumor_mask & ~tumor_interior
        boundary_hu = raw_hu_volume[tumor_boundary] if tumor_boundary.sum() > 0 else tumor_hu
        interior_hu = raw_hu_volume[tumor_interior] if tumor_interior.sum() > 0 else tumor_hu
        
        # Heterogeneity (coefficient of variation)
        heterogeneity = float(tumor_hu.std() / (abs(tumor_hu.mean()) + 1e-8))
        
        # Enhancement ratio vs normal pancreas
        if len(panc_hu) > 0 and panc_hu.mean() > 0:
            enhancement_ratio = float(tumor_hu.mean() / panc_hu.mean())
        else:
            enhancement_ratio = 1.0
        
        # Percentage of voxels in different HU ranges
        pct_very_low = float((tumor_hu < 20).sum() / len(tumor_hu) * 100)   # Cystic/fluid
        pct_low = float(((tumor_hu >= 20) & (tumor_hu < 60)).sum() / len(tumor_hu) * 100)  # Necrotic/edema
        pct_medium = float(((tumor_hu >= 60) & (tumor_hu < 100)).sum() / len(tumor_hu) * 100)  # Solid tumor
        pct_high = float(((tumor_hu >= 100) & (tumor_hu < 150)).sum() / len(tumor_hu) * 100)  # Normal/inflamed
        pct_very_high = float((tumor_hu >= 150).sum() / len(tumor_hu) * 100)  # Calcification
        
        features['tumor'] = {
            'voxel_count': int(tumor_mask.sum()),
            'volume_cm3': float(tumor_mask.sum() * voxel_vol_mm3 / 1000),
            'extent_mm': tumor_extent_mm.tolist(),
            'max_dimension_mm': max_dimension_mm,
            'center_mm': tumor_center_mm.tolist(),
            'hu_mean': float(tumor_hu.mean()),
            'hu_std': float(tumor_hu.std()),
            'hu_median': float(np.median(tumor_hu)),
            'hu_min': float(tumor_hu.min()),
            'hu_max': float(tumor_hu.max()),
            'hu_skewness': float(skew(tumor_hu)) if len(tumor_hu) > 10 else 0,
            'hu_kurtosis': float(kurtosis(tumor_hu)) if len(tumor_hu) > 10 else 0,
            'hu_25pct': float(np.percentile(tumor_hu, 25)),
            'hu_75pct': float(np.percentile(tumor_hu, 75)),
            'heterogeneity': heterogeneity,
            'enhancement_ratio': enhancement_ratio,
            'boundary_hu_mean': float(boundary_hu.mean()),
            'interior_hu_mean': float(interior_hu.mean()),
            'pct_very_low_hu': pct_very_low,
            'pct_low_hu': pct_low,
            'pct_medium_hu': pct_medium,
            'pct_high_hu': pct_high,
            'pct_very_high_hu': pct_very_high,
        }
    else:
        features['tumor'] = {
            'voxel_count': 0,
            'volume_cm3': 0,
            'extent_mm': [0, 0, 0],
            'max_dimension_mm': 0,
        }
    
    # ---- Peripancreatic Fat Analysis (for pancreatitis signs) ----
    # Dilate pancreas mask to capture surrounding tissue
    if panc_mask.sum() > 0:
        dilated = binary_dilation(panc_mask | tumor_mask, iterations=5)
        peripancreatic = dilated & ~(panc_mask | tumor_mask)
        peri_hu = raw_hu_volume[peripancreatic] if peripancreatic.sum() > 0 else np.array([])
        
        # Fat stranding indicator: normal fat is -100 to -50 HU
        # Inflamed fat has higher HU (fat stranding)
        fat_voxels = peri_hu[(peri_hu >= -120) & (peri_hu <= 0)]
        
        features['peripancreatic'] = {
            'voxel_count': int(peripancreatic.sum()),
            'hu_mean': float(peri_hu.mean()) if len(peri_hu) > 0 else 0,
            'hu_std': float(peri_hu.std()) if len(peri_hu) > 0 else 0,
            'fat_hu_mean': float(fat_voxels.mean()) if len(fat_voxels) > 0 else -80,
            'fat_stranding_score': float(
                np.clip((fat_voxels.mean() + 80) / 40, 0, 1)  # Normalized 0-1
            ) if len(fat_voxels) > 10 else 0,
        }
    else:
        features['peripancreatic'] = {
            'voxel_count': 0,
            'hu_mean': 0,
            'fat_stranding_score': 0,
        }
    
    return features


# ============================================================
# 2. PATHOLOGY CLASSIFICATION (HU-Based Rules)
# ============================================================

def classify_pathology(features):
    """
    Classify the detected lesion based on HU radiomics features.
    Uses clinical HU thresholds from radiology literature.
    
    Returns:
        dict with classification, confidence, evidence, and recommendations
    """
    result = {
        'primary_diagnosis': 'Normal',
        'confidence': 0.0,
        'differential': [],
        'evidence': [],
        'risk_level': 'LOW',
        'recommendations': [],
    }
    
    tumor = features.get('tumor', {})
    pancreas = features.get('pancreas', {})
    peri = features.get('peripancreatic', {})
    
    tumor_voxels = tumor.get('voxel_count', 0)
    
    if tumor_voxels == 0:
        result['primary_diagnosis'] = 'No Lesion Detected'
        result['confidence'] = 0.85
        result['risk_level'] = 'LOW'
        result['evidence'].append('No abnormal region identified by segmentation model')
        result['recommendations'].append('Routine follow-up as clinically indicated')
        return result
    
    # ---- Extract key HU features ----
    tumor_hu_mean = tumor.get('hu_mean', 0)
    tumor_hu_std = tumor.get('hu_std', 0)
    tumor_hu_median = tumor.get('hu_median', 0)
    panc_hu_mean = pancreas.get('hu_mean', 0)
    heterogeneity = tumor.get('heterogeneity', 0)
    enhancement_ratio = tumor.get('enhancement_ratio', 1.0)
    pct_very_low = tumor.get('pct_very_low_hu', 0)
    pct_low = tumor.get('pct_low_hu', 0)
    pct_medium = tumor.get('pct_medium_hu', 0)
    pct_high = tumor.get('pct_high_hu', 0)
    pct_very_high = tumor.get('pct_very_high_hu', 0)
    fat_stranding = peri.get('fat_stranding_score', 0)
    max_dim = tumor.get('max_dimension_mm', 0)
    volume = tumor.get('volume_cm3', 0)
    boundary_hu = tumor.get('boundary_hu_mean', 0)
    interior_hu = tumor.get('interior_hu_mean', 0)
    
    # ---- Scoring system for each pathology ----
    scores = {
        'solid_tumor_pdac': 0,
        'cystic_neoplasm': 0,
        'pancreatitis_acute': 0,
        'pancreatitis_chronic': 0,
        'edema': 0,
        'necrosis': 0,
    }
    evidence_map = {k: [] for k in scores}
    
    # ===== SOLID TUMOR (PDAC) =====
    # PDAC is hypoattenuating: 40-80 HU (lower than normal pancreas 100-150 HU)
    if 30 <= tumor_hu_mean <= 90:
        scores['solid_tumor_pdac'] += 3
        evidence_map['solid_tumor_pdac'].append(
            f'Mean HU {tumor_hu_mean:.0f} is hypoattenuating (typical PDAC: 40-80 HU)')
    
    if enhancement_ratio < 0.8:
        scores['solid_tumor_pdac'] += 2
        evidence_map['solid_tumor_pdac'].append(
            f'Enhancement ratio {enhancement_ratio:.2f} indicates hypoenhancement vs normal pancreas')
    
    if pct_medium > 40:
        scores['solid_tumor_pdac'] += 2
        evidence_map['solid_tumor_pdac'].append(
            f'{pct_medium:.0f}% of voxels in solid tumor HU range (60-100)')
    
    if 0.1 < heterogeneity < 0.5:
        scores['solid_tumor_pdac'] += 1
        evidence_map['solid_tumor_pdac'].append(
            f'Moderate heterogeneity ({heterogeneity:.2f}) consistent with solid mass')
    
    if max_dim >= 10:  # At least 1cm
        scores['solid_tumor_pdac'] += 1
        evidence_map['solid_tumor_pdac'].append(
            f'Lesion size {max_dim:.1f}mm consistent with pancreatic mass')
    
    # ===== CYSTIC NEOPLASM =====
    # Very low HU near water density (0-20 HU)
    if pct_very_low > 50:
        scores['cystic_neoplasm'] += 4
        evidence_map['cystic_neoplasm'].append(
            f'{pct_very_low:.0f}% of voxels near water density (<20 HU)')
    
    if tumor_hu_mean < 30:
        scores['cystic_neoplasm'] += 3
        evidence_map['cystic_neoplasm'].append(
            f'Mean HU {tumor_hu_mean:.0f} suggests cystic/fluid content')
    
    if tumor_hu_std < 15:
        scores['cystic_neoplasm'] += 1
        evidence_map['cystic_neoplasm'].append(
            f'Low HU variance ({tumor_hu_std:.0f}) suggests homogeneous fluid')
    
    if interior_hu < boundary_hu - 20:
        scores['cystic_neoplasm'] += 1
        evidence_map['cystic_neoplasm'].append(
            f'Interior HU ({interior_hu:.0f}) lower than boundary ({boundary_hu:.0f}) — cystic pattern')
    
    # ===== ACUTE PANCREATITIS =====
    # Heterogeneous enhancement, peripancreatic fat stranding, diffuse changes
    if fat_stranding > 0.5:
        scores['pancreatitis_acute'] += 3
        evidence_map['pancreatitis_acute'].append(
            f'Peripancreatic fat stranding score {fat_stranding:.2f} (elevated)')
    
    if heterogeneity > 0.5:
        scores['pancreatitis_acute'] += 2
        evidence_map['pancreatitis_acute'].append(
            f'High tissue heterogeneity ({heterogeneity:.2f}) suggests inflammatory changes')
    
    if 80 <= tumor_hu_mean <= 130 and tumor_hu_std > 25:
        scores['pancreatitis_acute'] += 2
        evidence_map['pancreatitis_acute'].append(
            f'HU profile (mean={tumor_hu_mean:.0f}, std={tumor_hu_std:.0f}) consistent with inflamed tissue')
    
    if pancreas.get('hu_std', 0) > 30:
        scores['pancreatitis_acute'] += 1
        evidence_map['pancreatitis_acute'].append(
            f'Pancreas parenchyma heterogeneity (std={pancreas.get("hu_std",0):.0f}) suggests diffuse inflammation')
    
    # ===== CHRONIC PANCREATITIS =====
    # Calcifications, ductal dilation, atrophy
    if pct_very_high > 10:
        scores['pancreatitis_chronic'] += 3
        evidence_map['pancreatitis_chronic'].append(
            f'{pct_very_high:.0f}% of voxels show calcification (>150 HU)')
    
    if tumor_hu_max > 200:
        scores['pancreatitis_chronic'] += 2
        evidence_map['pancreatitis_chronic'].append(
            f'High-density foci (max HU={tumor_hu_max:.0f}) suggest calcifications')
    
    if heterogeneity > 0.6 and pct_very_high > 5:
        scores['pancreatitis_chronic'] += 1
        evidence_map['pancreatitis_chronic'].append(
            'Mixed calcified and non-calcified tissue pattern')
    
    # ===== EDEMA =====
    # Low HU (20-40), homogeneous
    if 15 <= tumor_hu_mean <= 50 and tumor_hu_std < 20:
        scores['edema'] += 3
        evidence_map['edema'].append(
            f'HU profile (mean={tumor_hu_mean:.0f}, std={tumor_hu_std:.0f}) consistent with edematous tissue')
    
    if pct_low > 50:
        scores['edema'] += 2
        evidence_map['edema'].append(
            f'{pct_low:.0f}% of voxels in edema range (20-60 HU)')
    
    # ===== NECROSIS =====
    # Very low HU, often within pancreatitis
    if tumor_hu_mean < 25 and fat_stranding > 0.3:
        scores['necrosis'] += 3
        evidence_map['necrosis'].append(
            f'Low density (mean HU={tumor_hu_mean:.0f}) with peripancreatic changes suggests necrosis')
    
    if pct_very_low > 30 and pct_low > 20:
        scores['necrosis'] += 2
        evidence_map['necrosis'].append(
            f'Large portion of very low density tissue ({pct_very_low:.0f}% <20 HU)')
    
    # ---- Determine primary diagnosis ----
    max_score = max(scores.values())
    if max_score == 0:
        result['primary_diagnosis'] = 'Indeterminate Lesion'
        result['confidence'] = 0.3
        result['risk_level'] = 'MODERATE'
        result['evidence'].append('HU characteristics do not clearly match known patterns')
        result['recommendations'].append('Further imaging (MRI with MRCP) recommended')
        result['recommendations'].append('Consider endoscopic ultrasound (EUS)')
        return result
    
    # Get top diagnosis and differential
    sorted_diagnoses = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    diagnosis_labels = {
        'solid_tumor_pdac': 'Pancreatic Ductal Adenocarcinoma (PDAC)',
        'cystic_neoplasm': 'Cystic Neoplasm',
        'pancreatitis_acute': 'Acute Pancreatitis',
        'pancreatitis_chronic': 'Chronic Pancreatitis',
        'edema': 'Pancreatic Edema',
        'necrosis': 'Pancreatic Necrosis',
    }
    
    primary_key = sorted_diagnoses[0][0]
    primary_score = sorted_diagnoses[0][1]
    
    result['primary_diagnosis'] = diagnosis_labels[primary_key]
    result['confidence'] = min(0.95, primary_score / 10.0 + 0.3)
    result['evidence'] = evidence_map[primary_key]
    
    # Differential diagnoses (other possibilities)
    for key, score in sorted_diagnoses[1:]:
        if score > 0:
            diff_confidence = min(0.9, score / 10.0 + 0.2)
            result['differential'].append({
                'diagnosis': diagnosis_labels[key],
                'confidence': round(diff_confidence, 2),
                'evidence': evidence_map[key],
            })
    
    # ---- Risk level and recommendations based on diagnosis ----
    if primary_key == 'solid_tumor_pdac':
        result['risk_level'] = 'HIGH'
        result['recommendations'] = [
            'Urgent referral to hepatobiliary surgery / pancreatic oncology',
            'Complete staging workup (chest CT, liver MRI, CA 19-9 serology)',
            'Multidisciplinary tumor board review recommended',
            'Consider endoscopic ultrasound with FNA for tissue diagnosis',
            'Assess vascular involvement for resectability evaluation',
        ]
    elif primary_key == 'cystic_neoplasm':
        result['risk_level'] = 'MODERATE'
        result['recommendations'] = [
            'MRI with MRCP for cyst characterization',
            'Measure cyst fluid CEA and amylase if EUS-FNA performed',
            'Follow ACG/Fukuoka guidelines for cyst management',
            'Surveillance interval based on cyst size and features',
        ]
    elif primary_key == 'pancreatitis_acute':
        result['risk_level'] = 'HIGH'
        result['recommendations'] = [
            'Assess severity using Revised Atlanta Classification',
            'Check for organ failure (Modified Marshall Score)',
            'IV fluid resuscitation and pain management',
            'Follow-up CT at 72-96 hours if clinical deterioration',
            'Evaluate for gallstone etiology (RUQ ultrasound)',
        ]
    elif primary_key == 'pancreatitis_chronic':
        result['risk_level'] = 'MODERATE'
        result['recommendations'] = [
            'Assess exocrine function (fecal elastase)',
            'Screen for diabetes (HbA1c)',
            'Pain management protocol',
            'Consider MRCP for ductal evaluation',
            'Note: Chronic pancreatitis increases cancer risk — surveillance recommended',
        ]
    elif primary_key == 'edema':
        result['risk_level'] = 'MODERATE'
        result['recommendations'] = [
            'Correlate with clinical presentation',
            'Evaluate for underlying cause (pancreatitis, obstruction)',
            'Follow-up imaging in 4-6 weeks to assess resolution',
        ]
    elif primary_key == 'necrosis':
        result['risk_level'] = 'HIGH'
        result['recommendations'] = [
            'Urgent gastroenterology consultation',
            'CT severity index (CTSI) scoring',
            'Monitor for infected necrosis (gas bubbles, clinical sepsis)',
            'Consider drainage if walled-off necrosis develops',
        ]
    
    return result


# ============================================================
# 3. T-STAGING (Tumor Size-Based)
# ============================================================

def compute_t_stage(features):
    """
    Compute approximate T-stage from tumor measurements.
    Based on AJCC 8th Edition TNM staging for pancreatic cancer.
    
    T1: ≤2cm maximum dimension
      T1a: ≤0.5cm
      T1b: >0.5cm and ≤1cm
      T1c: >1cm and ≤2cm
    T2: >2cm and ≤4cm
    T3: >4cm
    T4: Involves celiac axis/SMA (cannot determine from segmentation)
    
    N and M staging require additional data (lymph nodes, distant metastases).
    """
    tumor = features.get('tumor', {})
    
    if tumor.get('voxel_count', 0) == 0:
        return {
            'stage': 'N/A',
            'substage': 'N/A',
            'description': 'No tumor detected',
            'max_dimension_mm': 0,
            'volume_cm3': 0,
            'tnm_string': 'N/A',
            'notes': [],
        }
    
    max_dim = tumor.get('max_dimension_mm', 0)
    volume = tumor.get('volume_cm3', 0)
    extent = tumor.get('extent_mm', [0, 0, 0])
    
    result = {
        'max_dimension_mm': round(max_dim, 1),
        'volume_cm3': round(volume, 2),
        'extent_mm': [round(e, 1) for e in extent],
        'notes': [],
    }
    
    # T classification
    if max_dim <= 5:
        result['stage'] = 'T1'
        result['substage'] = 'T1a'
        result['description'] = f'Tumor ≤0.5cm ({max_dim:.1f}mm) — confined to pancreas'
    elif max_dim <= 10:
        result['stage'] = 'T1'
        result['substage'] = 'T1b'
        result['description'] = f'Tumor 0.5-1cm ({max_dim:.1f}mm) — confined to pancreas'
    elif max_dim <= 20:
        result['stage'] = 'T1'
        result['substage'] = 'T1c'
        result['description'] = f'Tumor 1-2cm ({max_dim:.1f}mm) — confined to pancreas'
    elif max_dim <= 40:
        result['stage'] = 'T2'
        result['substage'] = 'T2'
        result['description'] = f'Tumor 2-4cm ({max_dim:.1f}mm) — confined to pancreas'
    else:
        result['stage'] = 'T3'
        result['substage'] = 'T3'
        result['description'] = f'Tumor >4cm ({max_dim:.1f}mm) — may extend beyond pancreas'
    
    # TNM string (N and M unknown from imaging alone)
    result['tnm_string'] = f'{result["substage"]}NxMx'
    
    # Clinical notes
    result['notes'].append(
        f'Maximum tumor dimension: {max_dim:.1f}mm '
        f'({extent[0]:.1f} × {extent[1]:.1f} × {extent[2]:.1f} mm)')
    result['notes'].append(f'Tumor volume: {volume:.2f} cm³')
    
    if max_dim <= 20:
        result['notes'].append(
            'T1 tumors have the best prognosis with 5-year survival ~35% if resected')
    elif max_dim <= 40:
        result['notes'].append(
            'T2 tumors are potentially resectable with 5-year survival ~20% if resected')
    else:
        result['notes'].append(
            'T3 tumors require careful assessment of vascular involvement for resectability')
    
    result['notes'].append(
        'N-staging (lymph node involvement) requires separate analysis '
        'and is not determinable from this segmentation')
    result['notes'].append(
        'M-staging (distant metastases) requires full-body imaging '
        '(chest CT, liver MRI) and is not assessed here')
    
    # Resectability assessment (simplified)
    if max_dim <= 40:
        result['resectability'] = 'Potentially Resectable'
        result['resectability_detail'] = (
            'Based on size criteria only. Vascular involvement assessment '
            '(celiac axis, SMA, portal vein) is required for definitive '
            'resectability determination.')
    else:
        result['resectability'] = 'Borderline / Locally Advanced'
        result['resectability_detail'] = (
            'Large tumor size (>4cm) suggests possible borderline resectable '
            'or locally advanced disease. Vascular involvement assessment required. '
            'Consider neoadjuvant chemotherapy evaluation.')
    
    return result


# ============================================================
# 4. PANCREATITIS SEVERITY SCORING (if pancreatitis detected)
# ============================================================

def compute_pancreatitis_severity(features, classification):
    """
    Compute pancreatitis severity based on CT findings.
    Uses Modified CT Severity Index (CTSI) adapted for automated analysis.
    
    Original CTSI (Mortele 2004):
      Pancreatic inflammation:
        Normal (0), Intrinsic abnormalities only (2), 
        Peripancreatic changes (4)
      Pancreatic necrosis:
        None (0), <30% (2), >30% (4)
      Extrapancreatic: 
        Pleural effusion, ascites, vascular complications (2 each)
    
    We can estimate inflammation and necrosis from HU analysis.
    """
    result = {
        'applicable': False,
        'severity': 'N/A',
        'score': 0,
        'max_score': 10,
        'components': {},
        'revised_atlanta': 'N/A',
        'details': [],
    }
    
    # Only applicable if pancreatitis was detected
    primary = classification.get('primary_diagnosis', '')
    if 'Pancreatitis' not in primary and 'Necrosis' not in primary:
        return result
    
    result['applicable'] = True
    
    tumor = features.get('tumor', {})
    pancreas = features.get('pancreas', {})
    peri = features.get('peripancreatic', {})
    
    total_score = 0
    
    # Component 1: Pancreatic inflammation (0-4 points)
    panc_hu_std = pancreas.get('hu_std', 0)
    fat_stranding = peri.get('fat_stranding_score', 0)
    
    if panc_hu_std < 15 and fat_stranding < 0.2:
        inflammation_score = 0
        inflammation_desc = 'Normal pancreas'
    elif panc_hu_std < 25 and fat_stranding < 0.4:
        inflammation_score = 2
        inflammation_desc = 'Intrinsic pancreatic abnormalities (focal/diffuse enlargement)'
    else:
        inflammation_score = 4
        inflammation_desc = 'Peripancreatic inflammatory changes with fat stranding'
    
    total_score += inflammation_score
    result['components']['inflammation'] = {
        'score': inflammation_score,
        'max': 4,
        'description': inflammation_desc,
    }
    
    # Component 2: Necrosis estimation (0-4 points)
    tumor_voxels = tumor.get('voxel_count', 0)
    panc_voxels = pancreas.get('voxel_count', 0)
    total_panc = tumor_voxels + panc_voxels
    
    # Estimate necrosis from very low HU regions
    pct_very_low = tumor.get('pct_very_low_hu', 0)
    pct_low = tumor.get('pct_low_hu', 0)
    necrosis_pct = pct_very_low + pct_low * 0.5  # Weighted estimate
    
    if necrosis_pct < 10:
        necrosis_score = 0
        necrosis_desc = 'No pancreatic necrosis detected'
    elif necrosis_pct < 30:
        necrosis_score = 2
        necrosis_desc = f'Estimated <30% necrosis ({necrosis_pct:.0f}% low-density voxels)'
    else:
        necrosis_score = 4
        necrosis_desc = f'Estimated >30% necrosis ({necrosis_pct:.0f}% low-density voxels)'
    
    total_score += necrosis_score
    result['components']['necrosis'] = {
        'score': necrosis_score,
        'max': 4,
        'description': necrosis_desc,
    }
    
    # Component 3: Extrapancreatic complications (0-2 points)
    # Limited assessment from CT — check for peripancreatic fluid
    peri_hu_mean = peri.get('hu_mean', 0)
    if peri_hu_mean < 10 and peri.get('voxel_count', 0) > 500:
        extra_score = 2
        extra_desc = 'Peripancreatic fluid collection detected'
    else:
        extra_score = 0
        extra_desc = 'No significant extrapancreatic complications detected on CT'
    
    total_score += extra_score
    result['components']['extrapancreatic'] = {
        'score': extra_score,
        'max': 2,
        'description': extra_desc,
    }
    
    result['score'] = total_score
    
    # Severity classification based on Modified CTSI
    if total_score <= 2:
        result['severity'] = 'MILD'
        result['revised_atlanta'] = 'Mild Acute Pancreatitis'
        result['details'] = [
            'No organ failure, no local or systemic complications',
            'Expected recovery within 1 week',
            'Supportive care: IV fluids, NPO/clear liquid diet, pain management',
        ]
    elif total_score <= 6:
        result['severity'] = 'MODERATE'
        result['revised_atlanta'] = 'Moderately Severe Acute Pancreatitis'
        result['details'] = [
            'Possible transient organ failure (<48 hours)',
            'Local complications may include peripancreatic fluid collections',
            'Requires close monitoring in hospital setting',
            'Consider repeat CT at 72-96 hours',
        ]
    else:
        result['severity'] = 'SEVERE'
        result['revised_atlanta'] = 'Severe Acute Pancreatitis'
        result['details'] = [
            'Persistent organ failure likely (>48 hours)',
            'Significant pancreatic necrosis detected',
            'ICU admission recommended',
            'Monitor for infected necrosis (may require drainage)',
            'Multidisciplinary management (GI, surgery, critical care)',
        ]
    
    return result


# ============================================================
# 5. GENERATE COMPLETE CLINICAL REPORT
# ============================================================

def generate_clinical_report(features, classification, t_stage, severity):
    """
    Generate a comprehensive clinical report combining all analyses.
    
    Returns structured dict for the dashboard.
    """
    report = {
        'summary': '',
        'risk_level': classification.get('risk_level', 'LOW'),
        'primary_finding': classification.get('primary_diagnosis', 'Normal'),
        'confidence': round(classification.get('confidence', 0), 2),
        
        'tumor_detected': features['tumor'].get('voxel_count', 0) > 0,
        
        # Radiomics
        'radiomics': {
            'pancreas_volume_cm3': features['pancreas'].get('volume_cm3', 0),
            'pancreas_hu_mean': features['pancreas'].get('hu_mean', 0),
            'pancreas_hu_std': features['pancreas'].get('hu_std', 0),
        },
        
        # Classification
        'classification': classification,
        
        # T-staging (if tumor)
        't_staging': t_stage,
        
        # Pancreatitis severity (if applicable)
        'pancreatitis_severity': severity,
        
        # Evidence trail
        'evidence': classification.get('evidence', []),
        'differential': classification.get('differential', []),
        'recommendations': classification.get('recommendations', []),
    }
    
    # Generate summary text
    primary = classification.get('primary_diagnosis', 'Normal')
    
    if 'PDAC' in primary or 'Adenocarcinoma' in primary:
        stage_str = t_stage.get('substage', 'Unknown')
        dim_str = f"{t_stage.get('max_dimension_mm', 0):.1f}mm"
        vol_str = f"{features['tumor'].get('volume_cm3', 0):.2f} cm³"
        resect = t_stage.get('resectability', 'Unknown')
        
        report['summary'] = (
            f"FINDINGS: A hypoattenuating mass consistent with pancreatic ductal "
            f"adenocarcinoma (PDAC) is identified. The lesion measures {dim_str} in "
            f"maximum dimension (volume: {vol_str}), corresponding to {stage_str} "
            f"staging. Resectability assessment: {resect}. "
            f"Confidence: {classification['confidence']*100:.0f}%."
        )
    
    elif 'Cystic' in primary:
        dim_str = f"{t_stage.get('max_dimension_mm', 0):.1f}mm"
        report['summary'] = (
            f"FINDINGS: A cystic lesion is identified in the pancreas, measuring "
            f"{dim_str}. The HU characteristics suggest a cystic neoplasm. "
            f"Further characterization with MRI/MRCP is recommended. "
            f"Confidence: {classification['confidence']*100:.0f}%."
        )
    
    elif 'Pancreatitis' in primary:
        sev = severity.get('severity', 'Unknown')
        score = severity.get('score', 0)
        report['summary'] = (
            f"FINDINGS: CT findings are consistent with {primary.lower()}. "
            f"Modified CT Severity Index: {score}/10 ({sev}). "
            f"{severity.get('revised_atlanta', '')}. "
            f"Confidence: {classification['confidence']*100:.0f}%."
        )
    
    elif 'Edema' in primary:
        report['summary'] = (
            f"FINDINGS: Low-density changes in the pancreatic region suggest edema. "
            f"Clinical correlation recommended to determine underlying etiology. "
            f"Confidence: {classification['confidence']*100:.0f}%."
        )
    
    elif 'Necrosis' in primary:
        report['summary'] = (
            f"FINDINGS: Areas of very low density within the pancreatic region are "
            f"consistent with necrosis. This is a serious finding requiring urgent "
            f"clinical evaluation. "
            f"Confidence: {classification['confidence']*100:.0f}%."
        )
    
    else:
        report['summary'] = (
            f"FINDINGS: {primary}. "
            f"Confidence: {classification['confidence']*100:.0f}%."
        )
    
    return report
