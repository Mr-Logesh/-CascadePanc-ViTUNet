/* ============================================================
   CascadePanc-ViTUNet — Clinical Dashboard Logic
   Two-step flow: Upload → Preview → Analyze → Results
   ============================================================ */

let uploadData = null;
let resultData = null;
let currentView = 'segmentation';
let resultSliceIdx = 0;
let xaiSlice = 0, totalXaiSlices = 0, xaiSliceImages = [];

// ============================================================
// Live Header Clock
// ============================================================
function updateClock() {
    const now = new Date();
    const h = now.getHours().toString().padStart(2, '0');
    const m = now.getMinutes().toString().padStart(2, '0');
    const s = now.getSeconds().toString().padStart(2, '0');
    const clockEl = document.getElementById('header-clock');
    if (clockEl) clockEl.textContent = `${h}:${m}:${s}`;
}
updateClock();
setInterval(updateClock, 1000);

// ============================================================
// File Upload
// ============================================================
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const fileInfoBar = document.getElementById('file-info-bar');

uploadZone.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    fileInput.click();
});

uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', e => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

document.getElementById('remove-file-btn').addEventListener('click', () => {
    fileInput.value = '';
    fileInfoBar.classList.remove('visible');
    document.getElementById('ct-preview-section').classList.remove('visible');
    document.getElementById('upload-section').classList.remove('has-file');
    uploadData = null;
});

function handleFile(file) {
    const name = file.name.toLowerCase();
    if (!name.endsWith('.nii') && !name.endsWith('.nii.gz') && !name.endsWith('.gz') && !name.endsWith('.dcm') && !name.endsWith('.zip')) {
        alert('⚠️ Supported formats: .nii, .nii.gz, .dcm, .zip\n\nFor a folder of DICOM slices, zip the folder and upload the .zip file.');
        return;
    }

    // Show file info
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-size').textContent = `(${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
    fileInfoBar.classList.add('visible');
    document.getElementById('upload-section').classList.add('has-file');

    // Upload to server for preview
    uploadZone.innerHTML = `
        <div class="upload-icon-wrap">
            <div class="upload-loading-spinner"></div>
        </div>
        <div class="upload-title">Loading CT Volume...</div>
        <div class="upload-desc">Reading file and generating preview slices</div>
    `;

    const form = new FormData();
    form.append('file', file);

    fetch('/upload', { method: 'POST', body: form })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                resetUploadZone();
                alert('Error: ' + data.error + (data.message ? '\n\n' + data.message : ''));
                return;
            }
            uploadData = data;
            showCTPreview(data);
            // Volume validation warnings
            showVolumeWarnings(data);
        })
        .catch(err => {
            resetUploadZone();
            alert('Upload failed: ' + err);
        });
}

function resetUploadZone() {
    uploadZone.innerHTML = `
        <div class="upload-icon-wrap">
            <svg class="upload-icon-svg" width="48" height="48" viewBox="0 0 24 24" fill="none"
                stroke="currentColor" stroke-width="1.5">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
        </div>
        <div class="upload-title">Upload CT Scan</div>
        <div class="upload-desc">Drag & drop a contrast-enhanced abdominal CT scan or click to browse</div>
        <div class="upload-formats">
            <span class="format-badge">.nii</span>
            <span class="format-badge">.nii.gz</span>
            <span class="format-badge format-badge-dicom">.dcm</span>
            <span class="format-badge format-badge-dicom">.zip (DICOM folder)</span>
            <span class="format-badge-info">Max 500MB</span>
        </div>
        <div class="upload-note">⚠️ 3D NIfTI or DICOM volumes required — For DICOM folders with many slices, zip the folder and upload the .zip</div>
    `;
}

// ============================================================
// Reset to New Scan (without page reload)
// ============================================================
function resetToNewScan() {
    // Clear all state
    uploadData = null;
    resultData = null;
    currentView = 'segmentation';
    resultSliceIdx = 0;
    xaiSlice = 0;
    totalXaiSlices = 0;
    xaiSliceImages = [];

    // Clear file input
    fileInput.value = '';
    fileInfoBar.classList.remove('visible');
    document.getElementById('upload-section').classList.remove('has-file');

    // Hide all sections
    document.getElementById('ct-preview-section').classList.remove('visible');
    document.getElementById('progress-section').classList.remove('visible');
    document.getElementById('results-section').classList.remove('visible');
    document.getElementById('completion-bar').classList.remove('visible');

    // Hide sub-panels in results (reset their display)
    ['mip-panel', 'hu-panel', 'clinical-panel', 'staging-panel',
        'severity-panel', 'xai-panel', 'timing-panel'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.display = 'none';
        });

    // Hide header new-scan button
    const headerBtn = document.getElementById('header-new-scan-btn');
    if (headerBtn) headerBtn.classList.remove('visible');

    // Reset risk banner
    const riskBanner = document.getElementById('risk-banner');
    if (riskBanner) riskBanner.className = 'risk-banner';

    // Reset differential and recommendations visibility
    const diffSection = document.getElementById('differential-section');
    if (diffSection) diffSection.style.display = 'none';
    const recSection = document.getElementById('recommendations-section');
    if (recSection) recSection.style.display = 'none';

    // Restore upload zone
    uploadZone.style.display = '';
    resetUploadZone();

    // Re-enable analyze button
    const btn = document.getElementById('analyze-btn');
    btn.disabled = false;
    btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>
    </svg> Run Cascade Analysis`;

    // Reset progress bar
    const progressFill = document.getElementById('progress-fill');
    if (progressFill) progressFill.style.width = '0%';
    document.querySelectorAll('.pstage').forEach(el => {
        el.classList.remove('active', 'complete');
    });

    // Clear new UI elements
    const errorCard = document.getElementById('error-card');
    if (errorCard) errorCard.classList.remove('visible');
    const volWarn = document.getElementById('volume-warning');
    if (volWarn) volWarn.classList.remove('visible');
    const patBar = document.getElementById('patient-results-bar');
    if (patBar) { patBar.innerHTML = ''; patBar.classList.remove('visible'); }
    const etaEl = document.getElementById('progress-eta');
    if (etaEl) etaEl.textContent = '';

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Wire up all New Scan buttons
const newScanBtn = document.getElementById('new-scan-btn');
if (newScanBtn) newScanBtn.addEventListener('click', resetToNewScan);
const headerNewScanBtn = document.getElementById('header-new-scan-btn');
if (headerNewScanBtn) headerNewScanBtn.addEventListener('click', resetToNewScan);

// ============================================================
// CT Volume Full Preview
// ============================================================
function showCTPreview(data) {
    // Hide upload zone, show preview
    uploadZone.style.display = 'none';
    const previewSection = document.getElementById('ct-preview-section');
    previewSection.classList.add('visible');

    document.getElementById('ct-filename').textContent =
        `${data.filename} (${data.file_type || 'NIfTI'})`;

    // Volume info grid
    const infoGrid = document.getElementById('ct-info-grid');
    infoGrid.innerHTML = `
        <div class="ct-info-item">
            <div class="ct-info-label">Dimensions</div>
            <div class="ct-info-value">${data.shape.join(' × ')}</div>
        </div>
        <div class="ct-info-item">
            <div class="ct-info-label">Slices</div>
            <div class="ct-info-value">${data.num_slices}</div>
        </div>
        <div class="ct-info-item">
            <div class="ct-info-label">Spacing (mm)</div>
            <div class="ct-info-value">${data.spacing.map(s => s.toFixed(1)).join(' × ')}</div>
        </div>
        <div class="ct-info-item">
            <div class="ct-info-label">HU Range</div>
            <div class="ct-info-value">${Math.round(data.hu_range[0])} to ${Math.round(data.hu_range[1])}</div>
        </div>
    `;

    // MIP Projections Row
    document.getElementById('preview-mip-row').innerHTML = `
        <div class="preview-mip-item">
            <img src="data:image/png;base64,${data.mip_axial}" alt="Axial MIP">
            <div class="preview-mip-label">Axial (Top-Down)</div>
        </div>
        <div class="preview-mip-item">
            <img src="data:image/png;base64,${data.mip_coronal}" alt="Coronal MIP">
            <div class="preview-mip-label">Coronal (Front)</div>
        </div>
        <div class="preview-mip-item">
            <img src="data:image/png;base64,${data.mip_sagittal}" alt="Sagittal MIP">
            <div class="preview-mip-label">Sagittal (Side)</div>
        </div>
    `;

    // Center Orthogonal Slices
    document.getElementById('preview-ortho-row').innerHTML = `
        <div class="preview-mip-item">
            <img src="data:image/png;base64,${data.center_axial}" alt="Center Axial">
            <div class="preview-mip-label">Axial (Slice ${Math.floor(data.shape[0] / 2)})</div>
        </div>
        <div class="preview-mip-item">
            <img src="data:image/png;base64,${data.center_coronal}" alt="Center Coronal">
            <div class="preview-mip-label">Coronal (Slice ${Math.floor(data.shape[1] / 2)})</div>
        </div>
        <div class="preview-mip-item">
            <img src="data:image/png;base64,${data.center_sagittal}" alt="Center Sagittal">
            <div class="preview-mip-label">Sagittal (Slice ${Math.floor(data.shape[2] / 2)})</div>
        </div>
    `;

    // Montage Grid
    const montageGrid = document.getElementById('preview-montage-grid');
    montageGrid.innerHTML = '';
    data.montage.forEach((b64, i) => {
        const idx = data.montage_indices[i];
        montageGrid.innerHTML += `
            <div class="montage-item">
                <img src="data:image/png;base64,${b64}" alt="Slice ${idx}">
                <div class="montage-label">S${idx}</div>
            </div>
        `;
    });
}

// ============================================================
// Analysis (with status polling)
// ============================================================
document.getElementById('analyze-btn').addEventListener('click', startAnalysis);

function startAnalysis() {
    if (!uploadData) return;

    const btn = document.getElementById('analyze-btn');
    btn.disabled = true;
    btn.innerHTML = '<div class="btn-spinner"></div> Starting Analysis...';

    document.getElementById('progress-section').classList.add('visible');
    document.getElementById('results-section').classList.remove('visible');
    // Hide error card if retrying
    const errorCard = document.getElementById('error-card');
    if (errorCard) errorCard.classList.remove('visible');
    // Clear ETA
    const etaEl = document.getElementById('progress-eta');
    if (etaEl) etaEl.textContent = '';
    analysisStartTime = Date.now();
    updateProgress('preprocessing', 5);

    fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            session_id: uploadData.session_id,
            file_path: uploadData.file_path,
            patient_id: (document.getElementById('patient-id') || {}).value || '',
            patient_name: (document.getElementById('patient-name') || {}).value || '',
            exam_date: (document.getElementById('exam-date') || {}).value || '',
        })
    })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                btn.disabled = false;
                btn.innerHTML = '🧠 Run Cascade Analysis';
                return;
            }
            pollStatus(data.session_id);
        })
        .catch(err => {
            alert('Analysis request failed: ' + err);
            btn.disabled = false;
            btn.innerHTML = '🧠 Run Cascade Analysis';
            document.getElementById('progress-section').classList.remove('visible');
        });
}

const STAGE_LABELS = {
    'preprocessing': 'Preprocessing CT volume...',
    'stage1': 'Stage 1: Localizing pancreas (sliding window)...',
    'roi_extraction': 'Extracting region of interest...',
    'stage2': 'Stage 2: Segmenting tumor (sliding window)...',
    'clinical_analysis': 'Running HU tissue characterization & T-staging...',
    'xai': 'Computing Attention Rollout & Grad-CAM...',
    'generating_images': 'Generating visualization images...',
    'complete': '✓ Analysis complete!',
    'error': 'Error during analysis',
};

function pollStatus(sessionId) {
    const poll = setInterval(() => {
        fetch(`/status/${sessionId}`)
            .then(r => r.json())
            .then(s => {
                updateProgress(s.stage, s.progress);

                if (s.stage === 'complete') {
                    clearInterval(poll);
                    resultData = s.results;
                    setTimeout(() => displayResults(s.results), 400);
                } else if (s.stage === 'error') {
                    clearInterval(poll);
                    document.getElementById('progress-section').classList.remove('visible');
                    // Show error card
                    const errorCard = document.getElementById('error-card');
                    if (errorCard) {
                        document.getElementById('error-stage').textContent = 'Failed at: ' + (s.error_stage || 'unknown');
                        document.getElementById('error-message').textContent = s.error || 'An unexpected error occurred.';
                        errorCard.classList.add('visible');
                    }
                    document.getElementById('analyze-btn').disabled = false;
                } else {
                    // Update ETA
                    updateETA(s.stage, s.progress);
                }
            })
            .catch(() => { /* network hiccup, keep polling */ });
    }, 1500);
}

function updateProgress(stage, pct) {
    const fill = document.getElementById('progress-fill');
    fill.style.width = pct + '%';

    const allStages = ['preprocessing', 'stage1', 'roi_extraction', 'stage2', 'clinical_analysis', 'xai', 'generating_images', 'complete'];
    const stageIdx = allStages.indexOf(stage);

    document.querySelectorAll('.pstage').forEach((el, i) => {
        el.classList.remove('active', 'complete');
        if (i < stageIdx) el.classList.add('complete');
        else if (i === stageIdx && stage !== 'complete') el.classList.add('active');
    });

    if (stage === 'complete') {
        document.querySelectorAll('.pstage').forEach(el => el.classList.add('complete'));
        fill.style.width = '100%';
    }

    document.getElementById('progress-status').textContent = STAGE_LABELS[stage] || stage;
}

// ============================================================
// Display Results
// ============================================================
function displayResults(r) {
    document.getElementById('progress-section').classList.remove('visible');
    const resultsSection = document.getElementById('results-section');
    resultsSection.classList.add('visible');

    // ---- Risk Banner ----
    if (r.clinical) {
        const banner = document.getElementById('risk-banner');
        const risk = r.clinical.risk_level || 'LOW';
        banner.className = 'risk-banner visible risk-' + risk.toLowerCase();

        const icons = { LOW: '✓', MODERATE: '⚠', HIGH: '⚠' };
        document.getElementById('risk-icon').textContent = icons[risk] || '?';
        document.getElementById('risk-label').textContent = risk + ' RISK';
        document.getElementById('risk-summary').textContent = r.clinical.summary || r.clinical.primary_finding || '';
        document.getElementById('risk-confidence').textContent = Math.round((r.clinical.confidence || 0) * 100) + '%';
    }

    // ---- Metrics Grid ----
    const td = r.tumor_detected;
    const mg = document.getElementById('metrics-grid');
    mg.innerHTML = `
        <div class="metric-card ${td ? 'tumor-detected' : 'tumor-none'}">
            <div class="metric-icon ${td ? 'mi-red' : 'mi-green'}">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/><path d="M12 8v4"/><path d="M12 16h.01"/>
                </svg>
            </div>
            <div class="metric-label">Tumor Status</div>
            <div class="metric-value ${td ? 'text-red' : 'text-green'}">${td ? 'DETECTED' : 'NOT FOUND'}</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon mi-red">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                </svg>
            </div>
            <div class="metric-label">Tumor Volume</div>
            <div class="metric-value text-red">${r.tumor_volume_cm3}</div>
            <div class="metric-unit">cm³</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon mi-cyan">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78L12 21.23l8.84-8.84a5.5 5.5 0 0 0 0-7.78z"/>
                </svg>
            </div>
            <div class="metric-label">Pancreas Volume</div>
            <div class="metric-value text-cyan">${r.pancreas_volume_cm3}</div>
            <div class="metric-unit">cm³</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon mi-amber">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
                </svg>
            </div>
            <div class="metric-label">Max Dimension</div>
            <div class="metric-value text-amber">${r.max_dimension_mm}</div>
            <div class="metric-unit">mm</div>
        </div>
    `;

    // ---- Result Slices ----
    if (r.result_slices && r.result_slices.length > 0) {
        const slider = document.getElementById('result-slider');
        slider.max = r.result_slices.length - 1;
        resultSliceIdx = Math.floor(r.result_slices.length / 2);
        slider.value = resultSliceIdx;
        slider.oninput = () => showResultSlice(parseInt(slider.value));
        showResultSlice(resultSliceIdx);

        // Tab switching
        document.querySelectorAll('.rtab').forEach(btn => {
            btn.addEventListener('click', function () {
                document.querySelectorAll('.rtab').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                currentView = this.dataset.view;
                showResultSlice(resultSliceIdx);
            });
        });
    }

    // ---- MIP Views ----
    if (r.mip_axial) {
        document.getElementById('mip-panel').style.display = 'block';
        document.getElementById('mip-row').innerHTML = `
            <div class="mip-item">
                <img src="data:image/png;base64,${r.mip_axial}" alt="Axial MIP">
                <div class="mip-label">Axial MIP</div>
            </div>
            <div class="mip-item">
                <img src="data:image/png;base64,${r.mip_coronal}" alt="Coronal MIP">
                <div class="mip-label">Coronal MIP</div>
            </div>
            <div class="mip-item">
                <img src="data:image/png;base64,${r.mip_sagittal}" alt="Sagittal MIP">
                <div class="mip-label">Sagittal MIP</div>
            </div>
        `;
    }

    // ---- HU Analysis Section ----
    if (td && r.tumor_hu_mean !== 0) {
        const huPanel = document.getElementById('hu-panel');
        huPanel.style.display = 'block';
        document.getElementById('hu-body').innerHTML = `
            <div class="ct-info-grid">
                <div class="ct-info-item"><div class="ct-info-label">Tumor Mean HU</div><div class="ct-info-value text-red">${r.tumor_hu_mean}</div></div>
                <div class="ct-info-item"><div class="ct-info-label">Tumor HU Std</div><div class="ct-info-value">${r.tumor_hu_std}</div></div>
                <div class="ct-info-item"><div class="ct-info-label">Pancreas Mean HU</div><div class="ct-info-value text-green">${r.panc_hu_mean}</div></div>
                <div class="ct-info-item"><div class="ct-info-label">Enhancement Ratio</div><div class="ct-info-value text-amber">${r.enhancement_ratio}</div></div>
                <div class="ct-info-item"><div class="ct-info-label">HU Confidence</div><div class="ct-info-value ${r.hu_confidence === 'HIGH' ? 'text-green' : r.hu_confidence === 'MEDIUM' ? 'text-amber' : 'text-red'}">${r.hu_confidence}</div></div>
                <div class="ct-info-item"><div class="ct-info-label">Classification</div><div class="ct-info-value">${r.hu_classification}</div></div>
            </div>
            <div class="hu-interpretation">
                <strong>Interpretation:</strong> ${r.hu_classification}.
                PDAC typically shows 40-80 HU (hypoattenuating vs. normal pancreas at 80-120 HU).
                Enhancement ratio < 0.7 suggests hypovascularity consistent with PDAC.
                <br><em style="color:var(--accent-amber);">Note: HU thresholds are for post-segmentation validation only, not primary diagnosis.</em>
            </div>
        `;
    }

    // ---- Clinical Diagnosis Panel ----
    if (r.clinical && r.clinical.primary_finding) {
        const cp = document.getElementById('clinical-panel');
        cp.style.display = 'block';

        document.getElementById('diagnosis-primary').textContent = r.clinical.primary_finding || 'N/A';

        const badge = document.getElementById('diagnosis-badge');
        badge.textContent = r.clinical.primary_finding || '';
        const risk = (r.clinical.risk_level || 'LOW').toLowerCase();
        badge.style.background = risk === 'high' ? 'var(--glow-red)' : risk === 'moderate' ? 'var(--glow-amber)' : 'var(--glow-green)';
        badge.style.color = risk === 'high' ? 'var(--accent-red)' : risk === 'moderate' ? 'var(--accent-amber)' : 'var(--accent-green)';

        // Evidence
        const el = document.getElementById('evidence-list');
        el.innerHTML = '';
        (r.clinical.evidence || []).forEach(ev => {
            const div = document.createElement('div');
            div.className = 'evidence-item';
            div.innerHTML = `<span class="evidence-bullet">●</span><span>${ev}</span>`;
            el.appendChild(div);
        });

        // Differential
        if (r.clinical.differential && r.clinical.differential.length > 0) {
            document.getElementById('differential-section').style.display = 'block';
            const dl = document.getElementById('differential-list');
            dl.innerHTML = '';
            r.clinical.differential.forEach(d => {
                const div = document.createElement('div');
                div.className = 'diff-item';
                div.innerHTML = `<span class="diff-name">${d.diagnosis}</span><span class="diff-conf">${Math.round(d.confidence * 100)}%</span>`;
                dl.appendChild(div);
            });
        }

        // Recommendations
        if (r.clinical.recommendations && r.clinical.recommendations.length > 0) {
            document.getElementById('recommendations-section').style.display = 'block';
            const rl = document.getElementById('recommendations-list');
            rl.innerHTML = '';
            r.clinical.recommendations.forEach(rec => {
                const li = document.createElement('li');
                li.textContent = rec;
                rl.appendChild(li);
            });
        }
    }

    // ---- T-Staging Panel ----
    if (r.t_staging && r.t_staging.stage && r.t_staging.stage !== 'N/A') {
        const sp = document.getElementById('staging-panel');
        sp.style.display = 'block';

        const sb = document.getElementById('stage-badge');
        sb.textContent = r.t_staging.substage || r.t_staging.stage;
        sb.style.background = 'var(--glow-amber)';
        sb.style.color = 'var(--accent-amber)';

        const sg = document.getElementById('staging-grid');
        sg.innerHTML = '';
        [
            { label: 'T-Stage', value: r.t_staging.substage || r.t_staging.stage },
            { label: 'TNM String', value: r.t_staging.tnm_string },
            { label: 'Max Dimension', value: r.t_staging.max_dimension_mm.toFixed(1) + ' mm' },
            { label: 'Volume', value: r.t_staging.volume_cm3.toFixed(2) + ' cm³' },
            { label: 'Resectability', value: r.t_staging.resectability },
        ].forEach(it => {
            sg.innerHTML += `<div class="staging-item"><div class="s-label">${it.label}</div><div class="s-value">${it.value}</div></div>`;
        });

        const sn = document.getElementById('staging-notes');
        sn.innerHTML = '';
        (r.t_staging.notes || []).forEach(n => {
            sn.innerHTML += `<div class="staging-note">${n}</div>`;
        });
    }

    // ---- Pancreatitis Severity Panel ----
    if (r.pancreatitis && r.pancreatitis.severity) {
        const pvp = document.getElementById('severity-panel');
        pvp.style.display = 'block';

        const svb = document.getElementById('severity-badge');
        svb.textContent = r.pancreatitis.severity;
        const sevColors = { MILD: 'var(--accent-green)', MODERATE: 'var(--accent-amber)', SEVERE: 'var(--accent-red)' };
        svb.style.color = sevColors[r.pancreatitis.severity] || 'var(--text-primary)';
        svb.style.background = r.pancreatitis.severity === 'SEVERE' ? 'var(--glow-red)' : r.pancreatitis.severity === 'MODERATE' ? 'var(--glow-amber)' : 'var(--glow-green)';

        const sb2 = document.getElementById('severity-body');
        let html = `<div class="severity-score-bar">
            <div class="severity-score-track">
                <div class="severity-score-fill" style="width:${(r.pancreatitis.score / r.pancreatitis.max_score) * 100}%;background:${sevColors[r.pancreatitis.severity]}"></div>
            </div>
            <div class="severity-score-label" style="color:${sevColors[r.pancreatitis.severity]}">${r.pancreatitis.score}/${r.pancreatitis.max_score}</div>
        </div>`;
        html += `<div style="font-size:13px;color:var(--text-secondary);margin-bottom:1rem;">${r.pancreatitis.revised_atlanta}</div>`;

        if (r.pancreatitis.components) {
            Object.entries(r.pancreatitis.components).forEach(([key, comp]) => {
                html += `<div class="severity-component"><span class="severity-comp-label">${comp.description}</span><span class="severity-comp-score">${comp.score}/${comp.max}</span></div>`;
            });
        }
        sb2.innerHTML = html;
    }

    // ---- XAI Images ----
    if (r.xai_images && r.xai_images.length > 0) {
        document.getElementById('xai-panel').style.display = 'block';
        xaiSliceImages = r.xai_images;
        totalXaiSlices = xaiSliceImages.length;
        xaiSlice = Math.floor(totalXaiSlices / 2);
        showXaiSlice(xaiSlice);
    }

    // ---- Timing ----
    if (r.timing && r.timing.total) {
        document.getElementById('timing-panel').style.display = 'block';
        document.getElementById('timing-total-label').textContent = r.timing.total + 's total';
        document.getElementById('timing-total-big').innerHTML =
            `<span class="metric-value text-cyan">${r.timing.total}s</span>
             <span class="metric-unit">Total inference time • ${r.original_shape.join('×')} → ${r.resampled_shape.join('×')}</span>`;
    }

    // Re-enable analyze button
    const btn = document.getElementById('analyze-btn');
    btn.disabled = false;
    btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>
    </svg> Run Cascade Analysis`;

    // Show completion bar with context-aware summary
    const completionBar = document.getElementById('completion-bar');
    const completionSummary = document.getElementById('completion-summary');
    let summaryText = '';
    if (r.tumor_detected) {
        summaryText = `Tumor detected (${r.tumor_volume_cm3} cm³) • ${r.max_dimension_mm}mm max dimension`;
        if (r.clinical && r.clinical.primary_finding) summaryText += ` • ${r.clinical.primary_finding}`;
    } else {
        summaryText = `No tumor detected • Pancreas volume: ${r.pancreas_volume_cm3} cm³`;
    }
    if (r.timing && r.timing.total) summaryText += ` • ${r.timing.total}s`;
    completionSummary.textContent = summaryText;
    completionBar.classList.add('visible');

    // Show header new-scan button
    const headerBtn = document.getElementById('header-new-scan-btn');
    if (headerBtn) headerBtn.classList.add('visible');

    resultsSection.scrollIntoView({ behavior: 'smooth' });

    // ---- Hook up new features ----
    showPatientInResults();
    addToHistory(uploadData && uploadData.file_name ? uploadData.file_name : 'scan', r);
    setTimeout(() => attachLightbox(), 200);
}

// ============================================================
// Result Slice Navigation
// ============================================================
function showResultSlice(idx) {
    if (!resultData || !resultData.result_slices) return;
    if (idx < 0 || idx >= resultData.result_slices.length) return;
    resultSliceIdx = idx;
    const s = resultData.result_slices[idx];
    const img = document.getElementById('result-img');
    img.src = 'data:image/png;base64,' + s[currentView];
    const tag = s.has_tumor ? ' 🔴 TUMOR' : s.has_pancreas ? ' 🟢 Pancreas' : '';
    document.getElementById('result-slice-label').textContent = `Slice ${s.index}${tag}`;
    document.getElementById('result-slider').value = idx;
}

document.getElementById('result-prev-btn').addEventListener('click', () => showResultSlice(resultSliceIdx - 1));
document.getElementById('result-next-btn').addEventListener('click', () => showResultSlice(resultSliceIdx + 1));

// ============================================================
// XAI Slice Navigation
// ============================================================
function showXaiSlice(idx) {
    if (idx < 0 || idx >= totalXaiSlices) return;
    xaiSlice = idx;
    document.getElementById('xai-slice-image').src = xaiSliceImages[idx];
    document.getElementById('xai-counter').textContent = `${idx + 1} / ${totalXaiSlices}`;
}

document.getElementById('xai-prev-btn').addEventListener('click', () => showXaiSlice(xaiSlice - 1));
document.getElementById('xai-next-btn').addEventListener('click', () => showXaiSlice(xaiSlice + 1));

// ============================================================
// Keyboard navigation
// ============================================================
document.addEventListener('keydown', e => {
    // Close lightbox on ESC
    if (e.key === 'Escape') {
        const lb = document.getElementById('lightbox-overlay');
        if (lb) lb.classList.remove('active');
    }
    if (e.key === 'ArrowLeft') {
        if (resultData) showResultSlice(resultSliceIdx - 1);
    }
    if (e.key === 'ArrowRight') {
        if (resultData) showResultSlice(resultSliceIdx + 1);
    }
});

// ============================================================
// Lightbox — Click any image to expand fullscreen
// ============================================================
function openLightbox(src, caption) {
    const overlay = document.getElementById('lightbox-overlay');
    const img = document.getElementById('lightbox-img');
    const cap = document.getElementById('lightbox-caption');
    if (!overlay || !img) return;
    img.src = src;
    if (cap) cap.textContent = caption || '';
    overlay.classList.add('active');
}

// Attach lightbox to clickable images (runs after results load)
function attachLightbox() {
    document.querySelectorAll('.viewer-img, .mip-item img').forEach(img => {
        img.style.cursor = 'zoom-in';
        img.onclick = function (e) {
            e.stopPropagation();
            openLightbox(this.src, this.alt || '');
        };
    });
}

// ============================================================
// Volume Validation Warnings
// ============================================================
function showVolumeWarnings(data) {
    const warningEl = document.getElementById('volume-warning');
    const textEl = document.getElementById('volume-warning-text');
    if (!warningEl || !textEl) return;

    const warnings = [];
    if (data.num_slices < 20) {
        warnings.push('Volume has only ' + data.num_slices + ' slices (recommended: 50+). Results may be unreliable.');
    }
    if (data.hu_range && data.hu_range[0] > 0) {
        warnings.push('HU range starts at ' + Math.round(data.hu_range[0]) + ' (expected negative for CT). This may not be a raw CT volume.');
    }
    if (data.hu_range && (data.hu_range[1] - data.hu_range[0]) < 100) {
        warnings.push('Very narrow HU range (' + Math.round(data.hu_range[1] - data.hu_range[0]) + '). This may not be a standard CT scan.');
    }

    if (warnings.length > 0) {
        textEl.textContent = warnings.join(' ');
        warningEl.classList.add('visible');
    } else {
        warningEl.classList.remove('visible');
    }
}

// ============================================================
// Progress ETA
// ============================================================
let analysisStartTime = null;

function updateETA(stage, progress) {
    const etaEl = document.getElementById('progress-eta');
    if (!etaEl || !analysisStartTime || progress <= 5) return;

    const elapsed = (Date.now() - analysisStartTime) / 1000;
    const rate = progress / elapsed;
    const remaining = Math.max(0, (100 - progress) / rate);

    if (remaining > 0 && remaining < 600) {
        const mins = Math.floor(remaining / 60);
        const secs = Math.round(remaining % 60);
        etaEl.textContent = mins > 0 ? '~' + mins + 'm ' + secs + 's remaining' : '~' + secs + 's remaining';
    } else {
        etaEl.textContent = '';
    }
}

// ============================================================
// Scan History
// ============================================================
let scanHistory = [];

function addToHistory(filename, results) {
    const entry = {
        filename: filename,
        tumor: results.tumor_detected,
        volume: results.tumor_volume_cm3,
        panc: results.pancreas_volume_cm3,
        time: new Date().toLocaleTimeString(),
        results: results
    };
    scanHistory.push(entry);
    renderHistory();
}

function renderHistory() {
    const panel = document.getElementById('scan-history-panel');
    const list = document.getElementById('history-list');
    if (!panel || !list || scanHistory.length === 0) return;

    panel.style.display = 'block';
    list.innerHTML = '';
    scanHistory.forEach((entry, i) => {
        const div = document.createElement('div');
        div.className = 'history-item';
        div.innerHTML = `
            <div class="history-item-left">
                <div class="history-dot ${entry.tumor ? 'tumor' : 'clear'}"></div>
                <div>
                    <div class="history-filename">${entry.filename}</div>
                    <div class="history-meta">${entry.tumor ? 'Tumor: ' + entry.volume + ' cm\u00b3' : 'No tumor'} \u2022 ${entry.time}</div>
                </div>
            </div>
        `;
        list.appendChild(div);
    });
}

// ============================================================
// Patient Info in Results
// ============================================================
function getPatientInfo() {
    return {
        id: (document.getElementById('patient-id') || {}).value || '',
        name: (document.getElementById('patient-name') || {}).value || '',
        date: (document.getElementById('exam-date') || {}).value || ''
    };
}

function showPatientInResults() {
    const info = getPatientInfo();
    const bar = document.getElementById('patient-results-bar');
    if (!bar) return;

    if (!info.id && !info.name && !info.date) {
        bar.classList.remove('visible');
        return;
    }

    let html = '';
    if (info.id) html += '<div class="pr-item"><span class="pr-label">ID:</span> <span class="pr-value">' + info.id + '</span></div>';
    if (info.name) html += '<div class="pr-item"><span class="pr-label">Name:</span> <span class="pr-value">' + info.name + '</span></div>';
    if (info.date) html += '<div class="pr-item"><span class="pr-label">Date:</span> <span class="pr-value">' + info.date + '</span></div>';
    bar.innerHTML = html;
    bar.classList.add('visible');
}
