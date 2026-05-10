"""
DICOM Utility Module for CascadePanc-ViTUNet
Converts DICOM files/series to 3D NumPy volumes with spacing information.
"""

import os
import numpy as np

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


def _safe_float(val, default=1.0):
    """Safely convert a value to float, returning default if None or invalid."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_spacing(ds, default_pixel=1.0, default_slice=1.0):
    """Extract spacing from a DICOM dataset, handling None/missing values."""
    # Pixel spacing
    ps = getattr(ds, 'PixelSpacing', None)
    if ps is not None and len(ps) >= 2:
        px = _safe_float(ps[0], default_pixel)
        py = _safe_float(ps[1], default_pixel)
    else:
        px, py = default_pixel, default_pixel

    # Slice thickness
    st = getattr(ds, 'SliceThickness', None)
    sz = _safe_float(st, default_slice)

    return px, py, max(sz, 0.1)


def is_dicom_file(filepath):
    """Check if a file is a DICOM file."""
    if not PYDICOM_AVAILABLE:
        return False
    try:
        pydicom.dcmread(filepath, stop_before_pixels=True)
        return True
    except Exception:
        return False


def load_dicom_series(path):
    """
    Load a DICOM series from a file or directory.

    Args:
        path: Path to a single .dcm file or a directory of DICOM slices

    Returns:
        volume: 3D numpy array (D, H, W) in Hounsfield Units
        spacing: numpy array [pixel_spacing_x, pixel_spacing_y, slice_thickness]

    Raises:
        ImportError: if pydicom is not installed
        ValueError: if no valid DICOM files found
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError(
            "pydicom is required for DICOM support. "
            "Install it with: pip install pydicom"
        )

    # Determine if path is a single file or directory
    if os.path.isfile(path):
        ds = pydicom.dcmread(path)

        # Multi-frame DICOM
        if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames is not None and int(ds.NumberOfFrames) > 1:
            return _load_multiframe_dicom(ds)

        # Check for sibling DICOM files in the same directory
        parent_dir = os.path.dirname(path)
        sibling_dicoms = _find_dicom_files(parent_dir)

        if len(sibling_dicoms) > 1:
            return _load_dicom_directory(parent_dir)
        else:
            # Single slice — treat as a 1-deep 3D volume
            return _load_single_slice(ds)

    elif os.path.isdir(path):
        return _load_dicom_directory(path)
    else:
        raise ValueError(f"Path does not exist: {path}")


def _find_dicom_files(directory):
    """Find all valid DICOM files in a directory (non-recursive)."""
    dicom_files = []
    try:
        for fname in os.listdir(directory):
            fpath = os.path.join(directory, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True)
                if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
                    dicom_files.append(fpath)
            except Exception:
                continue
    except Exception:
        pass
    return dicom_files


def _load_single_slice(ds):
    """Load a single-frame DICOM file as a 1-slice 3D volume."""
    arr = ds.pixel_array.astype(np.float32)

    # HU conversion
    slope = _safe_float(getattr(ds, 'RescaleSlope', 1), 1.0)
    intercept = _safe_float(getattr(ds, 'RescaleIntercept', 0), 0.0)
    arr = arr * slope + intercept

    # Make 3D: (1, H, W)
    if arr.ndim == 2:
        volume = arr[np.newaxis, :, :]
    else:
        volume = arr

    px, py, sz = _safe_spacing(ds)
    spacing = np.array([px, py, sz])

    print(f"[INFO] Loaded single DICOM slice -> {volume.shape}")
    print(f"[INFO] Spacing: {spacing} mm, HU range: [{volume.min():.0f}, {volume.max():.0f}]")

    return volume, spacing


def _load_multiframe_dicom(ds):
    """Load a multi-frame (enhanced) DICOM file."""
    pixel_data = ds.pixel_array.astype(np.float32)

    # Apply rescale slope/intercept for HU
    slope = _safe_float(getattr(ds, 'RescaleSlope', 1), 1.0)
    intercept = _safe_float(getattr(ds, 'RescaleIntercept', 0), 0.0)
    volume = pixel_data * slope + intercept

    px, py, sz = _safe_spacing(ds)
    spacing = np.array([px, py, sz])

    print(f"[INFO] Loaded multi-frame DICOM -> {volume.shape}")
    print(f"[INFO] Spacing: {spacing} mm, HU range: [{volume.min():.0f}, {volume.max():.0f}]")

    return volume, spacing


def _load_dicom_directory(directory):
    """Load all DICOM slices from a directory and stack into a 3D volume."""
    dicom_files = []

    for root, _, files in os.walk(directory):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True)
                # Check it has pixel data
                if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
                    dicom_files.append(fpath)
            except Exception:
                continue

    if len(dicom_files) == 0:
        raise ValueError(f"No valid DICOM files found in: {directory}")

    # Read all slices with full pixel data
    slices = []
    for fpath in dicom_files:
        try:
            ds = pydicom.dcmread(fpath)
            slices.append(ds)
        except Exception:
            continue

    if len(slices) == 0:
        raise ValueError("Failed to read any DICOM slices")

    # Sort by ImagePositionPatient[2] or InstanceNumber
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except (AttributeError, TypeError, IndexError):
        try:
            slices.sort(key=lambda s: int(s.InstanceNumber))
        except (AttributeError, TypeError):
            pass  # Use file order if no sorting info

    # Extract pixel data and apply HU conversion
    pixel_arrays = []
    for s in slices:
        arr = s.pixel_array.astype(np.float32)
        slope = _safe_float(getattr(s, 'RescaleSlope', 1), 1.0)
        intercept = _safe_float(getattr(s, 'RescaleIntercept', 0), 0.0)
        arr = arr * slope + intercept
        pixel_arrays.append(arr)

    volume = np.stack(pixel_arrays, axis=0)

    # Get spacing from first slice
    ds0 = slices[0]
    px, py, _ = _safe_spacing(ds0)

    # Calculate slice spacing from positions if available
    if len(slices) > 1:
        try:
            z0 = float(slices[0].ImagePositionPatient[2])
            z1 = float(slices[1].ImagePositionPatient[2])
            slice_spacing = abs(z1 - z0)
            if slice_spacing < 0.01:
                slice_spacing = _safe_float(getattr(ds0, 'SliceThickness', None), 1.0)
        except (AttributeError, TypeError, IndexError):
            slice_spacing = _safe_float(getattr(ds0, 'SliceThickness', None), 1.0)
    else:
        slice_spacing = _safe_float(getattr(ds0, 'SliceThickness', None), 1.0)

    spacing = np.array([px, py, max(slice_spacing, 0.1)])

    print(f"[INFO] Loaded {len(slices)} DICOM slices -> {volume.shape}")
    print(f"[INFO] Spacing: {spacing} mm, HU range: [{volume.min():.0f}, {volume.max():.0f}]")

    return volume, spacing

