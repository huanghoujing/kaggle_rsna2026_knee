"""Put a series into a canonical patient orientation and a fixed in-plane pixel size (torch: CPU or GPU).

Canonical display for each plane (LPS patient axes; the image column index j grows to the right of the screen,
the row index i grows to the bottom, the slice index k grows along the slice axis):

| plane    | columns (j)               | rows (i)                  | slices (k)                 |
|----------|---------------------------|---------------------------|----------------------------|
| Sagittal | +y (anterior -> posterior) | -z (superior -> inferior) | +x (patient right -> left) |
| Coronal  | +x (patient right -> left) | -z (superior -> inferior) | +y (anterior -> posterior) |
| Axial    | +x (patient right -> left) | +y (anterior -> posterior) | -z (superior -> inferior)  |

In this frame, the patient left-right axis is the slice axis of sagittal series and the column axis of coronal and
axial series. Mirroring a study along that axis turns a right knee into a left knee: medial stays medial, so the
labels do not change (anatomical mirror, used for augmentation and test-time augmentation).
"""
import numpy as np
import torch
import torch.nn.functional as F

AXES = {  # plane: (column axis, row axis, slice axis) as LPS unit vectors
    'Sagittal': ((0, 1, 0), (0, 0, -1), (1, 0, 0)),
    'Coronal': ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
    'Axial': ((1, 0, 0), (0, 1, 0), (0, 0, -1)),
}


def plane_of(normal):
    """Plane from the slice normal: the patient axis closest to the normal."""
    return ('Sagittal', 'Coronal', 'Axial')[int(np.argmax(np.abs(normal)))]


def canonical_transform(row_dir, col_dir, normal, plane):
    """Return (transpose, flip_rows, flip_cols, flip_slices) that maps the DICOM array to the canonical display.

    The DICOM array is [k, i, j] with j along row_dir and i along col_dir, k sorted along the normal.
    """
    tc, tr, ts = (np.asarray(a, dtype=float) for a in AXES[plane])
    keep = abs(row_dir @ tc) + abs(col_dir @ tr)
    swap = abs(row_dir @ tr) + abs(col_dir @ tc)
    transpose = bool(swap > keep)
    new_col, new_row = (col_dir, row_dir) if transpose else (row_dir, col_dir)
    return transpose, bool(new_row @ tr < 0), bool(new_col @ tc < 0), bool(normal @ ts < 0)


def to_canonical(vol, spacing_ij, row_dir, col_dir, normal, plane):
    """Reorient a [D, H, W] tensor to the canonical display. Return (volume, (d_row, d_col))."""
    transpose, flip_r, flip_c, flip_s = canonical_transform(row_dir, col_dir, normal, plane)
    d_i, d_j = spacing_ij
    if transpose:
        vol = vol.transpose(1, 2)
        d_i, d_j = d_j, d_i
    dims = [d for d, f in ((0, flip_s), (1, flip_r), (2, flip_c)) if f]
    if dims:
        vol = torch.flip(vol, dims)
    return vol, (d_i, d_j)


def resample_inplane(vol, spacing, target_mm, out_px):
    """Resample [D, H, W] to target_mm pixels and center crop / zero pad to out_px x out_px.

    Down-sampling uses antialiased bilinear interpolation (torch), so CPU and GPU give the same result up to rounding.
    """
    d, h, w = vol.shape
    new_h, new_w = max(1, round(h * spacing[0] / target_mm)), max(1, round(w * spacing[1] / target_mm))
    x = F.interpolate(vol[:, None].float(), size=(new_h, new_w), mode='bilinear', antialias=True,
                      align_corners=False)[:, 0]
    out = x.new_zeros((d, out_px, out_px))
    # center crop (source) and center pad (target), separately for each axis
    sy, ty = max(0, (new_h - out_px) // 2), max(0, (out_px - new_h) // 2)
    sx, tx = max(0, (new_w - out_px) // 2), max(0, (out_px - new_w) // 2)
    hh, ww = min(new_h, out_px), min(new_w, out_px)
    out[:, ty:ty + hh, tx:tx + ww] = x[:, sy:sy + hh, sx:sx + ww]
    return out


def merge_thin_slices(vol, slice_mm, min_slice_mm):
    """Average groups of adjacent thin slices (3D acquisitions) so that the slice spacing is about min_slice_mm."""
    k = int(round(min_slice_mm / max(slice_mm, 1e-3)))
    if k <= 1:
        return vol, slice_mm
    d = vol.shape[0] // k * k
    if d == 0:
        return vol, slice_mm
    return vol[:d].reshape(d // k, k, *vol.shape[1:]).mean(1), slice_mm * k


def scale_to_255(vol, lo_pct=0.5, hi_pct=99.5, max_samples=2_000_000):
    """Map the [lo_pct, hi_pct] percentiles of the volume to [0, 255] (float, clipped).

    Apply it before zero padding, so that the padding does not change the percentiles. The percentiles use a
    strided sample of the voxels (torch.quantile sorts, which is slow for large volumes).
    """
    flat = vol.reshape(-1).float()
    if flat.numel() > max_samples:
        flat = flat[:: flat.numel() // max_samples + 1]
    lo, hi = torch.quantile(flat, torch.tensor([lo_pct / 100, hi_pct / 100], device=flat.device))
    scale = 255.0 / max(float(hi - lo), 1e-6)
    return ((vol.float() - lo) * scale).clamp_(0, 255)
