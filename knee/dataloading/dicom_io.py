"""Read one DICOM series into a sorted 3D volume plus its geometry.

Geometry conventions (DICOM patient coordinates, LPS: +x = patient left, +y = posterior, +z = superior):
- row_dir (IOP[:3]) is the direction of increasing column index j.
- col_dir (IOP[3:]) is the direction of increasing row index i.
- PixelSpacing = (spacing between rows, spacing between columns) = (d_i, d_j).
- normal = row_dir x col_dir. Slices are sorted by the projection of ImagePositionPatient on the normal.
"""
import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pydicom


@dataclass
class Series:
    volume: np.ndarray          # [D, H, W] float32, slices sorted along `normal`
    row_dir: np.ndarray         # (3,) unit vector of increasing column index
    col_dir: np.ndarray         # (3,) unit vector of increasing row index
    normal: np.ndarray          # (3,) row_dir x col_dir
    positions: np.ndarray       # (D, 3) ImagePositionPatient of the sorted slices
    spacing_ij: tuple           # (d_i, d_j) in mm
    info: dict = field(default_factory=dict)  # tags of the first slice and read notes


def dicom_paths(series_dir):
    """The DICOM files of one series folder: *.dcm, else every file of the folder."""
    paths = sorted(glob.glob(os.path.join(series_dir, '*.dcm')))
    return paths or sorted(p for p in glob.glob(os.path.join(series_dir, '*')) if os.path.isfile(p))


def _orientation_key(ds):
    iop = np.round(np.asarray(ds.ImageOrientationPatient, dtype=float), 2)
    return tuple(iop) + (int(ds.Rows), int(ds.Columns))


def read_series(paths, keep_tags=('Laterality', 'PatientSex', 'Manufacturer', 'MagneticFieldStrength',
                                  'SeriesDescription', 'SliceThickness', 'PatientPosition')):
    """Read and sort the slices of one series. Return a Series, or None when no slice can be read.

    Slices of a different orientation or image size than the largest group are dropped (localizers, mixed series).
    Slices whose pixel data cannot be decoded are dropped. Duplicate positions keep the first slice.
    """
    headers = []
    for p in paths:
        try:
            headers.append((p, pydicom.dcmread(p, stop_before_pixels=True)))
        except Exception:
            continue
    headers = [(p, ds) for p, ds in headers if 'ImageOrientationPatient' in ds and 'ImagePositionPatient' in ds]
    if not headers:
        return None
    groups = {}
    for p, ds in headers:
        groups.setdefault(_orientation_key(ds), []).append((p, ds))
    group = max(groups.values(), key=len)
    ds0 = group[0][1]
    iop = np.asarray(ds0.ImageOrientationPatient, dtype=float)
    row_dir, col_dir = iop[:3] / np.linalg.norm(iop[:3]), iop[3:] / np.linalg.norm(iop[3:])
    normal = np.cross(row_dir, col_dir)
    pos = np.array([np.asarray(ds.ImagePositionPatient, dtype=float) for _, ds in group])
    order = np.argsort(pos @ normal, kind='stable')
    # drop duplicate positions (keep the first one in sorted order)
    proj = (pos @ normal)[order]
    keep = np.concatenate([[True], np.diff(proj) > 1e-3])
    order = order[keep]

    slices, kept_pos, n_bad = [], [], 0
    for k in order:
        p = group[k][0]
        try:
            ds = pydicom.dcmread(p)
            img = ds.pixel_array.astype(np.float32)
            slope, inter = float(ds.get('RescaleSlope', 1) or 1), float(ds.get('RescaleIntercept', 0) or 0)
            img = img * slope + inter
            if str(ds.get('PhotometricInterpretation', '')) == 'MONOCHROME1':
                img = img.max() - img
            if img.ndim != 2:
                img = img.reshape(img.shape[-2:]) if img.size == ds0.Rows * ds0.Columns else img[0]
            slices.append(img)
            kept_pos.append(pos[k])
        except Exception:
            n_bad += 1
    if not slices:
        return None
    sp = ds0.get('PixelSpacing', [1.0, 1.0])
    info = {t: (str(ds0.get(t, '')) if ds0.get(t, None) is not None else '') for t in keep_tags}
    info.update(n_files=len(paths), n_dropped_group=len(headers) - len(group), n_bad_pixels=n_bad,
                n_duplicates=int((~keep).sum()))
    return Series(volume=np.stack(slices), row_dir=row_dir, col_dir=col_dir, normal=normal,
                  positions=np.array(kept_pos), spacing_ij=(float(sp[0]), float(sp[1])), info=info)
