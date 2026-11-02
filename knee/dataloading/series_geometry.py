"""In-plane image center of every series in patient space (LPS mm), from the DICOM headers.

    .venv/bin/python -m knee.dataloading.series_geometry --split train   # data/meta/series_geometry_train.parquet

Image center of a slice = ImagePositionPatient + (Columns - 1) / 2 * column spacing * row direction
+ (Rows - 1) / 2 * row spacing * column direction (ImageOrientationPatient = row direction, column direction;
PixelSpacing = row spacing, column spacing); the series center is the mean over its slices. The cache (knee.dataloading.
build_cache) center-crops every series around this point, so a cached pixel lies at center + offsets along the
canonical display axes (knee.dataloading.canonical.AXES). series_table merges the table when it exists.
Also the patient directions (LPS unit vectors) of the canonical column, row and slice axes of each series (ax_c*, ax_r*,
ax_s*: the true, possibly oblique directions of the cache axes; knee.dataloading.canonical.canonical_transform) and its
FrameOfReferenceUID (series of one frame share the patient coordinates).
"""
import argparse
import os

import numpy as np
import pandas as pd

from knee.constants import META
from knee.dataloading.canonical import canonical_transform, plane_of


def vec(col, n):
    return np.array([np.array(str(v).split('\\'), dtype=float) if isinstance(v, str) else np.full(n, np.nan)
                     for v in col])


def series_centers(meta):
    ipp, iop, ps = vec(meta.ImagePositionPatient, 3), vec(meta.ImageOrientationPatient, 6), vec(meta.PixelSpacing, 2)
    rows, cols = meta.Rows.to_numpy(float), meta.Columns.to_numpy(float)
    center = ipp + ((cols - 1) / 2 * ps[:, 1])[:, None] * iop[:, :3] + ((rows - 1) / 2 * ps[:, 0])[:, None] * iop[:, 3:]
    t = pd.DataFrame(center, columns=['gx', 'gy', 'gz'])
    t['SeriesInstanceUID'] = meta.SeriesInstanceUID.to_numpy()
    return t.groupby('SeriesInstanceUID')[['gx', 'gy', 'gz']].mean().reset_index()


def canonical_axes(iop):
    """ImageOrientationPatient [6] -> the patient directions of the canonical column, row and slice axes [3, 3] (the
    cache volume [slice, row, column] of build_cache.prepare_series)."""
    row_dir, col_dir = iop[:3] / np.linalg.norm(iop[:3]), iop[3:] / np.linalg.norm(iop[3:])
    normal = np.cross(row_dir, col_dir)
    transpose, flip_r, flip_c, flip_s = canonical_transform(row_dir, col_dir, normal, plane_of(normal))
    new_col, new_row = (col_dir, row_dir) if transpose else (row_dir, col_dir)
    return np.stack([-new_col if flip_c else new_col, -new_row if flip_r else new_row, -normal if flip_s else normal])


def series_axes(meta):
    first = meta.drop_duplicates('SeriesInstanceUID').reset_index(drop=True)
    iop = vec(first.ImageOrientationPatient, 6)
    ax = np.stack([canonical_axes(v) if np.isfinite(v).all() else np.full((3, 3), np.nan) for v in iop])
    t = pd.DataFrame(ax.reshape(len(first), 9), columns=[f'ax_{a}{c}' for a in 'crs' for c in 'xyz'])
    t['SeriesInstanceUID'] = first.SeriesInstanceUID.to_numpy()
    t['frame'] = first.FrameOfReferenceUID.astype(str).to_numpy()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    args = ap.parse_args()
    meta = pd.read_parquet(os.path.join(META, f'{args.split}_dicom_meta.parquet'),
                           columns=['SeriesInstanceUID', 'ImagePositionPatient', 'ImageOrientationPatient',
                                    'PixelSpacing', 'Rows', 'Columns', 'FrameOfReferenceUID'])
    out = series_centers(meta).merge(series_axes(meta), on='SeriesInstanceUID', how='left')
    path = os.path.join(META, f'series_geometry_{args.split}.parquet')
    out.to_parquet(path, index=False)
    print(f'{len(out)} series ({out.gx.isna().sum()} without geometry) -> {path}')


if __name__ == '__main__':
    main()
