"""Build the image cache: one uint8 volume [D, H, W] for each series, canonical orientation, 0.5 mm pixels.

    .venv/bin/python -m knee.dataloading.build_cache --split train --version v1

Output (under data/cache/<version>/):
- <split>/<StudyInstanceUID>/<SeriesInstanceUID>.npy   uint8 [D, out_px, out_px]
- index_<split>.parquet                                one row for each series (geometry, tags, read notes)

The build is incremental: series with an .npy file are skipped, and only completely extracted studies are read
(see stream_unzip.complete_study_dirs). Run it again to add the studies that arrived since the last run.
"""
import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from knee.constants import DATA, RAW
from knee.dataloading.canonical import (merge_thin_slices, plane_of, resample_inplane, scale_to_255,
                                        to_canonical, AXES)
from knee.dataloading.dicom_io import dicom_paths, read_series
from knee.dataloading.stream_unzip import complete_study_dirs

CFG = {'v1': dict(target_mm=0.5, out_px=352, min_slice_mm=2.0, lo_pct=0.5, hi_pct=99.5)}     # 176 mm field of view


def prepare_series(series_dir, cfg):
    """Read one DICOM series and return (uint8 volume [D, out_px, out_px] in the canonical frame, info dict).

    Training (this cache) and inference (knee.inference) call this same function, so both see the same pixels.
    Return (None, info) when no slice can be read.
    """
    s = read_series(dicom_paths(series_dir))
    if s is None:
        return None, {'error': 'no readable slice'}
    plane = plane_of(s.normal)
    vol = torch.from_numpy(s.volume)
    vol, sp = to_canonical(vol, s.spacing_ij, s.row_dir, s.col_dir, s.normal, plane)
    ts = np.asarray(AXES[plane][2], dtype=float)
    proj = np.sort(s.positions @ ts)                   # slice positions along the canonical slice axis
    slice_mm = float(np.median(np.diff(proj))) if len(proj) > 1 else float(s.info.get('SliceThickness') or 3)
    vol, slice_mm = merge_thin_slices(vol, slice_mm, cfg['min_slice_mm'])
    vol = scale_to_255(vol, cfg['lo_pct'], cfg['hi_pct'])
    vol = resample_inplane(vol, sp, cfg['target_mm'], cfg['out_px']).round_().clamp_(0, 255).to(torch.uint8)
    center = s.positions.mean(0)
    info = dict(plane=plane, depth=int(vol.shape[0]), slice_mm=slice_mm,
                native_h=int(s.volume.shape[1]), native_w=int(s.volume.shape[2]),
                spacing_row=sp[0], spacing_col=sp[1], fov_row_mm=sp[0] * s.volume.shape[1],
                center_x=float(center[0]), center_y=float(center[1]), center_z=float(center[2]),
                first_mm=float(proj[0]), last_mm=float(proj[-1]),
                normal=json.dumps([round(float(v), 4) for v in s.normal]), **s.info)
    return vol.numpy(), info


def process_series(job):
    """Prepare and save one series. Return its index row."""
    study, series, series_dir, out_path, cfg = job
    torch.set_num_threads(1)
    row = {'StudyInstanceUID': study, 'SeriesInstanceUID': series, 'ok': False, 'error': ''}
    try:
        vol, info = prepare_series(series_dir, cfg)
        if vol is None:
            row.update(info)
            return row
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path + '.tmp.npy', vol)
        os.replace(out_path + '.tmp.npy', out_path)
        row.update(ok=True, **info)
    except Exception as e:
        row['error'] = f'{type(e).__name__}: {e}'
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train', help='<split>_series under --raw; the competition series csv')
    ap.add_argument('--version', default='v1', choices=sorted(CFG))
    ap.add_argument('--raw', default=RAW)
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() // 3))
    args = ap.parse_args()
    cfg = CFG[args.version]
    root = os.path.join(args.raw, f'{args.split}_series')
    out_root = os.path.join(DATA, 'cache', args.version)
    index_path = os.path.join(out_root, f'index_{args.split}.parquet')
    old = pd.read_parquet(index_path) if os.path.exists(index_path) else None
    done = set(old.loc[old.ok, 'SeriesInstanceUID']) if old is not None else set()

    jobs = []
    for study_dir in complete_study_dirs(root):
        study = os.path.basename(study_dir)
        for series_dir in sorted(glob.glob(os.path.join(study_dir, '*'))):
            series = os.path.basename(series_dir)
            if series in done:
                continue
            jobs.append((study, series, series_dir, os.path.join(out_root, args.split, study, series + '.npy'), cfg))
    print(f'{len(jobs)} series to process', flush=True)
    t0, rows = time.time(), []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in tqdm(ex.map(process_series, jobs, chunksize=1), total=len(jobs), mininterval=30):
            rows.append(r)
    new = pd.DataFrame(rows)
    if old is not None:  # failed rows are retried on the next run, so drop their old version
        old = old[~old.SeriesInstanceUID.isin(new.SeriesInstanceUID)] if len(new) else old
    index = new if old is None else pd.concat([old, new], ignore_index=True)
    os.makedirs(out_root, exist_ok=True)
    if len(index):
        index.to_parquet(index_path, index=False)
    n_err = int((~new.ok).sum()) if len(new) else 0
    print(f'{len(new)} series in {time.time() - t0:.0f} s, {n_err} errors; index: {len(index)} rows -> {index_path}')
    if n_err:
        print(new.loc[~new.ok, ['SeriesInstanceUID', 'error']].head(20).to_string())


if __name__ == '__main__':
    main()
