"""SKM-TEA (Stanford, NeurIPS 2021): 155 sagittal qDESS knee scans with masks of six tissues and 3D boxes of 16
pathologies (meniscal tears, ACL / PCL tears, cartilage lesions, effusion). The tissue segmentation model trains on it.

    REDIVIS_API_TOKEN=<token> python -m knee.dataloading.skm_tea download
    python -m knee.dataloading.skm_tea prepare      # the cache split skm_tea: 2 echoes x 3 planes + tissue labels

Source: Redivis dataset aimi.skm_tea:5r8z:v1_0 (Stanford AIMI; a Redivis account with access to the dataset is needed).
Only the DICOM track is downloaded: the scanner DICOMs ('dicom', 155 tar.gz, 19.8 GB), the tissue masks on the DICOM
grid ('segmentation_masks', dicom-track/*.nii.gz, 0.1 GB) and the box annotations ('annotations', COCO-style json).
Not downloaded: the raw k-space and reconstructions ('application', 857 GB) and the h5 copies of the images
('image_files', 65 GB). Masks: 0 background, 1 patellar cartilage, 2 femoral cartilage, 3 / 4 medial / lateral tibial
cartilage, 5 / 6 medial / lateral meniscus (skm_tea/data/register.py of the authors' code).
Licence: Stanford University School of Medicine research licence (the terms of the AIMI LERA dataset).
"""
import argparse
import glob
import json
import os
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from knee.constants import DATA

ROOT = os.path.join(DATA, 'external', 'skm_tea')
DATASET = 'skm_tea:5r8z:v1_0'
CACHE = os.path.join(DATA, 'cache', 'v1')
SPLIT = 'skm_tea'
SERIES_CSV = os.path.join(ROOT, 'skm_tea_series.csv')
N_TISSUES = 6   # label 0 = background
TABLES = {'annotations': None, 'segmentation_masks': 'dicom-track/', 'dicom': None}   # table -> file-name prefix


def download():
    import niquests
    import redivis
    init = niquests.AsyncSession.__init__

    def no_http3(self, *a, **kw):   # the HTTP/3 (QUIC) path of the Redivis downloader fails here
        kw.setdefault('disable_http3', True)
        init(self, *a, **kw)
    niquests.AsyncSession.__init__ = no_http3
    ds = redivis.organization('aimi').dataset(DATASET)
    for name, prefix in TABLES.items():
        table = ds.table(name)
        index = table.to_pandas_dataframe(progress=False)
        if prefix:
            index = index[index.file_name.str.startswith(prefix)]
        todo = [r for r in index.itertuples() if not os.path.exists(os.path.join(ROOT, name, r.file_name))]
        print(f'{name}: {len(index)} files ({index["size"].sum() / 1e9:.1f} GB), {len(todo)} to download', flush=True)
        for n, r in enumerate(todo):   # one file at a time, with retries (the parallel downloader hangs)
            dst = os.path.join(ROOT, name, r.file_name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            for attempt in range(5):
                try:
                    table.file(r.file_name).download(dst + '.part', overwrite=True, progress=False)
                    os.replace(dst + '.part', dst)
                    break
                except Exception as e:   # noqa: BLE001 (network errors: retry)
                    print(f'retry {attempt + 1} {r.file_name}: {type(e).__name__}', flush=True)
            if n % 20 == 0:
                print(f'{name}: {n}/{len(todo)}', flush=True)
    print('done ->', ROOT, flush=True)


def _echo_dirs(tar_path, work):
    """Unpack one scan; the series with both qDESS echoes (interleaved, the most files) is split by EchoNumbers into
    work/e1 and work/e2."""
    import pydicom
    with tarfile.open(tar_path) as tf:
        tf.extractall(work, filter='data')
    by_series = {}
    for f in sorted(glob.glob(os.path.join(work, '**', '*.dcm'), recursive=True)):
        h = pydicom.dcmread(f, stop_before_pixels=True)
        by_series.setdefault(h.SeriesInstanceUID, []).append((f, int(h.EchoNumbers)))
    dirs = {}
    for f, echo in by_series[max(by_series, key=lambda u: len(by_series[u]))]:
        dirs[echo] = os.path.join(work, f'e{echo}')
        os.makedirs(dirs[echo], exist_ok=True)
        os.replace(f, os.path.join(dirs[echo], os.path.basename(f)))
    return dirs


REFORMAT_MM = 3.5   # slab thickness of the coronal / axial reformats (our coronal / axial series: 3.5 / 3.6 mm median)
# canonical sagittal [slices +x, rows -z, columns +y] -> canonical coronal / axial (knee.dataloading.canonical.AXES)
REFORMATS = {'Coronal': (2, 1, 0), 'Axial': (1, 2, 0)}
SUFFIX = {'Sagittal': '', 'Coronal': '_cor', 'Axial': '_ax'}   # series skm_<scan><suffix>_e<echo>


def native_labels(mask_path, s):
    """Tissue labels (NIfTI on the DICOM grid) -> uint8 tensor [D, rows, columns] in the slice order of the DICOM
    series s (dicom_io.read_series)."""
    import nibabel as nib
    import torch
    m = nib.load(mask_path)
    lab = np.asarray(m.dataobj).astype(np.uint8)                         # [columns i, rows j, slices k]
    k = np.arange(lab.shape[2])
    ras = (m.affine @ np.stack([np.zeros_like(k), np.zeros_like(k), k, np.ones_like(k)]))[:3].T
    lps = ras * np.array([-1, -1, 1])                                    # NIfTI world (RAS) -> DICOM (LPS)
    dist = np.linalg.norm(s.positions[:, None, :] - lps[None], axis=-1)  # sorted DICOM slices x mask slices
    match = dist.argmin(1)   # the stored affine is rounded: up to about 0.1 mm off after 160 slices (spacing 0.8 mm)
    assert dist.min(1).max() < 0.3 and len(set(match)) == len(match), f'{mask_path}: mask and DICOM slices do not match'
    vol = torch.from_numpy(lab[:, :, match].transpose(2, 1, 0).copy())   # [D, rows, columns] as s.volume
    assert vol.shape == s.volume.shape, (vol.shape, s.volume.shape)
    return vol


def canonical_sagittal(x, s):
    """[D, rows, columns] in the slice order of s -> (canonical sagittal tensor at the native resolution, (row, column)
    pixel mm, slice mm)."""
    from knee.dataloading.canonical import AXES, to_canonical
    v, sp = to_canonical(x, s.spacing_ij, s.row_dir, s.col_dir, s.normal, 'Sagittal')
    proj = np.sort(s.positions @ np.asarray(AXES['Sagittal'][2], dtype=float))
    return v, sp, float(np.median(np.diff(proj)))


def to_cache_frame(v, sp, slice_mm, plane, cfg, image):
    """Canonical sagittal volume at the native resolution -> (cache volume of the plane, slice mm). Sagittal: the steps
    of build_cache.prepare_series (slices merged to about min_slice_mm); Coronal / Axial: reformat with REFORMAT_MM
    slabs. image: scaled to 0-255 as prepare_series; else float channels (one-hot labels)."""
    from knee.dataloading.canonical import merge_thin_slices, resample_inplane, scale_to_255
    if plane == 'Sagittal':
        v, slice_mm = merge_thin_slices(v.float(), slice_mm, cfg['min_slice_mm'])
    else:
        perm = REFORMATS[plane]
        mm = [((slice_mm,) + tuple(sp))[p] for p in perm]   # mm along the new (slices, rows, columns)
        v, slice_mm = merge_thin_slices(v.float().permute(*perm), mm[0], REFORMAT_MM)
        sp = (mm[1], mm[2])
    if image:
        v = scale_to_255(v, cfg['lo_pct'], cfg['hi_pct'])
    return resample_inplane(v, sp, cfg['target_mm'], cfg['out_px']), slice_mm


def labels_to_cache(lab, sp, slice_mm, plane, cfg, n_classes=N_TISSUES + 1):
    """Canonical sagittal labels at the native resolution -> uint8 labels in the cache frame of the plane (each one-hot
    channel through the image steps, then the arg max). n_classes: label 0 (background) included."""
    import torch
    chans = [to_cache_frame((lab == c).float(), sp, slice_mm, plane, cfg, image=False)[0] for c in range(n_classes)]
    return torch.stack(chans).argmax(0).to(torch.uint8).numpy()


def _prepare_scan(job):
    scan, split = job
    import torch
    from knee.dataloading.build_cache import CFG, prepare_series
    from knee.dataloading.dicom_io import dicom_paths, read_series
    torch.set_num_threads(2)
    cfg, study = CFG['v1'], f'skm_{scan}'
    work = os.path.join(ROOT, 'tmp', scan)
    shutil.rmtree(work, ignore_errors=True)
    rows = []
    try:
        dirs = _echo_dirs(os.path.join(ROOT, 'dicom', f'{scan}.tar.gz'), work)
        out_dir = os.path.join(CACHE, SPLIT, study)
        os.makedirs(out_dir, exist_ok=True)
        labels = None
        for echo in (1, 2):
            s = read_series(dicom_paths(dirs[echo]))
            if labels is None:   # one mask per scan: both echoes share the grid
                lab, sp, slice_mm = canonical_sagittal(native_labels(
                    os.path.join(ROOT, 'segmentation_masks', 'dicom-track', f'{scan}.nii.gz'), s), s)
                k = torch.arange(lab.shape[0]).view(-1, 1, 1).expand_as(lab)
                # canonical sagittal slices run to the patient left: the medial meniscus of a right knee lies further left
                side = 'R' if k[lab == 5].float().mean() > k[lab == 6].float().mean() else 'L'
                labels = {p: labels_to_cache(lab, sp, slice_mm, p, cfg) for p in SUFFIX}
                for p, v in labels.items():
                    np.save(os.path.join(out_dir, f'{study}{SUFFIX[p]}_seg.npy'), v)
            for plane, suffix in SUFFIX.items():
                if plane == 'Sagittal':   # as the competition series
                    vol, info = prepare_series(dirs[echo], cfg)
                else:
                    v, sp_i, smm = canonical_sagittal(torch.from_numpy(s.volume), s)
                    vol, smm = to_cache_frame(v, sp_i, smm, plane, cfg, image=True)
                    vol = vol.round_().clamp_(0, 255).to(torch.uint8).numpy()
                    info = dict(plane=plane, depth=len(vol), slice_mm=smm)
                assert vol.shape == labels[plane].shape, (plane, vol.shape, labels[plane].shape)
                series = f'{study}{suffix}_e{echo}'
                np.save(os.path.join(out_dir, series + '.npy'), vol)
                rows.append(dict(StudyInstanceUID=study, SeriesInstanceUID=series, ok=True, error='', split=split,
                                 echo=echo, side=side, seg_file=f'{study}{suffix}_seg.npy', **info))
    except Exception as e:   # noqa: BLE001 (one broken scan must not stop the others)
        rows = [dict(StudyInstanceUID=study, SeriesInstanceUID=f'{study}{suffix}_e{echo}', ok=False,
                     error=repr(e)[:300], split=split, echo=echo, plane=plane) for echo in (1, 2)
                for plane, suffix in SUFFIX.items()]
    shutil.rmtree(work, ignore_errors=True)
    return rows


def prepare(workers=24):
    """The cache split 'skm_tea': for each scan and echo a sagittal series (skm_<scan>_e<echo>, our DICOM preprocessing:
    canonical frame, slices merged to about 2 mm, 0.5 mm pixels, 352 px) and coronal / axial reformats of it
    (skm_<scan>_cor_e<echo>, skm_<scan>_ax_e<echo>: 3.5 mm slabs, as our coronal / axial series), and the tissue labels
    of each plane in the same frame (skm_<scan>[_cor | _ax]_seg.npy, uint8 0-6). The index has the knee side (from the
    medial and lateral menisci) and the label file of each series. Index: index_skm_tea.parquet; series table:
    skm_tea_series.csv (echo 2, the fluid-bright echo, counts as fat-suppressed / fluid-sensitive)."""
    jobs = []
    for split in ('train', 'val', 'test'):
        ann = json.load(open(os.path.join(ROOT, 'annotations', 'v1.0.0', f'{split}.json')))
        jobs += [(im['scan_id'], split) for im in ann['images']]
    with ProcessPoolExecutor(workers) as pool:
        rows = [r for rs in pool.map(_prepare_scan, jobs) for r in rs]
    idx = pd.DataFrame(rows)
    idx.to_parquet(os.path.join(CACHE, f'index_{SPLIT}.parquet'), index=False)
    ok = idx[idx.ok]
    fs = (ok.echo == 2).astype(int)
    pd.DataFrame({'StudyInstanceUID': ok.StudyInstanceUID, 'SeriesInstanceUID': ok.SeriesInstanceUID,
                  'Fluid_Sensitive': fs, 'Fat_Suppression': fs, 'Anatomical_Plane': ok.plane,
                  'SeriesDescription': 'qDESS E' + ok.echo.astype(str)}).to_csv(SERIES_CSV, index=False)
    print(f'{int(idx.ok.sum())} of {len(idx)} series ok;', idx[~idx.ok].error.head(3).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['download', 'prepare'])
    ap.add_argument('--workers', type=int, default=24)
    args = ap.parse_args()
    if args.step == 'download':
        download()
    else:
        prepare(args.workers)


if __name__ == '__main__':
    main()
