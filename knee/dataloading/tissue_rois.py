"""The knee compartments in every series of our studies, found by the tissue segmentation model, and the boxes made
from them: the study box of each compartment and the knee joint box of each study.

    python -m knee.dataloading.tissue_rois --split train [--limit N] [--model F]   # data/meta/tissue_rois_train.parquet
    python -m knee.dataloading.tissue_rois --split train --merge         # data/meta/tissue_rois_train_study.parquet
    python -m knee.dataloading.tissue_rois --split train --joint_boxes   # data/meta/joint_box_train.parquet

The tissue segmentation model of all 3 planes (configs/seg_skm3_r50rad_288.yaml, save_model) labels every slice of
every series in the cache, with the evaluation preprocessing of the classification model (crop 160 mm, 288 px, center
slice, left knees mirrored). A compartment is where its tissues are: medial tibiofemoral = medial tibial cartilage +
medial meniscus; lateral tibiofemoral = lateral tibial cartilage + lateral meniscus; patellofemoral = patellar cartilage
(the trochlea lies just behind it). One row per series and found compartment (at least MIN_PX pixels on a slice):
- z_lo, z_hi: the first / last slice with at least MIN_PX pixels; z_peak: the slice with the most pixels (cache slice
  index),
- u, v: the center in the image plane (median of the pixels on these slices; mm from the image center along the
  columns / rows, in the mirrored frame that GpuAugment produces), u_lo ... v_hi: the 5 / 95 % pixel extent,
- n_px: the pixel count over these slices.

--merge: the series of a study share the patient coordinates, so each found compartment becomes a box in patient space
(LPS mm; the true, possibly oblique axes of each series: knee.dataloading.series_geometry). The study box is the median
over the fat-suppressed series that found it (all series that found it when no fat-suppressed series did; fat-suppressed
series find the tissues more reliably). It is projected back into every series of the study, also those where the model
did not find the compartment (many non-fat-suppressed series). Columns as above, plus n_found (series in the median),
own (the series found it itself) and dist_mm (distance between its own center and the study center).
--joint_boxes: the knee joint box of each study = the union of its medial, lateral and patellofemoral study boxes
(patient LPS mm, lo_x ... hi_z): the region of the joint transformer.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from knee.constants import META, REPO
from knee.dataloading.canonical import AXES
from knee.dataloading.dataset import series_csv, series_table
from knee.dataloading.gpu_augment import GpuAugment
from knee.training.seg_pretrain import TISSUES, SegNet

ROIS = {'medial': (3, 5), 'lateral': (4, 6), 'pf': (1,)}   # compartment -> tissue labels (knee.dataloading.skm_tea)
MIN_PX = 20     # pixels of 0.56 mm on a slice (about 6 mm2)
MODEL = os.path.join(REPO, 'data/models/skm_seg3_r50rad_288.pt')


class Windows(Dataset):
    """All slices of a series as windows (z-1, z, z+1): uint8 [D, 3, H, W]."""

    def __init__(self, series):
        self.rows = list(series.itertuples())

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        v = np.load(self.rows[i].path)
        z = np.arange(len(v))
        win = np.stack([np.clip(z - 1, 0, len(v) - 1), z, np.clip(z + 1, 0, len(v) - 1)], 1)
        return torch.from_numpy(v[win]), i


@torch.no_grad()
def tissue_maps(model, aug, img, series_type, left, chunk=128):
    """uint8 windows [D, 3, H, W] of one series -> tissue labels [D, out_px, out_px] (GPU) in the mirrored frame.
    series_type: plane * 2 + FS; left: the study is a left knee (mirrored)."""
    d = len(img)
    batch = {'images': img[:, None], 'types': torch.full((d,), series_type), 'pos': torch.zeros(d, 1),
             'study_idx': torch.arange(d), 'flip': torch.full((d,), left)}   # each slice its own study: K = 1
    x, _ = aug(batch, train=False)
    out = []
    for s in range(0, d, chunk):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(x[s:s + chunk, 0].contiguous(memory_format=torch.channels_last))
        out.append(logits.argmax(1).to(torch.uint8))
    return torch.cat(out)


def compartments(seg, half_mm):
    """Tissue labels [D, P, P] -> one dict per found compartment (see the module text)."""
    p = seg.shape[-1]
    mm = (torch.arange(p, device=seg.device, dtype=torch.float32) + 0.5) * (2 * half_mm / p) - half_mm
    q = torch.tensor([0.05, 0.5, 0.95], device=seg.device)
    out = []
    for name, tissues in ROIS.items():
        m = torch.isin(seg, torch.tensor(tissues, device=seg.device))
        count = m.flatten(1).sum(1)
        ok = torch.nonzero(count >= MIN_PX).flatten()
        if len(ok) == 0:
            continue
        keep = torch.zeros_like(count, dtype=torch.bool)
        keep[ok] = True
        _, ii, jj = torch.nonzero(m & keep[:, None, None], as_tuple=True)
        u, v = torch.quantile(mm[jj], q).tolist(), torch.quantile(mm[ii], q).tolist()
        out.append(dict(roi=name, n_px=int(count[ok].sum()), z_lo=int(ok.min()), z_hi=int(ok.max()),
                        z_peak=int(count.argmax()), u=u[1], v=v[1], u_lo=u[0], u_hi=u[2], v_lo=v[0], v_hi=v[2]))
    return out


def series_frames(split):
    """Per series of the cache: study, plane, depth, image center c (gx, gy, gz), canonical axes A [3, 3] (column, row,
    slice; series_geometry) and the slice spacing along the slice axis (the index spacing is along the ideal axis)."""
    idx = pd.read_parquet(os.path.join(REPO, f'data/cache/v1/index_{split}.parquet'))
    idx = idx[idx.ok][['StudyInstanceUID', 'SeriesInstanceUID', 'plane', 'depth', 'slice_mm']]
    geo = pd.read_parquet(os.path.join(META, f'series_geometry_{split}.parquet'))
    t = idx.merge(geo, on='SeriesInstanceUID')
    a = t[[f'ax_{x}{c}' for x in 'crs' for c in 'xyz']].to_numpy().reshape(-1, 3, 3)
    ideal = np.stack([np.asarray(AXES[p][2], dtype=float) for p in t.plane])
    t['spacing'] = t.slice_mm / np.abs((a[:, 2] * ideal).sum(1)).clip(0.5)
    return t, a


def box_corners(lo, hi):
    """Corners [8, 3] of the box lo-hi (3 axes)."""
    return np.array([[(lo, hi)[b >> k & 1][k] for k in range(3)] for b in range(8)], dtype=float)


class Frame:
    """Geometry of one series in the cache frame: image center c (patient LPS mm), canonical axes A (rows: column, row
    and slice directions; series_geometry.canonical_axes), slice spacing along A[2], cache depth, and the column
    mirror that GpuAugment applies (left knee, coronal / axial series). Local coordinates: mm along the columns (in the
    mirrored frame) and rows from the image center, and along the slices from the middle slice."""

    def __init__(self, c, axes, spacing, depth, mirror):
        self.c, self.A = np.asarray(c, dtype=float), np.asarray(axes, dtype=float)
        self.sp, self.d, self.mirror = float(spacing), int(depth), bool(mirror)
        self.half = (self.d - 1) / 2

    def to_patient(self, u, v, s):
        u = -np.asarray(u, dtype=float) if self.mirror else np.asarray(u, dtype=float)
        return self.c + np.outer(u, self.A[0]) + np.outer(v, self.A[1]) + np.outer(s, self.A[2])

    def to_local(self, p):
        loc = (np.atleast_2d(p) - self.c) @ self.A.T
        if self.mirror:
            loc[:, 0] = -loc[:, 0]
        return loc


def found_box(frame, r):
    """A compartment found in one series (u, v, u_lo ... v_hi in mm; z_lo, z_hi slices) -> (center, lo, hi) in patient
    space (lo / hi: the axis-aligned box around the corners)."""
    lo = np.array([r['u_lo'], r['v_lo'], (r['z_lo'] - frame.half - 0.5) * frame.sp])
    hi = np.array([r['u_hi'], r['v_hi'], (r['z_hi'] - frame.half + 0.5) * frame.sp])
    corners = frame.to_patient(*box_corners(lo, hi).T)
    center = frame.to_patient([r['u']], [r['v']], [((r['z_lo'] + r['z_hi']) / 2 - frame.half) * frame.sp])[0]
    return center, corners.min(0), corners.max(0)


def study_box(boxes):
    """[(center, lo, hi, fs)] of one compartment in one study -> the median (center, lo, hi) over the FS series that
    found it (all of them when no FS series did; FS series find the tissues more reliably)."""
    use = [b for b in boxes if b[3]] or boxes
    return tuple(np.median(np.stack([b[j] for b in use]), 0) for j in range(3))


def project(frame, center, lo, hi):
    """A study box in one series -> dict(z_lo, z_hi, u, v, u_lo, u_hi, v_lo, v_hi), or None when the box lies outside
    the slices or its center outside the crop field."""
    u, v, _ = frame.to_local(center)[0]
    loc = frame.to_local(box_corners(lo, hi))
    z = loc[:, 2] / frame.sp + frame.half
    if z.max() < 0 or z.min() > frame.d - 1 or max(abs(u), abs(v)) > 80:
        return None
    return dict(z_lo=int(np.clip(np.floor(z.min() + 0.5), 0, frame.d - 1)),
                z_hi=int(np.clip(np.ceil(z.max() - 0.5), 0, frame.d - 1)), u=float(u), v=float(v),
                u_lo=float(loc[:, 0].min()), u_hi=float(loc[:, 0].max()), v_lo=float(loc[:, 1].min()),
                v_hi=float(loc[:, 1].max()))


def study_boxes(split):
    """The study box of every found compartment (patient space; from the per-series table tissue_rois_<split>.parquet)
    and the frame of every series:
    (boxes {(study, roi): (center, lo, hi, n_used, own {series: own center})}, frames {series: Frame},
    series of each study {study: [series]})."""
    rois = pd.read_parquet(os.path.join(META, f'tissue_rois_{split}.parquet'))
    frames, axes = series_frames(split)
    side = pd.read_parquet(os.path.join(META, f'laterality_{split}.parquet')).set_index('StudyInstanceUID').side
    mirror = (frames.StudyInstanceUID.map(side) == 'L').to_numpy() & (frames.plane != 'Sagittal').to_numpy()
    fr = {sid: Frame(frames[['gx', 'gy', 'gz']].iloc[i].to_numpy(float), axes[i], frames.spacing.iat[i],
                     frames.depth.iat[i], mirror[i]) for i, sid in enumerate(frames.SeriesInstanceUID)}
    by_study = frames.groupby('StudyInstanceUID').SeriesInstanceUID.apply(list).to_dict()
    found = {}   # (study, compartment) -> [(center, lo, hi, fs, series)]
    for r in rois.to_dict('records'):
        f = fr.get(r['SeriesInstanceUID'])
        if f is not None:
            found.setdefault((r['StudyInstanceUID'], r['roi']), []).append(
                (*found_box(f, r), r['type'] % 2 == 1, r['SeriesInstanceUID']))
    boxes = {}
    for key, fb in found.items():
        center, lo, hi = study_box([b[:4] for b in fb])
        boxes[key] = (center, lo, hi, sum(b[3] for b in fb) or len(fb), {b[4]: b[0] for b in fb})
    return boxes, fr, by_study


def merge_study(split):
    boxes, fr, by_study = study_boxes(split)
    rows = []
    for (study, roi), (center, lo, hi, n_used, own) in boxes.items():
        for sid in by_study[study]:
            p = project(fr[sid], center, lo, hi)
            if p is None:
                continue
            rows.append(dict(StudyInstanceUID=study, SeriesInstanceUID=sid, depth=fr[sid].d, roi=roi, **p,
                             n_found=n_used, own=sid in own,
                             dist_mm=float(np.linalg.norm(own[sid] - center)) if sid in own else np.nan))
    out = pd.DataFrame(rows)
    st = series_table(os.path.join(REPO, 'data/cache/v1'), split, series_csv(split))
    out.insert(2, 'type', out.SeriesInstanceUID.map(st.set_index('SeriesInstanceUID').type).astype(int))
    path = os.path.join(META, f'tissue_rois_{split}_study.parquet')
    out.to_parquet(path, index=False)
    n_type = st.groupby('type').size().reindex(range(6))
    print(f'{len(out)} compartments in {out.SeriesInstanceUID.nunique()} of {len(st)} series -> {path}')
    print('share of series with the compartment, by series type (plane * 2 + FS), own + merged:')
    print(out.groupby(['type', 'roi']).size().unstack().reindex(range(6)).div(n_type, axis=0).round(2).to_string())
    d = out[out.own & (out.n_found > 1)]
    print('distance (mm) between the own center and the study center, series that found it (studies with 2+ series):')
    print(d.groupby('type').dist_mm.describe(percentiles=[.5, .9]).round(1).to_string())


def joint_boxes(split):
    """The knee joint of each study in patient space: the union of its medial, lateral and patellofemoral study boxes
    (study_boxes; LPS mm, no knee side: the frames undo the mirror) -> joint_box_<split>.parquet (lo_x ... hi_z)."""
    boxes, _, _ = study_boxes(split)
    rows = {}
    for (study, roi), (center, lo, hi, n_used, own) in boxes.items():
        a, b = rows.get(study, (lo, hi))
        rows[study] = (np.minimum(a, lo), np.maximum(b, hi))
    out = pd.DataFrame([{'StudyInstanceUID': st, **{f'lo_{c}': lo[i] for i, c in enumerate('xyz')},
                         **{f'hi_{c}': hi[i] for i, c in enumerate('xyz')}} for st, (lo, hi) in rows.items()])
    path = os.path.join(META, f'joint_box_{split}.parquet')
    out.to_parquet(path, index=False)
    size = out[[f'hi_{c}' for c in 'xyz']].to_numpy() - out[[f'lo_{c}' for c in 'xyz']].to_numpy()
    print(f'{len(out)} studies -> {path}; size x / y / z median {np.median(size, 0).round(1)} mm')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    ap.add_argument('--model', default=MODEL, help='the tissue segmentation model (seg_pretrain save_model)')
    ap.add_argument('--limit', type=int, default=0, help='first n series only (checks)')
    ap.add_argument('--merge', action='store_true', help='study boxes from the per-series table (no model)')
    ap.add_argument('--joint_boxes', action='store_true', help='the knee joint box of each study in patient space')
    args = ap.parse_args()
    if args.joint_boxes:
        return joint_boxes(args.split)
    if args.merge:
        return merge_study(args.split)
    st = torch.load(args.model, map_location='cpu', weights_only=False)
    cfg, m = st['cfg'], st['cfg']['model']
    model = SegNet(m['backbone'], None, m['drop_path'], m['decoder_dim'], len(TISSUES) + 1)
    model.load_state_dict(st['model'])
    model = model.cuda().eval().to(memory_format=torch.channels_last)
    a = cfg['input']['aug']
    aug = GpuAugment(mean=model.encoder.pretrained_cfg['mean'], std=model.encoder.pretrained_cfg['std'], **a)
    series = series_table(os.path.join(REPO, 'data/cache/v1'), args.split, series_csv(args.split))
    if args.limit:
        series = series.head(args.limit)
    side = pd.read_parquet(os.path.join(META, f'laterality_{args.split}.parquet')).set_index('StudyInstanceUID').side
    rows = []
    loader = DataLoader(Windows(series), batch_size=None, num_workers=16, prefetch_factor=4)
    for n, (img, i) in enumerate(loader):
        r = series.iloc[i]
        seg = tissue_maps(model, aug, img, int(r.type), side.get(r.StudyInstanceUID) == 'L')
        for c in compartments(seg, a['crop_mm'] / 2):
            rows.append(dict(StudyInstanceUID=r.StudyInstanceUID, SeriesInstanceUID=r.SeriesInstanceUID,
                             type=int(r.type), depth=len(img), **c))
        if n % 2000 == 0:
            print(f'{n}/{len(series)} series', flush=True)
    out = pd.DataFrame(rows)
    path = os.path.join(META, f'tissue_rois_{args.split}.parquet')
    out.to_parquet(path, index=False)
    found = out.groupby(['type', 'roi']).size().unstack().reindex(range(6))
    n_type = series.groupby('type').size().reindex(range(6))
    print(f'{len(out)} compartments in {out.SeriesInstanceUID.nunique()} of {len(series)} series -> {path}')
    print('share of series with the compartment, by series type (plane * 2 + FS):')
    print(found.div(n_type, axis=0).round(2).to_string())


if __name__ == '__main__':
    main()
