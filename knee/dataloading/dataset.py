"""Study-level dataset: the cached series of one study, K slice positions per series, 3-slice windows as channels.

One item = one study:
    images  uint8  [S, K, 3, H, W]   S series (S <= max_series), K positions, channels = slices (k-1, k, k+1)
    types   int64  [S]               series type: plane * 2 + fat-suppressed (0..5), see TYPE_NAMES
    pos     float  [S, K]            relative position of the sampled slice in the series (0..1)
    labels  float  [12]              soft targets (NaN for unlabeled test studies)
    geom    float  [S, 6]            with geometry: image center (LPS mm), first / last slice position (mm), plane
    joint   float  [6]               with joint_boxes: the joint box of the study (lo, hi; LPS mm)
The GPU does the augmentation and the normalization (knee.dataloading.gpu_augment).
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from knee.constants import LABELS, META, PLANES, RAW


def series_csv(split):
    """The competition series table (plane, fat suppression) of a cache split."""
    return os.path.join(RAW, f'{split}_series.csv')


def series_table(cache_root, split, series_csv):
    """Join the cache index with the competition series table. One row for each cached series."""
    idx = pd.read_parquet(os.path.join(cache_root, f'index_{split}.parquet'))
    idx = idx[idx.ok]
    s = pd.read_csv(series_csv)[['SeriesInstanceUID', 'Fluid_Sensitive', 'Fat_Suppression', 'Anatomical_Plane']]
    t = idx.merge(s, on='SeriesInstanceUID', how='left')
    plane = t['plane'].fillna(t['Anatomical_Plane'])
    t['type'] = plane.map({p: i for i, p in enumerate(PLANES)}) * 2 + t['Fat_Suppression'].fillna(0).astype(int)
    t['path'] = [os.path.join(cache_root, split, a, b + '.npy') for a, b in zip(t.StudyInstanceUID, t.SeriesInstanceUID)]
    geometry = os.path.join(META, f'series_geometry_{split}.parquet')   # image centers (knee.dataloading.series_geometry)
    if os.path.exists(geometry):
        t = t.merge(pd.read_parquet(geometry), on='SeriesInstanceUID', how='left')
    return t


TYPE_NAMES = [f'{p}_FS{f}' for p in PLANES for f in (0, 1)]   # series type = plane index * 2 + fat sat
JOINT_HALF_MM = np.array([37.0, 36.0, 28.0])   # half the median joint box (data/meta/joint_box_train.parquet)


def load_joint_boxes(split):
    """{study: [6] lo, hi (LPS mm)} of knee.dataloading.tissue_rois --joint_boxes."""
    t = pd.read_parquet(os.path.join(META, f'joint_box_{split}.parquet'))
    cols = [f'{a}_{c}' for a in ('lo', 'hi') for c in 'xyz']
    return dict(zip(t.StudyInstanceUID, t[cols].to_numpy(np.float32)))


class StudyDataset(Dataset):
    def __init__(self, studies, series, targets=None, k_slices=16, max_series=8, coverage=(0.05, 0.95),
                 train=False, series_dropout=0.0, coverage_jitter=0.05, seed=0, geometry=False, joint_boxes=None):
        self.studies = list(studies)
        self.by_study = {k: g.sort_values(['type', 'SeriesInstanceUID']) for k, g in series.groupby('StudyInstanceUID')}
        self.targets = targets.set_index('StudyInstanceUID')[LABELS] if targets is not None else None
        self.k, self.max_series, self.coverage = k_slices, max_series, coverage
        self.train, self.series_dropout, self.coverage_jitter = train, series_dropout, coverage_jitter
        self.seed = seed
        self.geometry = geometry   # items carry the series geometry (the token positions in patient space)
        self.joint_boxes = joint_boxes   # {study: [6] lo, hi (LPS mm)}: items carry the joint box (joint transformer)

    def __len__(self):
        return len(self.studies)

    def _select_series(self, g, rng):
        rows = list(g.itertuples())
        if self.train and self.series_dropout > 0 and len(rows) > 1:
            keep = [r for r in rows if rng.random() >= self.series_dropout]
            rows = keep or [rows[rng.integers(len(rows))]]
        if len(rows) <= self.max_series:
            return rows
        # keep one series of each type first, then fill up (random in training, first ones otherwise)
        first, rest, seen = [], [], set()
        order = rng.permutation(len(rows)) if self.train else range(len(rows))
        for i in order:
            (rest if rows[i].type in seen else first).append(rows[i])
            seen.add(rows[i].type)
        return sorted((first + rest)[:self.max_series], key=lambda r: r.type)

    def _positions(self, depth, rng):
        lo, hi = self.coverage
        if self.train:
            lo += rng.uniform(-self.coverage_jitter, self.coverage_jitter)
            hi += rng.uniform(-self.coverage_jitter, self.coverage_jitter)
        rel = np.linspace(lo, hi, self.k)
        if self.train:  # jitter each position by up to half the spacing between positions
            rel = rel + rng.uniform(-0.5, 0.5, self.k) * (hi - lo) / max(self.k - 1, 1)
        rel = np.clip(rel, 0.0, 1.0)
        return rel, np.clip(np.round(rel * (depth - 1)).astype(int), 0, depth - 1)

    def series_input(self, vol, rng):
        """K windows of 3 adjacent slices: uint8 [K, 3, H, W] and the relative slice positions [K]."""
        rel, k = self._positions(vol.shape[0], rng)
        win = np.stack([np.clip(k - 1, 0, vol.shape[0] - 1), k, np.clip(k + 1, 0, vol.shape[0] - 1)], 1)  # [K, 3]
        return np.asarray(vol[win.reshape(-1)]).reshape(self.k, 3, *vol.shape[1:]), rel

    def joint_box(self, study):
        """The study's joint box (lo, hi; LPS mm), else a box of the median size around its series centers."""
        b = self.joint_boxes.get(study)
        if b is None:
            c = self.by_study[study][['gx', 'gy', 'gz']].mean().to_numpy(float)
            b = np.concatenate([c - JOINT_HALF_MM, c + JOINT_HALF_MM])
        return torch.tensor(np.asarray(b, dtype=np.float32))

    def read_volume(self, row):
        """uint8 [D, H, W] volume of one series (the cache; knee.inference reads the DICOM files instead)."""
        return np.load(row.path, mmap_mode='r')

    def __getitem__(self, i):
        study = self.studies[i]
        rng = np.random.default_rng() if self.train else np.random.default_rng(self.seed + i)
        rows = self._select_series(self.by_study[study], rng)
        images, types, pos = [], [], []
        for r in rows:
            img, rel = self.series_input(self.read_volume(r), rng)
            images.append(img)
            types.append(r.type)
            pos.append(rel)
        y = (self.targets.loc[study].values.astype(np.float32) if self.targets is not None and study in self.targets.index
             else np.full(len(LABELS), np.nan, np.float32))
        return {'images': torch.from_numpy(np.stack(images)), 'types': torch.tensor(types, dtype=torch.long),
                'pos': torch.from_numpy(np.stack(pos).astype(np.float32)), 'labels': torch.from_numpy(y),
                'study': study,
                **({'geom': torch.tensor([[r.gx, r.gy, r.gz, r.first_mm, r.last_mm, r.type // 2] for r in rows],
                                         dtype=torch.float32)} if self.geometry else {}),
                **({'joint': self.joint_box(study)} if self.joint_boxes is not None else {})}


def collate(items):
    """Concatenate the series of all studies. study_idx maps each series to its study in the batch. flip: the series
    of the study are shown mirrored (a left knee as a right knee); False unless an item sets it."""
    return {
        'images': torch.cat([it['images'] for it in items]),
        'types': torch.cat([it['types'] for it in items]),
        'pos': torch.cat([it['pos'] for it in items]),
        'study_idx': torch.cat([torch.full((len(it['types']),), b, dtype=torch.long) for b, it in enumerate(items)]),
        'labels': torch.stack([it['labels'] for it in items]),
        'flip': torch.stack([it.get('flip', torch.tensor(False)) for it in items]),
        **({'geom': torch.cat([it['geom'] for it in items])} if all('geom' in it for it in items) else {}),
        **({'joint': torch.stack([it['joint'] for it in items])} if all('joint' in it for it in items) else {}),
        'study': [it['study'] for it in items],
    }
