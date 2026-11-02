"""Make the cross-validation folds of the training studies.

    .venv/bin/python -m knee.dataloading.folds --targets data/labels/targets_v1.parquet --out data/folds_v1.csv

- 5 folds, multilabel-stratified on the binarized targets (> 0.5) plus the scanner manufacturer.
- Studies that share a report text or a PatientID form one group, and a group stays in one fold
  (4,407 studies have only about 4,276 different report texts).
- The 58 expert-labelled studies get fold -1: they are never in training and every fold model predicts them.
Make the folds once and keep them: the pseudo labels need the same folds for every model.
"""
import argparse
import os

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from knee.constants import LABELS, RAW


def groups_of(keys_per_study):
    """Union-find: studies that share any key get the same group id."""
    parent = list(range(len(keys_per_study)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = {}
    for i, keys in enumerate(keys_per_study):
        for k in keys:
            if k in first:
                parent[root(i)] = root(first[k])
            else:
                first[k] = i
    return np.array([root(i) for i in range(len(keys_per_study))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', required=True)
    ap.add_argument('--meta', default='data/meta/train_dicom_meta.parquet')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    t = pd.read_parquet(args.targets)
    meta = pd.read_parquet(args.meta, columns=['StudyInstanceUID', 'Manufacturer', 'PatientID'])
    first = meta.groupby('StudyInstanceUID').first()
    manu = first.Manufacturer.fillna('').str.lower()
    manu = manu.map(lambda m: 'siemens' if 'siemens' in m else 'ge' if m.startswith('ge') else
                    'philips' if 'philips' in m else 'toshiba/canon' if ('toshiba' in m or 'canon' in m) else 'other')
    t['manufacturer'] = t.StudyInstanceUID.map(manu).fillna('other')
    reports = pd.read_csv(os.path.join(RAW, 'train.csv')).set_index('StudyInstanceUID').Report
    t['report_key'] = t.StudyInstanceUID.map(reports.fillna('').str.split().str.join(' ').str.lower())
    t['patient'] = t.StudyInstanceUID.map(first.PatientID)
    t['fold'] = -1

    train = t[~t.is_gold].reset_index(drop=True)
    keys = [[('r', r)] + ([('p', p)] if isinstance(p, str) and p else []) for r, p in zip(train.report_key, train.patient)]
    train['group'] = groups_of(keys)
    g = train.groupby('group').first().reset_index()   # stratify the groups by their first study
    y = (g[LABELS].values > 0.5).astype(int)
    y = np.concatenate([y, pd.get_dummies(g.manufacturer).values.astype(int)], axis=1)
    mskf = MultilabelStratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    gfold = np.zeros(len(g), dtype=int)
    for k, (_, val_idx) in enumerate(mskf.split(g, y)):
        gfold[val_idx] = k
    train['fold'] = train.group.map(dict(zip(g.group, gfold)))
    t.loc[~t.is_gold, 'fold'] = train.fold.values
    print(f'{len(train)} training studies in {train.group.nunique()} groups '
          f'({(train.groupby("group").size() > 1).sum()} groups with more than one study)')
    t[['StudyInstanceUID', 'fold', 'manufacturer']].to_csv(args.out, index=False)
    print(t.groupby('fold').size().to_string())
    print(t[~t.is_gold].groupby('fold')[LABELS].mean().round(3).to_string())
    print(pd.crosstab(t.fold, t.manufacturer).to_string())


if __name__ == '__main__':
    main()
