"""The label table of ablation D from the label sets that the top teams shared on Kaggle after the competition.

Every set is one row per training study (StudyInstanceUID) with the 12 findings as probabilities. The files live in
data/external_labels/ (not in the repository); docs/labels.md lists the Kaggle dataset of each.

    python -m knee.labels.external_llm

Scores each set and their ensemble on the 58 expert-labelled studies (macro AUC against the expert labels, for the sets
that hold their own reads of these studies), then writes data/labels/targets_kd70f_x.parquet: 0.5 x targets_kd70f (the
predictions of our v70f ensemble, the v76 training labels) + 0.5 x the mean of the sets per (study, finding); a set
without the study is left out of the mean for that study.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from knee.constants import LABELS, LABELS_DIR, REPO
from knee.labels.sources import gold

EXT = os.path.join(REPO, 'data', 'external_labels')
# name: (file, column prefix, source)
SETS = {
    'kangaroo_mix5': ('kangaroo_mix5_weak4349.csv', '', '1st place (liushuzhi/rsna-knee-1st-place-training-supervision): '
                      '0.3 Astra v1 + 0.2 Astra v2 + 0.3 teammate Astra + 0.1 Hulu + 0.1 Qwen Next; 4,349 studies, no gold rows'),
    'kangaroo_astra_guide': ('kangaroo_astra_trained_guide.csv', '', '1st place (tascj0/rsna-2026-knee-agent-labels): GPT-6 Astra '
                             'read the report and the images with a reading guide learned from the 58 gold studies (gold rows optimistic)'),
    'songqizhou_final': ('songqizhou_9_final_label_with_gold58.csv', '', '7th place (songqizhou/rsna-knee-llm-labels file 9): '
                         '0.5 x Astra reads (report + images, 3 reads) + 0.5 x their image teacher; gold rows = their own reads'),
    'chengtingyi_opus2': ('chengtingyi_opus2_text_oa_frac.csv', '', 'chengtingyi/rsna-knee-llm-weak-labels: Claude Opus report '
                          'text + blind image reads (Opus effusion/synovitis, Astra lateral, Opus fracture/contusion) fused'),
    'drizlo': ('drizlo_llm_soft_labels.csv', '', '21st place (drizlo/rsna-knee-mri-llm-soft-labels): multimodal LLM reads'),
    'matsurihime_astra_image': ('matsurihime_llm_readings.csv', 'astra_image__', 'matsurihime/rsna-knee-llm-readings: GPT-6 Astra '
                                'on one montage per series + the report as an unverified reference'),
}


def load(name):
    f, prefix, _ = SETS[name]
    t = pd.read_csv(os.path.join(EXT, f), dtype={'StudyInstanceUID': str})
    t = t.set_index('StudyInstanceUID')[[prefix + c for c in LABELS]]
    t.columns = LABELS
    return t.astype(float)


def lab(name):
    return pd.read_parquet(os.path.join(LABELS_DIR, f'{name}.parquet')).set_index('StudyInstanceUID')[LABELS].astype(float)


def save(df, name):
    df.rename_axis('StudyInstanceUID').reset_index().to_parquet(os.path.join(LABELS_DIR, f'{name}.parquet'), index=False)


def gold_auc(df, ref):
    """Macro AUC on the gold studies the table holds (None when fewer than 40); the per-finding AUCs."""
    j = df.reindex(ref.index)
    ok = j.notna().all(axis=1)
    if ok.sum() < 40:
        return None, {}
    per = {c: roc_auc_score(ref.loc[ok, c], j.loc[ok, c]) for c in LABELS}
    return float(np.mean(list(per.values()))), per


def ensemble(sets):
    stack = pd.concat([s for s in sets.values()], keys=list(sets), names=['set', 'StudyInstanceUID'])
    mean = stack.groupby(level='StudyInstanceUID').mean()
    n = stack[LABELS[0]].groupby(level='StudyInstanceUID').count().rename('n_sets')
    return mean, n


def main():
    argparse.ArgumentParser(description=__doc__.split('\n\n')[0]).parse_args()
    ref = gold()[LABELS].astype(float)
    ref.index = ref.index.astype(str)
    sets = {n: load(n) for n in SETS}
    rows = []
    for n, t in sets.items():
        m, per = gold_auc(t, ref)
        rows.append({'set': n, 'rows': len(t), 'gold rows': int(t.index.isin(ref.index).sum()), 'gold-58 macro': m, **per})
    m, per = gold_auc(lab('targets_cx'), ref)
    rows.append({'set': 'ours targets_cx', 'rows': 4407, 'gold rows': 58, 'gold-58 macro': m, **per})
    ext, n_sets = ensemble(sets)
    gold_sets = [n for n in SETS if sets[n].index.isin(ref.index).sum() >= 40]
    for label, names in [('ensemble', list(SETS)), ('ensemble (gold-row sets only)', gold_sets)]:
        e, _ = ensemble({n: sets[n] for n in names})
        m, per = gold_auc(e, ref)
        rows.append({'set': label, 'rows': len(e), 'gold rows': int(e.index.isin(ref.index).sum()), 'gold-58 macro': m, **per})
    rep = pd.DataFrame(rows).set_index('set')
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)
    print(rep[['rows', 'gold rows', 'gold-58 macro']].to_string(float_format=lambda v: f'{v:.4f}'))
    print(rep[LABELS].to_string(float_format=lambda v: f'{v:.3f}'))
    print('n_sets per study:', n_sets.value_counts().sort_index().to_dict())
    cx, kd = lab('targets_cx'), lab('targets_kd70f')
    ext = ext.reindex(cx.index)
    missing = int(ext.isna().any(axis=1).sum())
    assert missing == 0, f'{missing} training studies without an external label'
    kd_x = 0.5 * kd + 0.5 * ext
    m, _ = gold_auc(kd_x, ref)
    print(f'targets_kd70f_x: gold-58 macro {m:.4f} (the gold rows are reads and predictions, not the expert labels)')
    save(kd_x, 'targets_kd70f_x')
    print(f'-> {LABELS_DIR}/targets_kd70f_x.parquet ({len(ext)} rows)')


if __name__ == '__main__':
    main()
