"""The left or right knee of each training study: the target of the orientation head (docs/knee_side.md).

    .venv/bin/python -m knee.dataloading.laterality --out data/meta/laterality_train.parquet \
        --cnn_pred data/models/laterality_r18_train_pred.parquet

Sources, in this order: the DICOM Laterality tag, the series descriptions (LT / RT / LEFT / RIGHT), the side read
from the radiology report, and for the rest the image classifier (knee.model.laterality_cnn, --cnn_pred) or, without
it, the knee position on the patient's left-right axis (x > -59 mm -> left; 98-99 % right for Siemens and Philips
scanners, unreliable for GE). The report sides come from an LLM table (--llm, column `laterality` = left / right) or
from an existing side table (--report_sides, default the output path when it exists: its rows with source 'report'),
because the LLM reading of the reports is not part of this repository. Output columns: StudyInstanceUID, side ('L' /
'R'), source ('tag', 'desc', 'report', 'cnn' or 'xpos').
"""
import argparse
import os

import pandas as pd

X_THRESHOLD_MM = -59.0


def tag_side(values):
    v = {str(s).strip().upper()[:1] for s in values} - {'', 'N'}
    return v.pop() if len(v) == 1 and next(iter(v)) in 'LR' else None


def desc_side(text):
    s = f' {text.upper()} '
    left = any(k in s for k in [' LT', 'LT_', '_LT', 'LEFT', ' L ', '_L_', 'KNEE L'])
    right = any(k in s for k in [' RT', 'RT_', '_RT', 'RIGHT', ' R ', '_R_', 'KNEE R'])
    return 'L' if left and not right else 'R' if right and not left else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', default='data/meta/train_dicom_meta.parquet')
    ap.add_argument('--cache_index', default='data/cache/v1/index_train.parquet')
    ap.add_argument('--llm', default=None, help='LLM report table with a laterality column (left / right)')
    ap.add_argument('--report_sides', default=None, help="a side table whose rows with source 'report' are reused "
                    "(default: the --out file when it exists)")
    ap.add_argument('--out', required=True)
    ap.add_argument('--cnn_pred', default=None, help='laterality_cnn predictions (column pred) instead of the position rule')
    args = ap.parse_args()
    m = pd.read_parquet(args.meta, columns=['StudyInstanceUID', 'Laterality', 'SeriesDescription'])
    g = m.groupby('StudyInstanceUID')
    df = pd.DataFrame({'tag': g.Laterality.agg(tag_side),
                       'desc': g.SeriesDescription.agg(lambda x: desc_side(' '.join(map(str, x))))})
    report_sides = args.report_sides or (args.out if os.path.exists(args.out) else None)
    if args.llm:
        llm = pd.read_parquet(args.llm).set_index('StudyInstanceUID').laterality.map({'left': 'L', 'right': 'R'})
        df['report'] = llm.reindex(df.index)
    elif report_sides:
        prev = pd.read_parquet(report_sides).set_index('StudyInstanceUID')
        df['report'] = prev.side[prev.source == 'report'].reindex(df.index)
        print(f'report sides: {int(df.report.notna().sum())} studies from {report_sides}')
    else:
        df['report'] = None
        print('no report sides (--llm or --report_sides): the tag, the descriptions and the classifier only')
    cx = pd.read_parquet(args.cache_index).groupby('StudyInstanceUID').center_x.median()
    df['xpos'] = (cx.reindex(df.index) > X_THRESHOLD_MM).map({True: 'L', False: 'R'})
    last = 'xpos'
    if args.cnn_pred:
        df['cnn'] = pd.read_parquet(args.cnn_pred).set_index('StudyInstanceUID').pred.reindex(df.index)
        last = 'cnn'
    df['side'], df['source'] = None, None
    for src in ['tag', 'desc', 'report', last]:
        todo = df.side.isna() & df[src].notna()
        df.loc[todo, 'side'] = df.loc[todo, src]
        df.loc[todo, 'source'] = src
    out = df[['side', 'source']].reset_index()
    out.to_parquet(args.out, index=False)
    print(out.source.value_counts().to_string()); print(out.side.value_counts().to_string())


if __name__ == '__main__':
    main()
