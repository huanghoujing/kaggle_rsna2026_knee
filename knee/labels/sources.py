"""The expert labels of the 58 gold studies, as a label table: a DataFrame indexed by StudyInstanceUID with the 12
findings as float columns."""
import os

import pandas as pd

from knee.constants import LABELS, RAW


def _std(df):
    df = df.set_index('StudyInstanceUID')
    return df[LABELS].apply(pd.to_numeric, errors='coerce').astype(float)


def gold():
    """The 58 studies with expert labels (train.csv)."""
    tr = pd.read_csv(os.path.join(RAW, 'train.csv'))
    return _std(tr.dropna(subset=LABELS))
