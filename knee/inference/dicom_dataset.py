"""The series table of a DICOM folder at test time: one row per series of the series csv whose folder exists, with
the folder path and the series type (plane and fat suppression) as in training."""
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from knee.constants import PLANES


def test_series_table(series_root, series_csv):
    """One row for each series of the series csv whose folder exists. type uses the csv plane and FS flag."""
    s = pd.read_csv(series_csv)
    s['series_dir'] = [os.path.join(series_root, a, b) for a, b in zip(s.StudyInstanceUID, s.SeriesInstanceUID)]
    with ThreadPoolExecutor(32) as pool:   # the folder checks in parallel: on Kaggle (NFS) each one is a network call
        exists = list(pool.map(os.path.isdir, s.series_dir))
    s = s[exists].copy()
    s['type'] = s.Anatomical_Plane.map({p: i for i, p in enumerate(PLANES)}).fillna(0).astype(int) * 2 \
        + s.Fat_Suppression.fillna(0).astype(int)
    return s
