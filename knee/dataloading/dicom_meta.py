"""Scan DICOM headers (no pixel data) into one table with one row for each slice.

    python -m knee.dataloading.dicom_meta --root data/raw/extracted/train_series --out data/meta/train_dicom_meta.parquet

The scan is incremental: series that are already in the output table are skipped. Use --workers to set the number
of processes (default: one third of the CPU threads).
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pydicom
from tqdm import tqdm

from knee.dataloading.dicom_io import dicom_paths
from knee.dataloading.stream_unzip import complete_study_dirs

# Tags that we keep. Multi-value tags are stored as tuples of float.
TAGS = [
    'SOPInstanceUID', 'InstanceNumber', 'SeriesNumber', 'AcquisitionNumber', 'SeriesDescription', 'ImageType',
    'Manufacturer', 'ManufacturerModelName', 'MagneticFieldStrength', 'PatientID', 'PatientSex', 'PatientPosition',
    'BodyPartExamined', 'Laterality', 'ScanningSequence', 'SequenceVariant', 'ScanOptions', 'SequenceName',
    'MRAcquisitionType', 'RepetitionTime', 'EchoTime', 'InversionTime', 'FlipAngle', 'EchoTrainLength',
    'SliceThickness', 'SpacingBetweenSlices', 'SliceLocation', 'ImagePositionPatient', 'ImageOrientationPatient',
    'PixelSpacing', 'Rows', 'Columns', 'BitsStored', 'PhotometricInterpretation', 'WindowCenter', 'WindowWidth',
    'RescaleIntercept', 'RescaleSlope', 'FrameOfReferenceUID', 'ContrastBolusAgent', 'ReceiveCoilName',
]


def _value(v):
    if v is None:
        return None
    if isinstance(v, (pydicom.multival.MultiValue, list, tuple)):
        try:
            return tuple(float(x) for x in v)
        except (TypeError, ValueError):
            return '\\'.join(str(x) for x in v)
    if isinstance(v, (pydicom.valuerep.DSfloat, pydicom.valuerep.IS, float, int)):
        return float(v)
    return str(v)


def scan_series(series_dir):
    """Read the headers of all slices of one series. Return a list of dicts."""
    rows = []
    study_uid = os.path.basename(os.path.dirname(series_dir))
    series_uid = os.path.basename(series_dir)
    for path in dicom_paths(series_dir):
        row = {'StudyInstanceUID': study_uid, 'SeriesInstanceUID': series_uid, 'path': path}
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            for tag in TAGS:
                row[tag] = _value(ds.get(tag, None))
            row['TransferSyntaxUID'] = str(ds.file_meta.get('TransferSyntaxUID', ''))
            row['read_error'] = ''
        except Exception as e:  # keep the slice in the table so that the error is visible
            row['read_error'] = f'{type(e).__name__}: {e}'
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='folder with <StudyInstanceUID>/<SeriesInstanceUID>/*.dcm')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() // 3))
    args = ap.parse_args()

    series_dirs = [d for s in complete_study_dirs(args.root) for d in sorted(glob.glob(os.path.join(s, '*')))]
    old = pd.read_parquet(args.out) if os.path.exists(args.out) else None
    if old is not None:
        done = set(old['SeriesInstanceUID'])
        series_dirs = [d for d in series_dirs if os.path.basename(d) not in done]
    print(f'{len(series_dirs)} series to scan', flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in tqdm(ex.map(scan_series, series_dirs, chunksize=4), total=len(series_dirs), mininterval=10):
            rows.extend(r)
    new = pd.DataFrame(rows)
    for col in new.columns:  # parquet needs one type per column; tuples of mixed length are stored as text
        if new[col].map(lambda x: isinstance(x, tuple)).any():
            new[col] = new[col].map(lambda x: '\\'.join(map(str, x)) if isinstance(x, tuple) else x)
    table = new if old is None else pd.concat([old, new], ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    table.to_parquet(args.out, index=False)
    print(f'{len(table)} slices, {table.SeriesInstanceUID.nunique()} series -> {args.out}')


if __name__ == '__main__':
    main()
