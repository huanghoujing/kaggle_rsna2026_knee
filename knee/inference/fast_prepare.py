"""The full read of one DICOM file, the fallback of the partial reads (knee.inference.partial_reader): a file that is
not explicit VR little endian with the pixel data at a known offset is read whole, header and pixels from the same
bytes (compressed transfer syntaxes use pydicom with pylibjpeg / gdcm), and its values are rescaled as in the cache.
"""
import io

import numpy as np
import pydicom

UNCOMPRESSED = {'1.2.840.10008.1.2', '1.2.840.10008.1.2.1'}
DTYPES = {(8, 0): np.uint8, (8, 1): np.int8, (16, 0): np.uint16, (16, 1): np.int16}   # (bits, signed)


def _read_one(path):
    """The 2D float32 pixels of one file (rescaled as the cache), or None if it cannot be read."""
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        ds = pydicom.dcmread(io.BytesIO(raw))
        if 'ImageOrientationPatient' not in ds or 'ImagePositionPatient' not in ds:
            return None
        rows, cols = int(ds.Rows), int(ds.Columns)
        if str(ds.file_meta.TransferSyntaxUID) in UNCOMPRESSED and int(ds.BitsAllocated) in (8, 16):
            dtype = DTYPES[(int(ds.BitsAllocated), int(ds.PixelRepresentation))]
            img = np.frombuffer(ds.PixelData, dtype=dtype, count=rows * cols).reshape(rows, cols)
        else:
            img = ds.pixel_array.reshape(rows, cols)
        return rescale(ds, img)
    except Exception:
        return None


def rescale(ds, img, keep_int=False):
    """Stored values -> float32 image values: rescale slope / intercept, MONOCHROME1 inverted (as the cache). With
    keep_int, the stored integers themselves when they are the image values (the caller converts them: exact)."""
    slope, inter = float(ds.get('RescaleSlope', 1) or 1), float(ds.get('RescaleIntercept', 0) or 0)
    mono1 = str(ds.get('PhotometricInterpretation', '')) == 'MONOCHROME1'
    if keep_int and slope == 1 and inter == 0 and not mono1:
        return img
    img = img.astype(np.float32)
    if slope != 1 or inter != 0:   # the identity in 99.6 % of the series: skip two passes over the image
        img = img * slope + inter
    if mono1:
        img = img.max() - img
    return img
