"""A small DICOM header parser for the efficient path. On the Kaggle machine (4 CPUs, 2.0 GHz) pydicom needs 0.7 ms
of CPU for one header and 2.8 ms with the 32 reading threads of a worker (the GIL); reading the first 4 KB of a file
costs 0.19 ms. This parser walks the elements of an explicit VR little endian header (the transfer syntax of all files
in this data) and decodes only the fields the efficient path uses, with pydicom's value rules (DS / IS as float / int,
text without the padding spaces). Anything else (another transfer syntax, a header past the read bytes, a malformed
element) returns None and the caller uses pydicom. Checked against pydicom on the training files:

    .venv/bin/python -m knee.inference.fast_header check [--files 20000]
"""
import struct

EXPLICIT_LE = '1.2.840.10008.1.2.1'
LONG_VRS = {b'OB', b'OD', b'OF', b'OL', b'OV', b'OW', b'SQ', b'SV', b'UC', b'UN', b'UR', b'UT', b'UV'}
UNDEFINED = 0xFFFFFFFF
FIELDS = {   # (group, element) -> (keyword, kind)
    (0x0018, 0x0050): ('SliceThickness', 'ds1'),
    (0x0020, 0x0032): ('ImagePositionPatient', 'ds'),
    (0x0020, 0x0037): ('ImageOrientationPatient', 'ds'),
    (0x0028, 0x0004): ('PhotometricInterpretation', 'text'),
    (0x0028, 0x0010): ('Rows', 'us'),
    (0x0028, 0x0011): ('Columns', 'us'),
    (0x0028, 0x0030): ('PixelSpacing', 'ds'),
    (0x0028, 0x0100): ('BitsAllocated', 'us'),
    (0x0028, 0x0103): ('PixelRepresentation', 'us'),
    (0x0028, 0x1052): ('RescaleIntercept', 'ds1'),
    (0x0028, 0x1053): ('RescaleSlope', 'ds1'),
}
PIXEL_DATA = (0x7FE0, 0x0010)


class _Meta:
    __slots__ = ('TransferSyntaxUID',)

    def __init__(self, ts):
        self.TransferSyntaxUID = ts


class FastHeader:
    """The decoded fields with the access of a pydicom dataset: ds.Rows, ds.get('EchoTime'), 'Rows' in ds."""
    __slots__ = ('_v', 'file_meta')

    def __init__(self, values, ts):
        self._v, self.file_meta = values, _Meta(ts)

    def __getattr__(self, key):
        try:
            return self._v[key]
        except KeyError:
            raise AttributeError(key) from None

    def __contains__(self, key):
        return key in self._v

    def get(self, key, default=None):
        return self._v.get(key, default)


def _decode(kind, raw):
    """pydicom's empty values: '' for text, None for numbers."""
    if kind == 'us':
        return struct.unpack_from('<H', raw)[0] if len(raw) >= 2 else None
    text = raw.decode('latin-1').strip(' \x00')
    if kind == 'text':
        return text
    if not text:
        return None
    parts = [p.strip() for p in text.split('\\')]
    if kind == 'ds1':
        return float(parts[0]) if len(parts) == 1 else [float(p) for p in parts]
    return [float(p) for p in parts]


def _skip_undefined(buf, pos):
    """Position after an undefined-length sequence (its items and the sequence delimiter), or raise."""
    n = len(buf)
    while pos + 8 <= n:
        group, elem, length = struct.unpack_from('<HHI', buf, pos)
        pos += 8
        if (group, elem) == (0xFFFE, 0xE0DD):          # sequence delimitation item
            return pos
        if (group, elem) != (0xFFFE, 0xE000):          # an item
            raise ValueError('unexpected element in a sequence')
        if length != UNDEFINED:
            pos += length
            continue
        while True:                                    # an item of undefined length: its elements, then E00D
            if pos + 8 > n:
                raise ValueError('header past the read bytes')
            g, e = struct.unpack_from('<HH', buf, pos)
            if (g, e) == (0xFFFE, 0xE00D):
                pos += 8
                break
            pos = _skip_element(buf, pos)
    raise ValueError('header past the read bytes')


def _skip_element(buf, pos):
    """Position after one explicit VR little endian element (nested sequences included)."""
    vr = buf[pos + 4:pos + 6]
    if vr in LONG_VRS:
        length = struct.unpack_from('<I', buf, pos + 8)[0]
        pos += 12
    else:
        length = struct.unpack_from('<H', buf, pos + 6)[0]
        pos += 8
    if length == UNDEFINED:
        if vr not in (b'SQ', b'UN'):
            raise ValueError('undefined length outside a sequence')
        return _skip_undefined(buf, pos)
    return pos + length


def parse(buf):
    """(FastHeader, byte offset of the pixel data element) of the first bytes of a file, or None."""
    try:
        if len(buf) < 132 or buf[128:132] != b'DICM':
            return None
        pos, n, ts, values = 132, len(buf), None, {}
        while pos + 8 <= n:
            group, elem = struct.unpack_from('<HH', buf, pos)
            if group != 0x0002:
                break
            vr = buf[pos + 4:pos + 6]
            if vr in LONG_VRS:
                length, start = struct.unpack_from('<I', buf, pos + 8)[0], pos + 12
            else:
                length, start = struct.unpack_from('<H', buf, pos + 6)[0], pos + 8
            if elem == 0x0010:
                ts = buf[start:start + length].decode('latin-1').strip(' \x00')
            pos = start + length
        if ts != EXPLICIT_LE:
            return None
        while pos + 8 <= n:
            group, elem = struct.unpack_from('<HH', buf, pos)
            if (group, elem) == PIXEL_DATA:
                return FastHeader(values, ts), pos
            vr = buf[pos + 4:pos + 6]
            if vr in LONG_VRS:
                if pos + 12 > n:
                    return None
                length, start = struct.unpack_from('<I', buf, pos + 8)[0], pos + 12
            else:
                length, start = struct.unpack_from('<H', buf, pos + 6)[0], pos + 8
            if length == UNDEFINED:
                if vr not in (b'SQ', b'UN'):
                    return None
                pos = _skip_undefined(buf, start)
                continue
            field = FIELDS.get((group, elem))
            if field is not None:
                if start + length > n:
                    return None
                values[field[0]] = _decode(field[1], buf[start:start + length])
            pos = start + length
        return None                                        # no pixel data in the read bytes
    except (ValueError, struct.error, UnicodeDecodeError, IndexError):
        return None


def check(n_files=20000, seed=0):
    """The parser against pydicom on random training files: every field and the pixel data offset."""
    import glob
    import io
    import os
    import random
    import numpy as np
    import pydicom
    from knee.constants import RAW
    files = sorted(glob.glob(os.path.join(RAW, 'train_series', '*', '*', '*.dcm')))
    random.seed(seed)
    files = random.sample(files, min(n_files, len(files)))
    bad, fallback = [], 0
    for f in files:
        with open(f, 'rb') as fh:
            raw = fh.read(4096)
        r = parse(raw)
        bio = io.BytesIO(raw)
        ref = pydicom.dcmread(bio, stop_before_pixels=True)
        if r is None:
            fallback += 1
            continue
        h, off = r
        if off != bio.tell():
            bad.append((f, 'offset', off, bio.tell()))
        for key, _ in FIELDS.values():
            a, b = h.get(key, 'MISSING'), ref.get(key, 'MISSING')
            if isinstance(b, pydicom.multival.MultiValue):
                b = [float(v) for v in b]
            elif isinstance(b, (pydicom.valuerep.DSfloat, float)):
                b = float(b)
            elif isinstance(b, pydicom.valuerep.IS):
                b = int(b)
            elif b is not None and not isinstance(b, (int, str, list)):
                b = str(b)
            if b is None:
                b = ''
            same = (np.allclose(a, b, rtol=0, atol=0) if isinstance(a, list) and isinstance(b, list) and len(a) == len(b)
                    else a == b)
            if not same:
                bad.append((f, key, a, b))
        if str(ref.file_meta.TransferSyntaxUID) != h.file_meta.TransferSyntaxUID:
            bad.append((f, 'TransferSyntaxUID', h.file_meta.TransferSyntaxUID, str(ref.file_meta.TransferSyntaxUID)))
    print(f'{len(files)} files: {len(bad)} differences, {fallback} fell back to pydicom')
    for b in bad[:10]:
        print(b)
    return bad


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['check'])
    ap.add_argument('--files', type=int, default=20000)
    a = ap.parse_args()
    check(a.files)
