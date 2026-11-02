"""Extract a zip file while it is still being downloaded.

The Kaggle zip stores the sizes in each local file header (flag bit 3 is not set), so the entries can be read in order
from the start of the file. The script follows the growing file, extracts each entry when its bytes are complete,
and saves the read offset in a state file. A restart continues at the saved offset.

    python -m knee.dataloading.stream_unzip --zip data/raw/rsna-knee-abnormality-detection.zip --out data/raw/extracted

The script stops when it reads the central directory (the end of the entries).
"""
import argparse
import glob
import json
import os
import struct
import time
import zlib

LOCAL_HEADER = 0x04034B50
CENTRAL_HEADER = 0x02014B50


def read_exact(f, n, path, poll_s, wait_for_growth):
    """Read n bytes. Wait for the file to grow if the bytes are not there yet."""
    start = f.tell()
    while True:
        f.seek(start)
        data = f.read(n)
        if len(data) == n:
            return data
        if not wait_for_growth():
            raise EOFError(f'{path}: file stopped at {start + len(data)} bytes, needed {start + n}')
        time.sleep(poll_s)


def zip64_sizes(extra, csz, usz):
    """Return the real sizes from the ZIP64 extra field when the header holds 0xFFFFFFFF."""
    p = 0
    while p + 4 <= len(extra):
        hid, hsz = struct.unpack('<HH', extra[p:p + 4])
        if hid == 1:
            vals = list(struct.unpack('<' + 'Q' * (hsz // 8), extra[p + 4:p + 4 + hsz]))
            if usz == 0xFFFFFFFF:
                usz = vals.pop(0)
            if csz == 0xFFFFFFFF:
                csz = vals.pop(0)
        p += 4 + hsz
    return csz, usz


def complete_study_dirs(series_root):
    """Return the study folders under series_root whose files are all extracted.

    The zip holds the entries in sorted order, and the state file holds the last entry before the saved offset.
    A study is complete when it comes before the study of that entry, or when the extraction is complete.
    A folder without a state file is taken as a normal, complete extraction.
    """
    studies = sorted(d for d in glob.glob(os.path.join(series_root, '*')) if os.path.isdir(d))
    state_path = os.path.join(os.path.dirname(os.path.normpath(series_root)), '.stream_unzip_state.json')
    if not os.path.exists(state_path):
        return studies
    state = json.load(open(state_path))
    if state.get('complete', False):
        return studies
    last = state.get('last_name', '')  # e.g. train_series/<study>/<series>/<sop>.dcm
    parts = last.split('/')
    root_name = os.path.basename(os.path.normpath(series_root))
    if len(parts) < 2 or parts[0] < root_name:
        return []  # the extraction has not passed this folder yet
    if parts[0] > root_name:
        return studies  # the extraction is past this folder
    return [d for d in studies if os.path.basename(d) < parts[1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--poll_s', type=float, default=5.0)
    ap.add_argument('--done_marker', default=None,
                    help='file that exists when the download is complete (default: <zip>.kaggle-partial is absent)')
    args = ap.parse_args()

    state_path = os.path.join(args.out, '.stream_unzip_state.json')
    os.makedirs(args.out, exist_ok=True)
    state = json.load(open(state_path)) if os.path.exists(state_path) else {'offset': 0, 'n_files': 0}
    partial_marker = args.zip + '.kaggle-partial'

    def download_running():
        return os.path.exists(partial_marker) if args.done_marker is None else not os.path.exists(args.done_marker)

    f = open(args.zip, 'rb')
    offset, n_files, t0, bytes0 = state['offset'], state['n_files'], time.time(), state['offset']
    while True:
        f.seek(offset)
        header = read_exact(f, 30, args.zip, args.poll_s, download_running)
        sig = struct.unpack('<I', header[:4])[0]
        if sig == CENTRAL_HEADER:
            print(f'central directory at offset {offset}: all {n_files} entries extracted', flush=True)
            break
        if sig != LOCAL_HEADER:
            raise ValueError(f'unexpected signature {sig:#x} at offset {offset}')
        _, _, flag, method, _, _, crc, csz, usz, nlen, elen = struct.unpack('<IHHHHHIIIHH', header)
        if flag & 8:
            raise ValueError('entry with a data descriptor: streaming is not supported')
        name = read_exact(f, nlen, args.zip, args.poll_s, download_running).decode('utf-8')
        extra = read_exact(f, elen, args.zip, args.poll_s, download_running)
        csz, usz = zip64_sizes(extra, csz, usz)
        data = read_exact(f, csz, args.zip, args.poll_s, download_running)
        if method == 8:
            data = zlib.decompress(data, -15)
        elif method != 0:
            raise ValueError(f'{name}: compression method {method} is not supported')
        if len(data) != usz or zlib.crc32(data) != crc:
            raise ValueError(f'{name}: size or CRC mismatch')
        dst = os.path.join(args.out, name)
        if not name.endswith('/'):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + '.part'
            with open(tmp, 'wb') as g:
                g.write(data)
            os.replace(tmp, dst)
        offset += 30 + nlen + elen + csz
        n_files += 1
        if n_files % 5000 == 0:
            json.dump({'offset': offset, 'n_files': n_files, 'last_name': name}, open(state_path, 'w'))
            mb_s = (offset - bytes0) / 1e6 / max(time.time() - t0, 1e-6)
            print(f'{n_files} files, offset {offset / 1e9:.1f} GB, {mb_s:.1f} MB/s, last {name}', flush=True)
    json.dump({'offset': offset, 'n_files': n_files, 'last_name': name, 'complete': True}, open(state_path, 'w'))


if __name__ == '__main__':
    main()
