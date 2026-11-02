"""Partial DICOM reading for the efficiency track (models with the center-slice input, 'channels: center').

For each series: read only the first bytes of every file (the header ends before 2 KB in this data) to sort the
slices, then read the pixel data of the K slices at the window positions only. The intensity percentiles come from
those K slices (the same AUC as the percentiles of all slices). Series with thin slices (the cache averages groups of
them, knee.dataloading.canonical.merge_thin_slices) are read the same way: only the thin slices of the K cached
slices, then averaged like the cache.

cpu_select (a DataLoader worker) returns the raw slices and the geometry of one series; gpu_windows (the main process)
makes the windows [K, 3, H, W] with the center slice in the 3 channels, which is what a 'channels: center' model sees.
"""
import glob
import io
import os

import numpy as np
import pydicom
import torch

from knee.dataloading.canonical import (AXES, canonical_transform, merge_thin_slices, plane_of, resample_inplane,
                                        scale_to_255, to_canonical)
from knee.inference.fast_header import parse
from knee.inference.fast_prepare import DTYPES, _read_one, rescale


def read_header(path, nbytes=4096):
    """(header dataset, byte offset where the parse stopped: the pixel data element, or None) of one file. The fast
    parser (knee.inference.fast_header: the same values, 4x less CPU) first; pydicom for anything it does not take."""
    # an unbuffered read of exactly nbytes: a buffered open() reads st_blksize at once, which on the Kaggle NFS mount
    # is the whole file (the header pass then read 2.2x the bytes of the whole run; locally 4 KB blocks hide it)
    with open(path, 'rb', buffering=0) as f:
        raw = f.read(nbytes)
        while 0 < len(raw) < nbytes:   # a raw read may return fewer bytes than asked (NFS): complete it
            more = f.read(nbytes - len(raw))
            if not more:
                break
            raw += more
    fast = parse(raw)
    if fast is not None:
        return fast
    bio = io.BytesIO(raw)
    try:
        return pydicom.dcmread(bio, stop_before_pixels=True), bio.tell()
    except Exception:  # header longer than nbytes: read the whole file
        return pydicom.dcmread(path, stop_before_pixels=True), None


def read_pixels(path, head):
    """Pixels [H, W] of one file (as fast_prepare._read_one) from its header-pass (dataset, offset): for uncompressed
    explicit little endian data only the pixel bytes are read, without a second parse, and the stored integers stay
    integers when they are the image values (half the bytes to the GPU). None if not readable."""
    ds, offset = head
    try:
        rows, cols, bits = int(ds.Rows), int(ds.Columns), int(ds.BitsAllocated)
        n = rows * cols * bits // 8
        if offset is not None and str(ds.file_meta.TransferSyntaxUID) == '1.2.840.10008.1.2.1' and bits in (8, 16):
            with open(path, 'rb') as f:
                f.seek(offset)
                tag, data = f.read(12), f.read(n)
            if tag[:6] in (b'\xe0\x7f\x10\x00OW', b'\xe0\x7f\x10\x00OB') and len(data) == n:   # (7FE0,0010) PixelData
                img = np.frombuffer(data, dtype=DTYPES[(bits, int(ds.PixelRepresentation))]).reshape(rows, cols)
                return rescale(ds, img, keep_int=True)
    except Exception:
        pass
    return _read_one(path)   # other transfer syntaxes or an unexpected layout: the full parse


def sorted_slices(series_dir, pool):
    """Paths of the kept slices in sorted order (same rules as dicom_io.read_series), the geometry and the
    (dataset, pixel offset) of each kept slice."""
    paths = sorted(glob.glob(os.path.join(series_dir, '*.dcm')))

    def head(p):
        try:
            return p, read_header(p)
        except Exception:
            return p, (None, None)
    hs = [(p, h) for p, h in pool.map(head, paths)
          if h[0] is not None and 'ImageOrientationPatient' in h[0] and 'ImagePositionPatient' in h[0]]
    if not hs:
        return None
    groups = {}
    for p, h in hs:
        iop = np.round(np.asarray(h[0].ImageOrientationPatient, dtype=float), 2)
        groups.setdefault(tuple(iop) + (int(h[0].Rows), int(h[0].Columns)), []).append((p, h))
    group = max(groups.values(), key=len)
    ds0 = group[0][1][0]
    iop = np.asarray(ds0.ImageOrientationPatient, dtype=float)
    row_dir, col_dir = iop[:3] / np.linalg.norm(iop[:3]), iop[3:] / np.linalg.norm(iop[3:])
    normal = np.cross(row_dir, col_dir)
    pos = np.array([np.asarray(h[0].ImagePositionPatient, dtype=float) for _, h in group])
    order = np.argsort(pos @ normal, kind='stable')
    proj = (pos @ normal)[order]
    order = order[np.concatenate([[True], np.diff(proj) > 1e-3])]
    sp = ds0.get('PixelSpacing', [1.0, 1.0])
    return ([group[k][0] for k in order], pos[order], row_dir, col_dir, normal, (float(sp[0]), float(sp[1])), ds0,
            [group[k][1] for k in order])


def rchar():
    """Bytes this process has read with read(2) and friends (/proc/self/io), or 0."""
    try:
        with open('/proc/self/io') as fh:
            return int([line for line in fh if line.startswith('rchar')][0].split()[1])
    except (OSError, IndexError, ValueError):
        return 0


def cpu_select(series_dir, cfg, k, coverage, pool):
    """CPU part (DataLoader worker): read the slices of the K window positions. Returns a dict or None.

    'stack' holds the raw slices [K * merge, H, W] in canonical slice order: for thin-slice series ('merge' > 1) the
    `merge` thin slices of each window position (the GPU part then averages them like the cache). 'io' holds the I/O
    accounting of the series (read_stack).
    """
    r0 = rchar()
    r = sorted_slices(series_dir, pool)
    r_headers = rchar() - r0
    if r is None:
        return None
    paths, pos, row_dir, col_dir, normal, spacing, ds0, heads = r
    plane = plane_of(normal)
    proj = np.sort(pos @ np.asarray(AXES[plane][2], dtype=float))
    slice_mm = float(np.median(np.diff(proj))) if len(proj) > 1 else float(ds0.get('SliceThickness', 3) or 3)
    rel = np.clip(np.linspace(coverage[0], coverage[1], k), 0, 1)
    item = {'row_dir': row_dir, 'col_dir': col_dir, 'normal': normal, 'spacing': spacing, 'plane': plane,
            'slice_mm': slice_mm, 'rel': rel}
    merge = int(round(cfg['min_slice_mm'] / max(slice_mm, 1e-3)))
    n = len(paths)
    merge = merge if merge > 1 and n >= merge else 1   # thin slices: the cache averages groups of `merge` slices
    d = n // merge                                     # slices of the cached volume
    idx = np.clip(np.round(rel * (d - 1)).astype(int), 0, d - 1)        # canonical slice indices in the cached volume
    idx = (idx[:, None] * merge + np.arange(merge)).reshape(-1)         # their thin slices (merge_thin_slices groups)
    flip_s = canonical_transform(row_dir, col_dir, normal, plane)[3]
    src = (n - 1 - idx) if flip_s else idx                                # indices in the sorted file list
    stack, io = read_stack(paths, heads, src, pool)   # the common case: the pixel bytes straight into shared memory
    io.update(files=len(paths), header_bytes=4096 * len(paths), reads=len(src), merge=merge, depth=n)
    if stack is None:
        uniq, inv = np.unique(src, return_inverse=True)
        imgs = list(pool.map(read_pixels, [paths[j] for j in uniq], [heads[j] for j in uniq]))
        if any(im is None for im in imgs):
            return None
        if len({im.dtype for im in imgs}) > 1:   # stored integers next to rescaled slices: all float32
            imgs = [im.astype(np.float32) for im in imgs]
        stack = torch.from_numpy(np.stack(imgs)[inv])   # stored integers or float32 (gpu_windows: float32)
        io['path'], io['reads'] = 'fallback', len(uniq)
        io['pixel_bytes'] += sum(os.path.getsize(paths[j]) for j in uniq)   # read_pixels reads the whole file
    io['rchar_headers'], io['rchar_pixels'] = r_headers, rchar() - r0 - r_headers   # what the process really read
    item['stack'], item['io'] = stack, io
    item['flip_s'], item['merge'] = flip_s, merge
    return item


TORCH_INT = {np.uint8: torch.uint8, np.int8: torch.int8, np.uint16: torch.uint16, np.int16: torch.int16}


def read_stack(paths, heads, src, pool):
    """The stored integers of the files paths[src] (in this order) read into one tensor in shared memory (the
    DataLoader then passes it without a copy; no np.stack copy), or None when a file is not explicit VR little endian
    with an offset, needs a rescale (rescale(keep_int=True) would give float32) or the slices differ in type: then
    the caller reads them as before (read_pixels). The values are those of read_pixels. The second return value is the
    I/O accounting of the series: the read path ('fast', or 'fallback' set by the caller), why the fast path was not
    taken, the pixel bytes read and the number of short reads."""
    io = {'path': 'fast', 'pixel_bytes': 0, 'short_reads': 0, 'why': ''}
    first = heads[src[0]][0]
    try:
        rows, cols, bits, signed = int(first.Rows), int(first.Columns), int(first.BitsAllocated), int(first.PixelRepresentation)
    except Exception:
        io['why'] = 'header'
        return None, io
    dtype = DTYPES.get((bits, signed))
    if dtype is None:
        io['why'] = 'dtype'
        return None, io
    for j in set(src.tolist()):
        ds, offset = heads[j]
        if (offset is None or str(ds.file_meta.TransferSyntaxUID) != '1.2.840.10008.1.2.1'
                or (int(ds.Rows), int(ds.Columns), int(ds.BitsAllocated), int(ds.PixelRepresentation)) != (rows, cols, bits, signed)
                or float(ds.get('RescaleSlope', 1) or 1) != 1 or float(ds.get('RescaleIntercept', 0) or 0) != 0
                or str(ds.get('PhotometricInterpretation', '')) == 'MONOCHROME1'):
            io['why'] = ('offset' if offset is None else 'syntax' if str(ds.file_meta.TransferSyntaxUID) != '1.2.840.10008.1.2.1'
                         else 'rescale' if float(ds.get('RescaleSlope', 1) or 1) != 1 or float(ds.get('RescaleIntercept', 0) or 0) != 0
                         else 'shape')
            return None, io
    out = torch.empty((len(src), rows, cols), dtype=TORCH_INT[dtype]).share_memory_()
    arr, n = out.numpy(), rows * cols * bits // 8

    def one(i):
        """(ok, bytes read, short reads): the pixel bytes of one slice straight into the shared tensor; a short read
        (NFS) continues until n bytes or the end of the file."""
        path, offset = paths[src[i]], heads[src[i]][1]
        with open(path, 'rb') as f:
            f.seek(offset)
            tag = f.read(12)
            if tag[:6] not in (b'\xe0\x7f\x10\x00OW', b'\xe0\x7f\x10\x00OB'):   # (7FE0,0010) PixelData
                return False, 12, 0
            mv, got, short = memoryview(arr[i]).cast('B'), 0, 0
            while got < n:
                r = f.readinto(mv[got:])
                if not r:
                    break
                got += r
                short += 1
            return got == n, 12 + got, max(short - 1, 0)
    res = list(pool.map(one, range(len(src))))
    io['pixel_bytes'] = sum(b for _, b, _ in res)
    io['short_reads'] = sum(sh for _, _, sh in res)
    if not all(ok for ok, _, _ in res):
        io['why'] = 'tag' if any(b == 12 for ok, b, _ in res if not ok) else 'short'
        return None, io
    return out, io


def gpu_windows(item, cfg, device='cuda'):
    """GPU part: [K, 3, out_px, out_px] uint8 windows with the center slice in the 3 channels."""
    stack, sp = to_canonical(item['stack'].to(device).float(), item['spacing'], item['row_dir'], item['col_dir'],
                             item['normal'], item['plane'])
    if item['flip_s']:  # the slice order is already canonical (flip_s was applied to the indices)
        stack = stack.flip(0)
    if item['merge'] > 1:  # thin slices: [K * merge] -> [K] like the cache
        stack, _ = merge_thin_slices(stack, item['slice_mm'], cfg['min_slice_mm'])
    assert stack.shape[0] == len(item['rel']), 'the merge must be the one of cpu_select'
    stack = scale_to_255(stack, cfg['lo_pct'], cfg['hi_pct'])
    sel = resample_inplane(stack, sp, cfg['target_mm'], cfg['out_px']).round_().clamp_(0, 255).to(torch.uint8)
    return sel[:, None].expand(-1, 3, -1, -1)
