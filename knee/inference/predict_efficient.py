"""Efficiency-track prediction from the DICOM files: partial reads of the window slices, GPU preprocessing, the models.

    python -m knee.inference.predict_efficient --series_root <dir>/test_series --series_csv <dir>/test_series.csv \
        --checkpoints a.pt [b.pt ...] --out submission.csv [--tta_mirror] [--workers 4] [--threads 8] [--k_slices 16] \
        [--weights w_a w_b ...] [--studies_csv <csv>] [--log_every N] [--time_limit_h H]

The models must use the center-slice input ('channels: center') and share k_slices, coverage and max_series: the
DICOM files are read once. DataLoader workers (CPU): a header-only read of every file of a series, then the pixels of
the K window slices only (thin-slice series: the thin slices of the K windows; knee.inference.partial_reader). Main
process (GPU): canonical orientation, intensity scaling from the read slices, resampling, the models. These are the
windows of training, except that the intensity percentiles come from the K slices instead of the whole volume.
The output is the weighted mean probability of the checkpoints (and of the mirror view with --tta_mirror). A time
guard writes 0.5 for the studies that are not done when the time limit is reached.
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from knee.constants import LABELS
from knee.dataloading.build_cache import CFG
from knee.dataloading.dataset import StudyDataset, collate
from knee.dataloading.gpu_augment import GpuAugment
from knee.inference.dicom_dataset import test_series_table
from knee.inference.partial_reader import cpu_select, gpu_windows
from knee.model.knee_net import build_model


def cache_version(cfg):
    """Preprocessing version of a model: the name of its training cache folder (data/cache/<version>)."""
    return os.path.basename(os.path.normpath(cfg['data']['cache']))


def load_models(paths, device):
    """The models of the checkpoints (eval mode, channels_last, on device) and the config of the first one."""
    models, cfg0 = [], None
    for p in paths:
        st = torch.load(p, map_location='cpu', weights_only=False)
        cfg0 = cfg0 or st['cfg']
        model = build_model(st['cfg']['model'], pretrained=False, pretrained_path=None)  # the weights: the checkpoint
        model.load_state_dict(st['model'])
        models.append(model.to(device).eval().to(memory_format=torch.channels_last))   # the inputs are channels_last
    return models, cfg0


class PartialStudies(Dataset):
    """The series of each study as partial reads: the series of the training selection (max_series, one series of
    each type first: knee.dataloading.dataset.StudyDataset._select_series), each read by cpu_select with a thread
    pool. An unreadable series is left out."""

    def __init__(self, studies, series, k, max_series, coverage, cache_cfg, threads):
        self.sel = StudyDataset(studies, series, targets=None, k_slices=k, max_series=max_series, coverage=coverage)
        self.k, self.coverage, self.cache_cfg, self.threads = k, coverage, cache_cfg, threads

    def __len__(self):
        return len(self.sel)

    def __getitem__(self, i):
        study = self.sel.studies[i]
        rows = self.sel._select_series(self.sel.by_study[study], np.random.default_rng(self.sel.seed + i))
        out = []
        with ThreadPoolExecutor(self.threads) as pool:
            for r in rows:
                try:
                    item = cpu_select(r.series_dir, self.cache_cfg, self.k, self.coverage, pool)
                except Exception:
                    item = None
                if item is not None:
                    item['type'] = int(r.type)
                    out.append(item)
        return study, out


def first_item(batch):
    return batch[0]


def low_priority(_):
    """DataLoader worker start: nice 10, so that the main process gets a CPU at once when it wakes up after a GPU wait
    (the reading threads keep the 4 Kaggle CPUs busy)."""
    os.nice(10)


def machine_counters():
    """CPU time (busy, iowait, steal, all), bytes read from the disks, and bytes read by the programs / from the
    server on the NFS mounts (the Kaggle data) since boot (Linux /proc), or None."""
    try:
        # the first line of /proc/stat: user nice system idle iowait irq softirq steal (in clock ticks)
        f = [int(v) for v in open('/proc/stat').readline().split()[1:9]]
        disks = [d.split() for d in open('/proc/diskstats')]
        disk = sum(int(d[5]) * 512 for d in disks if re.fullmatch(r'(sd|vd|xvd)[a-z]+|nvme\d+n\d+', d[2]))
        nfs, dev = {}, None   # 'bytes:' of each NFS export once (an export can show at several mount points)
        for line in open('/proc/self/mountstats'):
            if line.startswith('device '):
                dev = line.split()[1]
            elif line.strip().startswith('bytes:'):
                nfs[dev] = [int(v) for v in line.split()[1:]]   # normal read, ..., server read (index 4)
        return (sum(f) - f[3] - f[4] - f[7], f[4], f[7], sum(f), disk, sum(v[0] for v in nfs.values()),
                sum(v[4] for v in nfs.values()))
    except (OSError, ValueError, IndexError):
        return None


def file_system(path):
    """Type and source of the file system that holds path (Linux /proc/mounts), e.g. 'ext4 /dev/sdb'."""
    try:
        mounts = [m.split()[:3] for m in open('/proc/mounts')]
        real = os.path.realpath(path) + '/'
        src, _, fs = max((m for m in mounts if real.startswith(m[1].rstrip('/') + '/')), key=lambda m: len(m[1]))
        return f'{fs} {src}, block size {os.stat(path).st_blksize}'
    except (OSError, ValueError):
        return 'unknown'


def blocking_gpu_wait():
    """Let the CPU sleep while it waits for the GPU. CUDA spins by default when a process has fewer contexts than
    CPUs: the main process then takes one of the 4 Kaggle CPUs from the reader workers during every model batch.
    Sets CU_CTX_SCHED_BLOCKING_SYNC on the primary context of the visible GPU (driver API); True if done."""
    import ctypes
    try:
        cuda, dev = ctypes.CDLL('libcuda.so.1'), ctypes.c_int()
        return (cuda.cuInit(0) == 0 and cuda.cuDeviceGet(ctypes.byref(dev), 0) == 0
                and cuda.cuDevicePrimaryCtxSetFlags(dev, 4) == 0)   # 4 = CU_CTX_SCHED_BLOCKING_SYNC
    except OSError:
        return False


def load_groups(paths, device, weights=None):
    """Load the checkpoints and group them by input settings (cache version + input block); weights: one per
    checkpoint (the blend; default equal).

    All groups must share the CPU-side selection (k_slices, coverage, max_series): the DICOM files are read once.
    """
    groups = {}
    weights = weights or [1.0] * len(paths)
    assert len(weights) == len(paths), 'one weight per checkpoint'
    for p, wt in zip(paths, weights):
        models, cfg = load_models([p], device)
        key = json.dumps({'cache': cache_version(cfg), 'input': cfg['input']}, sort_keys=True)
        g = groups.setdefault(key, {'cfg': cfg, 'models': [], 'weights': []})
        g['models'] += models
        g['weights'] += [float(wt)] * len(models)
    cfgs = [g['cfg']['input'] for g in groups.values()]
    for c in cfgs[1:]:
        assert all(c[k] == cfgs[0][k] for k in ('k_slices', 'coverage', 'max_series')), \
            'all models must use the same k_slices, coverage and max_series'
    for g in groups.values():
        assert g['cfg']['input']['aug'].get('channels') == 'center', 'partial reads need center-slice models'
        assert not g['cfg']['model'].get('joint_layers'), \
            'the joint transformer needs the joint box at test time: drop it first (knee.submission.drop_joint)'
    return list(groups.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--series_root', required=True)
    ap.add_argument('--series_csv', required=True)
    ap.add_argument('--checkpoints', nargs='+', required=True)
    ap.add_argument('--out', default='submission.csv')
    ap.add_argument('--studies_csv', default=None, help='csv with the StudyInstanceUID to predict (default: all)')
    ap.add_argument('--tta_mirror', action='store_true', help='add the mirror view')
    ap.add_argument('--batch_size', type=int, default=8, help='studies per model call')
    ap.add_argument('--workers', type=int, default=os.cpu_count(), help='DataLoader workers (the reading processes)')
    ap.add_argument('--threads', type=int, default=8, help='file reading threads in each worker')
    ap.add_argument('--time_limit_h', type=float, default=8.5)
    ap.add_argument('--log_every', type=int, default=200, help='a progress line every N studies')
    ap.add_argument('--k_slices', type=int, default=None, help='override the windows per series (fewer reads)')
    ap.add_argument('--weights', type=float, nargs='+', default=None,
                    help='blend weight of each checkpoint (default equal)')
    args = ap.parse_args()
    t_start, device = time.time(), 'cuda'
    sleep_wait = blocking_gpu_wait()   # before the first CUDA call
    half = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    if os.environ.get('KNEE_HALF') == 'float16':   # the dtype of the Kaggle T4 on a machine with bfloat16: local checks
        half = torch.float16
    groups = load_groups(args.checkpoints, device, args.weights)
    n_models = sum(len(g['models']) for g in groups)
    total_w = sum(sum(g['weights']) for g in groups)
    views = [False, True] if args.tta_mirror else [False]
    for g in groups:
        if half == torch.float16:
            g['models'] = [m.half() for m in g['models']]
        dcfg = g['models'][0].encoder.pretrained_cfg
        g['aug'] = GpuAugment(mean=dcfg.get('mean', (0.485, 0.456, 0.406)), std=dcfg.get('std', (0.229, 0.224, 0.225)),
                              **g['cfg']['input']['aug'])
        g['cache_cfg'] = CFG[cache_version(g['cfg'])]
    i0 = dict(groups[0]['cfg']['input'])
    if args.k_slices:   # fewer windows than in training: fewer reads
        i0['k_slices'] = args.k_slices
    t_models = time.time()
    series = test_series_table(args.series_root, args.series_csv)
    studies = (pd.read_csv(args.studies_csv).StudyInstanceUID.tolist() if args.studies_csv
               else sorted(series.StudyInstanceUID.unique()))
    ds = PartialStudies(studies, series, i0['k_slices'], i0['max_series'], tuple(i0['coverage']),
                        groups[0]['cache_cfg'], args.threads)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=first_item, num_workers=args.workers,
                    prefetch_factor=4 if args.workers else None, worker_init_fn=low_priority)
    print(f'{len(studies)} studies, {n_models} models in {len(groups)} input groups, dtype {half} (startup: models '
          f'{t_models - t_start:.0f} s, series table {time.time() - t_models:.0f} s; DICOM files on '
          f'{file_system(args.series_root)}; blocking GPU wait {sleep_wait}; torch {torch.__version__}, cudnn '
          f'{torch.backends.cudnn.version()}, GPU {torch.cuda.get_device_name(0)})', flush=True)
    preds, pending = {}, []
    io_tot = {}   # the readers' I/O accounting per read path (knee.inference.partial_reader cpu_select: item['io'])
    parts = {'windows': 0.0, 'models': 0.0}   # model-batch time by part

    def tick(key, t0):
        torch.cuda.synchronize()
        parts[key] += time.time() - t0
        return time.time()

    def flush():
        todo = [(st, it) for st, it in pending if it]
        pending.clear()
        if not todo:
            return
        built = {}   # the windows of each series and cache setting, built once for all models

        def windows(s, cfg):
            key = (id(s), id(cfg))
            if key not in built:
                built[key] = gpu_windows(s, cfg, device)
            return built[key]

        t0 = time.time()
        p = 0
        for g in groups:
            items = [{'images': torch.stack([windows(s, g['cache_cfg']) for s in series_items]),
                      'types': torch.tensor([s['type'] for s in series_items]),
                      'pos': torch.stack([torch.from_numpy(s['rel'].astype(np.float32)) for s in series_items]),
                      'labels': torch.full((len(LABELS),), float('nan')), 'study': study}
                     for study, series_items in todo]
            t0 = tick('windows', t0)
            # studies per model call: as many as fit like a batch at 288 px and 16 windows (a 384 px reader needs
            # 16.4 GB for 8 studies of 8 series, and 32 windows twice the memory of 16: the T4 has 15 GB); the studies
            # are independent, so the chunks change nothing
            px_area = g['aug'].out_px ** 2
            chunk = max(1, int(args.batch_size * min(1.0, 288 ** 2 / px_area * 16 / i0['k_slices'])))
            pg = []
            with torch.no_grad():
                for c0 in range(0, len(items), chunk):
                    batch = collate(items[c0:c0 + chunk])
                    pc = 0
                    for model, wt in zip(g['models'], g['weights']):
                        pm = 0
                        for m in views:
                            x, ps = g['aug'](batch, train=False, mirror_all=m)
                            x, t, si = x.to(half), batch['types'].to(device), batch['study_idx'].to(device)
                            with torch.autocast('cuda', dtype=half):
                                pm = pm + torch.sigmoid(model(x, t, ps.to(half), si, len(batch['study'])).float())
                        pc = pc + wt * pm / len(views)
                    pg.append(pc)
            p = p + torch.cat(pg)
            t0 = tick('models', t0)
        preds.update(zip([st for st, _ in todo], (p / total_w).cpu().numpy()))

    t_wait = t_model = 0.0   # where the time goes: waiting for the reader workers vs model batches
    m0, main_cpu0 = machine_counters(), time.process_time()   # and what limits the reader workers: CPU or disk
    t_mark = time.time()
    for n, item in enumerate(dl):
        t_wait += time.time() - t_mark
        if time.time() - t_start > args.time_limit_h * 3600:
            print('time limit reached: the remaining studies get 0.5', flush=True)
            break
        pending.append(item)
        for s in item[1]:   # (study, [series items])
            io = s['io']
            key = io['path'] + (f" ({io['why']})" if io['why'] else '')
            tot = io_tot.setdefault(key, {'series': 0, 'reads': 0, 'files': 0, 'header_bytes': 0, 'pixel_bytes': 0,
                                          'short_reads': 0, 'merged': 0, 'rchar_headers': 0, 'rchar_pixels': 0})
            tot['series'] += 1
            tot['merged'] += int(io['merge'] > 1)
            for k_ in ('reads', 'files', 'header_bytes', 'pixel_bytes', 'short_reads', 'rchar_headers', 'rchar_pixels'):
                tot[k_] += int(io[k_])
        if len(pending) == args.batch_size:
            t1 = time.time()
            flush()   # ends with .cpu(): the GPU work is done
            t_model += time.time() - t1
        if n % args.log_every == 0 or n == len(studies) - 1:
            print(f'{n}/{len(studies)} studies, {time.time() - t_start:.0f} s', flush=True)
        t_mark = time.time()
    t1 = time.time()
    flush()
    t_model += time.time() - t1
    print(f'time: waiting for the reader workers {t_wait:.0f} s, model batches {t_model:.0f} s ('
          + ', '.join(f'{k} {v:.0f} s' for k, v in parts.items()) + ')', flush=True)
    for key, v in io_tot.items():
        print(f"reader I/O, {key}: {v['series']} series, {v['reads'] / max(v['series'], 1):.1f} slice reads and "
              f"{v['files'] / max(v['series'], 1):.0f} files per series, {v['merged']} with merged thin slices; "
              f"headers {v['header_bytes'] / 1e9:.1f} GB, pixels {v['pixel_bytes'] / 1e9:.1f} GB, {v['short_reads']} "
              f"short reads; really read: {v['rchar_headers'] / 1e9:.1f} GB for the headers, "
              f"{v['rchar_pixels'] / 1e9:.1f} GB for the pixels", flush=True)
    m1 = machine_counters()
    if m0 and m1:
        busy, iowait, steal, total, disk, nfs_read, nfs_server = (b - a for a, b in zip(m0, m1))
        print(f'machine: CPU busy {busy / max(total, 1):.0%}, iowait {iowait / max(total, 1):.0%}, steal '
              f'{steal / max(total, 1):.0%} ({os.cpu_count()} CPUs), main process '
              f'{time.process_time() - main_cpu0:.0f} s of CPU; read from the disks {disk / 1e9:.1f} GB; NFS: '
              f'{nfs_read / 1e9:.1f} GB read by the programs, {nfs_server / 1e9:.1f} GB from the server', flush=True)
    sub = pd.DataFrame([preds.get(s, np.full(len(LABELS), 0.5)) for s in studies], columns=LABELS)
    sub.insert(0, 'StudyInstanceUID', studies)
    sub.to_csv(args.out, index=False)
    print(f'wrote {args.out}: {len(sub)} studies ({len(studies) - len(preds)} with the 0.5 fallback), '
          f'{time.time() - t_start:.0f} s total', flush=True)


if __name__ == '__main__':
    main()
