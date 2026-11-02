"""Kaggle submission script (GPU T4, no internet): install the DICOM codecs offline, then run
knee.inference.predict_efficient on the test set, one process per GPU, and write submission.csv.

Attached inputs (kernel-metadata.json; the slugs come from the submit script, pipeline/7_submit.sh):
- the competition data
- <owner>/<prefix>-wheels   offline wheels: pylibjpeg, pylibjpeg-libjpeg, pylibjpeg-openjpeg, python-gdcm
- <owner>/<prefix>-code     this repository's `knee` package
- <owner>/<prefix>-models   the checkpoints (*.pt) and models.txt: one checkpoint name per line, and the test-time
                             settings as comment lines ('# k_slices: 16', '# threads: 12', '# workers: 6',
                             '# weights: ...', '# tta_mirror: 0', '# gpus: 2')
"""
import glob
import os
import subprocess
import sys
import time

t0 = time.time()


def find(pattern, max_depth=4):
    """First path under /kaggle/input that matches the pattern, searched level by level.

    The mount layout changes between Kaggle images. A recursive glob would crawl the ~1M DICOM files, so the
    search stops at max_depth folder levels.
    """
    for depth in range(max_depth + 1):
        hits = sorted(glob.glob('/kaggle/input/' + '*/' * depth + pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(pattern)


py = f'cp{sys.version_info.major}{sys.version_info.minor}'
wheel_dir = os.path.dirname(find('pylibjpeg-*.whl'))
wheels = [w for w in glob.glob(os.path.join(wheel_dir, '*.whl')) if 'none-any' in w or py in w]
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '-q'] + wheels, check=True)

# the code dataset holds the `knee` package; Kaggle may unpack it with or without the `knee/` folder level
pkg_root = os.path.dirname(os.path.dirname(find('inference/predict_efficient.py', max_depth=5)))
code_dir = '/tmp/kneepkg'
os.makedirs(code_dir, exist_ok=True)
if not os.path.exists(os.path.join(code_dir, 'knee')):
    os.symlink(pkg_root, os.path.join(code_dir, 'knee'))
sys.path.insert(0, code_dir)
comp_csv = find('test_series.csv')
comp_dir = os.path.dirname(comp_csv)
models_txt = find('models.txt')
lines = [n.strip() for n in open(models_txt) if n.strip()]
ckpts = [os.path.join(os.path.dirname(models_txt), n) for n in lines if not n.startswith('#')]
meta = lambda key: [l.split(':', 1)[1].strip() for l in lines if l.startswith(f'# {key}:')]   # noqa: E731
print(f'python {py}, code {code_dir}, data {comp_dir}, {len(ckpts)} checkpoints, setup {time.time() - t0:.0f} s')

work = '/kaggle/working'
env = {**os.environ, 'PYTHONPATH': code_dir}
out = os.path.join(work, 'submission.csv')
settings = []   # the test-time settings of models.txt
for key in ('k_slices', 'threads'):
    if meta(key):
        settings += [f'--{key}', *meta(key)[0].split()]
workers = meta('workers')[0] if meta('workers') else str(os.cpu_count())   # DataLoader workers (the readers)
if meta('weights'):   # blend weight of each checkpoint
    settings += ['--weights', *meta('weights')[0].split()]
mirror = [] if meta('tta_mirror') and meta('tta_mirror')[0] == '0' else ['--tta_mirror']
cmd = [sys.executable, '-m', 'knee.inference.predict_efficient', '--series_root', os.path.join(comp_dir, 'test_series'),
       '--series_csv', comp_csv, '--checkpoints', *ckpts, '--out', out, *mirror, '--workers', workers,
       '--time_limit_h', '8.3', *settings]


def run_split(cmd, n_gpus):
    """One process per GPU (the T4 machine has 2 GPUs and 4 CPUs), each on every n-th study (its own series table)
    with its share of the reader workers; the outputs are joined in study order. The same predictions as one process:
    every study is predicted alone inside its batch."""
    import csv
    rows = list(csv.DictReader(open(comp_csv)))
    studies = sorted({r['StudyInstanceUID'] for r in rows})
    procs, outs = [], []
    for k in range(n_gpus):
        mine = set(studies[k::n_gpus])
        part_csv = os.path.join(work, f'series_part{k}.csv')
        with open(part_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(r for r in rows if r['StudyInstanceUID'] in mine)
        a = list(cmd)
        a[a.index('--series_csv') + 1] = part_csv
        a[a.index('--out') + 1] = os.path.join(work, f'part{k}.csv')
        a[a.index('--workers') + 1] = str(max(1, int(a[a.index('--workers') + 1]) // n_gpus))
        outs.append(a[a.index('--out') + 1])
        procs.append(subprocess.Popen(a, env={**env, 'CUDA_VISIBLE_DEVICES': str(k)}))
    codes = [p.wait() for p in procs]
    assert not any(codes), f'exit codes {codes}'
    import pandas as pd
    g = pd.concat([pd.read_csv(o) for o in outs]).set_index('StudyInstanceUID').reindex(studies).reset_index()
    g.to_csv(out, index=False)
    print(f'{n_gpus} GPUs: {len(g)} studies', flush=True)


n_gpus = int(meta('gpus')[0]) if meta('gpus') else 1
if n_gpus > 1:   # as many GPUs as the machine has, at most the package's number
    try:
        found = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.count('GPU ')
    except OSError:
        found = 1
    n_gpus = max(1, min(n_gpus, found))
if n_gpus > 1:
    run_split(cmd, n_gpus)
else:
    subprocess.run(cmd, check=True, env=env)
print(f'total {time.time() - t0:.0f} s')
