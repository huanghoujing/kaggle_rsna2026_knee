"""Prepare the Kaggle datasets of a submission: the code, the codec wheels and the models.

    .venv/bin/python -m knee.submission.package --models <checkpoint.pt> [...] --fp16 --k_slices 16 --threads 12 \
        --workers 6 --gpus 2 --no_tta_mirror

Writes kaggle/datasets/knee-code/ (the knee package) and kaggle/datasets/knee-models/ (the checkpoints with the model
weights only, plus models.txt: one checkpoint name per line and the test-time settings of the notebook as comment
lines: '# k_slices:', '# threads:', '# workers:', '# weights:', '# tta_mirror: 0', '# gpus:'). kaggle/datasets/
knee-wheels/ is made once (pipeline/0_wheels.sh); its metadata is written here. The dataset owner, slug prefix and
licences come from the environment: KNEE_KAGGLE_OWNER, KNEE_KAGGLE_PREFIX, KNEE_KAGGLE_LICENSE_{CODE,MODELS,WHEELS};
KNEE_KAGGLE_CODE_PREFIX names the code and wheels datasets when several notebooks share them (default: the prefix).
"""
import argparse
import json
import os
import shutil

import torch

from knee.constants import REPO

OWNER = os.environ.get('KNEE_KAGGLE_OWNER', 'houjinghuang')      # the Kaggle account that holds the datasets
PREFIX = os.environ.get('KNEE_KAGGLE_PREFIX', 'knee')           # dataset slug <PREFIX>-models
CODE_PREFIX = os.environ.get('KNEE_KAGGLE_CODE_PREFIX', PREFIX)   # dataset slugs <CODE_PREFIX>-code, -wheels
# Kaggle's licence names have no CC BY-NC 4.0 (the competition's winner licence of the code): 'other', with the licence
# named in the dataset description and the LICENSE file inside the dataset
LICENSES = {'code': os.environ.get('KNEE_KAGGLE_LICENSE_CODE', 'other'),
            'models': os.environ.get('KNEE_KAGGLE_LICENSE_MODELS', 'CC-BY-NC-SA-4.0'),   # RadImageNet-derived weights
            'wheels': os.environ.get('KNEE_KAGGLE_LICENSE_WHEELS', 'other')}


def write_meta(folder, part):
    """dataset-metadata.json of kaggle/datasets/knee-<part>: slug <prefix>-<part> of OWNER."""
    prefix = PREFIX if part == 'models' else CODE_PREFIX
    meta = {'title': f'{prefix}-{part}', 'id': f'{OWNER}/{prefix}-{part}', 'licenses': [{'name': LICENSES[part]}]}
    if part == 'code':
        meta['description'] = ('The knee package of https://github.com/huanghoujing/kaggle_rsna2026_knee (code licence '
                               'CC BY-NC 4.0: the LICENSE file in this dataset).')
    json.dump(meta, open(os.path.join(folder, 'dataset-metadata.json'), 'w'), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True, help='checkpoints (final.pt / best.pt / final_nojoint.pt)')
    ap.add_argument('--fp16', action='store_true', help='store the weights in float16 (half the size)')
    ap.add_argument('--k_slices', type=int, default=None, help='windows per series at test time (the models keep '
                    'their own K when not set)')
    ap.add_argument('--threads', type=int, default=None, help='file reading threads per DataLoader worker')
    ap.add_argument('--workers', type=int, default=None, help='DataLoader workers (default: the CPUs)')
    ap.add_argument('--weights', type=float, nargs='+', default=None, help='blend weight of each model')
    ap.add_argument('--no_tta_mirror', action='store_true', help='one view only (no mirror view)')
    ap.add_argument('--gpus', type=int, default=1, help='one process per GPU, each on a share of the studies (the '
                    'Kaggle T4 machine has 2 GPUs)')
    args = ap.parse_args()
    base = os.path.join(REPO, 'kaggle', 'datasets')

    code = os.path.join(base, 'knee-code')
    shutil.rmtree(os.path.join(code, 'knee'), ignore_errors=True)
    shutil.copytree(os.path.join(REPO, 'knee'), os.path.join(code, 'knee'),
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy(os.path.join(REPO, 'LICENSE'), os.path.join(code, 'LICENSE'))
    write_meta(code, 'code')

    models = os.path.join(base, 'knee-models')
    shutil.rmtree(models, ignore_errors=True)
    os.makedirs(models)

    def save(p, name):
        st = torch.load(p, map_location='cpu', weights_only=False)
        sd = {k: (v.half() if args.fp16 and v.is_floating_point() else v) for k, v in st['model'].items()}
        torch.save({'model': sd, 'cfg': st['cfg'], 'epoch': st.get('epoch'), 'source': p}, os.path.join(models, name))
        return name

    def run(p):
        """experiments/<run>/checkpoints/<file>.pt -> <run>; any other path -> the file name without .pt."""
        d = os.path.dirname(os.path.abspath(p))
        if os.path.basename(d) == 'checkpoints':
            return os.path.basename(os.path.dirname(d))
        return os.path.splitext(os.path.basename(p))[0]

    names = [save(p, f'{i:02d}_{run(p)}.pt') for i, p in enumerate(args.models)]
    lines = list(names)
    lines += [f'# k_slices: {args.k_slices}'] if args.k_slices else []
    lines += [f'# threads: {args.threads}'] if args.threads else []
    lines += [f'# workers: {args.workers}'] if args.workers else []
    if args.weights:
        assert len(args.weights) == len(args.models), 'one weight per model'
        lines += ['# weights: ' + ' '.join(f'{w:g}' for w in args.weights)]
    lines += ['# tta_mirror: 0'] if args.no_tta_mirror else []
    lines += [f'# gpus: {args.gpus}'] if args.gpus > 1 else []
    open(os.path.join(models, 'models.txt'), 'w').write('\n'.join(lines) + '\n')
    write_meta(models, 'models')
    write_meta(os.path.join(base, 'knee-wheels'), 'wheels')
    print(f'code -> {code}\n{len(names)} models -> {models}')


if __name__ == '__main__':
    main()
