"""A deployable copy of a run without its joint transformer branch: the branch needs the series geometry and the knee
joint box at test time, which the efficient Kaggle path does not have. Without the branch the model scores the same
(its gain comes from training), so the copy is a plain spatial reader: model.joint_layers 0 and no joint.* /
joint_gate weights. Writes checkpoints/final_nojoint.pt next to final.pt.

    .venv/bin/python -m knee.submission.drop_joint --run r50rad_288_spatial_os10_kd70fx_joint3d_orient_e48_f-1
"""
import argparse
import glob
import os

import torch

from knee.constants import REPO
from knee.model.knee_net import build_model


def find_run(name):
    """The newest finished run folder experiments/*_<name> (one with checkpoints/final.pt)."""
    runs = [r for r in sorted(glob.glob(os.path.join(REPO, 'experiments', f'*_{name}')))
            if os.path.exists(os.path.join(r, 'checkpoints', 'final.pt'))]
    assert runs, f'no finished run {name}'
    return runs[-1]


def drop_joint(run):
    """Writes <run>/checkpoints/final_nojoint.pt and returns its path."""
    src = os.path.join(find_run(run), 'checkpoints', 'final.pt')
    st = torch.load(src, map_location='cpu', weights_only=False)
    sd = {k: v for k, v in st['model'].items() if not (k.startswith('joint.') or k == 'joint_gate')}
    cfg = st['cfg']
    cfg['model']['joint_layers'] = 0
    build_model(cfg['model'], pretrained=False, pretrained_path=None).load_state_dict(sd)   # strict: nothing missing
    dst = src.replace('final.pt', 'final_nojoint.pt')
    torch.save({'model': sd, 'cfg': cfg, 'epoch': st.get('epoch'), 'source': src}, dst)
    print(f'{dst}: {len(st["model"]) - len(sd)} tensors of the joint branch dropped')
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    drop_joint(ap.parse_args().run)


if __name__ == '__main__':
    main()
