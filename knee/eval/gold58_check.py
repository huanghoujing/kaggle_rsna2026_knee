"""Gold-58 check of a trained model as the Kaggle path predicts it: the checkpoint without the joint transformer branch
(checkpoints/final_nojoint.pt, made here when it is missing), 16 windows per series, one view, no augmentation. Prints
the AUC of each finding and the macro AUC on the 58 expert-labelled studies (train.csv). With train.include_gold these
studies were in training, so the score is in-sample: it checks the model and the path, it does not select models.

    .venv/bin/python -m knee.eval.gold58_check --run <run folder name without the time prefix, e.g. <config name>_f-1>
    .venv/bin/python -m knee.eval.gold58_check --checkpoint data/models/<file>.pt      # any checkpoint

Writes the probabilities next to the checkpoint (<run>/gold58_k16.csv) or next to the file (--checkpoint).
"""
import argparse
import copy
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from knee.constants import LABELS
from knee.dataloading.dataset import collate
from knee.dataloading.gpu_augment import GpuAugment
from knee.eval.metrics import auc_table
from knee.model.knee_net import build_model
from knee.submission.drop_joint import drop_joint, find_run
from knee.training.train import build_data, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', help='run folder name without the time prefix (experiments/*_<run>)')
    ap.add_argument('--checkpoint', help='a checkpoint file instead of a run')
    ap.add_argument('--k_slices', type=int, default=16, help='windows per series (the submission: 16)')
    ap.add_argument('--tta_mirror', action='store_true', help='add the mirror view (not used by v76)')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    assert bool(args.run) != bool(args.checkpoint), 'give --run or --checkpoint'
    ckpt = args.checkpoint
    if args.run:
        ckpt = os.path.join(find_run(args.run), 'checkpoints', 'final_nojoint.pt')
        if not os.path.exists(ckpt):
            ckpt = drop_joint(args.run)
    st = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfg = copy.deepcopy(st['cfg'])
    cfg['input']['k_slices'] = args.k_slices
    model = build_model(cfg['model'], pretrained=False, pretrained_path=None)
    model.load_state_dict(st['model'])
    model = model.cuda().eval().requires_grad_(False)
    aug = GpuAugment(mean=model.encoder.pretrained_cfg['mean'], std=model.encoder.pretrained_cfg['std'],
                     **cfg['input']['aug'])
    ds_gold = build_data(cfg, -1)[2]
    dl = DataLoader(ds_gold, batch_size=args.batch_size, collate_fn=collate, num_workers=args.workers, pin_memory=True)
    studies, probs, labels = predict(model, dl, aug, tta_mirror=args.tta_mirror)
    out = pd.DataFrame(probs, columns=LABELS)
    out.insert(0, 'StudyInstanceUID', studies)
    tag = f'gold58_k{args.k_slices}' + ('_mirror' if args.tta_mirror else '')
    folder = os.path.dirname(os.path.dirname(ckpt)) if args.run else os.path.dirname(os.path.abspath(ckpt))
    path = os.path.join(folder, f'{tag}.csv')
    out.to_csv(path, index=False)
    table = auc_table(labels, probs)
    print(pd.Series(table, name='AUC').round(4).to_string())
    print(f'{len(studies)} expert-labelled studies, {args.k_slices} windows per series, '
          f'{"two views" if args.tta_mirror else "one view"} -> {path}')


if __name__ == '__main__':
    main()
