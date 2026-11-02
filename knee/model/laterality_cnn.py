"""Image laterality classifier: left or right knee from the middle slice of coronal and axial series.

    CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m knee.model.laterality_cnn --out data/models/laterality_r18.pt
    .venv/bin/python -m knee.model.laterality_cnn --out data/models/laterality_r18.pt --predict_only

--predict_only: no training; the model saved at --out predicts every training study and writes only the prediction
table <out without .pt>_train_pred.parquet (columns StudyInstanceUID, logit_left, pred).

Training labels: the studies whose side comes from the DICOM tag, the series description or the report
(data/meta/laterality_train.parquet, not the knee-position fallback). Input: the middle slice of each coronal and
axial series in the canonical frame (cache v1), 224 px over 160 mm, center slice in 3 channels. A study gets the
mean logit of its slices. The script prints the held-out accuracy and the agreement with the position rule.
"""
import argparse
import os

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F

from knee.constants import RAW
from knee.dataloading.dataset import series_table

MEAN, STD = 0.45, 0.225


def middle_slices(series, cache_px=352, crop_px=320, out_px=224):
    """[M, 1, out_px, out_px] float tensor of the middle slice of every coronal / axial series, and the study ids."""
    rows = series[series.plane.isin(['Coronal', 'Axial'])]
    imgs, studies = [], []
    for r in rows.itertuples():
        v = np.load(r.path, mmap_mode='r')
        s = np.asarray(v[v.shape[0] // 2], dtype=np.float32)
        o = (cache_px - crop_px) // 2
        imgs.append(s[o:o + crop_px, o:o + crop_px])
        studies.append(r.StudyInstanceUID)
    x = torch.from_numpy(np.stack(imgs))[:, None] / 255.0
    return F.interpolate(x, size=(out_px, out_px), mode='bilinear', align_corners=False), np.array(studies)


def make_model():
    return timm.create_model('resnet18.a1_in1k', pretrained=True, num_classes=1)


def logits(model, x, device, bs=256):
    out = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(x), bs):
            xb = ((x[i:i + bs].to(device) - MEAN) / STD).expand(-1, 3, -1, -1)
            out.append(model(xb).float().squeeze(1).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--laterality', default='data/meta/laterality_train.parquet')
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--predict_only', action='store_true', help='no training: the model saved at --out predicts every '
                    'training study and only the prediction table is written')
    args = ap.parse_args()
    device = 'cuda'
    torch.manual_seed(0)
    series = series_table('data/cache/v1', 'train', os.path.join(RAW, 'train_series.csv'))
    x, studies = middle_slices(series)
    if args.predict_only:
        st = torch.load(args.out, map_location='cpu', weights_only=False)
        assert (st['mean'], st['std'], st['cache_px'], st['crop_px'], st['out_px']) == (MEAN, STD, 352, 320, 224), \
            'the saved model expects other inputs than middle_slices makes'
        model = timm.create_model(st['arch'], pretrained=False, num_classes=1)
        model.load_state_dict(st['model'])
        model = model.to(device)
    else:
        lat = pd.read_parquet(args.laterality).set_index('StudyInstanceUID')
        side = lat.side.reindex(studies).values
        known = lat.source.reindex(studies).isin(['tag', 'desc', 'report']).values
        y = torch.tensor(side == 'L', dtype=torch.float32)
        uniq = np.array(sorted(set(studies[known])))
        rng = np.random.default_rng(0)
        val_studies = set(rng.choice(uniq, size=len(uniq) // 5, replace=False))
        is_val = np.array([s in val_studies for s in studies])
        tr_idx = np.where(known & ~is_val)[0]
        model = make_model().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
        steps = args.epochs * (len(tr_idx) // 64)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps)
        model.train()
        for ep in range(args.epochs):
            perm = rng.permutation(tr_idx)
            for i in range(0, len(perm) - 63, 64):
                b = perm[i:i + 64]
                xb = x[b].to(device)
                # augmentation that keeps the side: small rotation / scale / shift, intensity
                ang = (torch.rand(len(b), device=device) - 0.5) * 0.4
                sc = 1 + (torch.rand(len(b), device=device) - 0.5) * 0.3
                tx, ty = ((torch.rand(len(b), device=device) - 0.5) * 0.2 for _ in range(2))
                th = torch.stack([torch.stack([sc * torch.cos(ang), -sc * torch.sin(ang), tx], 1),
                                  torch.stack([sc * torch.sin(ang), sc * torch.cos(ang), ty], 1)], 1)
                xb = F.grid_sample(xb, F.affine_grid(th, xb.shape, align_corners=False), align_corners=False)
                xb = xb * (0.8 + 0.4 * torch.rand(len(b), 1, 1, 1, device=device))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = model(((xb - MEAN) / STD).expand(-1, 3, -1, -1)).float().squeeze(1)
                loss = F.binary_cross_entropy_with_logits(out, y[b].to(device))
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            print(f'epoch {ep} loss {float(loss):.4f}', flush=True)
    model.eval()
    lg = logits(model, x, device)
    df = pd.DataFrame({'study': studies, 'logit': lg.numpy()}).groupby('study').logit.mean()
    pred = pd.Series(np.where(df > 0, 'L', 'R'), index=df.index)
    if not args.predict_only:
        vs = sorted(val_studies)
        acc = (pred.loc[vs] == lat.side.reindex(vs)).mean()
        print(f'held-out study accuracy {acc:.4f} on {len(vs)} studies')
        xpos = lat[lat.source == 'xpos']   # the table built without a classifier: the position rule for these
        if len(xpos):
            agree = (pred.reindex(xpos.index) == xpos.side).mean()
            print(f'agreement with the position rule on the {len(xpos)} position-only studies {agree:.4f}')
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save({'model': model.state_dict(), 'arch': 'resnet18.a1_in1k', 'mean': MEAN, 'std': STD,
                    'crop_px': 320, 'cache_px': 352, 'out_px': 224}, args.out)
    pd.DataFrame({'StudyInstanceUID': df.index, 'logit_left': df.values, 'pred': pred.values}).to_parquet(
        args.out.replace('.pt', '_train_pred.parquet'), index=False)


if __name__ == '__main__':
    main()
