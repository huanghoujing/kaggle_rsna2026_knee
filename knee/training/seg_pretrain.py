"""Tissue segmentation training of the 2D encoder on SKM-TEA. Resumable: --resume continues a run.

    python -m knee.training.seg_pretrain --config configs/seg_skm3_r50rad_288.yaml [--set key.sub=value ...]
    python -m knee.training.seg_pretrain --resume experiments/<run>

Data: the cache split skm_tea (knee.dataloading.skm_tea prepare): 155 sagittal qDESS scans with 2 echoes and the tissue
labels in the cache frame (0 background, 1 patellar cartilage, 2 femoral cartilage, 3 / 4 medial / lateral tibial
cartilage, 5 / 6 medial / lateral meniscus). Training: the SKM-TEA train and val splits (119 scans, both echoes); test:
the 36 scans of the SKM-TEA test split (3D Dice of each tissue on each echo volume, mean over the volumes). data.planes:
the sagittal series and the coronal / axial reformats (knee.dataloading.skm_tea; left knees mirrored, as our studies).
Model: the encoder of the classification model (timm ResNet-50 with the RadImageNet weights) and a small U-Net decoder
(1 x 1 convs on the stem and the 4 stages, upsample + add, 3 x 3 convs) with logits at stride 2, upsampled to the input
size. Loss: soft cross-entropy + batch Dice (as nnU-Net), and a cross-entropy on the stage-4 cells against the tissue
share of each cell (the classification heads read these cells). Input and augmentation as the classification model
(GpuAugment, one window per sample, 'channels: center'); the masks get the same spatial transform. All volumes stay on
the GPU (about 4 GB): no data loader.
Outputs in experiments/MM_DD_HH_mm_<name>/: config.yaml, steps.csv, log.csv (test Dice), curves.png,
checkpoints/last.pt (resume). The EMA encoder alone goes to save_encoder (the format of model.pretrained_path of a
classification config), the whole EMA model with its config to save_model (knee.dataloading.tissue_rois runs it on the
series of our studies).
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import argparse
import copy
import math
import time
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from knee.constants import PLANES, REPO  # noqa: E402
from knee.dataloading.gpu_augment import GpuAugment  # noqa: E402
from knee.model.knee_net import load_torchvision_weights  # noqa: E402
from knee.training.train import load_config, param_groups, save_ckpt  # noqa: E402

TISSUES = ['patellar cartilage', 'femoral cartilage', 'medial tibial cartilage', 'lateral tibial cartilage',
           'medial meniscus', 'lateral meniscus']   # labels 1-6 (knee.dataloading.skm_tea)


class Volumes:
    """The SKM-TEA volumes of some splits and planes on the GPU, the slices of all volumes concatenated: images
    [S, H, W] uint8 (each echo of each plane is one volume), labels [S, H, W] uint8, and for each slice its volume, the
    first / last slice of it, its series type (plane * 2 + FS; echo 2 counts as FS) and the left-knee flag (GpuAugment
    mirrors left knees, as the knee-side table does for our studies)."""

    def __init__(self, cache, splits, planes):
        idx = pd.read_parquet(os.path.join(cache, 'index_skm_tea.parquet'))
        idx = idx[idx.ok & idx.split.isin(splits) & idx.plane.isin(planes)].sort_values('SeriesInstanceUID')
        imgs, labs, vol, first, last, typ, left, n = [], [], [], [], [], [], [], 0
        for v, r in enumerate(idx.itertuples()):
            folder = os.path.join(cache, 'skm_tea', r.StudyInstanceUID)
            imgs.append(np.load(os.path.join(folder, r.SeriesInstanceUID + '.npy')))
            labs.append(np.load(os.path.join(folder, r.seg_file)))
            d = len(imgs[-1])
            vol += [v] * d; first += [n] * d; last += [n + d - 1] * d
            typ += [PLANES.index(r.plane) * 2 + int(r.echo == 2)] * d; left += [r.side == 'L'] * d
            n += d
        self.images = torch.from_numpy(np.concatenate(imgs)).to('cuda')
        self.labels = torch.from_numpy(np.concatenate(labs)).to('cuda')
        self.vol, self.first, self.last, self.type, self.left = (torch.tensor(a).to('cuda')
                                                                 for a in (vol, first, last, typ, left))
        self.series = idx.SeriesInstanceUID.tolist()
        self.series_echo, self.series_plane = idx.echo.to_numpy(), idx.plane.to_numpy()
        self.fg = torch.nonzero(self.labels.flatten(1).amax(1) > 0).flatten()   # slices with any tissue

    def batch(self, z):
        """Slices z [N] -> a batch for GpuAugment: one window (z-1, z, z+1) per sample; each sample is its own study.
        seg_mask: the one-hot tissue labels of the center slice [N, K = 1, n_tissues, H, W]."""
        w = (z[:, None] + torch.arange(-1, 2, device=z.device)).clamp(self.first[z, None], self.last[z, None])
        onehot = self.labels[z][:, None] == torch.arange(1, len(TISSUES) + 1, device=z.device)[None, :, None, None]
        pos = (z - self.first[z]).float() / (self.last[z] - self.first[z]).clamp(min=1).float()
        return {'images': self.images[w][:, None], 'types': self.type[z], 'flip': self.left[z],
                'pos': pos[:, None], 'study_idx': torch.arange(len(z), device=z.device),
                'seg_mask': onehot[:, None].to(torch.uint8)}


def conv_bn(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SegNet(nn.Module):
    """The encoder of KneeNet (same timm model and weights) + a U-Net decoder; also logits on the stage-4 cells."""

    def __init__(self, backbone, pretrained_path, drop_path=0.1, decoder_dim=128, n_classes=len(TISSUES) + 1):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool='',
                                         drop_path_rate=drop_path)
        if pretrained_path:
            load_torchvision_weights(self.encoder, pretrained_path)
        chs = [fi['num_chs'] for fi in self.encoder.feature_info]   # stem (stride 2), stages 1-4 (stride 4-32)
        self.lateral = nn.ModuleList(nn.Conv2d(c, decoder_dim, 1) for c in chs)
        self.fuse = nn.ModuleList(conv_bn(decoder_dim, decoder_dim) for _ in chs[:-1])
        self.out = nn.Conv2d(decoder_dim, n_classes, 1)
        self.cells = nn.Conv2d(chs[-1], n_classes, 1)

    def forward(self, x):
        e = self.encoder
        f = [e.act1(e.bn1(e.conv1(x)))]
        y = e.maxpool(f[0])
        for i in range(1, 5):
            y = getattr(e, f'layer{i}')(y)
            f.append(y)
        y = self.lateral[-1](f[-1])
        for i in range(len(f) - 2, -1, -1):   # stride 16, 8, 4, 2
            y = F.interpolate(y, size=f[i].shape[-2:], mode='bilinear', align_corners=False) + self.lateral[i](f[i])
            y = self.fuse[i](y)
        return F.interpolate(self.out(y), size=x.shape[-2:], mode='bilinear', align_corners=False), self.cells(f[-1])


def with_background(soft):
    """Soft tissue targets [N, 6, H, W] -> [N, 7, H, W]: background = the share that no tissue covers."""
    return torch.cat([(1 - soft.sum(1, keepdim=True)).clamp(min=0), soft], 1)


def seg_loss(logits, cells, soft, cell_weight):
    """Soft cross-entropy + batch Dice of the tissues (as nnU-Net) at the input size, and the cross-entropy of the
    stage-4 cells against the tissue share of each cell."""
    t = with_background(soft)
    logp = F.log_softmax(logits.float(), 1)
    ce = -(t * logp).sum(1).mean()
    p = logp.exp()[:, 1:]
    inter, den = (p * soft).sum((0, 2, 3)), (p + soft).sum((0, 2, 3))
    dice = 1 - ((2 * inter + 1) / (den + 1)).mean()
    tc = F.adaptive_avg_pool2d(t, cells.shape[-2:])
    ce_cells = -(tc * F.log_softmax(cells.float(), 1)).sum(1).mean()
    return ce + dice + cell_weight * ce_cells


@torch.no_grad()
def evaluate(model, vols, aug, batch_size=128):
    """3D Dice of each tissue on each test volume (arg max of the logits against the arg max of the transformed mask, at
    the input size), mean over the volumes; also the mean Dice of each echo and each plane."""
    model.eval()
    inter = torch.zeros(len(vols.series), len(TISSUES), device='cuda')
    total = torch.zeros_like(inter)
    for s in range(0, len(vols.images), batch_size):
        z = torch.arange(s, min(s + batch_size, len(vols.images)), device='cuda')
        b = vols.batch(z)
        x, _ = aug(b, train=False)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(x[:, 0].contiguous(memory_format=torch.channels_last))
        pred, tgt = logits.argmax(1), with_background(b['seg_target'][:, 0]).argmax(1)
        for c in range(1, len(TISSUES) + 1):
            pc, tc = (pred == c).flatten(1), (tgt == c).flatten(1)
            inter[:, c - 1].index_add_(0, vols.vol[z], (pc & tc).sum(1).float())
            total[:, c - 1].index_add_(0, vols.vol[z], (pc.sum(1) + tc.sum(1)).float())
    dice = torch.where(total > 0, 2 * inter / total.clamp(min=1), torch.nan).cpu().numpy()   # [volumes, tissues]
    out = {f'dice {k}': float(np.nanmean(dice[:, i])) for i, k in enumerate(TISSUES)}
    out['dice mean'] = float(np.nanmean(dice))
    for e in (1, 2):
        out[f'dice echo {e}'] = float(np.nanmean(dice[vols.series_echo == e]))
    for p in np.unique(vols.series_plane):
        out[f'dice {p.lower()}'] = float(np.nanmean(dice[vols.series_plane == p]))
    return out


def plot(out_dir, steps, log):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    if len(steps):
        s = pd.DataFrame(steps)
        ax[0].plot(s.epoch_f, s.loss, alpha=0.25, lw=0.5, label='train loss (step)')
        ax[0].plot(s.epoch_f, s.loss.ewm(alpha=0.02).mean(), label='train loss (EMA)')
    if len(log):
        lg = pd.DataFrame(log)
        for k in ['mean'] + TISSUES:
            ax[1].plot(lg.epoch, lg[f'dice {k}'], 'o-', lw=2.5 if k == 'mean' else 1, label=k)
    ax[0].set_xlabel('epoch'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel('epoch'); ax[1].set_ylabel('test Dice (EMA model)'); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'curves.png'), dpi=80); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    ap.add_argument('--resume', default=None, help='experiment folder to continue')
    ap.add_argument('--set', nargs='*', default=[], help='config overrides key.sub=value')
    args = ap.parse_args()
    if args.resume:
        out_dir = args.resume
        cfg = yaml.safe_load(open(os.path.join(out_dir, 'config.yaml')))
    else:
        cfg = load_config(args.config, args.set)
        out_dir = os.path.join(REPO, 'experiments', f"{datetime.now().strftime('%m_%d_%H_%M')}_{cfg['name']}")
        os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
        yaml.safe_dump(cfg, open(os.path.join(out_dir, 'config.yaml'), 'w'), sort_keys=False)
    d, m, t = cfg['data'], cfg['model'], cfg['train']
    torch.manual_seed(t['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True   # fixed input size

    train = Volumes(d['cache'], d['train_splits'], d['planes'])
    test = Volumes(d['cache'], d['test_splits'], d['planes'])
    print(f'{out_dir}\ntrain {len(train.series)} volumes ({len(train.images)} slices, {len(train.fg)} with tissue) | '
          f'test {len(test.series)} volumes', flush=True)
    model = SegNet(m['backbone'], m['pretrained_path'], m['drop_path'], m['decoder_dim']).cuda()
    model = model.to(memory_format=torch.channels_last)
    aug = GpuAugment(mean=model.encoder.pretrained_cfg['mean'], std=model.encoder.pretrained_cfg['std'],
                     **cfg['input']['aug'])
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(param_groups(model, t['lr_backbone'], t['lr_head'], t['weight_decay']), betas=(0.9, 0.999))
    ipe = t['iters_per_epoch']
    total, warm = t['epochs'] * ipe, int(t['warmup_epochs'] * ipe)
    base_lrs = [g['lr'] for g in opt.param_groups]

    def set_lr(step):   # as knee.training.train: linear warm-up, cosine decay
        f = step / max(warm, 1) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g['lr'] = lr * max(f, t['min_lr_ratio'])

    gstep, log, steps, pending = 0, [], [], []
    last = os.path.join(out_dir, 'checkpoints', 'last.pt')
    if os.path.exists(last):
        st = torch.load(last, map_location='cuda', weights_only=False)
        model.load_state_dict(st['model']); ema.load_state_dict(st['ema']); opt.load_state_dict(st['opt'])
        gstep, log, steps = st['gstep'], st['log'], st['steps']
        print(f'resumed at step {gstep}', flush=True)

    def checkpoint():
        steps.extend({'epoch_f': e, 'loss': float(v)} for e, v in pending)
        pending.clear()
        save_ckpt(last, model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), gstep=gstep, log=log,
                  steps=steps)

    gen = torch.Generator(device='cuda')
    n = t['batch_size']
    last_save = t0 = time.time()
    while gstep < total:
        model.train()
        gen.manual_seed(t['seed'] * 1_000_003 + gstep)   # the same slices after a resume
        fg = torch.rand(n, device='cuda', generator=gen) < d['p_foreground']
        z = torch.where(fg, train.fg[torch.randint(len(train.fg), (n,), device='cuda', generator=gen)],
                        torch.randint(len(train.images), (n,), device='cuda', generator=gen))
        b = train.batch(z)
        x, _ = aug(b, train=True)
        set_lr(gstep)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, cells = model(x[:, 0].contiguous(memory_format=torch.channels_last))
        loss = seg_loss(logits, cells, b['seg_target'][:, 0], t['cell_weight'])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t['grad_clip'])
        opt.step()
        with torch.no_grad():   # EMA of the weights, as knee.training.train
            dcy = min(t['ema_decay'], (1 + gstep) / (10 + gstep))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - dcy)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        gstep += 1
        pending.append((gstep / ipe, loss.detach()))
        if gstep % 50 == 0:   # one GPU sync per 50 steps
            steps.extend({'epoch_f': e, 'loss': float(v)} for e, v in pending)
            pending.clear()
            print(f"step {gstep}/{total} loss {np.mean([s['loss'] for s in steps[-50:]]):.4f} "
                  f"{50 * n / (time.time() - t0):.0f} slices/s", flush=True)
            t0 = time.time()
        if gstep % (t['eval_every'] * ipe) == 0 or gstep == total:
            r = evaluate(ema, test, aug)
            log.append({'epoch': gstep / ipe, **r})
            pd.DataFrame(log).to_csv(os.path.join(out_dir, 'log.csv'), index=False)
            print(f"== epoch {gstep / ipe:.0f}: test Dice " + ', '.join(f'{k[5:]} {v:.3f}' for k, v in r.items()),
                  flush=True)
            checkpoint(); last_save = time.time()
            pd.DataFrame(steps).to_csv(os.path.join(out_dir, 'steps.csv'), index=False)
            plot(out_dir, steps, log)
        elif time.time() - last_save > 60 * t['ckpt_minutes']:
            checkpoint(); last_save = time.time()
    path = os.path.join(REPO, cfg['save_encoder'])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_ckpt(path, model=ema.encoder.state_dict())   # {'model': state dict}: load_torchvision_weights reads it
    print('encoder ->', path, flush=True)
    save_ckpt(os.path.join(REPO, cfg['save_model']), model=ema.state_dict(), cfg=cfg)   # knee.dataloading.tissue_rois
    print('model ->', cfg['save_model'], flush=True)


if __name__ == '__main__':
    main()
