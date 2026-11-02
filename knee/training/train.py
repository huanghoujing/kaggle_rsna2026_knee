"""Train the 2.5D study model. Resumable: run the same command again (or --resume) to continue.

    .venv/bin/python -m knee.training.train --config configs/<name>.yaml --fold -1 [--set train.epochs=1 ...]
    .venv/bin/python -m knee.training.train --resume experiments/<dir>   # continue a run

fold -1 (the released models): every study with a fold trains, with include_gold also the 58 expert-labelled studies;
the 58 expert-labelled studies validate, once with the report targets (val_targets) and once with the expert labels.

Outputs in experiments/MM_DD_HH_mm_<name>_f<fold>/:
    config.yaml, log.csv (one row per evaluation), steps.csv (train loss), curves.png,
    checkpoints/last.pt (saved every ckpt_minutes and at each epoch end; holds the step inside the epoch),
    checkpoints/best.pt (EMA weights with the best validation macro AUC), checkpoints/final.pt (the final EMA weights),
    oof.parquet, gold.parquet (the final predictions of the validation and the expert-labelled studies)
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')  # batches have variable shapes

import argparse
import copy
import math
import time
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Sampler

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from knee.constants import LABELS, RAW, REPO  # noqa: E402
from knee.dataloading.dataset import StudyDataset, collate, load_joint_boxes, series_table  # noqa: E402
from knee.dataloading.gpu_augment import GpuAugment  # noqa: E402
from knee.eval.metrics import auc_table  # noqa: E402
from knee.model.knee_net import build_model  # noqa: E402


class EpochSampler(Sampler):
    """A fixed random order for each epoch (seed + epoch). `start` skips the samples done before a resume.

    With `weights`, each epoch draws n studies with replacement in proportion to the weights (oversampling).
    """

    def __init__(self, n, seed, weights=None):
        self.n, self.seed, self.epoch, self.start = n, seed, 0, 0
        self.weights = None if weights is None else torch.as_tensor(weights, dtype=torch.double)

    def set_epoch(self, epoch, start=0):
        self.epoch, self.start = epoch, start

    def __iter__(self):
        if self.weights is None:
            order = np.random.default_rng(self.seed + self.epoch).permutation(self.n)
        else:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.multinomial(self.weights, self.n, replacement=True, generator=g).numpy()
        return iter(order[self.start:].tolist())

    def __len__(self):
        return self.n - self.start


def rarity_weights(targets, power):
    """Sampling weight of each training study: the weight of its rarest finding, weighted by its probability.

    Label weight a = (largest prevalence / prevalence) ** power (>= 1; prevalence = share of targets > 0.5).
    Study weight = max(1, max over labels of target * a), normalised to mean 1.
    """
    y = targets[LABELS].values.astype(np.float64)
    known = ~np.isnan(y)   # unknown labels (NaN) count neither as positive nor as negative
    prev = np.clip(((y > 0.5) & known).sum(0) / np.maximum(known.sum(0), 1), 1e-3, None)
    a = (prev.max() / prev) ** power
    w = np.maximum(1.0, np.nan_to_num(y * a, nan=0.0).max(1))
    return w / w.mean()


def load_config(path, overrides):
    cfg = yaml.safe_load(open(path))
    for o in overrides:  # key.sub=value
        key, val = o.split('=', 1)
        d = cfg
        *parents, last = key.split('.')
        for p in parents:
            d = d.setdefault(p, {})
        d[last] = yaml.safe_load(val)
    return cfg


def build_data(cfg, fold):
    """The training, validation and expert-labelled (gold-58) datasets -> (ds_tr, ds_va, ds_gold). fold >= 0: the
    other folds train and this fold validates; fold -1: all folds train and the 58 expert-labelled studies validate."""
    d = cfg['data']
    series = series_table(d['cache'], 'train', os.path.join(RAW, 'train_series.csv'))
    targets = pd.read_parquet(d['targets'])
    folds = pd.read_csv(d['folds'])
    have = set(series.StudyInstanceUID)
    folds = folds[folds.StudyInstanceUID.isin(have)]
    tr = folds[(folds.fold != fold) & (folds.fold >= 0)].StudyInstanceUID   # fold -1: all studies (full-data model)
    if cfg['train'].get('max_train_studies'):  # debugging: a small subset (overfit test)
        tr = tr.iloc[:cfg['train']['max_train_studies']]
    va = folds[folds.fold == fold].StudyInstanceUID if fold >= 0 else folds[folds.fold == -1].StudyInstanceUID
    gold = folds[folds.fold == -1].StudyInstanceUID
    if cfg['train'].get('include_gold') and fold == -1:   # the 58 expert-labelled studies train too, with their
        tr = pd.concat([tr, gold])                         # expert labels (0 / 1) instead of the pseudo labels
        g = gold_targets().set_index('StudyInstanceUID')[LABELS]
        targets = targets.set_index('StudyInstanceUID')
        assert set(g.index) <= set(targets.index), 'the targets table lacks gold studies'
        targets.loc[g.index, LABELS] = g.to_numpy(np.float32)
        targets = targets.reset_index()
    i = cfg['input']
    common = dict(series=series, targets=targets, k_slices=i['k_slices'], max_series=i['max_series'],
                  coverage=tuple(i['coverage']), geometry=bool(cfg['model'].get('joint_layers')))
    if cfg['model'].get('joint_layers'):   # the joint transformer reads the knee joint box of each study
        common['joint_boxes'] = load_joint_boxes('train')
    ds_tr = StudyDataset(tr, train=True, series_dropout=i.get('series_dropout', 0.0),
                         coverage_jitter=i.get('coverage_jitter', 0.05), **common)
    # validation targets: the fixed report targets, also when training on pseudo labels (comparable CV)
    val_targets = pd.read_parquet(d['val_targets']) if d.get('val_targets') else targets
    ds_va = StudyDataset(va, train=False, **{**common, 'targets': val_targets})
    ds_gold = StudyDataset(gold, train=False, **{**common, 'targets': gold_targets()})
    return ds_tr, ds_va, ds_gold


def gold_targets():
    tr = pd.read_csv(os.path.join(RAW, 'train.csv'))
    return tr.dropna(subset=LABELS)[['StudyInstanceUID'] + LABELS]


def param_groups(model, lr_bb, lr_head, wd, lr_rope=None):
    """AdamW groups: backbone (encoder.*) / head, each with and without weight decay (no decay for biases, norms,
    embeddings and the label queries). With lr_rope the joint transformer (joint.*) gets its own learning rate; its
    groups come after the others."""
    groups = {('bb', True): [], ('bb', False): [], ('head', True): [], ('head', False): []}
    for name, p in model.named_parameters():
        decay = p.ndim >= 2 and 'emb' not in name and 'queries' not in name
        backbone = name.startswith('encoder.')
        joint = name.startswith('joint.')
        part = 'bb' if backbone else 'rope' if lr_rope is not None and joint else 'head'
        groups.setdefault((part, decay), []).append(p)
    lrs = {'bb': lr_bb, 'head': lr_head, 'rope': lr_rope}
    return [{'params': ps, 'lr': lrs[part], 'weight_decay': wd if decay else 0.0}
            for (part, decay), ps in groups.items() if ps]


@torch.no_grad()
def orient_targets(batch, side):
    """Where two anatomical axes point in the model input of each series after the augmentation: unit vectors
    (x = columns, y = rows) [N, 2] u and v, and a mask [N] of the series whose v is known. u: the canonical row axis
    (sagittal / coronal: superior-inferior, axial: anterior-posterior); v: the canonical column axis (sagittal:
    anterior-posterior; coronal / axial: medial-lateral, which needs the knee side: side {study: 'L' / 'R'}; a study
    without a side has no v). The in-plane flips come from the mirror and the up / down and front / back flips
    (knee.dataloading.gpu_augment); the rotation from theta (output -> input coordinates, so an input direction d shows
    as theta^T d)."""
    theta = batch['theta']
    n, dev = theta.shape[0], theta.device
    planes = batch['types'].to(dev) // 2
    lr, ud, fb = (batch[k].to(dev) for k in ('flip_series', 'flip_up_down', 'flip_front_back'))
    row_flip = torch.where(planes == 2, fb, ud)            # axial rows: front-back; sagittal / coronal rows: up-down
    col_flip = torch.where(planes == 0, fb, lr)            # sagittal columns: front-back; coronal / axial: left-right
    s = torch.tensor([{'R': 1.0, 'L': -1.0}.get(side.get(st), 0.0) for st in batch['study']], device=dev)
    s = s[batch['study_idx'].to(dev)]
    sign = torch.where(planes == 0, torch.ones(n, device=dev), s)
    known = (planes == 0) | (s != 0)
    u = torch.stack([torch.zeros(n, device=dev), 1 - 2 * row_flip.float()], 1)
    v = torch.stack([(1 - 2 * col_flip.float()) * sign, torch.zeros(n, device=dev)], 1)
    a_t = theta[:, :, :2].transpose(1, 2).float()
    return F.normalize((a_t @ u[..., None])[..., 0], dim=-1), F.normalize((a_t @ v[..., None])[..., 0], dim=-1), known


def orient_loss(pred, batch, side):
    """1 - cosine between the predicted and the true directions (orient_targets) of every window; pred [N, K, 4]."""
    u, v, known = orient_targets(batch, side)
    pu, pv = pred[..., :2].float(), pred[..., 2:].float()
    lu = 1 - F.cosine_similarity(pu, u[:, None].expand_as(pu), dim=-1)
    lv = 1 - F.cosine_similarity(pv, v[:, None].expand_as(pv), dim=-1)
    return lu.mean() + (lv * known[:, None]).sum() / (known.sum() * pred.shape[1]).clamp(min=1)


@torch.no_grad()
def extra_inputs(batch):
    """The model inputs next to the images: the series geometry with the augmentation draws of GpuAugment (theta, the
    mirror, in training also the up / down and front / back flips) for the patient-space token positions, and the knee
    joint box of each study for the joint transformer."""
    out = {}
    if 'geom' in batch and 'theta' in batch:
        out['geometry'] = (batch['geom'].cuda(non_blocking=True), batch['theta'], batch['flip_series'],
                           *([batch['flip_up_down'], batch['flip_front_back']] if 'flip_up_down' in batch else []))
    if 'joint' in batch:
        out['joint_box'] = batch['joint'].cuda(non_blocking=True)
    return out


def predict(model, loader, aug, tta_mirror=False):
    """Probabilities of the 12 findings on a loader -> (studies, probs [B, 12], labels [B, 12]). tta_mirror: the mean
    of the predictions on the images and on their left-right mirror."""
    model.eval()
    preds, studies, labels = [], [], []
    for batch in loader:
        n = len(batch['study'])
        views = [False, True] if tta_mirror else [False]   # mirror_all of each pass
        p = 0
        for mirror_all in views:
            x, pos = aug(batch, train=False, mirror_all=mirror_all)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x, batch['types'].cuda(), pos, batch['study_idx'].cuda(), n, **extra_inputs(batch))
            p = p + torch.sigmoid(logits.float()) / len(views)
        preds.append(p.cpu().numpy())
        studies += batch['study']
        labels.append(batch['labels'].numpy())
    return studies, np.concatenate(preds), np.concatenate(labels)


def plot(out_dir, steps, log):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    if len(steps):
        s = pd.DataFrame(steps)
        ax[0].plot(s.epoch_f, s.loss, alpha=0.25, lw=0.5, label='train loss (step)')
        ax[0].plot(s.epoch_f, s.loss.ewm(alpha=0.02).mean(), label='train loss (EMA)')
    if len(log):
        lg = pd.DataFrame(log)
        ax[0].plot(lg.epoch, lg.val_loss, 'o-', label='val loss')
        ax[1].plot(lg.epoch, lg.val_auc, 'o-', label='val macro AUC (report targets > 0.5)')
        ax[1].plot(lg.epoch, lg.gold_auc, 's-', label='gold-58 macro AUC')
    ax[0].set_xlabel('epoch'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel('epoch'); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'curves.png'), dpi=80); plt.close(fig)


def save_ckpt(path, **state):
    torch.save(state, path + '.tmp')
    os.replace(path + '.tmp', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    ap.add_argument('--fold', type=int, default=0, help='validation fold; -1: all studies train (the released models)')
    ap.add_argument('--resume', default=None, help='experiment folder to continue')
    ap.add_argument('--set', nargs='*', default=[], help='config overrides key.sub=value')
    args = ap.parse_args()

    if args.resume:
        out_dir = args.resume
        cfg = yaml.safe_load(open(os.path.join(out_dir, 'config.yaml')))
        fold = cfg['fold']
    else:
        cfg = load_config(args.config, args.set)
        fold = cfg['fold'] = args.fold
        stamp = datetime.now().strftime('%m_%d_%H_%M')
        out_dir = os.path.join(REPO, 'experiments', f"{stamp}_{cfg['name']}_f{fold}")
        os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
        yaml.safe_dump(cfg, open(os.path.join(out_dir, 'config.yaml'), 'w'), sort_keys=False)
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    t = cfg['train']
    torch.manual_seed(t['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ds_tr, ds_va, ds_gold = build_data(cfg, fold)
    print(f'{out_dir}\ntrain {len(ds_tr)} | val {len(ds_va)} | gold {len(ds_gold)} studies', flush=True)
    weights = None
    if t.get('sampling') == 'rarity':  # oversample the studies with rare findings
        tgt = ds_tr.targets.reindex(ds_tr.studies)
        weights = rarity_weights(tgt, t.get('rarity_power', 0.5))
        print(f'rarity sampling: weights {weights.min():.2f}-{weights.max():.2f}', flush=True)
    if t.get('include_gold'):   # the 58 expert-labelled studies are drawn gold_weight times as often
        is_gold = np.isin(ds_tr.studies, gold_targets().StudyInstanceUID)
        weights = np.where(is_gold, t.get('gold_weight', 4.0), 1.0) * (1.0 if weights is None else weights)
        print(f'expert studies in training: {int(is_gold.sum())}, weight x {t.get("gold_weight", 4.0)}', flush=True)
    loader_kw = dict(collate_fn=collate, num_workers=t['num_workers'], pin_memory=True, persistent_workers=True,
                     prefetch_factor=4)
    sampler = EpochSampler(len(ds_tr), t['seed'], weights)
    dl_tr = DataLoader(ds_tr, batch_size=t['batch_size'], sampler=sampler, drop_last=True, **loader_kw)
    dl_va = DataLoader(ds_va, batch_size=t['eval_batch_size'], shuffle=False, **loader_kw)
    dl_gold = DataLoader(ds_gold, batch_size=t['eval_batch_size'], shuffle=False, **loader_kw)

    m = cfg['model']
    model = build_model(m).cuda()
    if t.get('init_from'):   # start from a trained model of the same architecture (v76: from single model 8)
        sd = torch.load(t['init_from'], map_location='cpu', weights_only=False)['model']
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"init from {t['init_from']}: new (not loaded) {missing}; not used {unexpected}", flush=True)
    orient_side = None   # the left / right knee of each study: only the target of the medial direction of the windows
    if m.get('orient_head'):
        orient_side = pd.read_parquet(cfg['input']['orient_side']).set_index('StudyInstanceUID').side.to_dict()
    aug = GpuAugment(mean=model.encoder.pretrained_cfg['mean'], std=model.encoder.pretrained_cfg['std'],
                     **cfg['input']['aug'])
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    groups = param_groups(model, t['lr_backbone'], t['lr_head'], t['weight_decay'], t.get('lr_rope'))
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    steps_per_epoch = sampler.n // t['batch_size']   # the sampler draws one study per training study and epoch
    total = steps_per_epoch * t['epochs']
    warm = int(steps_per_epoch * t['warmup_epochs'])
    base_lrs = [g['lr'] for g in opt.param_groups]

    def set_lr(step):   # linear warm-up, then a cosine decay to min_lr_ratio of the peak
        f = step / max(warm, 1) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g['lr'] = lr * max(f, t.get('min_lr_ratio', 0.01))

    epoch, step_in_epoch, gstep, best, log, steps, pending = 0, 0, 0, -1.0, [], [], []
    last = os.path.join(ckpt_dir, 'last.pt')
    if os.path.exists(last):
        st = torch.load(last, map_location='cuda', weights_only=False)
        model.load_state_dict(st['model']); ema.load_state_dict(st['ema']); opt.load_state_dict(st['opt'])
        epoch, step_in_epoch, gstep, best = st['epoch'], st['step_in_epoch'], st['gstep'], st['best']
        log, steps = st['log'], st['steps']
        print(f'resumed at epoch {epoch} step {step_in_epoch} (global {gstep})', flush=True)

    def flush():   # the losses not yet logged (one GPU sync)
        steps.extend({'epoch_f': e, 'loss': float(v), **{k: float(x) for k, x in p.items()}} for e, v, p in pending)
        pending.clear()

    def checkpoint():
        flush()
        save_ckpt(last, model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), epoch=epoch,
                  step_in_epoch=step_in_epoch, gstep=gstep, best=best, log=log, steps=steps)

    ema_decay, last_save = t['ema_decay'], time.time()
    while epoch < t['epochs']:
        model.train()
        sampler.set_epoch(epoch, start=step_in_epoch * t['batch_size'])
        t0 = time.time()
        for batch in dl_tr:
            x, pos = aug(batch, train=True)
            y = batch['labels'].cuda(non_blocking=True)
            set_lr(gstep)
            known = ~torch.isnan(y)   # a label without a value is NaN: no loss
            parts = {}   # loss parts for steps.csv
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(x, batch['types'].cuda(non_blocking=True), pos, batch['study_idx'].cuda(non_blocking=True),
                            y.shape[0], return_aux=True, return_orient=bool(m.get('orient_head')),
                            **extra_inputs(batch))
            logits = out[0]
            loss = F.binary_cross_entropy_with_logits(logits.float()[known], y[known])
            if m.get('orient_head'):   # the orientation of each window after the flips and rotation
                parts['orient'] = orient_loss(out[2]['orient'], batch, orient_side)
                loss = loss + t.get('orient_weight', 0.1) * parts['orient']
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t['grad_clip'])
            opt.step()
            with torch.no_grad():  # EMA of the weights; short warm-up of the decay
                d = min(ema_decay, (1 + gstep) / (10 + gstep))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            gstep += 1
            step_in_epoch += 1
            pending.append((epoch + step_in_epoch / steps_per_epoch, loss.detach(),
                            {k: v.detach() for k, v in parts.items()}))
            if step_in_epoch % 50 == 0 or step_in_epoch >= steps_per_epoch:  # one GPU sync per 50 steps
                flush()
            if step_in_epoch % 50 == 0:
                rate = step_in_epoch * t['batch_size'] / (time.time() - t0)
                recent = np.mean([s['loss'] for s in steps[-50:]])
                print(f'epoch {epoch} step {step_in_epoch}/{steps_per_epoch} loss {recent:.4f} {rate:.1f} studies/s',
                      flush=True)
            if time.time() - last_save > 60 * t['ckpt_minutes']:
                checkpoint(); last_save = time.time()
            if step_in_epoch >= steps_per_epoch:
                break
        epoch += 1
        step_in_epoch = 0
        if epoch % t['eval_every'] == 0 or epoch == t['epochs']:
            _, pv, yv = predict(ema, dl_va, aug)
            _, pg, yg = predict(ema, dl_gold, aug)
            va, go = auc_table(yv, pv), auc_table(yg, pg)
            ok = np.isfinite(yv)   # a finding without a target is NaN
            vloss = float(F.binary_cross_entropy(torch.tensor(pv[ok]).clamp(1e-6, 1 - 1e-6), torch.tensor(yv[ok])))
            row = {'epoch': epoch, 'val_loss': vloss, 'val_auc': va['macro'], 'gold_auc': go['macro'],
                   'time_min': (time.time() - t0) / 60, **{f'val_{k}': va[k] for k in LABELS},
                   **{f'gold_{k}': go[k] for k in LABELS}}
            log.append(row)
            pd.DataFrame(log).to_csv(os.path.join(out_dir, 'log.csv'), index=False)
            print(f"== epoch {epoch}: val loss {vloss:.4f} val AUC {va['macro']:.4f} gold AUC {go['macro']:.4f}",
                  flush=True)
            if va['macro'] > best:
                best = va['macro']
                save_ckpt(os.path.join(ckpt_dir, 'best.pt'), model=ema.state_dict(), cfg=cfg, epoch=epoch, val_auc=best)
        pd.DataFrame(steps).to_csv(os.path.join(out_dir, 'steps.csv'), index=False)
        plot(out_dir, steps, log)
        checkpoint(); last_save = time.time()

    # final predictions of the last EMA weights; input.tta_mirror (default on): also on the left-right mirror
    tta = cfg['input'].get('tta_mirror', True)
    for name, dl in (('oof', dl_va), ('gold', dl_gold)):
        s, p, y = predict(ema, dl, aug, tta_mirror=tta)
        df = pd.DataFrame(p, columns=LABELS)
        df.insert(0, 'StudyInstanceUID', s)
        df.to_parquet(os.path.join(out_dir, f'{name}.parquet'), index=False)
        print(f'final {name}{" (mirror TTA)" if tta else ""} macro AUC {auc_table(y, p)["macro"]:.4f}', flush=True)
    save_ckpt(os.path.join(ckpt_dir, 'final.pt'), model=ema.state_dict(), cfg=cfg, epoch=epoch)


if __name__ == '__main__':
    main()
