"""GPU augmentation, crop, resize and normalization of a collated batch (see knee.dataloading.dataset).

The input is uint8 [N, K, 3, H0, W0] (N series, K slice windows) in the canonical frame (knee.dataloading.canonical).
One spatial transform is drawn for each series and applied to all its K windows. The anatomical mirror and the up / down
and front / back flips are drawn for each study and applied to all its series (the mirror: sagittal series reverse the
slice order, coronal and axial series flip the columns). The intensity transforms follow the nnU-Net default set and
probabilities, drawn for each series. Tissue masks in batch['seg_mask'] (uint8 one-hot [N, K, C, H0, W0]) get the same
flips and spatial transform: batch['seg_target'] holds the share of each output pixel that each class covers.
"""
import math

import torch
import torch.nn.functional as F

SAGITTAL = 0
# up / down and front / back flips (nnU-Net mirrors every axis): the image axis of each plane (index in PLANES) in the
# canonical frame (knee.dataloading.canonical.AXES): rows, columns or the slice order
FLIP_UP_DOWN = ('rows', 'rows', 'slices')            # sagittal, coronal, axial
FLIP_FRONT_BACK = ('cols', 'slices', 'rows')


def mirror(x, pos, planes, flip):
    """Anatomical left-right mirror of the series where flip is True. x [N, K, 3, H, W], pos [N, K]."""
    sag, other = flip & (planes == SAGITTAL), flip & (planes != SAGITTAL)
    if sag.any():
        x[sag] = x[sag].flip(1).flip(2)          # slice order and the 3 slices of each window
        pos[sag] = (1 - pos[sag]).flip(1)
    if other.any():
        x[other] = x[other].flip(-1)
    return x, pos


def flip_axis(x, pos, planes, flip, axes):
    """Flip the series where flip is True along one anatomical axis (axes: the image axis of each plane: 'rows',
    'cols' or 'slices'). The 3 slices of a window follow the slice order. Medial and lateral do not change, so the
    labels stay."""
    for plane, axis in enumerate(axes):
        sel = flip & (planes == plane)
        if not sel.any():
            continue
        if axis == 'rows':
            x[sel] = x[sel].flip(-2)
        elif axis == 'cols':
            x[sel] = x[sel].flip(-1)
        else:
            x[sel] = x[sel].flip(1).flip(2)
            pos[sel] = (1 - pos[sel]).flip(1)
    return x, pos


def flip_masks(m, planes, flip, axes):
    """The masks [N, K, C, h, w] of flip_axis: in-plane flips flip the mask rows / columns; a slice-order flip reverses
    the windows (each mask is the one of its center slice, which stays the center)."""
    for plane, axis in enumerate(axes):
        sel = flip & (planes == plane)
        if sel.any():
            m[sel] = m[sel].flip({'rows': -2, 'cols': -1, 'slices': 1}[axis])
    return m


def _gaussian_blur(x, sigma):
    """Separable Gaussian blur of [M, C, H, W] with one sigma (pixels)."""
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, 1, -1)
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='replicate'), k.expand(c, 1, 1, -1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode='replicate'), k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


class GpuAugment:
    def __init__(self, out_px=224, crop_mm=160.0, fov_mm=176.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 p_mirror=0.5, p_rotate=0.5, rot_deg=15.0, p_scale=0.5, scale=(0.85, 1.2), translate=0.06,
                 p_noise=0.1, noise_std=(0.0, 0.05), p_blur=0.2, blur_sigma=(0.5, 1.0), p_brightness=0.15,
                 brightness=(0.75, 1.25), p_contrast=0.15, contrast=(0.75, 1.25), p_lowres=0.25, lowres=(0.5, 1.0),
                 p_gamma=0.3, p_gamma_inv=0.1, gamma=(0.7, 1.5), channels='center', p_flip_up_down=0.0,
                 p_flip_front_back=0.0):
        """out_px: the model input size (pixels). crop_mm: the input window (mm) out of the fov_mm field of the cached
        slices. mean, std: the normalization of the encoder. channels: 'center' is the only mode (the center slice of
        each window in all 3 channels: the grayscale contract of the RadImageNet weights). p_*: the probability of each
        transform (the mirror and the flips per study, the others per series); rot_deg: the rotation range."""
        assert channels == 'center', 'the center slice in all 3 channels is the only input mode'
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
        self.out_px, self.crop_mm = int(out_px), float(crop_mm)
        self.base = self.crop_mm / fov_mm   # the input window as a share of the field: output -> input coordinates

    def _per_series(self, n, p, lo, hi, device, neutral=1.0):
        """Factor for each series: uniform in [lo, hi] with probability p, else neutral."""
        f = torch.empty(n, device=device).uniform_(lo, hi)
        return torch.where(torch.rand(n, device=device) < p, f, torch.full_like(f, neutral))

    def __call__(self, batch, train, mirror_all=False):
        """(x [N, K, 3, out_px, out_px] normalized, pos [N, K]) of the batch on the GPU. train: random augmentation;
        else the fixed view, mirrored when mirror_all (test-time mirror). Adds to batch: flip_series [N] (the mirror
        of each series), flip_up_down and flip_front_back [N] (in training with these flips), theta [N, 2, 3] (output
        -> input image coordinates of every series) and seg_target when the batch has seg_mask."""
        x = batch['images'].cuda(non_blocking=True)
        pos = batch['pos'].cuda(non_blocking=True)
        types = batch['types'].cuda(non_blocking=True)
        planes = types // 2
        study_idx = batch['study_idx'].cuda(non_blocking=True)
        n, k, c, h0, w0 = x.shape
        dev = x.device
        x = x[:, :, 1:2].expand(-1, -1, 3, -1, -1)   # the center slice in all 3 channels
        x = x.float().div_(255)

        # anatomical mirror: one draw per study (every study when mirror_all), combined with batch['flip'] (the tissue
        # model's batches mark left knees, which it sees mirrored to right knees)
        n_studies = int(study_idx.max()) + 1
        flip_study = (torch.rand(n_studies, device=dev) < self.p_mirror) if train else \
            torch.full((n_studies,), mirror_all, device=dev, dtype=torch.bool)
        if 'flip' in batch:
            flip_study = flip_study ^ batch['flip'].to(dev)
        if flip_study.any():
            x, pos = mirror(x, pos, planes, flip_study[study_idx])
        batch['flip_series'] = flip_study[study_idx]   # for the token positions in patient space
        if train and (self.p_flip_up_down > 0 or self.p_flip_front_back > 0):   # one draw per study and axis
            if 'seg_mask' in batch:   # the tissue masks follow the flips
                batch['seg_mask'] = batch['seg_mask'].to(dev, non_blocking=True)
            for p, axes, key in ((self.p_flip_up_down, FLIP_UP_DOWN, 'flip_up_down'),
                                 (self.p_flip_front_back, FLIP_FRONT_BACK, 'flip_front_back')):
                f = (torch.rand(n_studies, device=dev) < p)[study_idx]
                batch[key] = f   # for the token positions in patient space
                if f.any():
                    x, pos = flip_axis(x, pos, planes, f, axes)
                    if 'seg_mask' in batch:
                        batch['seg_mask'] = flip_masks(batch['seg_mask'], planes, f, axes)

        # spatial: crop to crop_mm, rotate, scale, translate, resize to out_px (one bilinear resampling)
        ang = torch.zeros(n, device=dev)
        s = torch.ones(n, device=dev)
        txy = torch.zeros(n, 2, device=dev)
        if train:
            ang = self._per_series(n, self.p_rotate, -self.rot_deg, self.rot_deg, dev, 0.0) * math.pi / 180
            s = torch.exp(self._per_series(n, self.p_scale, math.log(self.scale[0]), math.log(self.scale[1]), dev, 0.0))
            txy = torch.empty(n, 2, device=dev).uniform_(-self.translate, self.translate)
        b = self.base                                        # R(ang) x b: output -> input coordinates
        cos, sin = torch.cos(ang) * s, torch.sin(ang) * s
        theta = torch.stack([torch.stack([cos * b, -sin * b, txy[:, 0]], 1),
                             torch.stack([sin * b, cos * b, txy[:, 1]], 1)], 1)
        batch['theta'] = theta   # output -> input image coordinates of every series
        px = self.out_px
        grid = F.affine_grid(theta.repeat_interleave(k, 0), (n * k, c, px, px), align_corners=False)
        x = F.grid_sample(x.view(n * k, c, h0, w0), grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        x = x.view(n, k, c, px, px)
        if 'seg_mask' in batch:   # the tissue masks, one-hot: soft targets at the output size
            batch['seg_target'] = self._boxes(batch['seg_mask'].to(dev, non_blocking=True).float(),
                                              flip_study[study_idx], planes, theta, px)

        if train:
            x = self._intensity(x)
        mean = torch.tensor(self.mean[:c], device=dev).view(1, 1, c, 1, 1)
        std = torch.tensor(self.std[:c], device=dev).view(1, 1, c, 1, 1)
        return (x - mean) / std, pos

    @staticmethod
    def _boxes(m, flip, planes, theta, size):
        """Masks [N, K, C, h, w] (0 / 1) -> the share of each cell of a size x size output grid that each class covers,
        after the mirror and the spatial transform of the images."""
        sag, other = flip & (planes == SAGITTAL), flip & (planes != SAGITTAL)
        if sag.any():
            m[sag] = m[sag].flip(1)               # slice order (as the windows)
        if other.any():
            m[other] = m[other].flip(-1)          # columns
        n, k, c, h, w = m.shape
        grid = F.affine_grid(theta, (n, k * c, size, size), align_corners=False)
        m = F.grid_sample(m.view(n, k * c, h, w), grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return m.view(n, k, c, size, size)

    def _intensity(self, x):
        n, dev = x.shape[0], x.device
        view = (n, 1, 1, 1, 1)
        # Gaussian noise
        sd = self._per_series(n, self.p_noise, *self.noise_std, dev, 0.0).view(view)
        x = x + torch.randn_like(x) * sd
        # Gaussian blur (one sigma for the selected series of this batch)
        sel = torch.rand(n, device=dev) < self.p_blur
        if sel.any():
            xs = x[sel]
            sigma = float(torch.empty(1).uniform_(*self.blur_sigma))
            x[sel] = _gaussian_blur(xs.flatten(0, 1), sigma).view_as(xs)
        # brightness (multiplicative)
        x = x * self._per_series(n, self.p_brightness, *self.brightness, dev).view(view)
        # contrast around the series mean, clipped to the range before the change
        f = self._per_series(n, self.p_contrast, *self.contrast, dev).view(view)
        m = x.mean(dim=(1, 2, 3, 4), keepdim=True)
        lo, hi = x.amin(dim=(1, 2, 3, 4), keepdim=True), x.amax(dim=(1, 2, 3, 4), keepdim=True)
        x = torch.minimum(torch.maximum((x - m) * f + m, lo), hi)
        # low resolution simulation (nearest down, bilinear up)
        sel = torch.rand(n, device=dev) < self.p_lowres
        if sel.any():
            xs = x[sel]
            z = float(torch.empty(1).uniform_(*self.lowres))
            size = tuple(xs.shape[-2:])   # (rows, cols)
            small = F.interpolate(xs.flatten(0, 1), scale_factor=z, mode='nearest')
            x[sel] = F.interpolate(small, size=size, mode='bilinear', align_corners=False).view_as(xs)
        # gamma (and inverted gamma) on the [0, 1] range of each series
        for p, invert in ((self.p_gamma_inv, True), (self.p_gamma, False)):
            g = self._per_series(n, p, *self.gamma, dev).view(view)
            if bool((g != 1).any()):
                lo, hi = x.amin(dim=(1, 2, 3, 4), keepdim=True), x.amax(dim=(1, 2, 3, 4), keepdim=True)
                u = (x - lo) / (hi - lo + 1e-6)
                u = 1 - (1 - u).pow(g) if invert else u.pow(g)
                x = u * (hi - lo) + lo
        return x
