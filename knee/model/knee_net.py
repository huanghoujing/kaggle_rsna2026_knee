"""The 2.5D study model: a ResNet-50 reads the 3-slice windows of all series of a study, label queries read its cells.

1. The encoder (timm ResNet-50, RadImageNet weights) encodes every 3-slice window of every series:
   [N*K, 3, H, W] -> the last feature map [N*K, C, h, w] (N series, K windows per series).
2. Every cell of that map is a token: a 1 x 1 projection plus a learned position table (resized to h x w), plus the
   embedding of the series type and of the slice position of the window.
3. Twelve label queries (one per finding) attend over all tokens of a study (all series, all windows; the padding of
   the shorter studies is masked); a linear read-out per finding gives its logit.
4. Training only: the orientation head predicts two in-plane direction vectors of every window from the mean of the
   last feature map (knee.training.train.orient_loss), and the joint transformer fuses the stage-3 cells inside the
   knee joint box of the study from all series (patient-space positions, 3D rotary attention); the label queries read
   the fused cells through a gate. knee.submission.drop_joint removes the joint branch for test time, where the series
   geometry and the joint box are not available.
"""
import math
import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """Generalized-mean pooling of a feature map [M, C, h, w] -> [M, C], with a learned exponent p."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p, self.eps = nn.Parameter(torch.tensor(p)), eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.float().clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p).flatten(1)


def position_features(pos, n_freq=8):
    """Sin/cos features of the relative slice position pos in [0, 1]: [..., 2 * n_freq]."""
    freq = (2.0 ** torch.arange(n_freq, device=pos.device, dtype=pos.dtype)) * math.pi
    a = pos[..., None] * freq
    return torch.cat([a.sin(), a.cos()], -1)


def pack(tokens, study_idx, n_studies):
    """[N, K, D] series tokens -> [B, S_max * K, D] study tokens and a validity mask [B, S_max * K].

    The series of one study must be contiguous (knee.dataloading.dataset.collate makes them so).
    """
    n, k, d = tokens.shape
    counts = torch.bincount(study_idx, minlength=n_studies)
    starts = torch.cumsum(counts, 0) - counts
    slot = torch.arange(n, device=tokens.device) - starts[study_idx]
    s_max = int(counts.max())
    out = tokens.new_zeros(n_studies, s_max, k, d)
    out[study_idx, slot] = tokens
    mask = torch.zeros(n_studies, s_max, k, dtype=torch.bool, device=tokens.device)
    mask[study_idx, slot] = True
    return out.view(n_studies, s_max * k, d), mask.view(n_studies, s_max * k)


def load_torchvision_weights(encoder, path):
    """Load a torchvision-style ResNet checkpoint (the RadImageNet ResNet-50) into a timm encoder.

    path: a local file, or a HuggingFace file <repo_id>/<filename> (fetched into the HuggingFace cache). Accepts a
    plain state dict or {'model': state dict}; strips the prefix '_orig_mod.' (torch.compile) and drops the classifier.
    """
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        repo, fname = path.rsplit('/', 1)
        path = hf_hub_download(repo, fname)
    sd = torch.load(path, map_location='cpu', weights_only=False)
    sd = sd.get('model', sd)
    out = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    out = {k: v for k, v in out.items() if not k.startswith('fc.')}
    missing, unexpected = encoder.load_state_dict(out, strict=False)
    assert not unexpected and not [m for m in missing if not m.startswith('fc.')], (missing, unexpected)


def rope_angles(p, n_freq, min_mm=8.0, max_mm=512.0):
    """Patient-space positions p [B, T, 3] (mm) -> rotation angles [B, T, 3 * n_freq] for 3D rotary attention:
    n_freq wavelengths per axis, geometric from min_mm to max_mm."""
    steps = torch.arange(n_freq, device=p.device, dtype=torch.float32) / max(n_freq - 1, 1)
    lam = min_mm * (max_mm / min_mm) ** steps
    return (p.float()[..., None] * (2 * math.pi / lam)).flatten(-2)


def apply_rope(x, ang):
    """x [B, H, T, d]: its first 2 * A channels (pairs) rotated by ang [B, T, A]; the other channels unchanged."""
    a = ang.shape[-1]
    xr, xp = x[..., :2 * a], x[..., 2 * a:]
    x1, x2 = xr[..., 0::2], xr[..., 1::2]
    cos, sin = ang.cos()[:, None].to(x.dtype), ang.sin()[:, None].to(x.dtype)
    r = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)
    return torch.cat([r, xp], -1)


class RoPE3DLayer(nn.Module):
    """Pre-LN transformer layer over the joint tokens of a study: the self-attention rotates the queries and keys by the
    3D patient-space position of each token, so the attention depends on the relative position of two tokens in the
    knee (across slices and series). The queries and keys of each head go through an RMSNorm before the rotation,
    against growing attention logits. The output projections start at zero: the layer starts as the identity."""

    def __init__(self, dim, heads, dropout, n_freq=10):
        super().__init__()
        self.heads, self.hd = heads, dim // heads
        assert 6 * n_freq <= self.hd, 'rope: 3 axes x n_freq pairs must fit in the head dimension'
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.q_norm, self.k_norm = nn.RMSNorm(self.hd), nn.RMSNorm(self.hd)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)
        for lin in (self.proj, self.ff[2]):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x, ang, mask):
        """x [B, T, D], ang [B, T, 3 * n_freq], mask [B, T] (True = token)."""
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(self.q_norm(q), ang), apply_rope(self.k_norm(k), ang)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])   # True = may attend
        x = x + self.drop(self.proj(o.transpose(1, 2).reshape(b, t, d)))
        return x + self.drop(self.ff(self.ln2(x)))


class KneeNet(nn.Module):
    """The study model (see the module docstring). Only the spatial head without a token transformer is supported
    (head 'spatial', depth 0): the label queries read the cells of the last feature map directly.

    cache_half_mm: half the side of the cached square image (mm); the augmentation affine (theta) maps the model input
    onto it, so theta x cache_half_mm gives the position of a cell in the series image, for the joint transformer.
    rope_freq: the wavelengths per axis of the 3D rotary attention of the joint transformer.
    """

    def __init__(self, backbone='resnet50.a1_in1k', pretrained=True, dim=384, depth=0, heads=6, n_types=6,
                 n_labels=12, drop_path=0.1, dropout=0.1, head='spatial', pretrained_path=None, cache_half_mm=88.0,
                 rope_freq=10, joint_layers=0, joint_margin_mm=10.0, joint_max_tokens=12000, orient_head=False):
        super().__init__()
        assert depth == 0 and head == 'spatial', 'the model is the spatial head without a token transformer'
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool='',
                                         drop_path_rate=drop_path)
        if pretrained_path:
            load_torchvision_weights(self.encoder, pretrained_path)
        c = self.encoder.num_features
        # the orientation head (training only): two in-plane unit vectors (x, y) of every window after the flips and
        # the rotation, from the mean of the last feature map (knee.training.train.orient_loss)
        self.orient = nn.Sequential(nn.Linear(c, 256), nn.GELU(), nn.Linear(256, 4)) if orient_head else None
        # GeM pooling and the projection of the pooled feature: the spatial head does not use them; kept because the
        # released checkpoints hold their weights (pool.p, proj.*)
        self.pool = GeM()
        self.proj = nn.Sequential(nn.Linear(c, dim), nn.LayerNorm(dim))
        # every cell of the last feature map is a token: a 1 x 1 projection + a position table resized to the map
        self.spatial_proj = nn.Conv2d(c, dim, 1)
        self.spatial_pos = nn.Parameter(torch.zeros(1, dim, 16, 16))
        # the joint transformer (training only): the stage-3 cells inside the knee joint of the study (its joint box +
        # margin, patient space) from all series, fused by joint_layers rotary self-attention layers; the label queries
        # read them through their own attention, added through a gate that starts at zero (the start is the base reader)
        self.joint, self.cache_half_mm, self.rope_freq = None, cache_half_mm, rope_freq
        if joint_layers > 0:
            from knee.dataloading.canonical import AXES
            from knee.constants import PLANES
            info = {fi['module']: fi['num_chs'] for fi in self.encoder.feature_info}
            assert 'layer3' in info, 'the joint transformer reads the stage-3 maps of a ResNet'
            self.register_buffer('plane_axes', torch.tensor([AXES[p] for p in PLANES], dtype=torch.float32),
                                 persistent=False)   # [plane, (column, row, slice) axis, xyz]
            self.joint = nn.ModuleDict(dict(
                proj=nn.Conv2d(info['layer3'], dim, 1), norm=nn.LayerNorm(dim),
                layers=nn.ModuleList([RoPE3DLayer(dim, heads, dropout, rope_freq) for _ in range(joint_layers)]),
                attn=nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)))
            self.joint_gate = nn.Parameter(torch.zeros(n_labels, 1))
            self.joint_margin_mm, self.joint_max_tokens = joint_margin_mm, joint_max_tokens
        # the embedding of the series type and of the slice position of a window, added to its tokens
        self.type_emb = nn.Embedding(n_types, dim)
        self.pos_proj = nn.Linear(16, dim)
        self.norm = nn.LayerNorm(dim)
        # one label query per finding, one cross-attention over the tokens, one linear read-out per finding
        self.queries = nn.Parameter(torch.randn(n_labels, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.head_w = nn.Parameter(torch.randn(n_labels, dim) * 0.02)
        self.head_b = nn.Parameter(torch.zeros(n_labels))
        self.dropout = nn.Dropout(dropout)

    def features(self, x):
        """[N, K, 3, H, W] -> the last feature map of the encoder [N*K, C, h, w]."""
        return self.encoder.forward_features(x.flatten(0, 1).contiguous(memory_format=torch.channels_last))

    def stage_maps(self, x):
        """[N, K, 3, H, W] -> the stage-3 and stage-4 maps of the ResNet, [N*K, C3, h3, w3] and [N*K, C4, h4, w4] (the
        operations of timm ResNet.forward_features, keeping the stage-3 output for the joint transformer)."""
        e = self.encoder
        x = x.flatten(0, 1).contiguous(memory_format=torch.channels_last)
        f3 = e.layer3(e.layer2(e.layer1(e.maxpool(e.act1(e.bn1(e.conv1(x)))))))
        return f3, e.layer4(f3)

    def spatial_tokens(self, f, n, k):
        """The last feature map [N*K, C, h, w] -> the cell tokens [N, K, h*w, D], and (h, w)."""
        h, w = f.shape[-2:]
        t = self.spatial_proj(f) + F.interpolate(self.spatial_pos, size=(h, w), mode='bilinear', align_corners=False)
        return t.flatten(2).transpose(1, 2).reshape(n, k, h * w, -1), (h, w)

    def read(self, h, mask, n_studies, joint=None):
        """The label queries read the normalized tokens h [B, T, D] (mask [B, T]: True = token) -> one logit per
        finding [B, n_labels]. joint = (fused joint tokens [B, Tj, D], their mask [B, Tj]): read by the queries through
        their own attention and added through the gate."""
        q = self.queries[None].expand(n_studies, -1, -1)
        o, _ = self.attn(q, h, h, key_padding_mask=~mask, need_weights=False)
        if joint is not None:
            hj, mj = joint
            oj, _ = self.joint['attn'](q, hj, hj, key_padding_mask=~mj, need_weights=False)
            o = o + self.joint_gate * oj
        return (self.dropout(o) * self.head_w).sum(-1) + self.head_b

    def series_emb(self, types, pos):
        """[N, K, D] embedding of the series type [N] and the slice position [N, K] of every window."""
        return self.type_emb(types)[:, None] + self.pos_proj(position_features(pos))

    def token_positions(self, geometry, pos, h, w):
        """The patient-space (LPS, mm) position of every cell of every window [N, K, h*w, 3], the augmentation undone.

        geometry = (geom [N, 6]: image center xyz, first / last slice position along the canonical slice axis, plane;
        theta [N, 2, 3]: the augmentation affine (output -> input image coordinates); flip [N]: the mirror of the
        series (applied before the affine); optionally flip_up_down, flip_front_back [N]: the other flips of
        knee.dataloading.gpu_augment, each on one image axis per plane)."""
        geom, theta, flip = geometry[:3]
        fud, ffb = geometry[3:5] if len(geometry) > 3 else (torch.zeros_like(flip), torch.zeros_like(flip))
        dev = pos.device
        ys, xs = torch.meshgrid((torch.arange(h, device=dev, dtype=torch.float32) + 0.5) / h * 2 - 1,
                                (torch.arange(w, device=dev, dtype=torch.float32) + 0.5) / w * 2 - 1, indexing='ij')
        grid = torch.stack([xs.flatten(), ys.flatten(), torch.ones(h * w, device=dev)], 0)        # [3, h*w]
        uv = (theta.float() @ grid) * self.cache_half_mm                                           # [N, 2, h*w] mm
        plane = geom[:, 5].long()
        sag, cor, axi = plane == 0, plane == 1, plane == 2
        # undo the flips (gpu_augment.mirror, FLIP_UP_DOWN, FLIP_FRONT_BACK): columns, rows or slice order per plane
        flip_u = (flip & ~sag) | (ffb & sag)
        flip_v = (fud & (sag | cor)) | (ffb & axi)
        flip_s = (flip & sag) | (fud & axi) | (ffb & cor)
        u = torch.where(flip_u[:, None], -uv[:, 0], uv[:, 0])
        v = torch.where(flip_v[:, None], -uv[:, 1], uv[:, 1])
        rel = torch.where(flip_s[:, None], 1 - pos.float(), pos.float())
        ax = self.plane_axes[plane]                                             # [N, 3, 3]
        center, first, last = geom[:, :3], geom[:, 3:4], geom[:, 4:5]
        ds = (rel - 0.5) * (last - first)                                       # [N, K] mm from the middle slice
        return (center[:, None, None] + u[:, None, :, None] * ax[:, None, None, 0]                   # [N, K, h*w, 3]
                + v[:, None, :, None] * ax[:, None, None, 1] + ds[:, :, None, None] * ax[:, None, None, 2])

    def joint_tokens(self, f3, emb, geometry, pos, study_idx, n_studies, box):
        """The stage-3 maps [N*K, C, h, w] -> the fused joint tokens of each study [B, T, D] and their mask [B, T]:
        the cells inside the study's joint box (+ margin; true patient positions, the augmentation undone), at most
        joint_max_tokens per study (training: a random subset, else a fixed one), the rotary positions from the joint
        center with the flips of the study (as its images)."""
        assert geometry is not None and box is not None, 'the joint transformer needs the geometry and the joint boxes'
        n, k = pos.shape
        t = self.joint['proj'](f3)
        h, w = t.shape[-2:]
        t = t.flatten(2).transpose(1, 2).reshape(n, k, h * w, -1) + emb                     # [N, K, T, D]
        p = self.token_positions(geometry, pos, h, w)                                        # [N, K, T, 3] LPS mm
        b = box.float()[study_idx]                                                           # [N, 6] lo, hi
        m = self.joint_margin_mm
        inside = ((p >= b[:, None, None, :3] - m) & (p <= b[:, None, None, 3:] + m)).all(-1)
        q = p - ((b[:, :3] + b[:, 3:]) / 2)[:, None, None]
        flip, fud, ffb = (geometry[i] if len(geometry) > i else torch.zeros_like(geometry[2]) for i in (2, 3, 4))
        q = q * torch.stack([1 - 2 * flip.float(), 1 - 2 * ffb.float(), 1 - 2 * fud.float()], -1)[:, None, None]
        sel = inside.flatten(1)                                                              # [N, K*T]
        tok, pq = t.flatten(1, 2)[sel], q.flatten(1, 2)[sel]
        sid = study_idx[:, None].expand(-1, sel.shape[1])[sel]
        dev = tok.device
        idx = torch.arange(len(sid), device=dev)
        key = (torch.rand(len(sid), device=dev) if self.training   # a fixed pseudo-random order at test time
               else (idx * 2654435761 % 4294967296).float() / 4294967296)
        order = torch.argsort(sid.float() * 2 + key)
        s_sorted = sid[order]
        counts = torch.bincount(s_sorted, minlength=n_studies)
        rank = torch.arange(len(order), device=dev) - (torch.cumsum(counts, 0) - counts)[s_sorted]
        keep = rank < self.joint_max_tokens
        order, s_sorted, rank = order[keep], s_sorted[keep], rank[keep]
        t_max = max(1, int(min(counts.max(), self.joint_max_tokens)))
        out = tok.new_zeros(n_studies, t_max, tok.shape[-1])
        pos3 = pq.new_zeros(n_studies, t_max, 3)
        mask = torch.zeros(n_studies, t_max, dtype=torch.bool, device=dev)
        out[s_sorted, rank], pos3[s_sorted, rank], mask[s_sorted, rank] = tok[order], pq[order], True
        mask[:, 0] |= ~mask.any(1)   # a study without a joint cell: one empty token, so that the attention is defined
        ang = rope_angles(pos3, self.rope_freq)
        hj = self.dropout(out)
        for layer in self.joint['layers']:
            hj = layer(hj, ang, mask)
        return self.joint['norm'](hj), mask

    def forward_spatial(self, x, types, pos, study_idx, n_studies, geometry=None, extra=None, joint_box=None):
        """The logits [B, n_labels] of the windows x [N, K, 3, H, W]; extra (a dict, or None) gets 'orient', the
        orientation head's output [N, K, 4]."""
        n, k = x.shape[:2]
        if self.joint is not None:
            f3, f = self.stage_maps(x)
        else:
            f = self.features(x)
        if extra is not None:   # the orientation of each window
            extra['orient'] = self.orient(f.float().mean((-2, -1))).view(n, k, 4)
        t, (h, w) = self.spatial_tokens(f, n, k)                                      # [N, K, h*w, D]
        emb = self.series_emb(types, pos)[:, :, None]
        joint = (self.joint_tokens(f3, emb, geometry, pos, study_idx, n_studies, joint_box)
                 if self.joint is not None else None)
        t = t + emb
        tok, mask = pack(t.reshape(n, k * h * w, -1), study_idx, n_studies)
        return self.read(self.norm(self.dropout(tok)), mask, n_studies, joint)

    def forward(self, x, types, pos, study_idx, n_studies, return_aux=False, return_orient=False, geometry=None,
                joint_box=None):
        """x [N, K, 3, H, W]: the K windows of N series; types [N]: the series type; pos [N, K]: the relative slice
        position of each window; study_idx [N]: the study of each series (contiguous); n_studies = B.
        Returns the logits of the scored findings [B, 12]; with return_aux also the auxiliary logits (none: [B, 0]);
        with return_orient a dict comes last with 'orient', two in-plane direction vectors of every window [N, K, 4].
        geometry (knee.training.train.extra_inputs) and joint_box [B, 6] feed the joint transformer."""
        extra = {'orient': None} if return_orient else None
        logits = self.forward_spatial(x, types, pos, study_idx, n_studies, geometry, extra, joint_box)
        out = (logits, logits[:, :0]) if return_aux else logits
        if extra is None:
            return out
        return (*out, extra) if isinstance(out, tuple) else (out, extra)


def build_model(model_cfg, **overrides):
    """KneeNet from the model section of a config (its keys are the keyword arguments), with overrides."""
    return KneeNet(**{**model_cfg, **overrides})
