import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from brainmarks_smri.models import EmbeddingOutput, ImageInput, register_model
from einops import rearrange
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from jaxtyping import Float, Int

from .fino import FinoHeads
from .modules import Block, LayerNorm, NativeEmbed, NativeHead, SeparablePosEmbed3D
from .data import (
    crop_to_mask,
    fit_to_shape,
    native_box,
    normalize_values,
    reorient_to_ras,
    resample_to_spacing,
)


class ViTMAE3D(nn.Module):
    name = "vit_mae_3d"

    def __init__(
        self,
        grid_size: tuple[int, int, int] = (192, 240, 192),
        target_spacing: float = 1.0,
        patch_size: int = 8,
        # native resolution (forward_native): images keep their voxel grids, patches of about
        # patch_size mm per axis. grid_size / patch_size stay the 1 mm grid of the evals
        native: bool = False,
        depth: int = 12,
        embed_dim: int = 768,
        num_heads: int = 12,
        decoder_depth: int = 4,
        decoder_embed_dim: int = 512,
        decoder_num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mlp_ratio: int | float = 4,
        class_tokens: int = 1,
        drop_path_rate: float = 0.0,
        clip_quantiles: tuple[float, float] = (0.005, 0.995),
        qk_norm: bool = False,
        decoder_qk_norm: bool = False,
        qk_scale_init: float = 1.0,
        mask_cube: int = 0,
        # training only: hard-patch loss weight (detached per-token error ** gamma, capped) and the
        # fraction of encoder outputs the decoder sees
        hard_gamma: float = 0.0,
        hard_max: float = 4.0,
        decoder_keep: float = 1.0,
        # zero-init linear path from the pooled encoder mean to every mask token
        decoder_global: bool = False,
        # linear head on the visible encoder tokens reconstructing their own patch (weight; 0 = off)
        self_recon_weight: float = 0.0,
        # sequences only use patches with at least this brain fraction (unless a scan has too few)
        min_mask_frac: float = 0.25,
        # linear head on the visible encoder tokens: BCE to the in-mask fraction of each level_pool^3
        # sub-block of their own patch above each level threshold (weight; 0 = off)
        level_weight: float = 0.0,
        level_thresholds: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5),
        level_pool: int = 2,
        # the nanobrain-v3 loss stack, grafted from v3/src/nanobrain/model.py:
        # z-score each target sequence before the mse; foreground-weighted mse; and the
        # xyz gradient penalty on the target-ref patches (dense_neighbor_pool needs
        # encoder plumbing and is not part of this graft)
        norm_target: bool = False,
        grad_loss_weight: float = 0.0,
        fg_loss_weight: float = 0.0,
        # fino3d config block (nanobrain.fino), training only heads on the global embedding;
        # None = off (no params, same init)
        fino3d: dict | None = None,
        # cross-subject replaced-token detection (ELECTRA / CIM RevDet, with REAL replacements):
        # in native training a rtd_ratio share of each sequence's visible patches is swapped for
        # the patch at the same token coordinate of another image of its spacing group, quantile
        # matched (rtd_quantiles levels) to the host image's in-brain intensities; a zero-init
        # linear head on the visible encoder tokens detects them (BCE, weight rtd_weight; 0 = off).
        # the MAE targets stay the clean host patches. training only
        rtd_weight: float = 0.0,
        rtd_ratio: float = 0.15,
        rtd_quantiles: int = 32,
        # level_mirror (ported from r18_mir_dec, extended to the native path; () = off): extra level
        # channels on a separate zero-init head: in-mask fraction of each 2 mm sub-block above each
        # threshold AND above margin + the max within +-2-4 mm of its x(L-R)-mirrored 2 mm cell
        level_mirror: tuple[float, ...] = (),
        level_mirror_margin: float = 0.5,
        # context level channels (0 = off): a zero-init linear head on the visible encoder tokens
        # predicts, per level threshold, the in-mask fraction of voxels above it over the
        # ctx_windows^3 token neighborhood (24 / 40 mm), so each token carries its surroundings the way
        # the seg probe needs to tell lesions from normally bright tissue. targets come from the full
        # image, not from other tokens, so they work with sparse visible sets. training only
        ctx_weight: float = 0.0,
        ctx_windows: tuple[int, ...] = (3, 5),
        # local anomaly channels (0 = off): per 2 mm sub-block, the in-mask fraction of voxels above
        # mean + k * sd of the anom_window^3 token neighborhood, on a separate zero-init head: the
        # population-free, tilt-free analog of level_mirror (midline and bilateral lesions). training only
        anom_weight: float = 0.0,
        anom_k: tuple[float, ...] = (1.5, 2.5),
        anom_window: int = 5,
        # synthetic lesions (training only; ported from mihirneal r24_syn): small focal hyperintensities
        # (blob_*), one lobulated tumour (tumor_*: ring / diffuse / nodule), and (new) one stroke-like
        # irregular infarct (stroke_*); lesion_mask_weight > 0 adds a zero-init linear head on the visible
        # tokens predicting each 2 mm sub-block's lesion fraction (the seg probe's form)
        blob_prob: float = 0.0,
        blob_max: int = 3,
        blob_radius: tuple[float, float] = (2.0, 9.0),
        blob_contrast: tuple[float, float] = (1.0, 3.0),
        blob_edge: float = 1.5,
        tumor_prob: float = 0.0,
        tumor_radius: tuple[float, float] = (8.0, 30.0),
        tumor_ring: float = 0.5,
        tumor_contrast: tuple[float, float] = (0.8, 2.5),
        tumor_edge: tuple[float, float] = (3.0, 8.0),
        tumor_nodule: float = 0.25,
        stroke_prob: float = 0.0,
        stroke_foci: int = 6,
        stroke_spread: float = 8.0,
        stroke_radius: tuple[float, float] = (3.0, 12.0),
        stroke_contrast: tuple[float, float] = (1.5, 3.5),
        stroke_edge: float = 1.0,
        lesion_mask_weight: float = 0.0,
        # global lesion head (new): from each sequence's mean visible token (the training analog of the
        # eval's global embedding), predict whether its visible voxels hold an inserted lesion and how
        # much (log burden), so the pooled embedding encodes pathology burden (IDH, survival, mRS)
        global_lesion_weight: float = 0.0,
        # hybrid conv stem (new): see PatchConvStem; 0 = off (pure linear patch embedding)
        conv_stem_width: int = 0,
        # contrast-aware lesions (new; needs fino3d metadata for the per-image MRI sequence): each lesion
        # kind's intensity change is scaled per sequence as it appears clinically (tumour rim bright on
        # T1c, dark on T1w; infarct bright on DWI, dark on ADC; foci bright on FLAIR / T2), tumours get a
        # FLAIR / T2-bright peritumoral edema halo (in the whole-lesion label, like BraTS whole tumour), and
        # with lesion_subregions the lesion head gets a second channel: the enhancing part on T1c (BraTS ET)
        lesion_contrast_aware: bool = False,
        lesion_subregions: bool = False,
    ):
        super().__init__()
        self.rtd_weight = float(rtd_weight)
        self.rtd_ratio = float(rtd_ratio)
        self.rtd_quantiles = int(rtd_quantiles)
        self.norm_target = norm_target
        self.grad_loss_weight = float(grad_loss_weight)
        self.fg_loss_weight = float(fg_loss_weight)
        self.grid_size = grid_size
        self.target_spacing = target_spacing
        self.native = native
        assert not native or target_spacing == 1.0, "native kernels are defined on 1 mm voxels"
        # tuple for the brainmarks-smri model contract
        self.patch_size = (patch_size, patch_size, patch_size)
        self.embed_dim = embed_dim
        self.class_tokens = class_tokens
        patch_grid_size = tuple(size // patch_size for size in grid_size)
        self.patch_grid_size = patch_grid_size
        self.clip_quantiles = clip_quantiles
        self.mask_cube = mask_cube
        self.hard_gamma = hard_gamma
        self.hard_max = hard_max
        self.decoder_keep = decoder_keep

        patch_dim = patch_size**3
        self.register_buffer("coord_grid", make_coord_grid(grid_size, patch_size), persistent=False)

        # encoder
        if native:
            self.patch_embed = NativeEmbed(patch_size, embed_dim)
        else:
            self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = SeparablePosEmbed3D(patch_grid_size, embed_dim)
        if class_tokens:
            self.cls_token = nn.Parameter(torch.empty(1, class_tokens, embed_dim))
        else:
            self.cls_token = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[ii],
                    qk_norm=qk_norm,
                    qk_scale_init=qk_scale_init,
                )
                for ii in range(depth)
            ]
        )
        self.norm = LayerNorm(embed_dim)

        # decoder
        self.decoder_proj = nn.Linear(embed_dim, decoder_embed_dim)
        self.global_proj = nn.Linear(embed_dim, decoder_embed_dim) if decoder_global else None
        self.self_recon_weight = self_recon_weight
        self.self_recon_head = nn.Linear(embed_dim, patch_dim) if self_recon_weight else None
        self.min_mask_frac = min_mask_frac
        self.level_weight = level_weight
        self.level_pool = level_pool
        self.level_head = None
        if level_weight:
            levels = [float(t) for t in level_thresholds]
            self.level_head = nn.Linear(embed_dim, len(levels) * (patch_size // level_pool) ** 3)
            self.register_buffer("level_thresholds", torch.tensor(levels), persistent=False)
        # training speed, set by train.py (not a model kwarg): see decode_features
        self.decoder_prune_last = False
        # training speed, set by train.py (not a model kwarg): forward_native's level loss in one pass
        self.level_batched = False
        self.decoder_pos_embed = SeparablePosEmbed3D(patch_grid_size, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.empty(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_ratio=mlp_ratio,
                    qk_norm=decoder_qk_norm,
                    qk_scale_init=qk_scale_init,
                )
                for ii in range(decoder_depth)
            ]
        )

        self.decoder_norm = LayerNorm(decoder_embed_dim)
        if native:
            self.decoder_head = NativeHead(decoder_embed_dim, patch_size)
        else:
            self.decoder_head = nn.Linear(decoder_embed_dim, patch_dim)
        self.init_weights()
        # built after the base init, so every base parameter is initialized as with fino3d off
        self.fino_heads = None
        self.fino_stats = {}
        if fino3d is not None:
            assert native, "fino3d: native training only"
            self.fino_heads = FinoHeads(embed_dim, fino3d)
            self.fino_heads.apply(_init_weights)
        # built last, so every other parameter is initialized as with rtd off
        self.rtd_head = None
        self.rtd_stats = {}
        if self.rtd_weight > 0.0:
            assert native, "rtd: native training only"
            self.rtd_head = nn.Linear(embed_dim, 1)
            nn.init.zeros_(self.rtd_head.weight)
            nn.init.zeros_(self.rtd_head.bias)

        self.level_mirror = tuple(float(t) for t in level_mirror)
        self.level_mirror_margin = float(level_mirror_margin)
        self.level_mirror_head = None
        self.level_mirror_stats = {}
        if self.level_mirror:
            assert level_weight and native, "level_mirror: needs the level head, native path"
            assert patch_size == 8 and level_pool == 2, "level_mirror: 2 mm cells of 8 mm tokens"
            # background (0) is in the mirror window at the mask edge: thresholds >= margin keep it inert
            assert min(self.level_mirror) >= self.level_mirror_margin, "level_mirror: thresholds >= margin"
            # off the rng stream and zero init: every other weight initializes as without it
            with torch.random.fork_rng(devices=[]):
                self.level_mirror_head = nn.Linear(embed_dim, len(self.level_mirror) * 64)
            nn.init.zeros_(self.level_mirror_head.weight)
            nn.init.zeros_(self.level_mirror_head.bias)
            self.register_buffer("level_mirror_thresholds", torch.tensor(self.level_mirror), persistent=False)

        # context + local anomaly heads: off the rng stream and zero init, like level_mirror
        self.ctx_weight = float(ctx_weight)
        self.ctx_windows = tuple(int(w) for w in ctx_windows)
        self.anom_weight = float(anom_weight)
        self.anom_k = tuple(float(k) for k in anom_k)
        self.anom_window = int(anom_window)
        # python floats: thresholds looped on the gpu without a host sync
        self.ctx_thresholds = tuple(float(t) for t in level_thresholds)
        self.ctx_head = None
        self.anom_head = None
        self.ctx_stats = {}
        if self.ctx_weight > 0.0 or self.anom_weight > 0.0:
            assert level_weight and native, "ctx / anom: need the level head (thresholds), native path"
            assert patch_size == 8 and level_pool == 2, "ctx / anom: 2 mm cells of 8 mm tokens"
            assert all(w % 2 == 1 for w in self.ctx_windows + (self.anom_window,)), "odd windows"
        if self.ctx_weight > 0.0:
            with torch.random.fork_rng(devices=[]):
                self.ctx_head = nn.Linear(embed_dim, len(level_thresholds) * len(self.ctx_windows))
            nn.init.zeros_(self.ctx_head.weight)
            nn.init.zeros_(self.ctx_head.bias)
        if self.anom_weight > 0.0:
            with torch.random.fork_rng(devices=[]):
                self.anom_head = nn.Linear(embed_dim, len(self.anom_k) * 64)
            nn.init.zeros_(self.anom_head.weight)
            nn.init.zeros_(self.anom_head.bias)
            self.register_buffer("anom_k_t", torch.tensor(self.anom_k), persistent=False)

        # synthetic lesions, lesion heads and the conv stem: off the rng stream (zero-init outputs), so
        # every other weight initializes as with them off
        self.blob_prob = float(blob_prob)
        self.blob_max = int(blob_max)
        self.blob_radius = tuple(float(v) for v in blob_radius)
        self.blob_contrast = tuple(float(v) for v in blob_contrast)
        self.blob_edge = float(blob_edge)
        self.tumor_prob = float(tumor_prob)
        self.tumor_radius = tuple(float(v) for v in tumor_radius)
        self.tumor_ring = float(tumor_ring)
        self.tumor_contrast = tuple(float(v) for v in tumor_contrast)
        self.tumor_edge = tuple(float(v) for v in tumor_edge)
        self.tumor_nodule = float(tumor_nodule)
        self.stroke_prob = float(stroke_prob)
        self.stroke_foci = int(stroke_foci)
        self.stroke_spread = float(stroke_spread)
        self.stroke_radius = tuple(float(v) for v in stroke_radius)
        self.stroke_contrast = tuple(float(v) for v in stroke_contrast)
        self.stroke_edge = float(stroke_edge)
        self.lesion_mask_weight = float(lesion_mask_weight)
        self.global_lesion_weight = float(global_lesion_weight)
        any_lesion = self.blob_prob > 0 or self.tumor_prob > 0 or self.stroke_prob > 0
        assert not any_lesion or native, "synthetic lesions: native path only"
        assert self.stroke_prob == 0 or self.stroke_foci >= 2, "stroke_foci >= 2"
        self.lesion_head = None
        self.global_lesion_head = None
        self.lesion_stats = {}
        if self.lesion_mask_weight > 0 or self.global_lesion_weight > 0:
            assert any_lesion, "lesion heads: turn on blob_prob, tumor_prob or stroke_prob"
            assert patch_size == 8 and level_pool == 2, "lesion heads: 2 mm cells of 8 mm tokens"
        self.lesion_contrast_aware = bool(lesion_contrast_aware)
        self.lesion_subregions = bool(lesion_subregions)
        assert not self.lesion_subregions or self.lesion_contrast_aware, "lesion_subregions: needs contrast-aware lesions"
        if self.lesion_mask_weight > 0:
            with torch.random.fork_rng(devices=[]):
                n_ch = 2 if self.lesion_subregions else 1
                self.lesion_head = nn.Linear(embed_dim, n_ch * (patch_size // level_pool) ** 3)
            nn.init.zeros_(self.lesion_head.weight)
            nn.init.zeros_(self.lesion_head.bias)
        if self.global_lesion_weight > 0:
            with torch.random.fork_rng(devices=[]):
                self.global_lesion_head = nn.Linear(embed_dim, 2)  # presence logit, log-burden logit
            nn.init.zeros_(self.global_lesion_head.weight)
            nn.init.zeros_(self.global_lesion_head.bias)
        self.conv_stem = None
        if conv_stem_width:
            assert patch_size == 8, "conv stem: 8 mm tokens (8^3 voxel cells)"
            with torch.random.fork_rng(devices=[]):
                self.conv_stem = PatchConvStem(embed_dim, int(conv_stem_width))

    @torch.no_grad()
    def rtd_replace(
        self, vis_patches: Tensor, vis_mask: Tensor, order: Tensor, vols: Tensor, b: int, num: int,
        num_samples: int, num_visible: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Swap a rtd_ratio share of the visible patches of every row for the patch at the same
        token coordinate of the next image of the group (rolled by one), quantile matched to the
        host row's in-brain voxels. Returns (patches, mask, replaced [rows, num_visible] bool)."""
        rows = b * num_samples
        vis_order = order[:, :num_visible]  # [rows, V], token ids of the host
        partner_img = (torch.arange(b, device=order.device) + 1) % b
        partner_img = partner_img.repeat_interleave(num_samples)  # [rows]
        flat = vis_order + (partner_img * num)[:, None]
        p_patches = vols[0].flatten(0, 1).index_select(0, flat.flatten()).view_as(vis_patches)
        p_mask = vols[1].flatten(0, 1).index_select(0, flat.flatten()).view_as(vis_patches)
        # the partner patch must be brain tissue too (half its voxels in its brain mask)
        ok = p_mask.float().mean(-1) >= 0.5
        replaced = (torch.rand(vis_order.shape, device=order.device) < self.rtd_ratio) & ok
        # per-row in-brain quantiles of the host's visible patches and of the partner image's
        # visible patches in its own row (row of image j+1, same sample index)
        nq = self.rtd_quantiles + 1
        qs = torch.linspace(0, 1, nq, device=order.device)

        def row_quantiles(x: Tensor, m: Tensor) -> Tensor:
            x = x.float().flatten(1)
            m = m.flatten(1) > 0.5
            cnt = m.sum(1).clamp_min(1)
            srt = torch.where(m, x, torch.full_like(x, float("inf"))).sort(dim=1).values
            idx = (qs[None] * (cnt[:, None] - 1).float()).round().long()
            q = srt.gather(1, idx)
            q = torch.where(torch.isfinite(q), q, torch.zeros_like(q))
            # strictly increasing for the interpolation
            return q + 1e-5 * torch.arange(nq, device=x.device)[None]

        q_host = row_quantiles(vis_patches, vis_mask)
        row_of = torch.arange(rows, device=order.device)
        partner_row = (row_of // num_samples + 1) % b * num_samples + row_of % num_samples
        q_part = q_host[partner_row]
        x = p_patches.float().flatten(1)
        pos = torch.searchsorted(q_part.contiguous(), x.contiguous()).clamp(1, nq - 1)
        lo, hi = q_part.gather(1, pos - 1), q_part.gather(1, pos)
        t = ((x - lo) / (hi - lo)).clamp(0, 1)
        mapped = q_host.gather(1, pos - 1) + t * (q_host.gather(1, pos) - q_host.gather(1, pos - 1))
        mapped = mapped.view_as(p_patches).to(vis_patches.dtype)
        sel = replaced[..., None]
        return (
            torch.where(sel, mapped, vis_patches),
            torch.where(sel, p_mask, vis_mask),
            replaced,
        )

    @classmethod
    def from_config(cls, args: dict) -> "ViTMAE3D":
        return cls(
            grid_size=tuple(args["grid_size"]),
            target_spacing=args["target_spacing"],
            clip_quantiles=tuple(args["clip_quantiles"]),
            patch_size=args["patch_size"],
            native=args["native"],
            fino3d=fino if (fino := args.get("fino3d")) and fino.get("enabled") else None,
            **args["model_kwargs"],
        )

    def extra_repr(self):
        return (
            f"{self.grid_size}, {self.patch_size}, class_tokens={self.class_tokens}, "
            f"native={self.native}"
        )

    def init_weights(self):
        self.apply(_init_weights)
        if self.class_tokens:
            nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        # initialize the decoder head to zero to prevent hockey stick
        self.decoder_head.weight.data.mul_(0.0)
        if self.level_head is not None:
            nn.init.zeros_(self.level_head.weight)
            nn.init.zeros_(self.level_head.bias)
        if self.self_recon_head is not None:
            nn.init.zeros_(self.self_recon_head.weight)
            nn.init.zeros_(self.self_recon_head.bias)
        if self.global_proj is not None:
            nn.init.zeros_(self.global_proj.weight)
            nn.init.zeros_(self.global_proj.bias)

    def forward_encoder(
        self,
        patches: Float[Tensor, "B N P"],
        coord: Float[Tensor, "B N 3"],
    ) -> Float[Tensor, "B R+N D"]:
        B, N, P = patches.shape
        assert coord.shape == (B, N, 3)
        x = self.patch_embed(patches)
        if self.conv_stem is not None:
            # hybrid stem: same 8^3 voxel cells as training (1 mm grid at eval)
            x = x + self.conv_stem(patches.reshape(B * N, 1, 8, 8, 8).to(x.dtype)).view(B, N, -1)
        return self.encode_tokens(x, coord)

    def encode_tokens(
        self,
        x: Float[Tensor, "B N D"],
        coord: Float[Tensor, "B N 3"],
    ) -> Float[Tensor, "B R+N D"]:
        # forward_encoder after the patch embedding
        B = x.shape[0]
        x = self.pos_embed(x, coord)

        if self.class_tokens:
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

    def forward_decoder(
        self,
        embeds: Float[Tensor, "B R+N D"],
        target_coord: Float[Tensor, "B M 3"],
    ) -> Float[Tensor, "B M P"]:
        return self.decoder_head(self.decode_features(embeds, target_coord))

    def decode_features(
        self,
        embeds: Float[Tensor, "B R+N D"],
        target_coord: Float[Tensor, "B M 3"],
    ) -> Float[Tensor, "B M D"]:
        # forward_decoder before the head
        B = embeds.shape[0]
        M = target_coord.shape[1]
        assert target_coord.shape == (B, M, 3)

        if self.global_proj is not None:
            pooled = self.global_proj(embeds[:, self.class_tokens :].mean(dim=1, keepdim=True))
        if self.training and self.decoder_keep < 1.0:
            # visible order is a random shuffle, so a prefix is a uniform subset
            keep = int((embeds.shape[1] - self.class_tokens) * self.decoder_keep)
            embeds = embeds[:, : self.class_tokens + keep]
        embeds = self.decoder_proj(embeds)

        mask_tokens = self.mask_token.expand(B, M, -1)
        mask_tokens = self.decoder_pos_embed(mask_tokens, target_coord)
        if self.global_proj is not None:
            mask_tokens = mask_tokens + pooled

        x = torch.cat([embeds, mask_tokens], dim=1)

        if self.decoder_prune_last and len(self.decoder_blocks):
            # the head reads only the mask tokens, so the last block computes just their rows
            for block in self.decoder_blocks[:-1]:
                x = block(x)
            x = self.decoder_blocks[-1].forward_last_rows(x, M)
        else:
            for block in self.decoder_blocks:
                x = block(x)
            x = x[:, -M:]
        x = self.decoder_norm(x)
        return x

    def forward(
        self,
        images: Float[Tensor, "B X Y Z"] | list[Float[Tensor, "X Y Z"]],
        mask: Float[Tensor, "B X Y Z"] | list[Float[Tensor, "X Y Z"]],
        num_visible: int,
        num_predict: int | None = None,
        num_samples: int = 1,
        with_state: bool = True,
        spacing: list[tuple[float, float, float]] | None = None,
        fino_meta: Tensor | None = None,
        fino_gamma: float = 0.0,
    ):
        if isinstance(images, list):
            # native resolution batch, images on their own voxel grids
            return self.forward_native(
                images, mask, spacing, num_visible, num_predict, num_samples, with_state,
                fino_meta=fino_meta, fino_gamma=fino_gamma,
            )
        seqs, order = self.sample_patches(images, mask, num_visible, num_predict, num_samples)
        if not with_state:
            return self.patch_loss(*seqs)
        loss, embeds, pred_patches = self.patch_loss(*seqs, with_outputs=True)
        vis_patches, vis_mask_patches, _, target_patches, target_mask_patches, _ = seqs
        num_predict = target_patches.shape[1]
        B = images.shape[0]

        patch_ids = order.flatten(0, 1)
        image_ids = torch.arange(B, device=images.device).repeat_interleave(num_samples)
        vis_ids, target_ids = torch.split(patch_ids, [num_visible, num_predict], dim=1)

        state = {
            "vis_patches": vis_patches,
            "target_patches": target_patches,
            "vis_mask_patches": vis_mask_patches.to(images.dtype),
            "target_mask_patches": target_mask_patches.to(images.dtype),
            "embeds": embeds,
            "pred_patches": pred_patches,
            "vis_ids": vis_ids,
            "target_ids": target_ids,
            "image_ids": image_ids,
        }
        return loss, state

    def sample_patches(
        self,
        images: Float[Tensor, "B X Y Z"],
        mask: Float[Tensor, "B X Y Z"],
        num_visible: int,
        num_predict: int | None = None,
        num_samples: int = 1,
    ):
        # forward's sequence sampling and patch gather: (vis patches, vis mask patches, vis coords,
        # target patches, target mask patches, target coords), order
        B = images.shape[0]
        assert tuple(images.shape[1:]) == tuple(self.grid_size)

        patches = patchify3d(images, self.patch_size[0])
        mask_patches = patchify3d(mask, self.patch_size[0])
        coord = self.coord_grid.expand(B, -1, -1)

        # sample num_samples sequences per image, then flatten them into the batch
        # num_predict None predicts all remaining patches (dense decoding)
        seq_length = None if num_predict is None else num_visible + num_predict
        cube = (self.patch_grid_size, self.mask_cube) if self.mask_cube else None
        order = sample_sequences(
            mask_patches, num_samples, seq_length, min_mask_frac=self.min_mask_frac, cube=cube
        )
        num_predict = order.shape[2] - num_visible
        batch_ids = torch.arange(B, device=images.device)[:, None, None]
        patches = patches[batch_ids, order].flatten(0, 1)
        # uint8 masks from the fast prep, cast in patch_loss
        mask_patches = mask_patches[batch_ids, order].flatten(0, 1)
        coord = coord[batch_ids, order].flatten(0, 1)

        vis_patches, target_patches = torch.split(patches, [num_visible, num_predict], dim=1)
        vis_mask_patches, target_mask_patches = torch.split(
            mask_patches, [num_visible, num_predict], dim=1
        )
        vis_coord, target_coord = torch.split(coord, [num_visible, num_predict], dim=1)
        seqs = (vis_patches, vis_mask_patches, vis_coord, target_patches, target_mask_patches, target_coord)
        return seqs, order

    def sample_patches_contiguous(self, *args):
        # cuda graph inputs: separate dense buffers, uint8 masks stay uint8. the class method, as
        # compile_cudagraphs shadows self.sample_patches with this
        seqs, order = ViTMAE3D.sample_patches(self, *args)
        return tuple(t.contiguous() for t in seqs), order

    def patch_loss(
        self,
        vis_patches: Float[Tensor, "B N P"],
        vis_mask_patches: Tensor,
        vis_coord: Tensor,
        target_patches: Float[Tensor, "B M P"],
        target_mask_patches: Tensor,
        target_coord: Tensor,
        with_outputs: bool = False,
    ):
        # forward's loss from sampled sequences
        vis_mask_patches = vis_mask_patches.to(vis_patches.dtype)
        target_mask_patches = target_mask_patches.to(target_patches.dtype)
        embeds = self.forward_encoder(vis_patches, vis_coord)
        pred_patches = self.forward_decoder(embeds, target_coord)
        # graft: the v3 loss stack. norm_target z-scores each target sequence; the
        # mse and the gradient penalty run on target_ref, the fg mask weights the mse
        if self.norm_target:
            mean = target_patches.mean(dim=-1, keepdim=True)
            var = target_patches.var(dim=-1, keepdim=True)
            target_ref = (target_patches - mean) / (var + 1e-6) ** 0.5
        else:
            target_ref = target_patches
        loss = F.mse_loss(pred_patches, target_ref, reduction="none")
        if self.fg_loss_weight > 0.0:
            fg_mask = (target_patches.abs() > 0.05).float()
            fg_weights = 1.0 + self.fg_loss_weight * fg_mask
            loss = (target_mask_patches * fg_weights * loss).sum() / (target_mask_patches * fg_weights).sum().clamp(min=1.0)
        else:
            loss = (target_mask_patches * loss).sum() / target_mask_patches.sum()
        if self.grad_loss_weight > 0.0:
            p = self.patch_size[0]
            BM = pred_patches.shape[0] * pred_patches.shape[1]
            pred_vol = pred_patches.reshape(BM, p, p, p)
            tgt_vol = target_ref.reshape(BM, p, p, p)
            mask_vol = target_mask_patches.reshape(BM, p, p, p)
            dx_loss = F.mse_loss(
                pred_vol[:, 1:, :, :] - pred_vol[:, :-1, :, :],
                tgt_vol[:, 1:, :, :] - tgt_vol[:, :-1, :, :],
                reduction="none",
            )
            dy_loss = F.mse_loss(
                pred_vol[:, :, 1:, :] - pred_vol[:, :, :-1, :],
                tgt_vol[:, :, 1:, :] - tgt_vol[:, :, :-1, :],
                reduction="none",
            )
            dz_loss = F.mse_loss(
                pred_vol[:, :, :, 1:] - pred_vol[:, :, :, :-1],
                tgt_vol[:, :, :, 1:] - tgt_vol[:, :, :, :-1],
                reduction="none",
            )
            mx = mask_vol[:, 1:, :, :] * mask_vol[:, :-1, :, :]
            my = mask_vol[:, :, 1:, :] * mask_vol[:, :, :-1, :]
            mz = mask_vol[:, :, :, 1:] * mask_vol[:, :, :, :-1]
            grad_loss = (
                (mx * dx_loss).sum() / mx.sum().clamp(min=1.0)
                + (my * dy_loss).sum() / my.sum().clamp(min=1.0)
                + (mz * dz_loss).sum() / mz.sum().clamp(min=1.0)
            ) / 3.0
            loss = loss + self.grad_loss_weight * grad_loss

        if self.hard_gamma:
            vox = target_mask_patches.sum(-1)
            tok = (target_mask_patches * loss).sum(-1) / vox.clamp_min(1)
            has = (vox > 0).to(tok.dtype)
            h = tok.detach()
            w = (h / ((h * has).sum() / has.sum()).clamp_min(1e-8)).clamp_min(1e-8)
            w = w.pow(self.hard_gamma).clamp(max=self.hard_max)
            loss = (w * vox * tok).sum() / (w * vox).sum()
        else:
            loss = (target_mask_patches * loss).sum() / target_mask_patches.sum()
        if self.self_recon_head is not None:
            self_pred = self.self_recon_head(embeds[:, self.class_tokens :])
            self_loss = F.mse_loss(self_pred, vis_patches, reduction="none")
            self_loss = (vis_mask_patches * self_loss).sum() / vis_mask_patches.sum()
            loss = loss + self.self_recon_weight * self_loss
        if self.level_head is not None:
            loss = loss + self.level_weight * self.level_loss(
                embeds[:, self.class_tokens :], vis_patches, vis_mask_patches
            )
        if with_outputs:
            return loss, embeds, pred_patches
        return loss

    def level_terms(self, tokens: Tensor, patches: Tensor, mask: Tensor, patch) -> tuple[Tensor, Tensor]:
        # level_loss on native patches (voxels px x py x pz): targets on the same mm sub-blocks via
        # adaptive pooling (thick-slice axes replicate); weighted sum and its normalizer
        px, py, pz = patch
        x = patches.float().reshape(-1, 1, px, py, pz)
        m = mask.float().reshape(-1, 1, px, py, pz)
        above = (x > self.level_thresholds.view(1, -1, 1, 1, 1)).float() * m
        sub = self.patch_size[0] // self.level_pool
        weight = F.adaptive_avg_pool3d(m, sub)
        target = F.adaptive_avg_pool3d(above, sub) / weight.clamp_min(1e-6)
        logits = self.level_head(tokens).float().reshape(target.shape)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (weight * bce).sum(), weight.sum() * target.shape[1]

    def level_loss(self, tokens: Tensor, patches: Tensor, mask: Tensor) -> Tensor:
        # level head BCE, sub-blocks weighted by their in-mask fraction
        p, q = self.patch_size[0], self.level_pool
        x = patches.float().reshape(-1, 1, p, p, p)
        m = mask.float().reshape(-1, 1, p, p, p)
        above = (x > self.level_thresholds.view(1, -1, 1, 1, 1)).float() * m
        weight = F.avg_pool3d(m, q)
        target = F.avg_pool3d(above, q) / weight.clamp_min(1e-6)
        logits = self.level_head(tokens).float().reshape(target.shape)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (weight * bce).sum() / (weight.sum() * target.shape[1]).clamp_min(1e-6)

    @staticmethod
    def _to_cells8(t: Tensor, patch) -> Tensor:
        # native token voxels [R, px*py*pz] -> [R, 1, 8, 8, 8] (nearest; identity at 1 mm), so 2 mm
        # sub-blocks are 2^3 voxel blocks
        if tuple(patch) != (8, 8, 8):
            # F.interpolate's nearest (src = floor(dst * in / out)) as one gather: upsample_nearest3d
            # with ~1e5 batch rows is ~30x slower
            t = t.reshape(-1, math.prod(patch)).index_select(1, cells8_index(tuple(patch), t.device))
        return t.reshape(-1, 1, 8, 8, 8)

    def mirror_ref(self, vols: Tensor, patch, box, grid, sp) -> Tensor:
        # per token [b, N, 64]: the max within +-1 cell (2 mm cells) of each sub-block's x-mirrored cell
        # (mirror about the box centre, the fit_to_shape centre); 1e4 where that cell is out of the
        # mask, out of the box, or within 4 mm of the midline. at 1 mm = r18's mirror_map
        img, msk = vols[0], vols[1]
        b, N = img.shape[:2]
        gx, gy, gz = grid

        def cellvol(t: Tensor) -> Tensor:
            c = self._to_cells8(t, patch).view(b, N, 4, 2, 4, 2, 4, 2).amax((3, 5, 7))
            c = c.view(b, gx, gy, gz, 4, 4, 4).permute(0, 1, 4, 2, 5, 3, 6)
            return c.reshape(b, gx * 4, gy * 4, gz * 4)

        cells = cellvol(img)
        valid = cellvol(msk.to(img.dtype)) > 0
        hmax = F.max_pool3d(cells.float()[:, None], 3, 1, 1)[:, 0]
        # x mirror index per cell, in voxels of this spacing: cell width px / 4
        Cx, w = gx * 4, patch[0] / 4
        centre = (torch.arange(Cx, device=img.device, dtype=torch.float64) + 0.5) * w
        j = torch.floor((box[0] - centre) / w).long()
        ok = (j >= 0) & (j < Cx) & ((centre - box[0] / 2).abs() * sp[0] >= 4.0)
        j = j.clamp(0, Cx - 1)
        mvalid = valid.index_select(1, j) & ok.view(1, -1, 1, 1)
        ref = torch.where(mvalid, hmax.index_select(1, j), 1e4)
        ref = ref.view(b, gx, 4, gy, 4, gz, 4).permute(0, 1, 3, 5, 2, 4, 6)
        return ref.reshape(b, N, 64)

    def mirror_targets(self, patches: Tensor, mask: Tensor, patch, ref: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # mirror level channels without the head: (target [R, M, 4, 4, 4], valid weight, level weight [R, 1, 4, 4, 4])
        x = self._to_cells8(patches.float(), patch)
        m = self._to_cells8(mask.float(), patch)
        weight = F.avg_pool3d(m, 2)
        mh = ref.float().view(-1, 1, 4, 4, 4)
        thr = torch.maximum(self.level_mirror_thresholds.view(1, -1, 1, 1, 1), mh + self.level_mirror_margin)
        thr = thr.repeat_interleave(2, 2).repeat_interleave(2, 3).repeat_interleave(2, 4)
        target = F.avg_pool3d((x > thr).float() * m, 2) / weight.clamp_min(1e-6)
        m_weight = weight * (mh < 1e3).float()
        return target, m_weight, weight

    @torch.no_grad()
    def context_ref(self, vols: Tensor, grid) -> tuple[Tensor | None, Tensor | None]:
        # per token of the whole image [b, N, ...]: context level targets (in-mask fraction above each
        # level threshold over each ctx window of tokens) and the anom_window neighborhood intensity
        # (mean, sd) used for the local anomaly targets. sums over windows via avg_pool3d * w^3
        img, msk = vols[0], vols[1] > 0  # [b, N, P] voxels per token; no full-size float copies
        b, N = img.shape[:2]
        gx, gy, gz = grid

        def window_sum(t: Tensor, w: int) -> Tensor:
            # [b, N, C] token values -> [b, N, C] sums over the w^3 token neighborhood (zero padded)
            g = t.permute(0, 2, 1).reshape(b, -1, gx, gy, gz)
            s = F.avg_pool3d(g, w, 1, w // 2, count_include_pad=True) * (w ** 3)
            return s.reshape(b, -1, N).permute(0, 2, 1)

        cnt = msk.sum(-1, keepdim=True).float()  # [b, N, 1] in-mask voxels per token
        ctx = None
        if self.ctx_head is not None:
            # one threshold at a time: bool [b, N, P] -> counts [b, N]
            above = torch.stack([((img > t) & msk).sum(-1) for t in self.ctx_thresholds], -1).float()
            ctx = torch.cat(
                [window_sum(above, w) / window_sum(cnt, w).clamp_min(1.0) for w in self.ctx_windows], -1
            )
        stats = None
        if self.anom_head is not None:
            w = self.anom_window
            xm = torch.where(msk, img, 0).float()
            s0 = window_sum(cnt, w).clamp_min(1.0)
            s1 = window_sum(xm.sum(-1, keepdim=True), w)
            s2 = window_sum(xm.square().sum(-1, keepdim=True), w)
            del xm
            mu = s1 / s0
            sd = (s2 / s0 - mu * mu).clamp_min(1e-4).sqrt()
            stats = torch.cat([mu, sd], -1)  # [b, N, 2]
        return ctx, stats

    def anom_targets(self, patches: Tensor, mask: Tensor, patch, stats: Tensor) -> tuple[Tensor, Tensor]:
        # local anomaly channels without the head: (target [R, K, 4, 4, 4], weight [R, 1, 4, 4, 4]);
        # a voxel is positive above its neighborhood's mean + k * sd
        x = self._to_cells8(patches.float(), patch)
        m = self._to_cells8(mask.float(), patch)
        weight = F.avg_pool3d(m, 2)
        st = stats.float().reshape(-1, 2)
        thr = (st[:, :1] + st[:, 1:] * self.anom_k_t.view(1, -1)).view(-1, len(self.anom_k), 1, 1, 1)
        target = F.avg_pool3d((x > thr).float() * m, 2) / weight.clamp_min(1e-6)
        return target, weight

    def mirror_terms(self, tokens: Tensor, patches: Tensor, mask: Tensor, patch, ref: Tensor):
        # mirror level channels: (weighted bce sum, valid weight sum, level weight sum, weighted positives [M])
        target, m_weight, weight = self.mirror_targets(patches, mask, patch, ref)
        logits = self.level_mirror_head(tokens).float().reshape(target.shape)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (m_weight * bce).sum(), m_weight.sum(), weight.sum(), (m_weight * target).sum((0, 2, 3, 4))

    def forward_native(
        self,
        images: list[Float[Tensor, "X Y Z"]],
        mask: list[Float[Tensor, "X Y Z"]],
        spacing: list[tuple[float, float, float]],
        num_visible: int,
        num_predict: int,
        num_samples: int = 1,
        with_state: bool = True,
        fino_meta: Tensor | None = None,
        fino_gamma: float = 0.0,
    ):
        # images (float) and masks (bool) on their own voxel grids (sample_transform native), fit
        # here to the grid_size mm box in their voxels. images with the same spacing share one
        # token grid, sequence sampling and embedding / head weight. each predicted token counts
        # by its in-mask fraction whatever its voxel count; with 1 mm images only this is
        # forward's loss, bitwise
        assert self.native and num_predict is not None
        seq_length = num_visible + num_predict
        token_mm = self.patch_size[0] * self.target_spacing
        groups: dict[tuple, list[int]] = {}
        for ii, sp in enumerate(spacing):
            groups.setdefault(tuple(sp), []).append(ii)

        device = images[0].device
        geoms = {sp: native_geometry(sp, tuple(self.grid_size), token_mm, device) for sp in groups}
        # embedding and head weights of all non 1 mm spacings in one pass each
        native_sps = [sp for sp in groups if sp != (1.0, 1.0, 1.0)]
        embed_ws, head_ws = {}, {}
        if native_sps:
            sp_patches = [geoms[sp][0] for sp in native_sps]
            embed_ws = self.patch_embed.native_weights(native_sps, sp_patches)
            head_ws = self.decoder_head.native_weights(native_sps, sp_patches)
            embed_ws, head_ws = dict(zip(native_sps, embed_ws)), dict(zip(native_sps, head_ws))

        tumors = {}
        if self.training and self.tumor_prob > 0:
            tumors = plan_tumors(
                images, mask, groups, geoms, self.tumor_prob, self.tumor_radius, self.tumor_ring,
                self.tumor_contrast, self.tumor_edge, self.tumor_nodule,
            )
        want_lesion = self.training and (self.lesion_head is not None or self.global_lesion_head is not None)
        seq_all = None
        if self.training and self.lesion_contrast_aware and fino_meta is not None:
            from nanobrain.fino import COLUMNS as FINO_COLUMNS

            seq_all = torch.nan_to_num(fino_meta.cpu()[:, FINO_COLUMNS.index("seq")].float(), nan=6.0).long()
        parts = []
        for sp, ids in groups.items():
            patch, box, grid, _ = geoms[sp]
            # images and masks fit to the box (fit_to_shape) and padded up to whole patches, tiling
            # the box from its low corner, in one buffer
            padded = tuple(num * p for num, p in zip(grid, patch))
            vols = images[ids[0]].new_zeros((2, len(ids)) + padded)
            for jj, ii in enumerate(ids):
                src, dst = fit_slices(images[ii].shape, box)
                vols[0, jj][dst] = images[ii][src]
                vols[1, jj][dst] = mask[ii][src]
            # synthetic lesions before patchify: they enter the inputs and every target (MAE, level,
            # mirror, context, anomaly), shared by an image's sequences
            lesion, lesion_enh = None, None
            # contrast-aware lesions: per image MRI sequence -> signed per-kind gains (x U(0.75, 1.25))
            gseq = None if seq_all is None else seq_all[ids].to(device)
            nb = len(ids)

            def gains(table: tuple[float, ...]) -> Tensor | None:
                if gseq is None:
                    return None
                return torch.tensor(table, device=device)[gseq] * (0.75 + 0.5 * torch.rand(nb, device=device))

            lo = None
            if gseq is not None and (self.blob_prob > 0 or self.stroke_prob > 0 or sp in tumors):
                # each image's in-mask minimum: darkening lesions never go below it
                lo = torch.where(vols[1] > 0, vols[0], torch.inf).amin((1, 2, 3)).view(-1, 1, 1, 1)
            if self.training and self.blob_prob > 0:
                lesion = insert_blobs(
                    vols[0], vols[1], patch, sp, self.blob_prob, self.blob_max, self.blob_radius,
                    self.blob_contrast, self.blob_edge, with_mask=want_lesion, gain=gains(BLOB_GAIN),
                )
            if self.training and self.stroke_prob > 0:
                stroke = insert_strokes(
                    vols[0], vols[1], patch, sp, self.stroke_prob, self.stroke_foci, self.stroke_spread,
                    self.stroke_radius, self.stroke_contrast, self.stroke_edge, with_mask=want_lesion,
                    gain=gains(STROKE_GAIN),
                )
                if stroke is not None:
                    lesion = stroke if lesion is None else torch.maximum(lesion, stroke)
            if sp in tumors:
                before = vols[0].clone() if gseq is not None else None
                tumor = paste_tumors(vols[0], vols[1], tumors[sp], with_mask=want_lesion or gseq is not None)
                if gseq is not None:
                    diff = vols[0] - before
                    vols[0] = before + diff * gains(TUMOR_GAIN).view(-1, 1, 1, 1)
                    # peritumoral edema: a soft shell ~6-12 mm around the mass, bright on FLAIR / T2
                    halo_amp = torch.tensor(HALO_AMP, device=device)[gseq] * (0.6 + 0.8 * torch.rand(nb, device=device))
                    halo = edema_halo(tumor, vols[1], sp)
                    vols[0] = vols[0] + halo * halo_amp.view(-1, 1, 1, 1)
                    if self.lesion_subregions and want_lesion:
                        # enhancing part (BraTS ET): voxels the mass brightens by > 0.5 z, on T1c only
                        is_t1c = (gseq == SEQ_T1C).view(-1, 1, 1, 1)
                        lesion_enh = ((diff > 0.5) & is_t1c).to(vols.dtype) * tumor
                    # whole-lesion label (BraTS WT): mass + edema where the edema is visible (FLAIR / T2)
                    tumor = torch.maximum(tumor, halo * (halo_amp >= 0.5).view(-1, 1, 1, 1).to(halo.dtype))
                    del before, diff
                if want_lesion and tumor is not None:
                    lesion = tumor if lesion is None else torch.maximum(lesion, tumor)
            if lo is not None:
                vols[0] = torch.where(vols[1] > 0, torch.maximum(vols[0], lo), vols[0])
            lesion_patches = None if lesion is None else patchify3d(lesion, patch)
            lesion_enh_patches = None if lesion_enh is None else patchify3d(lesion_enh, patch)
            vols = patchify3d(vols.flatten(0, 1), patch).unflatten(0, (2, len(ids)))
            cube = (grid, self.mask_cube) if self.mask_cube else None
            scores = sequence_scores(vols[1], num_samples, self.min_mask_frac, cube=cube).flatten(0, 1)
            parts.append({"spacing": sp, "patch": patch, "ids": ids, "vols": vols})
            if want_lesion:
                parts[-1]["lesion"] = lesion_patches  # [b, N, P] soft lesion mask, or None
                if self.lesion_subregions:
                    parts[-1]["lesion_enh"] = lesion_enh_patches
            parts[-1]["scores"] = scores

        # one argsort for all groups, rows padded with 3.0 (after every score). sorts this long are
        # stable, so these are the per group argsorts
        num_tokens = [part["scores"].shape[1] for part in parts]
        num_rows = [part["scores"].shape[0] for part in parts]
        assert min(num_tokens) >= seq_length
        if len(parts) == 1:
            scores = parts[0]["scores"]
        else:
            scores = parts[0]["scores"].new_full((sum(num_rows), max(num_tokens)), 3.0)
            row = 0
            for part, num, rows in zip(parts, num_tokens, num_rows):
                scores[row : row + rows, :num] = part["scores"]
                row += rows
        orders = scores.argsort(dim=1, stable=True)[:, :seq_length].split(num_rows)

        for part, order, num in zip(parts, orders, num_tokens):
            sp, patch, vols = part["spacing"], part["patch"], part.pop("vols")
            del part["scores"]
            b = len(part["ids"])
            part["order"] = order.unflatten(0, (b, num_samples))
            # images and masks gathered together by flat token ids
            flat_ids = part["order"] + gather_offsets(b, num, device)
            patches = vols.flatten(0, 2).index_select(0, flat_ids.flatten())
            patches, mask_patches = patches.view(2, b * num_samples, seq_length, -1)
            coord = geoms[sp][3][0][order]
            vis_patches, part["target_patches"] = patches.split([num_visible, num_predict], 1)
            part["target_mask_patches"] = mask_patches[:, num_visible:]
            vis_mask = mask_patches[:, :num_visible]
            if self.rtd_head is not None and self.training:
                if b >= 2:
                    vis_patches, vis_mask, part["rtd_label"] = self.rtd_replace(
                        vis_patches, vis_mask, order, vols, b, num, num_samples, num_visible
                    )
                else:
                    part["rtd_label"] = torch.zeros(
                        vis_patches.shape[:2], dtype=torch.bool, device=device
                    )
            part["vis_coord"], part["target_coord"] = coord.split([num_visible, num_predict], 1)
            part["embeds"] = self.patch_embed.native(vis_patches, sp, patch, embed_ws.get(sp))
            if self.conv_stem is not None:
                # hybrid stem on the 8^3 voxel cells of each visible token (nearest resample off 1 mm)
                cells = self._to_cells8(vis_patches.to(part["embeds"].dtype), patch)
                part["embeds"] = part["embeds"] + self.conv_stem(cells).view(part["embeds"].shape)
            if "lesion" in part:
                vis_ids_l = part["order"][..., :num_visible] + (torch.arange(b, device=device) * num).view(-1, 1, 1)
                for key in ("lesion", "lesion_enh"):
                    if part.get(key) is not None:
                        part[key] = part[key].flatten(0, 1).index_select(0, vis_ids_l.flatten()).view(
                            b * num_samples, num_visible, -1
                        )
                # rtd replaced tokens show the donor, not the host: out of the lesion losses
                part["lesion_mask"] = vis_mask
                if "rtd_label" in part:
                    part["lesion_mask"] = vis_mask * ~part["rtd_label"][..., None]
            if self.level_head is not None:
                part["vis_patches"], part["vis_mask"] = vis_patches, vis_mask
            if self.level_mirror_head is not None:
                # mirror references of the visible tokens [b * S * num_visible, 64]
                ref = self.mirror_ref(vols, patch, geoms[sp][1], geoms[sp][2], sp)
                vis_ids = part["order"][..., :num_visible] + (torch.arange(b, device=device) * num).view(-1, 1, 1)
                part["mirror_ref"] = ref.flatten(0, 1).index_select(0, vis_ids.flatten())
            if (self.ctx_head is not None or self.anom_head is not None) and self.training:
                # context / local anomaly references of the visible tokens, from the whole host image
                ctx_t, anom_st = self.context_ref(vols, geoms[sp][2])
                vis_flat = (
                    part["order"][..., :num_visible] + (torch.arange(b, device=device) * num).view(-1, 1, 1)
                ).flatten()
                rows_v = (b * num_samples, num_visible)
                if ctx_t is not None:
                    part["ctx_t"] = ctx_t.flatten(0, 1).index_select(0, vis_flat).view(*rows_v, -1)
                if anom_st is not None:
                    part["anom_st"] = anom_st.flatten(0, 1).index_select(0, vis_flat).view(*rows_v, 2)
                    # the visible voxels the encoder saw (rtd replaced tokens are masked from the loss)
                    part["anom_x"], part["anom_m"] = vis_patches, vis_mask

        def cat(key: str) -> Tensor:
            # contiguous for one group too, the compiled encoder would recompile on view strides
            if len(parts) == 1:
                return parts[0][key].contiguous()
            return torch.cat([part[key] for part in parts])

        def flat_cat(tensors: list[Tensor]) -> Tensor:
            if len(tensors) == 1:
                return tensors[0].flatten()
            return torch.cat([t.flatten() for t in tensors])

        embeds = self.encode_tokens(cat("embeds"), cat("vis_coord"))
        feats = self.decode_features(embeds, cat("target_coord"))
        feats = torch.split(feats, [len(part["ids"]) * num_samples for part in parts])
        # graft: the v3 loss stack, extended to per-spacing native patches. norm_target
        # z-scores each token's voxels (spacing groups differ in patch size), the fg mask
        # weights the mse per voxel, and the xyz gradient penalty runs per group on its own
        # patch geometry; the global loss is the weight-masked mean across groups
        losses, wts = [], []
        grad_num, grad_den = [], []
        for part, feat in zip(parts, feats):
            sp = part["spacing"]
            pred = self.decoder_head.native(feat, sp, part["patch"], head_ws.get(sp))
            part["pred_patches"] = pred
            px, py, pz = part["patch"]
            tgt = part["target_patches"]
            if self.norm_target:
                tgt = (tgt - tgt.mean(dim=-1, keepdim=True)) / (tgt.var(dim=-1, keepdim=True) + 1e-6) ** 0.5
            mse = F.mse_loss(pred, tgt, reduction="none")
            mask = part["target_mask_patches"]
            w = mask / pred.shape[-1]
            if self.fg_loss_weight > 0.0:
                fg_w = 1.0 + self.fg_loss_weight * (part["target_patches"].abs() > 0.05).to(w.dtype)
                w = w * fg_w
            losses.append((w * mse).flatten())
            wts.append(w.flatten())
            if self.grad_loss_weight > 0.0:
                BM = pred.shape[0] * pred.shape[1]
                pv = pred.reshape(BM, px, py, pz)
                tv = tgt.reshape(BM, px, py, pz)
                mv = mask.reshape(BM, px, py, pz).to(w.dtype)
                dx = F.mse_loss(pv[:, 1:], tv[:, 1:] - tv[:, :-1], reduction="none")
                dy = F.mse_loss(pv[:, :, 1:], tv[:, :, 1:] - tv[:, :, :-1], reduction="none")
                dz = F.mse_loss(pv[:, :, :, 1:], tv[:, :, :, 1:] - tv[:, :, :, :-1], reduction="none")
                mx = mv[:, 1:] * mv[:, :-1]
                my = mv[:, :, 1:] * mv[:, :, :-1]
                mz = mv[:, :, :, 1:] * mv[:, :, :, :-1]
                grad_num += [(mx * dx).sum(), (my * dy).sum(), (mz * dz).sum()]
                grad_den += [mx.sum(), my.sum(), mz.sum()]
        loss = flat_cat(losses).sum() / flat_cat(wts).sum()
        if self.grad_loss_weight > 0.0:
            loss = loss + self.grad_loss_weight * (
                grad_num[0] / grad_den[0].clamp_min(1.0)
                + grad_num[1] / grad_den[1].clamp_min(1.0)
                + grad_num[2] / grad_den[2].clamp_min(1.0)
            ) / 3.0
        # an all 1 mm batch leaves the residuals without a gradient, and ddp needs one every step
        residuals = (self.patch_embed.residual, self.decoder_head.residual)
        residuals += (self.decoder_head.bias_residual,)
        loss = loss + 0.0 * sum(r.sum() for r in residuals)
        mterms = []
        if self.level_head is not None and self.level_batched:
            # level_terms with one head pass and one bce over all parts (speed, same sums): the
            # per-part targets are [rows, V, levels * sub^3] for every patch shape. level_mirror
            # likewise: per-part targets (cells8 resample per patch shape), one head pass and one bce
            def cat0(ts: list[Tensor]) -> Tensor:
                return torch.cat(ts) if len(ts) > 1 else ts[0]

            with torch.no_grad():
                tw, mt = [], []
                for part in parts:
                    vis_patches, vis_mask = part.pop("vis_patches"), part.pop("vis_mask")
                    tw.append(level_targets(
                        vis_patches, vis_mask, self.level_thresholds,
                        *(
                            adaptive_pool_matrix(n, self.patch_size[0] // self.level_pool, device)
                            for n in part["patch"]
                        ),
                    ))
                    if self.level_mirror_head is not None:
                        mt.append(self.mirror_targets(vis_patches, vis_mask, part["patch"], part.pop("mirror_ref")))
                target = cat0([t for t, _ in tw])
                weight = cat0([w for _, w in tw])
            tokens = embeds[:, self.class_tokens :]
            logits = self.level_head(tokens).float()
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            num = (weight * bce).sum()
            den = weight.sum()  # weight is repeated over the levels: level_terms' weight.sum() * L
            loss = loss + self.level_weight * num / den.clamp_min(1e-6)
            if mt:
                m_target, m_weight, l_weight = (cat0([t[k] for t in mt]) for k in range(3))
                m_logits = self.level_mirror_head(tokens).float().reshape(m_target.shape)
                m_bce = F.binary_cross_entropy_with_logits(m_logits, m_target, reduction="none")
                mterms.append((
                    (m_weight * m_bce).sum(), m_weight.sum(), l_weight.sum(),
                    (m_weight * m_target).sum((0, 2, 3, 4)),
                ))
        elif self.level_head is not None:
            num, den = 0.0, 0.0
            rows = torch.split(embeds, [len(part["ids"]) * num_samples for part in parts])
            for part, tokens in zip(parts, rows):
                vis_patches, vis_mask = part.pop("vis_patches"), part.pop("vis_mask")
                n, d = self.level_terms(tokens[:, self.class_tokens :], vis_patches, vis_mask, part["patch"])
                num, den = num + n, den + d
                if self.level_mirror_head is not None:
                    mterms.append(self.mirror_terms(
                        tokens[:, self.class_tokens :], vis_patches, vis_mask, part["patch"], part.pop("mirror_ref")
                    ))
            loss = loss + self.level_weight * num / den.clamp_min(1e-6)
        if mterms:
            # summed over groups: (weighted bce, valid weight, level weight, weighted positives [M])
            mb, mw, lw, mp = [sum(t[k] for t in mterms) for k in range(4)]
            T = len(self.level_thresholds)
            # same per-channel weight as the level channels (r18: / (valid weight * T))
            loss = loss + self.level_weight * mb / (mw * T).clamp_min(1e-6)
            M = len(self.level_mirror)
            self.level_mirror_stats = {
                "bce": (mb / (mw * M).clamp_min(1e-6)).detach(),
                "valid": (mw / lw.clamp_min(1e-6)).detach(),
                **{f"pos{t:g}": v for t, v in zip(self.level_mirror, (mp / mw.clamp_min(1e-6)).detach())},
            }
        if self.fino_heads is not None and fino_meta is not None:
            # the global embedding of each sequence: mean of its visible encoder tokens (eval:
            # mean of all brain-touching tokens); rows are part by part, image-major
            pooled = embeds[:, self.class_tokens :].mean(dim=1)
            # fino_meta stays on the cpu: one pinned non blocking copy, no sync with the gpu queue
            rows = torch.tensor([ii for part in parts for ii in part["ids"]]).repeat_interleave(num_samples)
            meta = fino_meta.cpu()[rows]
            if device.type == "cuda":
                meta = meta.pin_memory().to(device, non_blocking=True)
            fino_loss, self.fino_stats = self.fino_heads(pooled, meta, fino_gamma)
            loss = loss + fino_loss
        if self.rtd_head is not None and self.training and "rtd_label" in parts[0]:
            labels = cat("rtd_label").float()
            logits = self.rtd_head(embeds[:, self.class_tokens :]).squeeze(-1).float()
            rtd_loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss = loss + self.rtd_weight * rtd_loss
            with torch.no_grad():
                pos = labels > 0.5
                acc_pos = ((logits > 0) & pos).sum() / pos.sum().clamp_min(1)
                acc_neg = ((logits <= 0) & ~pos).sum() / (~pos).sum().clamp_min(1)
            # rtd_pos: share of visible tokens replaced (only spacing groups of >= 2 images swap, so
            # it tracks the batch composition: ~9% random bs 32, ~14% spacing bucketed)
            self.rtd_stats = {
                "rtd_loss": rtd_loss.detach(), "rtd_tpr": acc_pos, "rtd_tnr": acc_neg,
                "rtd_pos": pos.float().mean(),
            }
            self.fino_stats = {**self.fino_stats, **self.rtd_stats}

        if (self.ctx_head is not None or self.anom_head is not None) and self.training:
            tokens = embeds[:, self.class_tokens :]
            # rtd replaced tokens carry another subject's voxels: no context / anomaly target there
            keep = (
                (~cat("rtd_label")).float()
                if "rtd_label" in parts[0]
                else tokens.new_ones(tokens.shape[:2], dtype=torch.float32)
            )
            stats = {}
            if self.ctx_head is not None:
                target = cat("ctx_t")  # [rows, V, T * windows]
                logits = self.ctx_head(tokens).float()
                bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(-1)
                ctx_loss = (keep * bce).sum() / keep.sum().clamp_min(1.0)
                loss = loss + self.ctx_weight * ctx_loss
                stats["ctx_loss"] = ctx_loss.detach()
            if self.anom_head is not None:
                with torch.no_grad():
                    tw = [
                        self.anom_targets(part.pop("anom_x"), part.pop("anom_m"), part["patch"], part.pop("anom_st"))
                        for part in parts
                    ]
                    a_target = torch.cat([t for t, _ in tw]) if len(tw) > 1 else tw[0][0]
                    a_weight = torch.cat([w for _, w in tw]) if len(tw) > 1 else tw[0][1]
                    a_weight = a_weight * keep.reshape(-1, 1, 1, 1, 1)
                a_logits = self.anom_head(tokens).float().reshape(a_target.shape)
                a_bce = F.binary_cross_entropy_with_logits(a_logits, a_target, reduction="none")
                anom_loss = (a_weight * a_bce).sum() / (a_weight.sum() * len(self.anom_k)).clamp_min(1e-6)
                loss = loss + self.anom_weight * anom_loss
                with torch.no_grad():
                    pos = (a_weight * a_target).sum((0, 2, 3, 4)) / a_weight.sum().clamp_min(1e-6)
                stats["anom_loss"] = anom_loss.detach()
                stats.update({f"anom_pos{k:g}": v for k, v in zip(self.anom_k, pos)})
            self.ctx_stats = stats
            self.fino_stats = {**self.fino_stats, **stats}

        if (self.lesion_head is not None or self.global_lesion_head is not None) and self.training:
            tokens = embeds[:, self.class_tokens :]
            sub = self.patch_size[0] // self.level_pool
            stats = {}
            with torch.no_grad():
                # per part: [rows, V, P] lesion and in-mask (rtd-masked) voxels -> 2 mm sub-block targets
                wts, tgts, burden_num, burden_den = [], [], [], []
                for part in parts:
                    les, msk, patch = part.pop("lesion"), part.pop("lesion_mask"), part["patch"]
                    enh = part.pop("lesion_enh", None)
                    rows, V = msk.shape[:2]
                    px, py, pz = patch
                    m = msk.float().reshape(-1, 1, px, py, pz)
                    w = F.adaptive_avg_pool3d(m, sub).flatten(1)  # [rows*V, 64]

                    def frac(t: Tensor | None) -> Tensor:
                        # in-mask lesion fraction of each 2 mm sub-block
                        if t is None:
                            return torch.zeros_like(w)
                        return F.adaptive_avg_pool3d(t.float().reshape(m.shape) * m, sub).flatten(1) / w.clamp_min(1e-6)

                    t = frac(les)
                    if self.lesion_subregions:
                        t = torch.stack([t, frac(enh)], 1)  # [rows*V, 2, 64]: whole lesion, enhancing part
                    burden_num.append(m.new_zeros(rows) if les is None else (les.float().reshape(m.shape) * m).view(rows, -1).sum(1))
                    burden_den.append(m.view(rows, -1).sum(1))
                    wts.append(w)
                    tgts.append(t)
                weight, target = torch.cat(wts), torch.cat(tgts)
                if self.lesion_subregions:
                    weight = weight[:, None].expand_as(target)
                # visible lesion burden per sequence: presence and log10 burden scaled to [0, 1]
                burden = torch.cat(burden_num) / torch.cat(burden_den).clamp_min(1.0)
                present = (burden > 1e-3).float()
                log_b = ((torch.log10(burden + 1e-4) + 4.0) / 4.0).clamp(0.0, 1.0)
            if self.lesion_head is not None:
                logits = self.lesion_head(tokens).float().reshape(target.shape)
                bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
                lw = weight.sum().clamp_min(1e-6)
                les_loss = (weight * bce).sum() / lw
                loss = loss + self.lesion_mask_weight * les_loss
                stats["les_bce"] = les_loss.detach()
                if self.lesion_subregions:
                    stats["les_pos"] = ((weight[:, 0] * target[:, 0]).sum() / weight[:, 0].sum().clamp_min(1e-6)).detach()
                    stats["enh_pos"] = ((weight[:, 1] * target[:, 1]).sum() / weight[:, 1].sum().clamp_min(1e-6)).detach()
                else:
                    stats["les_pos"] = ((weight * target).sum() / lw).detach()
            if self.global_lesion_head is not None:
                pooled = tokens.mean(dim=1)  # [rows, D]: the eval global embedding's training analog
                g = self.global_lesion_head(pooled).float()
                g_loss = F.binary_cross_entropy_with_logits(g[:, 0], present) + F.binary_cross_entropy_with_logits(
                    g[:, 1], log_b
                )
                loss = loss + self.global_lesion_weight * g_loss
                stats["glob_les"] = g_loss.detach()
                stats["les_present"] = present.mean()
            self.lesion_stats = stats
            self.fino_stats = {**self.fino_stats, **stats}

        if not with_state:
            return loss
        return loss, {"parts": parts, "embeds": embeds}

    def compile_native(self, **kwargs) -> None:
        # the fixed-shape transformer parts. per-spacing tokenization, embedding and head stay eager
        self.encode_tokens = torch.compile(self.encode_tokens, **kwargs)
        self.decode_features = torch.compile(self.decode_features, **kwargs)

    def compile_cudagraphs(self) -> None:
        # training speed, same math: sampling + gather compiled as usual, the encoder / decoder / loss
        # fwd and bwd replayed as cuda graphs (one launch each instead of ~800 python launches)
        self.sample_patches = torch.compile(self.sample_patches_contiguous, dynamic=False)
        self.patch_loss = torch.compile(self.patch_loss, mode="reduce-overhead", dynamic=False)

    # the methods below implement the brainmarks model contract.
    #   - forward_embeddings: computes a global and dense embedding per input sample
    #   - transform: cpu data transform, runs in loader workers (optional)
    #   - transform_cuda: cuda data transform. you call it inside forward_embeddings (optional)

    def transform(self, sample: ImageInput) -> dict[str, Tensor]:
        image = torch.from_numpy(sample["image"].get_fdata(dtype=np.float32))
        mask = torch.from_numpy(np.asarray(sample["brain_mask"].dataobj) > 0)
        affine = sample["image"].affine
        image, mask, affine = crop_to_mask(image, mask, affine)
        return {"values": image[mask], "mask": mask, "affine": affine}

    def transform_cuda(self, sample: dict[str, Tensor]) -> dict:
        values = sample["values"]
        mask = sample["mask"]
        affine = sample["affine"]
        values = normalize_values(values, self.clip_quantiles)
        image = values.new_zeros(mask.shape).masked_scatter_(mask, values)
        image, mask, affine = reorient_to_ras(image, mask, affine)
        image, mask, affine = resample_to_spacing(image, mask, affine, self.target_spacing)
        image, mask, affine = fit_to_shape(image, mask, affine, self.grid_size)
        return {"image": image, "mask": mask, "affine": affine}

    @torch.inference_mode()
    def forward_embeddings(
        self, samples: list[dict], return_dense: bool = False
    ) -> list[EmbeddingOutput]:
        device = self.coord_grid.device
        patch_grid_size = [size // self.patch_size[0] for size in self.grid_size]

        # one forward per image since the number of patches in the mask varies
        outputs = []
        for sample in samples:
            sample = {
                "values": sample["values"].to(device),
                "mask": sample["mask"].to(device),
                "affine": sample["affine"],
            }
            sample = self.transform_cuda(sample)
            patches = patchify3d(sample["image"][None], self.patch_size[0])
            in_mask = (patchify3d(sample["mask"][None], self.patch_size[0]) > 0).any(dim=2)[0]
            # cudnn attention builds a new graph (~45 ms) for each new sequence length
            with (
                sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]),
                torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"),
            ):
                tokens = self.forward_encoder(patches[:, in_mask], self.coord_grid[:, in_mask])
            tokens = tokens[0, self.class_tokens :].float()

            output = EmbeddingOutput(
                global_embedding=tokens.mean(dim=0),
                dense_embedding=None,
                dense_affine=None,
                dense_mask=None,
            )
            if return_dense:
                dense = tokens.new_zeros(len(in_mask), self.embed_dim)
                dense[in_mask] = tokens
                output["dense_embedding"] = dense.reshape(*patch_grid_size, self.embed_dim)
                output["dense_affine"] = sample["affine"]
                output["dense_mask"] = in_mask.reshape(patch_grid_size)
            outputs.append(output)
        return outputs


@register_model
def vit_mae_3d(checkpoint: str, random_weights: bool = False) -> ViTMAE3D:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = ViTMAE3D.from_config(ckpt["args"])
    if not random_weights:
        model.load_state_dict(ckpt["model"])
    return model


def level_targets(
    patches: Tensor, mask: Tensor, thresholds: Tensor, ax: Tensor, ay: Tensor, az: Tensor
) -> tuple[Tensor, Tensor]:
    # ViTMAE3D.level_terms' (target, weight) without the head: [rows, V, L * sub^3] each (weight
    # repeated over the L levels), adaptive_avg_pool3d as separable bin-averaging matrices
    # (adaptive_pool_matrix, [sub, patch voxels] per axis)
    (sub, px), py, pz = ax.shape, ay.shape[1], az.shape[1]
    rows, V = patches.shape[:2]
    x = patches.float().reshape(rows * V, px, py, pz)
    m = mask.float().reshape(rows * V, px, py, pz)
    L = thresholds.numel()
    above = (x[:, None] > thresholds.view(1, -1, 1, 1, 1)).float() * m[:, None]
    both = torch.cat([m[:, None], above], dim=1)  # [RV, 1 + L, px, py, pz]
    pooled = torch.einsum("ncxyz,ix,jy,kz->ncijk", both, ax, ay, az)
    weight = pooled[:, :1]
    target = pooled[:, 1:] / weight.clamp_min(1e-6)
    target = target.reshape(rows, V, L * sub**3)
    weight = weight.expand(-1, L, -1, -1, -1).reshape(rows, V, L * sub**3)
    return target, weight



# synthetic lesions: insert_blobs, paste_tumors and plan_tumors are ported from mihirneal's
# nanobrain run r24_syn (labless run_sub_463a0d7b5e, github.com/mihirneal/nanobrain, MIT); the only local
# change is insert_blobs' optional per-image gain
def insert_blobs(
    images: Tensor,
    mask: Tensor,
    patch: tuple[int, int, int],
    sp: tuple[float, float, float],
    prob: float,
    num: int,
    radius: tuple[float, float],
    contrast: tuple[float, float],
    edge: float,
    box_mm: float = 40.0,
    with_mask: bool = False,
    gain: Tensor | None = None,
) -> Tensor | None:
    # (ported from mihirneal r24_syn; local addition: gain [B], a per-image signed contrast scale)
    # in place on a spacing group's padded volumes [B, X, Y, Z] (voxels of sp mm): per image (share
    # prob) 1 to num soft edged ellipsoids with a perturbed boundary, sized in mm, each in a box_mm
    # box and centred in a patch >= 90% in the mask. with_mask: returns the soft blob mask [B, X, Y, Z]
    B, X, Y, Z = images.shape
    dev = images.device
    dims = torch.tensor([X, Y, Z], device=dev)
    pt = torch.tensor(patch, device=dev)
    mm = torch.tensor(sp, device=dev, dtype=torch.float32)
    gx, gy, gz = X // patch[0], Y // patch[1], Z // patch[2]
    ok = mask.view(B, gx, patch[0], gy, patch[1], gz, patch[2]).float().mean((2, 4, 6)).view(B, 1, -1) >= 0.9
    # uniform over the qualifying patches (any patch when none qualify)
    idx = (torch.rand(B, num, ok.shape[2], device=dev) + ok.float()).argmax(-1)
    cell = torch.stack([idx // (gy * gz), idx // gz % gy, idx % gz], -1)
    centre = (cell * pt).float() + torch.rand(B, num, 3, device=dev) * pt
    box = [min(d, math.ceil(box_mm / s)) for d, s in zip((X, Y, Z), sp)]
    bt = torch.tensor(box, device=dev)
    lo = torch.minimum((centre.long() - bt // 2).clamp_min(0), dims - bt)
    r = torch.exp(torch.rand(B, num, 1, device=dev) * math.log(radius[1] / radius[0]) + math.log(radius[0]))
    axes = r * torch.exp(torch.rand(B, num, 3, device=dev) * 0.4 - 0.2)
    knots = 0.1 * torch.randn(B * num, 1, 4, 4, 4, device=dev)
    pert = F.interpolate(knots, size=box, mode="trilinear", align_corners=True).view(B, num, -1)
    grid = torch.meshgrid(*[torch.arange(n, device=dev) for n in box], indexing="ij")
    pos = lo[:, :, None] + torch.stack(grid, -1).view(1, 1, -1, 3)
    d = ((pos - centre[:, :, None]) * mm / axes[:, :, None]).norm(dim=-1) * (1 + pert)
    on = (torch.rand(B, 1, device=dev) < prob) & (
        torch.arange(num, device=dev) < torch.randint(1, num + 1, (B, 1), device=dev)
    )
    amp = (torch.rand(B, num, 1, device=dev) * (contrast[1] - contrast[0]) + contrast[0]) * on[..., None]
    if gain is not None:
        amp = amp * gain.view(B, 1, 1).to(amp.dtype)
    flat = (pos[..., 0] * Y + pos[..., 1]) * Z + pos[..., 2]
    flat = (flat + torch.arange(B, device=dev).view(-1, 1, 1) * (X * Y * Z)).view(-1)
    soft = torch.sigmoid((1 - d) * r / edge)
    inside = mask.reshape(-1)[flat].float()
    val = (amp * soft).view(-1) * inside
    images.view(-1).index_add_(0, flat, val.to(images.dtype))
    if not with_mask:
        return None
    out = torch.zeros_like(images)
    out.view(-1).scatter_reduce_(0, flat, ((soft * on[..., None]).view(-1) * inside).to(out.dtype), "amax")
    return out


def paste_tumors(images: Tensor, mask: Tensor, plan: tuple, with_mask: bool = False) -> Tensor | None:
    # in place on a spacing group's padded volumes [B, X, Y, Z], in the mask only, clipped to each
    # image's own in-mask range (a saturating enhancement). returns the soft lesion mask if asked
    flat, val, soft, lim = plan
    inside = mask.reshape(-1)[flat] > 0
    old = images.view(-1)[flat]
    new = torch.minimum(torch.maximum(old + val, lim[:, 0]), lim[:, 1]).to(old.dtype)
    images.view(-1).index_copy_(0, flat, torch.where(inside, new, old))
    if not with_mask:
        return None
    out = torch.zeros_like(images)
    out.view(-1).index_copy_(0, flat, (soft * inside).to(out.dtype))
    return out


def plan_tumors(
    images: list[Tensor],
    masks: list[Tensor],
    groups: dict[tuple, list[int]],
    geoms: dict[tuple, tuple],
    prob: float,
    radius: tuple[float, float],
    ring: float,
    contrast: tuple[float, float],
    edge: tuple[float, float],
    nodule: float,
) -> dict[tuple, tuple[Tensor, Tensor, Tensor, Tensor]]:
    # one lobulated mass sized in mm in a prob share of the batch, centred >= 6 mm inside the mask: a
    # solid bright 3-10 mm nodule (share nodule), else a bright 3-8 mm rim of uneven thickness on a
    # mildly darker core (share ring) or a diffuse bright mass. all spacing groups in one pass over
    # the masses' own boxes; per spacing: flat voxel indices into its padded [B, X, Y, Z] volumes, the
    # value to add, the soft lesion mask and the image's in-mask (min, max) there
    dev = masks[0].device
    k = min(len(masks), int(prob * len(masks) + torch.rand(()).item()))
    if k == 0:
        return {}
    chosen = set(torch.randperm(len(masks))[:k].tolist())
    geo, owners, counts = [], [], {}
    for sp, ids in groups.items():
        patch, box, grid, _ = geoms[sp]
        dims = [num * p for num, p in zip(grid, patch)]
        for jj, ii in enumerate(ids):
            if ii not in chosen:
                continue
            src, dst = fit_slices(masks[ii].shape, box)
            span = [d.stop - d.start for d in dst]
            geo.append([*sp, *dims, *(d.start for d in dst), *span, *(d.start - c.start for c, d in zip(src, dst)),
                        *masks[ii].shape, jj * math.prod(dims)])
            owners.append(ii)
            counts[sp] = counts.get(sp, 0) + 1
    # per mass draws and box sizes on the cpu stream (known sizes, no gpu sync)
    geo = torch.tensor(geo, dtype=torch.float64)
    spc, dims, dst0, span, shift, shape = geo[:, :18].split(3, 1)
    u = torch.rand(k, 10, dtype=torch.float64)
    is_nodule = u[:, 9] < nodule
    is_ring = ~is_nodule & (u[:, 4] < ring)
    hot = is_nodule | is_ring
    r = torch.where(is_nodule, 3.0 * (10.0 / 3.0) ** u[:, 0], radius[0] * (radius[1] / radius[0]) ** u[:, 0])
    axes = r[:, None] * torch.exp(u[:, 1:4] * 0.4 - 0.2)
    # ring and nodule edges 2 mm wide (10-90%), diffuse edges at most r wide
    width = torch.where(hot, 2.0, torch.minimum(edge[0] + u[:, 5] * (edge[1] - edge[0]), r))
    rim = torch.minimum(3.0 + 5.0 * u[:, 6], 0.6 * r)
    amp = torch.where(hot, 2.0 + 3.0 * u[:, 7], contrast[0] + u[:, 7] * (contrast[1] - contrast[0]))
    # box per axis: its semi-axis at the largest boundary bump (d = 1 / 0.75), + the edge
    num = torch.minimum(dims, torch.ceil(2 * (axes / 0.75 + width[:, None]) / spc)).long()
    size = num.prod(1)
    start = size.cumsum(0) - size
    total = int(size.sum())
    ints = torch.cat([
        dims, dst0, span, shift, shape, (6.0 / spc).round().clamp_min(1), num, start[:, None], geo[:, 18:],
    ], 1).long().to(dev)
    dims, dst0, span, shift, shape, probe, num = ints[:, :21].split(3, 1)
    start, base = ints[:, 21], ints[:, 22]
    # columns: mm per voxel, 1 / semi-axes, 1 / bump scale, r, 1 / logistic scale, rim, ring, contrasts, knot row
    cols = torch.cat([
        spc, 1 / axes, 1 / (1.6 * r[:, None]), r[:, None], 4.4 / width[:, None], rim[:, None], is_ring[:, None],
        amp[:, None], 0.2 + 0.6 * u[:, 8:9], 5.0 * torch.arange(k)[:, None],
    ], 1).float().to(dev)

    # centre: of 64 voxels in the image, the first with itself and its 6 mm axis neighbours in the mask
    cand = torch.minimum((torch.rand(k, 64, 3, device=dev) * span[:, None]).long(), span[:, None] - 1) + dst0[:, None]
    star = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], device=dev
    )
    pts = (cand - shift[:, None])[:, :, None] + star * probe[:, None, None]
    pts = torch.minimum(pts.clamp_min(0), shape[:, None, None] - 1)
    pts = (pts[..., 0] * shape[:, 1, None, None] + pts[..., 1]) * shape[:, 2, None, None] + pts[..., 2]
    hits = torch.stack([masks[ii].reshape(-1)[pts[t].reshape(-1)] for t, ii in enumerate(owners)])
    best = hits.view(k, 64, 7).sum(-1).argmax(1)
    centre = torch.take_along_dim(cand, best.view(k, 1, 1).expand(k, 1, 3), 1)[:, 0] + torch.rand(k, 3, device=dev)
    lo = torch.minimum((centre.long() - num // 2).clamp_min(0), dims - num)
    lim = torch.stack([
        torch.stack([
            torch.where(masks[ii].bool(), images[ii], math.inf).amin(),
            torch.where(masks[ii].bool(), images[ii], -math.inf).amax(),
        ])
        for ii in owners
    ]).float()

    # every box voxel of every mass in one ragged row: owner, then its box and volume indices
    owner = torch.zeros(total, dtype=torch.int32, device=dev)
    owner[start[1:]] = 1
    owner = owner.cumsum(0, dtype=torch.int32)
    per = torch.stack([
        start, num[:, 1] * num[:, 2], num[:, 1], num[:, 2], dims[:, 1] * dims[:, 2], dims[:, 2],
    ], 1).int().index_select(0, owner)
    q = torch.arange(total, dtype=torch.int32, device=dev) - per[:, 0]
    loc = torch.stack([q // per[:, 1], q // per[:, 3] % per[:, 2], q % per[:, 3]], 1)
    flat = (base + (lo[:, 0] * dims[:, 1] + lo[:, 1]) * dims[:, 2] + lo[:, 2]).index_select(0, owner)
    flat = flat + (loc[:, 0] * per[:, 4] + loc[:, 1] * per[:, 5] + loc[:, 2])
    par = torch.cat([cols, centre - lo], 1).index_select(0, owner)
    rel = (loc - par[:, 14:17]) * par[:, 0:3]  # mm from the centre
    d = (rel * par[:, 3:6]).square().sum(1).sqrt()
    # boundary bumps and rim thickness: trilinear noise on 5^3 knots spanning +-1.6 r of the mass,
    # masses stacked on depth
    knots = (0.15 * torch.randn(1, 2, 5 * k, 5, 5, device=dev)).clamp(-0.25, 0.25)
    g = (rel * par[:, 6:7]).clamp(-1, 1)
    row = (par[:, 13] + (g[:, 0] + 1) * 2) * (2 / (5 * k - 1)) - 1
    grid = torch.stack([g[:, 2], g[:, 1], row], -1).view(1, total, 1, 1, 3)
    bump, thick = F.grid_sample(knots, grid, mode="bilinear", padding_mode="border", align_corners=True).view(2, total)
    depth = (1 - d * (1 + bump)) * par[:, 7]  # ~mm inside the boundary
    outer = torch.sigmoid(depth * par[:, 8])
    # rim 0.5-1.5x its thickness around the mass, 4 mm wide transition to the core
    core = torch.sigmoid((depth - par[:, 9] * (1 + 2 * thick)) * (4.4 / 4.0)) * par[:, 10]
    val = par[:, 11] * (outer - core) - par[:, 12] * core
    lim = lim.index_select(0, owner)
    out, t = {}, 0
    size, start = size.tolist(), start.tolist()
    for sp, n in counts.items():
        a, b = start[t], start[t + n - 1] + size[t + n - 1]
        out[sp] = (flat[a:b], val[a:b], outer[a:b], lim[a:b])
        t += n
    return out



def insert_strokes(
    images: Tensor,
    mask: Tensor,
    patch: tuple[int, int, int],
    sp: tuple[float, float, float],
    prob: float,
    num: int,
    spread: float,
    radius: tuple[float, float],
    contrast: tuple[float, float],
    edge: float,
    box_mm: float = 64.0,
    with_mask: bool = False,
    gain: Tensor | None = None,
) -> Tensor | None:
    # stroke-like lesions (new): in place on a spacing group's padded volumes [B, X, Y, Z], per image
    # (share prob) one irregular infarct made of 2 to num overlapping elongated soft ellipsoids whose
    # centres scatter (sd spread mm) around a seed in a patch >= 90% in the mask: lobulated, uneven,
    # DWI/FLAIR-bright territories rather than the round masses of plan_tumors. same mechanics as
    # insert_blobs (sizes in mm, boundary noise, soft edge, in-mask only). with_mask: soft mask [B, X, Y, Z]
    B, X, Y, Z = images.shape
    dev = images.device
    dims = torch.tensor([X, Y, Z], device=dev)
    pt = torch.tensor(patch, device=dev)
    mm = torch.tensor(sp, device=dev, dtype=torch.float32)
    gx, gy, gz = X // patch[0], Y // patch[1], Z // patch[2]
    ok = mask.view(B, gx, patch[0], gy, patch[1], gz, patch[2]).float().mean((2, 4, 6)).view(B, -1) >= 0.9
    idx = (torch.rand(B, ok.shape[1], device=dev) + ok.float()).argmax(-1)
    cell = torch.stack([idx // (gy * gz), idx // gz % gy, idx % gz], -1)
    seed = (cell * pt).float() + torch.rand(B, 3, device=dev) * pt
    # foci around the seed, in voxels of this spacing, kept inside the volume
    centre = seed[:, None] + torch.randn(B, num, 3, device=dev) * spread / mm
    centre = torch.minimum(centre.clamp_min(0), (dims - 1).float())
    box = [min(d, math.ceil(box_mm / s)) for d, s in zip((X, Y, Z), sp)]
    bt = torch.tensor(box, device=dev)
    lo = torch.minimum((centre.long() - bt // 2).clamp_min(0), dims - bt)
    r = torch.exp(torch.rand(B, num, 1, device=dev) * math.log(radius[1] / radius[0]) + math.log(radius[0]))
    # elongated, unevenly shaped foci
    axes = r * torch.exp(torch.rand(B, num, 3, device=dev) - 0.5)
    knots = 0.15 * torch.randn(B * num, 1, 4, 4, 4, device=dev)
    pert = F.interpolate(knots, size=box, mode="trilinear", align_corners=True).view(B, num, -1)
    grid = torch.meshgrid(*[torch.arange(n, device=dev) for n in box], indexing="ij")
    pos = lo[:, :, None] + torch.stack(grid, -1).view(1, 1, -1, 3)
    d = ((pos - centre[:, :, None]) * mm / axes[:, :, None]).norm(dim=-1) * (1 + pert)
    on = (torch.rand(B, 1, device=dev) < prob) & (
        torch.arange(num, device=dev) < torch.randint(2, num + 1, (B, 1), device=dev)
    )
    # one brightness per infarct (foci of one lesion share their signal), slight per-focus jitter
    amp = (torch.rand(B, 1, 1, device=dev) * (contrast[1] - contrast[0]) + contrast[0])
    amp = amp * (0.85 + 0.3 * torch.rand(B, num, 1, device=dev)) * on[..., None]
    flat = (pos[..., 0] * Y + pos[..., 1]) * Z + pos[..., 2]
    flat = (flat + torch.arange(B, device=dev).view(-1, 1, 1) * (X * Y * Z)).view(-1)
    soft = torch.sigmoid((1 - d) * r / edge)
    inside = mask.reshape(-1)[flat].float()
    # overlapping foci saturate instead of stacking: per voxel max of amp * soft over the foci, then the
    # per-image signed gain (gain [B]: e.g. bright on DWI, dark on ADC)
    add = torch.zeros_like(images)
    add.view(-1).scatter_reduce_(0, flat, ((amp * soft).view(-1) * inside).to(add.dtype), "amax")
    if gain is not None:
        add = add * gain.view(B, 1, 1, 1).to(add.dtype)
    images.add_(add)
    if not with_mask:
        return None
    out = torch.zeros_like(images)
    out.view(-1).scatter_reduce_(0, flat, ((soft * on[..., None]).view(-1) * inside).to(out.dtype), "amax")
    return out


# contrast-aware lesion appearance (signed intensity gain per MRI sequence code, nanobrain.fino.sequence_code:
# 0 T1w, 1 T1c, 2 T2w / PDw / DWI b0, 3 FLAIR, 4 DWI, 5 ADC, 6 other). Tumours: enhancing rim / mass bright on T1c,
# bright on T2 / FLAIR, hypointense on T1w, mild on DWI / ADC. Acute infarcts: bright on DWI, dark on ADC, mildly
# bright on T2 / FLAIR, mildly dark on T1. Small foci (WMH-like): bright on FLAIR / T2, faint elsewhere.
SEQ_T1C = 1
TUMOR_GAIN = (-0.6, 1.0, 0.9, 1.0, 0.4, 0.5, 1.0)
STROKE_GAIN = (-0.5, -0.4, 0.8, 0.9, 1.0, -0.9, 1.0)
BLOB_GAIN = (-0.3, 0.3, 1.0, 1.0, 0.3, 0.3, 1.0)
# peritumoral edema amplitude (z): hyperintense on FLAIR / T2, faint on T1 and diffusion
HALO_AMP = (-0.2, 0.0, 0.9, 1.0, 0.2, 0.4, 0.5)


def edema_halo(tumor: Tensor, mask: Tensor, sp: tuple[float, float, float], reach_mm: float = 6.0) -> Tensor:
    # soft shell around each mass [B, X, Y, Z] in [0, 1]: the mass mask box-blurred twice (separable,
    # +-reach_mm per pass, so the edema reaches ~6-12 mm), saturated, minus the mass itself, in the brain only
    x = tumor.float()[:, None]
    for _ in range(2):
        for axis, s_mm in enumerate(sp):
            k = 2 * max(1, round(reach_mm / s_mm)) + 1
            size = [1, 1, 1]
            size[axis] = k
            pad = [0, 0, 0]
            pad[axis] = k // 2
            x = F.avg_pool3d(x, tuple(size), 1, tuple(pad), count_include_pad=True)
    halo = (x[:, 0] * 4.0).clamp(0.0, 1.0) * (1.0 - tumor.float()).clamp_min(0.0) * (mask > 0).float()
    return halo.to(tumor.dtype)


class PatchConvStem(nn.Module):
    """Hybrid stem (new): a small 3D CNN inside each 8 mm patch (8^3 voxels -> 16 x 8^3 -> 32 x 4^3 at
    2 mm cells), projected to the token width and added to the linear patch embedding. The projection
    is zero-initialized, so training starts exactly at the linear embedding (and its natural-image
    init); the CNN gives each token a nonlinear, spatially structured view of its own voxels, which
    the seg probe (each token -> its 512 voxel labels) relies on."""

    def __init__(self, dim: int, width: int = 16):
        super().__init__()
        self.c1 = nn.Conv3d(1, width, 3, padding=1)
        self.c2 = nn.Conv3d(width, 2 * width, 3, stride=2, padding=1)
        self.proj = nn.Linear(2 * width * 64, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cells: Tensor) -> Tensor:
        # cells [R, 1, 8, 8, 8] -> [R, dim]
        x = F.gelu(self.c1(cells))
        x = F.gelu(self.c2(x))
        return self.proj(x.flatten(1))


def patchify3d(x: Tensor, patch_size: int | tuple[int, int, int] = 8) -> Tensor:
    px, py, pz = (patch_size,) * 3 if isinstance(patch_size, int) else patch_size
    x = rearrange(x, "b (gx px) (gy py) (gz pz) -> b (gx gy gz) (px py pz)", px=px, py=py, pz=pz)
    return x


@lru_cache(maxsize=4096)
def cells8_index(patch: tuple[int, int, int], device: torch.device) -> Tensor:
    # flat voxel index in a px x py x pz patch of each 8^3 nearest-resampled voxel
    ax = [torch.div(torch.arange(8) * p, 8, rounding_mode="floor") for p in patch]
    i, j, k = torch.meshgrid(*ax, indexing="ij")
    return ((i * patch[1] + j) * patch[2] + k).flatten().to(device)


@lru_cache(maxsize=4096)
def native_geometry(
    spacing: tuple[float, float, float],
    box_mm: tuple[int, int, int],
    token_mm: float,
    device: torch.device | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], Float[Tensor, "1 N 3"]]:
    # per axis: patch voxels (~token_mm), the box in voxels, the token grid tiling the box from
    # its low corner, and token centres in position table units, centred like fit_to_shape. at
    # 1 mm these are make_coord_grid's coordinates
    patch = tuple(max(1, round(token_mm / sp)) for sp in spacing)
    box = native_box(spacing, box_mm)
    grid = tuple(math.ceil(size / p) for size, p in zip(box, patch))
    centres = []
    for sp, p, size, num, mm in zip(spacing, patch, box, grid, box_mm):
        voxel = (torch.arange(num, dtype=torch.float64, device=device) + 0.5) * p
        centres.append((((voxel - size / 2) * sp + mm / 2) / token_mm - 0.5).float())
    coord = torch.stack(torch.meshgrid(*centres, indexing="ij"), dim=-1).reshape(1, -1, 3)
    return patch, box, grid, coord


def unpatchify3d(x: Tensor, grid_size: tuple[int, int, int], patch_size: int = 8) -> Tensor:
    p = patch_size
    gx, gy, gz = [size // patch_size for size in grid_size]
    x = rearrange(
        x,
        "b (gx gy gz) (px py pz) -> b (gx px) (gy py) (gz pz)",
        gx=gx,
        gy=gy,
        gz=gz,
        px=p,
        py=p,
        pz=p,
    )
    return x


def patches_to_volume(
    patches: Float[Tensor, "B L P"],
    patch_ids: Int[Tensor, "B L"],
    grid_size: tuple[int, int, int],
    patch_size: int = 8,
) -> Float[Tensor, "B X Y Z"]:
    B, L, P = patches.shape
    gx, gy, gz = [size // patch_size for size in grid_size]
    grid = patches.new_zeros(B, gx * gy * gz, P)
    batch_ids = torch.arange(B, device=patches.device)[:, None]
    grid[batch_ids, patch_ids] = patches
    images = unpatchify3d(grid, grid_size=grid_size, patch_size=patch_size)
    return images


def make_coord_grid(
    grid_size: tuple[int, int, int],
    patch_size: int = 8,
    device: torch.device | None = None,
) -> Tensor:
    patch_grid_size = tuple(size // patch_size for size in grid_size)
    grid_ids = [torch.arange(size, device=device) for size in patch_grid_size]
    coords = torch.stack(torch.meshgrid(*grid_ids, indexing="ij"), dim=-1)
    coords = coords.reshape(1, -1, 3)
    return coords


def sample_sequences(
    mask_patches: torch.Tensor,
    num_samples: int,
    seq_length: int | None = None,
    min_mask_frac: float = 0.25,
    cube: tuple[tuple[int, int, int], int] | None = None,
) -> torch.Tensor:
    scores = sequence_scores(mask_patches, num_samples, min_mask_frac, cube)
    if seq_length is None:
        # one host sync to get the longest mask sequence in the batch
        mask_frac = mask_patches.float().mean(dim=2)
        seq_length = (mask_frac >= min_mask_frac).sum(dim=1).max().item()
    order = scores.argsort(dim=2)[:, :, :seq_length]
    return order


def sequence_scores(
    mask_patches: torch.Tensor,
    num_samples: int,
    min_mask_frac: float = 0.25,
    cube: tuple[tuple[int, int, int], int] | None = None,
) -> torch.Tensor:
    # sample_sequences' sort keys, [B, num_samples, N]
    batch_size, num_patches, _ = mask_patches.shape
    mask_frac = mask_patches.float().mean(dim=2)
    # one independent shuffle per sequence, with patches outside the mask sorted last so
    # they are only used when a volume has too few mask patches
    scores = torch.rand(batch_size, num_samples, num_patches, device=mask_patches.device)
    if cube is not None:
        # cube: (patch grid, edge in patches). one uniform score per cube, shared by its patches,
        # so they are ordered (visible / predicted) together; the white noise only breaks ties
        # inside a cube. The cube grid is shifted by a random offset per sequence, so no two
        # patches are always in the same cube
        grid, edge = cube
        num_seq = batch_size * num_samples
        coords = grid_coords(tuple(grid), scores.device)
        offset = torch.randint(0, edge, (num_seq, 1, 3), device=scores.device)
        cells = (coords + offset) // edge
        cx, cy, cz = [size // edge + 2 for size in grid]
        cell_ids = (cells[..., 0] * cy + cells[..., 1]) * cz + cells[..., 2]
        cell_scores = torch.rand(num_seq, cx * cy * cz, device=scores.device)
        cell_scores = torch.gather(cell_scores, 1, cell_ids).reshape(batch_size, num_samples, num_patches)
        scores = cell_scores + 1e-6 * scores
    return torch.where(mask_frac[:, None, :] >= min_mask_frac, scores, 2.0)


def fit_slices(
    shape: tuple[int, ...], box: tuple[int, ...]
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    # fit_to_shape's centre pad / crop as (source, destination) slices
    src, dst = [], []
    for size, target in zip(shape, box):
        before = (target - size) // 2
        num = min(size, target)
        src.append(slice(max(0, -before), max(0, -before) + num))
        dst.append(slice(max(0, before), max(0, before) + num))
    return tuple(src), tuple(dst)


@lru_cache(maxsize=4096)
def adaptive_pool_matrix(size: int, out: int, device: torch.device) -> Tensor:
    # [out, size] rows averaging adaptive_avg_pool's bins [floor(i size / out), ceil((i+1) size / out))
    a = torch.zeros(out, size, dtype=torch.float32)
    for i in range(out):
        lo, hi = (i * size) // out, -((-(i + 1) * size) // out)
        a[i, lo:hi] = 1.0 / (hi - lo)
    return a.to(device)


@lru_cache(maxsize=4096)
def gather_offsets(batch_size: int, num_patches: int, device: torch.device) -> Tensor:
    # [2, B, 1, 1] flat row offsets of (image | mask, image) in a [2, B, N, P] patch tensor
    return torch.arange(2 * batch_size, device=device).view(2, batch_size, 1, 1) * num_patches


@lru_cache(maxsize=4096)
def grid_coords(grid: tuple[int, int, int], device: torch.device) -> Tensor:
    axes = [torch.arange(size, device=device) for size in grid]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(1, -1, 3)


# JAX ViT xavier uniform init
# https://github.com/facebookresearch/capi/blob/main/model.py
def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
        nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
