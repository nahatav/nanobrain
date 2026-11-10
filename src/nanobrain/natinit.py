# Natural-image pretrained init for the 3D encoder (operator ruling 2026-11-08: "natural image
# model is ok"; the README bans weights pretrained on brain MRI only, this is the operator's
# reading of that rule). Attention and MLP act on token vectors, so the 12 blocks and the final
# norm of a 2D ViT-B (width 768, 12 heads) copy over as is. The 2D patch embedding inflates
# I3D-style into the 1 mm canonical NativeEmbed kernel. Position embeddings stay random.
#
# Supported checkpoints (local safetensors, compute nodes have no network; when natinit.path is
# missing, model.safetensors of natinit.hf_repo is downloaded through huggingface_hub instead):
#   mae:    facebook/vit-mae-base (HF ViTMAEForPreTraining, separate q/k/v)
#   dinov2: timm/vit_base_patch14_dinov2.lvd142m (fused qkv, LayerScale folded into proj / fc2)

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


def _family(sd: dict) -> str:
    if "vit.encoder.layer.0.attention.attention.query.weight" in sd:
        return "mae"
    if "blocks.0.attn.qkv.weight" in sd:
        return "dinov2"
    raise ValueError("natinit: unknown checkpoint layout")


def _block_tensors(sd: dict, family: str, ii: int) -> dict[str, torch.Tensor]:
    # our Block's state dict keys -> tensors
    if family == "mae":
        p = f"vit.encoder.layer.{ii}."
        return {
            "norm1.weight": sd[p + "layernorm_before.weight"],
            "norm1.bias": sd[p + "layernorm_before.bias"],
            "attn.q.weight": sd[p + "attention.attention.query.weight"],
            "attn.q.bias": sd[p + "attention.attention.query.bias"],
            "attn.k.weight": sd[p + "attention.attention.key.weight"],
            "attn.k.bias": sd[p + "attention.attention.key.bias"],
            "attn.v.weight": sd[p + "attention.attention.value.weight"],
            "attn.v.bias": sd[p + "attention.attention.value.bias"],
            "attn.proj.weight": sd[p + "attention.output.dense.weight"],
            "attn.proj.bias": sd[p + "attention.output.dense.bias"],
            "norm2.weight": sd[p + "layernorm_after.weight"],
            "norm2.bias": sd[p + "layernorm_after.bias"],
            "mlp.fc1.weight": sd[p + "intermediate.dense.weight"],
            "mlp.fc1.bias": sd[p + "intermediate.dense.bias"],
            "mlp.fc2.weight": sd[p + "output.dense.weight"],
            "mlp.fc2.bias": sd[p + "output.dense.bias"],
        }
    p = f"blocks.{ii}."
    q, k, v = sd[p + "attn.qkv.weight"].chunk(3, dim=0)
    qb, kb, vb = sd[p + "attn.qkv.bias"].chunk(3, dim=0)
    g1, g2 = sd[p + "ls1.gamma"], sd[p + "ls2.gamma"]
    # x + ls1 * proj(a) = x + (g1 W) a + g1 b
    return {
        "norm1.weight": sd[p + "norm1.weight"],
        "norm1.bias": sd[p + "norm1.bias"],
        "attn.q.weight": q, "attn.q.bias": qb,
        "attn.k.weight": k, "attn.k.bias": kb,
        "attn.v.weight": v, "attn.v.bias": vb,
        "attn.proj.weight": sd[p + "attn.proj.weight"] * g1[:, None],
        "attn.proj.bias": sd[p + "attn.proj.bias"] * g1,
        "norm2.weight": sd[p + "norm2.weight"],
        "norm2.bias": sd[p + "norm2.bias"],
        "mlp.fc1.weight": sd[p + "mlp.fc1.weight"],
        "mlp.fc1.bias": sd[p + "mlp.fc1.bias"],
        "mlp.fc2.weight": sd[p + "mlp.fc2.weight"] * g2[:, None],
        "mlp.fc2.bias": sd[p + "mlp.fc2.bias"] * g2,
    }


def inflate_patch_kernel(w2: torch.Tensor, p: int) -> torch.Tensor:
    """[D, C, k, k] 2D patch kernel -> [D, p*p*p] canonical 1 mm kernel (px py pz flattened).

    RGB summed into one channel (a grey image fed as R=G=B gives the same response), the k x k
    kernel sum-pooled to p x p in-plane cells (x <- kernel rows, y <- columns; each 1 mm voxel
    gathers the weights of the pixels it covers, so a smooth input keeps its response), then
    replicated along z and divided by p: an input constant along z gives the 2D response."""
    D, _, k, _ = w2.shape
    w = w2.float().sum(dim=1, keepdim=True)
    # adaptive average pooling handles k not a multiple of p (14 -> 8); x (k/p)^2 keeps the sum
    w = F.adaptive_avg_pool2d(w, (p, p)) * (k * k) / (p * p)
    w = w[:, 0, :, :, None].expand(D, p, p, p) / p
    return w.reshape(D, p * p * p).contiguous()


def resolve_checkpoint(cfg) -> Path:
    """cfg.path if it exists (local safetensors file or dir), else model.safetensors of cfg.hf_repo
    at cfg.hf_revision, downloaded once through huggingface_hub into cfg.cache_dir (null = the HF
    cache). For runtimes with network (Colab); the cluster path is used as is."""
    path = Path(cfg.path) if cfg.get("path") else None
    if path is not None and path.exists():
        return path / "model.safetensors" if path.is_dir() else path
    repo = cfg.get("hf_repo")
    assert repo, f"natinit: {path} not found and no hf_repo to download"
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=repo,
        filename="model.safetensors",
        revision=cfg.get("hf_revision"),
        cache_dir=cfg.get("cache_dir"),
    )
    logger.info(f"natinit: {path} not found, using {repo}@{cfg.get('hf_revision')} -> {local}")
    return Path(local)


@torch.no_grad()
def apply_natural_init(model, cfg) -> dict:
    """Copy blocks + final norm (and with cfg.embed the inflated patch embedding) into model."""
    path = resolve_checkpoint(cfg)
    sd = load_file(str(path))
    family = _family(sd)
    report = {"family": family, "path": str(path), "copied": [], "mismatch": [], "skipped": []}

    n_src = 1 + max(
        int(k.split(".")[3 if family == "mae" else 1])
        for k in sd
        if k.startswith("vit.encoder.layer." if family == "mae" else "blocks.")
    )
    if n_src != len(model.blocks):
        report["mismatch"].append(f"depth {n_src} vs ours {len(model.blocks)}")
    for ii, block in enumerate(model.blocks[: min(n_src, len(model.blocks))]):
        ours = block.state_dict()
        src = _block_tensors(sd, family, ii)
        for key, t in src.items():
            if key not in ours or ours[key].shape != t.shape:
                report["mismatch"].append(
                    f"blocks.{ii}.{key}: src {tuple(t.shape)} ours "
                    f"{tuple(ours[key].shape) if key in ours else None}"
                )
                continue
            ours[key].copy_(t.to(ours[key].dtype))
        extra = sorted(set(ours) - set(src))
        if extra:
            report["skipped"].append(f"blocks.{ii} kept random: {extra}")
        block.load_state_dict(ours)
    report["copied"].append(f"{min(n_src, len(model.blocks))} blocks (q k v proj fc1 fc2 norm1 norm2)")

    nw, nb = ("vit.layernorm.weight", "vit.layernorm.bias") if family == "mae" else ("norm.weight", "norm.bias")
    if model.norm.weight.shape == sd[nw].shape:
        model.norm.weight.copy_(sd[nw])
        model.norm.bias.copy_(sd[nb])
        report["copied"].append("final norm")
    else:
        report["mismatch"].append(f"final norm {tuple(sd[nw].shape)} vs {tuple(model.norm.weight.shape)}")

    if cfg.embed:
        pw = "vit.embeddings.patch_embeddings.projection." if family == "mae" else "patch_embed.proj."
        w2, b2 = sd[pw + "weight"], sd[pw + "bias"]
        p = model.patch_embed.patch_size if hasattr(model.patch_embed, "patch_size") else model.patch_size[0]
        w3 = inflate_patch_kernel(w2, p)
        if w3.shape != model.patch_embed.weight.shape:
            report["mismatch"].append(f"patch embed {tuple(w3.shape)} vs {tuple(model.patch_embed.weight.shape)}")
        else:
            model.patch_embed.weight.copy_(w3)
            model.patch_embed.bias.copy_(b2)
            report["copied"].append(
                f"patch embed inflated {tuple(w2.shape)} -> {tuple(w3.shape)} (rgb sum, pool to {p}x{p}, /{p} along z)"
            )
    else:
        report["skipped"].append("patch embed kept random")

    if cfg.pos_scale != 1.0:
        for wt in (model.pos_embed.weight_x, model.pos_embed.weight_y, model.pos_embed.weight_z):
            wt.mul_(cfg.pos_scale)
    report["skipped"].append(f"pos embed random (x{cfg.pos_scale}), decoder / heads random")
    report["skipped"].append(
        "src unused: cls token, pos embed" + (", mae decoder" if family == "mae" else "")
    )
    logger.info(f"natinit: {report}")
    return report


def is_natinit_param(name: str) -> bool:
    # the pretrained encoder parameters whose lr natinit.block_lr_mult scales
    return name.startswith("blocks.") or name in ("norm.weight", "norm.bias")
