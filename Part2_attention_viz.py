"""
Part 2 – ViT Classification
attention_viz.py: Visualize ViT attention maps using:
  1. Raw last-layer attention (CLS token → patches)
  2. Attention Rollout (Abnar & Zuidema, 2020)

Usage:
    python Part2_attention_viz.py \
        --checkpoint checkpoints/cls/vit_b16/best.pth \
        --image path/to/image.jpg \
        --output attention_maps.png
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import timm

from Part2_model import build_model


IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


# ──────────────────────────────────────────────
#  Hook manager to capture attention weights
# ──────────────────────────────────────────────
class AttentionHook:
    """Registers forward hooks on all self-attention layers of a timm ViT."""

    def __init__(self, model):
        self.attentions = []
        self.hooks      = []
        for block in model.blocks:
            h = block.attn.register_forward_hook(self._hook_fn)
            self.hooks.append(h)

    def _hook_fn(self, module, input, output):
        # timm ViT returns (output, attn_weights) when attn_drop is set
        # We need to register a modified forward to capture raw attention
        pass

    def remove(self):
        for h in self.hooks:
            h.remove()


class ViTAttentionExtractor:
    """
    Patches timm ViT blocks to return attention weights.
    Works for timm >= 0.9 with vit_base_patch16_224.
    """

    def __init__(self, model):
        self.model      = model
        self.attentions = []   # filled per forward pass
        self._patch_blocks()

    def _patch_blocks(self):
        for block in self.model.blocks:
            orig_fwd = block.attn.forward

            def make_patched(orig):
                extractor = self
                def patched_forward(x):
                    B, N, C = x.shape
                    qkv = block.attn.qkv(x).reshape(
                        B, N, 3, block.attn.num_heads,
                        C // block.attn.num_heads
                    ).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    scale   = (C // block.attn.num_heads) ** -0.5
                    attn    = (q @ k.transpose(-2, -1)) * scale
                    attn    = attn.softmax(dim=-1)
                    extractor.attentions.append(attn.detach().cpu())
                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = block.attn.proj(x)
                    x = block.attn.proj_drop(x)
                    return x
                return patched_forward

            block.attn.forward = make_patched(orig_fwd)

    def clear(self):
        self.attentions = []


# ──────────────────────────────────────────────
#  Attention Rollout (Abnar & Zuidema, 2020)
# ──────────────────────────────────────────────
def attention_rollout(attentions, discard_ratio: float = 0.9):
    """
    Compute attention rollout from a list of [B, H, N, N] attention matrices.
    Returns spatial attention map for CLS token.
    """
    result = torch.eye(attentions[0].shape[-1])
    for attn in attentions:
        # Average over heads
        attn_avg = attn[0].mean(0)  # [N, N]
        # Add residual connection
        attn_avg = attn_avg + torch.eye(attn_avg.shape[0])
        # Normalize
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
        result   = attn_avg @ result

    # CLS token → all patch tokens
    mask = result[0, 1:]   # skip CLS itself
    width = int(mask.shape[0] ** 0.5)
    mask  = mask.reshape(width, width).numpy()
    mask  = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


def raw_head_attention(attentions):
    """Last-layer, CLS-to-patch attention for each head."""
    last = attentions[-1][0]   # [H, N, N]
    H    = last.shape[0]
    width = int((last.shape[-1] - 1) ** 0.5)
    maps = []
    for h in range(H):
        m = last[h, 0, 1:].numpy().reshape(width, width)
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        maps.append(m)
    return maps


def preprocess_image(image_path: str, img_size: int = 224):
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMG_MEAN, IMG_STD),
    ])
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((img_size, img_size))
    tensor = transform(img_resized).unsqueeze(0)
    return tensor, np.array(img_resized)


def overlay_attention(img_np: np.ndarray, attn_map: np.ndarray,
                      cmap: str = "jet", alpha: float = 0.55) -> np.ndarray:
    h, w = img_np.shape[:2]
    attn_resized = np.array(
        Image.fromarray((attn_map * 255).astype(np.uint8)).resize((w, h),
        Image.BILINEAR)) / 255.0
    colormap = plt.get_cmap(cmap)
    heatmap  = (colormap(attn_resized)[:, :, :3] * 255).astype(np.uint8)
    overlay  = (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)
    return overlay


def visualize_attention(image_path: str, checkpoint: str,
                        output_path: str = "attention_maps.png",
                        num_classes: int = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt   = torch.load(checkpoint, map_location=device)
    n_cls  = num_classes or len(ckpt.get("classes", [0]*2))
    model  = build_model("vit_b16", n_cls).to(device).eval()
    model.load_state_dict(ckpt["model_state"])

    extractor = ViTAttentionExtractor(model)
    tensor, img_np = preprocess_image(image_path)

    with torch.no_grad():
        logits = model(tensor.to(device))
    pred_idx = logits.argmax(1).item()
    classes  = ckpt.get("classes", [str(i) for i in range(n_cls)])
    print(f"Predicted class: {classes[pred_idx]}")

    # Compute attention maps
    rollout  = attention_rollout(extractor.attentions)
    head_maps = raw_head_attention(extractor.attentions)

    num_heads = len(head_maps)
    fig = plt.figure(figsize=(4 * min(num_heads + 2, 8), 8))
    gs  = fig.add_gridspec(2, min(num_heads + 2, 8))

    # Row 1: original + rollout
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(img_np); ax_orig.set_title("Input"); ax_orig.axis("off")

    ax_roll = fig.add_subplot(gs[0, 1])
    ax_roll.imshow(overlay_attention(img_np, rollout))
    ax_roll.set_title("Attention Rollout"); ax_roll.axis("off")

    # Row 2: individual heads (up to 6)
    for i, hmap in enumerate(head_maps[:6]):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(overlay_attention(img_np, hmap))
        ax.set_title(f"Head {i+1}", fontsize=9)
        ax.axis("off")

    plt.suptitle(f"ViT-B/16 Attention Maps  |  Pred: {classes[pred_idx]}",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Attention map saved → {output_path}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   type=str, required=True)
    p.add_argument("--image",        type=str, required=True)
    p.add_argument("--output",       type=str, default="attention_maps.png")
    p.add_argument("--num_classes",  type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    visualize_attention(args.image, args.checkpoint,
                        args.output, args.num_classes)
