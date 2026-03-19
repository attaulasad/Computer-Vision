"""
Part 1 – Semantic Segmentation
visualize.py: Overlay predictions on images + ablation mIoU bar chart.

Usage:
    # Visualize single image
    python Part1_visualize.py --mode predict \
        --checkpoint checkpoints/seg/A4/best.pth \
        --image path/to/image.png

    # Plot ablation results
    python Part1_visualize.py --mode ablation \
        --ablation_json checkpoints/seg/ablation_results.json
"""

import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from Part1_dataset    import CITYSCAPES_CLASSES, NUM_CLASSES, encode_label
from Part1_transforms import get_val_transform, MEAN, STD
from Part1_model      import build_deeplabv3plus

# ──────────────────────────────────────────────
#  Cityscapes color palette (19 trainIds)
# ──────────────────────────────────────────────
PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70],   [102, 102, 156],
    [190, 153, 153],[153, 153, 153],[250, 170, 30],  [220, 220, 0],
    [107, 142, 35], [152, 251, 152],[70, 130, 180],  [220, 20, 60],
    [255, 0, 0],    [0, 0, 142],    [0, 0, 70],      [0, 60, 100],
    [0, 80, 100],   [0, 0, 230],    [119, 11, 32],
], dtype=np.uint8)

VOID_COLOR = np.array([0, 0, 0], dtype=np.uint8)


def label_to_color(label: np.ndarray) -> np.ndarray:
    """Convert 19-class label map to RGB color map."""
    h, w   = label.shape
    color  = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(NUM_CLASSES):
        color[label == cls_id] = PALETTE[cls_id]
    color[label == 255] = VOID_COLOR
    return color


def predict_image(model, image_path: str, device):
    """Run inference on a single image; return (rgb_image, color_pred)."""
    image_np = np.array(Image.open(image_path).convert("RGB"))
    transform = get_val_transform()
    tensor = transform(image=image_np)["image"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(tensor)["out"].argmax(1).squeeze().cpu().numpy()

    return image_np, label_to_color(pred)


def save_overlay(image_np: np.ndarray, pred_color: np.ndarray,
                 out_path: str, alpha: float = 0.55):
    """Blend prediction color map over original image."""
    overlay = (alpha * pred_color + (1 - alpha) * image_np).astype(np.uint8)

    patches = [mpatches.Patch(color=PALETTE[i] / 255., label=CITYSCAPES_CLASSES[i])
               for i in range(NUM_CLASSES)]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    axes[0].imshow(image_np);       axes[0].set_title("Input Image");         axes[0].axis("off")
    axes[1].imshow(pred_color);     axes[1].set_title("Prediction");          axes[1].axis("off")
    axes[2].imshow(overlay);        axes[2].set_title("Overlay (alpha=0.55)");axes[2].axis("off")
    fig.legend(handles=patches, loc="lower center", ncol=7,
               fontsize=7, frameon=False)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved overlay → {out_path}")


def plot_ablation(results: dict, out_path: str = "ablation_miou.png"):
    """Bar chart of mIoU per augmentation config."""
    configs = list(results.keys())
    mious   = [v * 100 for v in results.values()]

    labels = {
        "A0": "Baseline\n(resize)",
        "A1": "A0 + HFlip",
        "A2": "A1 + RandCrop",
        "A3": "A2 + ColorJitter",
        "A4": "A3 + Scale\n+ Noise + Distort",
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([labels.get(c, c) for c in configs], mious,
                  color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"])
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=11)
    ax.set_ylabel("Validation mIoU (%)", fontsize=12)
    ax.set_title("DeepLabV3+ | Augmentation Ablation on Cityscapes", fontsize=13)
    ax.set_ylim(0, max(mious) + 8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ablation chart → {out_path}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",           choices=["predict", "ablation"], required=True)
    p.add_argument("--checkpoint",     type=str)
    p.add_argument("--backbone",       type=str, default="resnet101")
    p.add_argument("--image",          type=str)
    p.add_argument("--output",         type=str, default="prediction_overlay.png")
    p.add_argument("--ablation_json",  type=str)
    p.add_argument("--ablation_chart", type=str, default="ablation_miou.png")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "predict":
        assert args.checkpoint and args.image, "--checkpoint and --image required"
        model = build_deeplabv3plus(args.backbone).to(device)
        ckpt  = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        img_np, pred_color = predict_image(model, args.image, device)
        save_overlay(img_np, pred_color, args.output)

    elif args.mode == "ablation":
        assert args.ablation_json, "--ablation_json required"
        with open(args.ablation_json) as f:
            results = json.load(f)
        plot_ablation(results, args.ablation_chart)


if __name__ == "__main__":
    main()
