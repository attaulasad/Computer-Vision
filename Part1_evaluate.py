"""
Part 1 – Semantic Segmentation
evaluate.py: Per-class mIoU evaluation + confusion matrix + report table.

Usage:
    python Part1_evaluate.py \
        --data_root /path/to/cityscapes \
        --checkpoint checkpoints/seg/A4/best.pth
"""

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Part1_dataset    import CityscapesDataset, NUM_CLASSES, CITYSCAPES_CLASSES
from Part1_transforms import get_val_transform
from Part1_model      import build_deeplabv3plus


# ──────────────────────────────────────────────
#  Core metric: streaming confusion matrix
# ──────────────────────────────────────────────
class SegMetrics:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index
        self.conf_matrix   = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray):
        valid = target != self.ignore_index
        pred, target = pred[valid], target[valid]
        idx = target * self.num_classes + pred
        self.conf_matrix += np.bincount(idx, minlength=self.num_classes ** 2) \
                              .reshape(self.num_classes, self.num_classes)

    def per_class_iou(self) -> np.ndarray:
        tp  = np.diag(self.conf_matrix)
        fp  = self.conf_matrix.sum(0) - tp
        fn  = self.conf_matrix.sum(1) - tp
        iou = tp / (tp + fp + fn + 1e-10)
        # mask classes with no GT pixels
        valid = self.conf_matrix.sum(1) > 0
        iou[~valid] = np.nan
        return iou

    def mean_iou(self) -> float:
        iou = self.per_class_iou()
        return float(np.nanmean(iou))

    def pixel_accuracy(self) -> float:
        tp_total = np.diag(self.conf_matrix).sum()
        total    = self.conf_matrix.sum()
        return float(tp_total / (total + 1e-10))


@torch.no_grad()
def evaluate_miou(model, loader, device, num_classes=NUM_CLASSES):
    """Returns (mean_iou, per_class_iou_list). Called during training."""
    model.eval()
    metrics = SegMetrics(num_classes)
    for images, masks in loader:
        images = images.to(device)
        outputs = model(images)["out"]
        preds = outputs.argmax(dim=1).cpu().numpy()
        for p, t in zip(preds, masks.numpy()):
            metrics.update(p, t)
    iou = metrics.per_class_iou()
    return metrics.mean_iou(), iou.tolist()


def print_iou_table(per_class_iou: np.ndarray, miou: float, pix_acc: float):
    header = f"{'Class':<20} {'IoU':>8}"
    print("\n" + "=" * 30)
    print(f"  Per-Class IoU Report")
    print("=" * 30)
    print(header)
    print("-" * 30)
    for cls, iou in zip(CITYSCAPES_CLASSES, per_class_iou):
        val = f"{iou*100:.2f}" if not np.isnan(iou) else " N/A"
        print(f"{cls:<20} {val:>8}")
    print("-" * 30)
    print(f"{'mIoU':<20} {miou*100:>7.2f}")
    print(f"{'Pixel Acc.':<20} {pix_acc*100:>7.2f}")
    print("=" * 30)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--backbone",    type=str, default="resnet101")
    p.add_argument("--batch_size",  type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_json", type=str, default="eval_results.json")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_ds = CityscapesDataset(args.data_root, "val",
                                transform=get_val_transform())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    model = build_deeplabv3plus(args.backbone).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint: mIoU={ckpt.get('miou', 'N/A')}")

    metrics  = SegMetrics(NUM_CLASSES)
    model.eval()
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            preds  = model(images)["out"].argmax(1).cpu().numpy()
            for p, t in zip(preds, masks.numpy()):
                metrics.update(p, t)

    per_class = metrics.per_class_iou()
    miou      = metrics.mean_iou()
    pix_acc   = metrics.pixel_accuracy()

    print_iou_table(per_class, miou, pix_acc)

    results = {
        "mIoU":         miou,
        "pixel_acc":    pix_acc,
        "per_class_iou": {
            cls: float(v) for cls, v in zip(CITYSCAPES_CLASSES, per_class)
        }
    }
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {args.output_json}")


if __name__ == "__main__":
    main()
