"""
Part 1 – Semantic Segmentation
train.py: Training loop + ablation study runner.

Usage:
    # Single run with full augmentation
    python Part1_train.py --data_root /path/to/cityscapes --aug_config A4

    # Run all ablation configs sequentially
    python Part1_train.py --data_root /path/to/cityscapes --ablation
"""

import os
import argparse
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Part1_dataset   import CityscapesDataset, NUM_CLASSES
from Part1_transforms import get_transform, get_val_transform
from Part1_model     import build_deeplabv3plus, SegmentationLoss
from Part1_evaluate  import evaluate_miou


# ──────────────────────────────────────────────
#  Polynomial LR scheduler (standard for DeepLab)
# ──────────────────────────────────────────────
class PolyLRScheduler:
    def __init__(self, optimizer, max_iters: int, power: float = 0.9):
        self.optimizer  = optimizer
        self.max_iters  = max_iters
        self.power      = power
        self.base_lrs   = [pg["lr"] for pg in optimizer.param_groups]
        self.iter       = 0

    def step(self):
        factor = (1 - self.iter / self.max_iters) ** self.power
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * factor
        self.iter += 1


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--backbone",    type=str, default="resnet101", choices=["resnet50", "resnet101"])
    p.add_argument("--aug_config",  type=str, default="A4",
                   choices=["A0", "A1", "A2", "A3", "A4"])
    p.add_argument("--epochs",      type=int, default=80)
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--lr",          type=float, default=0.01)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir",  type=str, default="checkpoints/seg")
    p.add_argument("--ablation",    action="store_true",
                   help="Run all aug configs A0–A4 sequentially")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="  train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train(args, aug_config: str = None):
    cfg = aug_config or args.aug_config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Training DeepLabV3+  |  backbone={args.backbone}  |  aug={cfg}")
    print(f"  device={device}")
    print(f"{'='*60}")

    # Datasets
    train_ds = CityscapesDataset(args.data_root, "train",
                                  transform=get_transform(cfg))
    val_ds   = CityscapesDataset(args.data_root, "val",
                                  transform=get_val_transform())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True,  num_workers=args.num_workers,
                               pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                               shuffle=False, num_workers=args.num_workers,
                               pin_memory=True)

    # Model
    model = build_deeplabv3plus(args.backbone).to(device)

    # Optimizer: separate LR for backbone vs head
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if "classifier" not in n and "aux" not in n],
         "lr": args.lr * 0.1},
        {"params": [p for n, p in model.named_parameters()
                    if "classifier" in n or "aux" in n],
         "lr": args.lr},
    ]
    optimizer = torch.optim.SGD(params, momentum=0.9,
                                 weight_decay=args.weight_decay)
    max_iters = len(train_loader) * args.epochs
    scheduler = PolyLRScheduler(optimizer, max_iters=max_iters)
    criterion = SegmentationLoss(NUM_CLASSES, aux_weight=0.4)
    scaler    = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # Logging
    ckpt_dir = os.path.join(args.output_dir, cfg)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer   = SummaryWriter(log_dir=os.path.join(ckpt_dir, "logs"))

    best_miou = 0.0
    history = {"train_loss": [], "val_miou": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     scheduler, criterion, device, scaler)
        miou, per_class = evaluate_miou(model, val_loader, device, NUM_CLASSES)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={train_loss:.4f}  mIoU={miou:.4f}  ({elapsed:.0f}s)")
        writer.add_scalar("train/loss",    train_loss, epoch)
        writer.add_scalar("val/mIoU",      miou,       epoch)
        writer.add_scalars("val/per_class_IoU",
                           dict(zip(["road","sidewalk","building","wall","fence",
                                     "pole","tlight","tsign","vegetation","terrain",
                                     "sky","person","rider","car","truck","bus",
                                     "train","motorcycle","bicycle"], per_class)), epoch)

        history["train_loss"].append(train_loss)
        history["val_miou"].append(miou)

        if miou > best_miou:
            best_miou = miou
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "miou": miou, "per_class_iou": per_class},
                       os.path.join(ckpt_dir, "best.pth"))

    writer.close()
    with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest mIoU for config {cfg}: {best_miou:.4f}")
    return best_miou


def main():
    args = get_args()
    if args.ablation:
        results = {}
        for cfg in ["A0", "A1", "A2", "A3", "A4"]:
            results[cfg] = train(args, aug_config=cfg)
        print("\n===== ABLATION RESULTS =====")
        for cfg, miou in results.items():
            print(f"  {cfg}: {miou:.4f}")
        with open(os.path.join(args.output_dir, "ablation_results.json"), "w") as f:
            json.dump(results, f, indent=2)
    else:
        train(args)


if __name__ == "__main__":
    main()
