"""
Part 2 – ViT Classification
train.py: Fine-tune ViT-B/16, ResNet-50, EfficientNet-B4.

Training strategy (ViT):
  Phase 1 (warm-up, 5 epochs): freeze backbone, train head only
  Phase 2 (fine-tune, full):   unfreeze all layers with differential LR

Usage:
    python Part2_train.py \
        --data_root /path/to/dataset \
        --model_name vit_b16 \
        --epochs 30

    # Train all three models for comparison
    python Part2_train.py --data_root /path/to/dataset --compare_all
"""

import os
import argparse
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Part2_dataset import ImageFolderDataset
from Part2_model   import build_model, freeze_backbone, unfreeze_all, count_parameters


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    type=str,  required=True)
    p.add_argument("--model_name",   type=str,  default="vit_b16",
                   choices=["vit_b16", "resnet50", "efficientnet_b4"])
    p.add_argument("--num_classes",  type=int,  default=None,
                   help="Auto-detected from data_root if not specified")
    p.add_argument("--img_size",     type=int,  default=224)
    p.add_argument("--epochs",       type=int,  default=30)
    p.add_argument("--warmup_epochs",type=int,  default=5,
                   help="Epochs to train head-only before unfreezing")
    p.add_argument("--batch_size",   type=int,  default=32)
    p.add_argument("--lr",           type=float,default=1e-3)
    p.add_argument("--backbone_lr",  type=float,default=1e-5,
                   help="Lower LR for backbone during full fine-tune")
    p.add_argument("--weight_decay", type=float,default=1e-4)
    p.add_argument("--patience",     type=int,  default=7,
                   help="Early stopping patience")
    p.add_argument("--num_workers",  type=int,  default=4)
    p.add_argument("--output_dir",   type=str,  default="checkpoints/cls")
    p.add_argument("--compare_all",  action="store_true")
    return p.parse_args()


def make_optimizer(model, model_name, lr, backbone_lr, weight_decay,
                   phase="head"):
    if phase == "head":
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay
        )
    # Phase 2: differential LR
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if any(k in name for k in ["head", "fc", "classifier"]):
            head_params.append(param)
        else:
            backbone_params.append(param)
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params,     "lr": lr},
    ], weight_decay=weight_decay)


def run_epoch(model, loader, criterion, optimizer, device, scaler, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, desc="  " + ("train" if train else "val"),
                                    leave=False):
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss   = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)
    return total_loss / total, correct / total


def train_model(args, model_name: str = None):
    name   = model_name or args.model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Training {name}  |  device={device}")
    print(f"{'='*55}")

    train_ds  = ImageFolderDataset(args.data_root, "train", args.img_size)
    val_ds    = ImageFolderDataset(args.data_root, "val",   args.img_size)
    num_cls   = args.num_classes or len(train_ds.classes)
    print(f"  Classes ({num_cls}): {train_ds.classes}")
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}")
    print(f"  Params: {count_parameters(build_model(name, num_cls)):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True,  num_workers=args.num_workers,
                               pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers,
                               pin_memory=True)

    model     = build_model(name, num_cls).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    ckpt_dir  = os.path.join(args.output_dir, name)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer    = SummaryWriter(os.path.join(ckpt_dir, "logs"))

    best_acc  = 0.0
    patience_ctr = 0
    history   = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Phase 1: warm-up → train head only
        if epoch == 1:
            freeze_backbone(model, name)
            optimizer = make_optimizer(model, name, args.lr, args.backbone_lr,
                                        args.weight_decay, phase="head")
            print("  [Warm-up phase] backbone frozen")
        # Phase 2: unfreeze at warmup_epochs+1
        if epoch == args.warmup_epochs + 1:
            unfreeze_all(model)
            optimizer = make_optimizer(model, name, args.lr, args.backbone_lr,
                                        args.weight_decay, phase="full")
            print("  [Fine-tune phase] all layers unfrozen (differential LR)")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs)

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                     optimizer, device, scaler, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion,
                                     optimizer, device, scaler, train=False)
        if epoch > args.warmup_epochs:
            scheduler.step()

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"loss={tr_loss:.4f}/{vl_loss:.4f}  "
              f"acc={tr_acc*100:.2f}%/{vl_acc*100:.2f}%  ({elapsed:.0f}s)")

        writer.add_scalar("train/loss", tr_loss, epoch)
        writer.add_scalar("val/loss",   vl_loss, epoch)
        writer.add_scalar("train/acc",  tr_acc,  epoch)
        writer.add_scalar("val/acc",    vl_acc,  epoch)

        for k, v in zip(["train_loss","val_loss","train_acc","val_acc"],
                         [tr_loss, vl_loss, tr_acc, vl_acc]):
            history[k].append(v)

        if vl_acc > best_acc:
            best_acc = vl_acc
            patience_ctr = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_acc": vl_acc, "classes": train_ds.classes},
                       os.path.join(ckpt_dir, "best.pth"))
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    writer.close()
    with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Best val acc [{name}]: {best_acc*100:.2f}%")
    return best_acc


def main():
    args = get_args()
    if args.compare_all:
        results = {}
        for m in ["vit_b16", "resnet50", "efficientnet_b4"]:
            results[m] = train_model(args, m)
        print("\n===== MODEL COMPARISON =====")
        for m, acc in results.items():
            print(f"  {m:<25}: {acc*100:.2f}%")
        with open(os.path.join(args.output_dir, "comparison.json"), "w") as f:
            json.dump(results, f, indent=2)
    else:
        train_model(args)


if __name__ == "__main__":
    main()
