"""
Part 2 – ViT Classification
evaluate.py: Accuracy, F1, confusion matrix, per-class report.

Usage:
    python Part2_evaluate.py \
        --data_root /path/to/dataset \
        --checkpoint checkpoints/cls/vit_b16/best.pth \
        --model_name vit_b16
"""

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                              top_k_accuracy_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from Part2_dataset import ImageFolderDataset
from Part2_model   import build_model


def evaluate(model, loader, device, classes):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    top1 = (all_preds == all_labels).mean()
    top5 = top_k_accuracy_score(all_labels, all_probs, k=min(5, len(classes)))

    print(f"\nTop-1 Accuracy: {top1*100:.2f}%")
    print(f"Top-5 Accuracy: {top5*100:.2f}%\n")
    print(classification_report(all_labels, all_preds, target_names=classes))

    return all_preds, all_labels, all_probs, top1, top5


def plot_confusion_matrix(all_labels, all_preds, classes, out_path):
    cm   = confusion_matrix(all_labels, all_preds)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(max(8, len(classes)), max(6, len(classes))))
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax,
                linewidths=0.3, linecolor="gray")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Normalized Confusion Matrix", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved → {out_path}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--model_name",  type=str, default="vit_b16")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir",  type=str, default="eval_results")
    return p.parse_args()


def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import os; os.makedirs(args.output_dir, exist_ok=True)

    val_ds = ImageFolderDataset(args.data_root, "val")
    loader = DataLoader(val_ds, batch_size=args.batch_size,
                         num_workers=args.num_workers)

    ckpt   = torch.load(args.checkpoint, map_location=device)
    model  = build_model(args.model_name, len(val_ds.classes)).to(device)
    model.load_state_dict(ckpt["model_state"])

    preds, labels, probs, top1, top5 = evaluate(model, loader, device, val_ds.classes)
    plot_confusion_matrix(labels, preds, val_ds.classes,
                          os.path.join(args.output_dir, "confusion_matrix.png"))

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"top1": top1, "top5": top5, "model": args.model_name}, f, indent=2)


if __name__ == "__main__":
    main()
