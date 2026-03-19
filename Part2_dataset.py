"""
Part 2 – ViT Classification
dataset.py: Generic multi-class image dataset loader.

Expected layout:
    data_root/
      train/
        class_a/  img1.jpg  img2.jpg ...
        class_b/  ...
      val/
        class_a/  ...
      test/       (optional)
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


def build_transforms(split: str, img_size: int = 224):
    """ImageNet-style transforms with augmentation for training."""
    if split == "train":
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize(int(img_size * 256 / 224)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


class ImageFolderDataset(Dataset):
    """
    Reads class sub-folders. Compatible with multi-class classification
    for ViT, ResNet-50, and EfficientNet models.
    """

    def __init__(self, root: str, split: str = "train",
                 img_size: int = 224, transform=None):
        self.split     = split
        self.root      = os.path.join(root, split)
        self.transform = transform or build_transforms(split, img_size)
        self.classes   = sorted(os.listdir(self.root))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples   = self._scan()

    def _scan(self):
        samples = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    samples.append((os.path.join(cls_dir, fname),
                                    self.class_to_idx[cls]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label
