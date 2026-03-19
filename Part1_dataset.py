"""
Part 1 – Semantic Segmentation
dataset.py: Cityscapes dataset loader with 19 trainId classes.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# Cityscapes 35-class → 19 trainId mapping (255 = ignore)
LABEL_MAPPING = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0,   8: 1,   9: 255, 10: 255,
    11: 2,  12: 3,  13: 4,  14: 255, 15: 255, 16: 255,
    17: 5,  18: 255, 19: 6,  20: 7,
    21: 8,  22: 9,
    23: 10, 24: 11, 25: 12,
    26: 13, 27: 14, 28: 15,
    29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: 255
}

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

NUM_CLASSES = 19


def encode_label(mask: np.ndarray) -> np.ndarray:
    """Convert full 35-class label to 19 trainId label."""
    label = np.full_like(mask, 255, dtype=np.int64)
    for k, v in LABEL_MAPPING.items():
        label[mask == k] = v
    return label


class CityscapesDataset(Dataset):
    """
    Cityscapes fine annotation dataset.

    Directory layout expected:
        root/
          leftImg8bit/{split}/{city}/*.png
          gtFine/{split}/{city}/*_gtFine_labelIds.png
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        assert split in ("train", "val", "test")
        self.root = root
        self.split = split
        self.transform = transform
        self.images, self.masks = self._collect_files()

    def _collect_files(self):
        img_dir = os.path.join(self.root, "leftImg8bit", self.split)
        mask_dir = os.path.join(self.root, "gtFine", self.split)
        images, masks = [], []
        for city in sorted(os.listdir(img_dir)):
            city_img = os.path.join(img_dir, city)
            city_mask = os.path.join(mask_dir, city)
            for fname in sorted(os.listdir(city_img)):
                if fname.endswith("_leftImg8bit.png"):
                    stem = fname.replace("_leftImg8bit.png", "")
                    mask_name = f"{stem}_gtFine_labelIds.png"
                    images.append(os.path.join(city_img, fname))
                    masks.append(os.path.join(city_mask, mask_name))
        return images, masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert("RGB"))
        mask = np.array(Image.open(self.masks[idx]))
        mask = encode_label(mask).astype(np.int64)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask
