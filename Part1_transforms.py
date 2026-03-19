"""
Part 1 – Semantic Segmentation
transforms.py: Albumentations augmentation pipelines for ablation study.

Ablation configs:
  A0 – baseline (resize + normalize)
  A1 – A0 + horizontal flip
  A2 – A1 + random crop
  A3 – A2 + color jitter
  A4 – A3 (full aug: + scale jitter + gaussian noise + grid distortion)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

CROP_H, CROP_W = 512, 1024   # standard Cityscapes training resolution


def get_transform(config: str = "A4"):
    """Return albumentations Compose transform for the given ablation config."""

    base = [
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ]

    if config == "A0":
        pipeline = [
            A.Resize(CROP_H, CROP_W),
        ] + base

    elif config == "A1":
        pipeline = [
            A.Resize(CROP_H, CROP_W),
            A.HorizontalFlip(p=0.5),
        ] + base

    elif config == "A2":
        pipeline = [
            A.SmallestMaxSize(max_size=CROP_H * 2),
            A.RandomCrop(height=CROP_H, width=CROP_W),
            A.HorizontalFlip(p=0.5),
        ] + base

    elif config == "A3":
        pipeline = [
            A.SmallestMaxSize(max_size=CROP_H * 2),
            A.RandomCrop(height=CROP_H, width=CROP_W),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        ] + base

    elif config == "A4":   # full augmentation
        pipeline = [
            A.RandomScale(scale_limit=(-0.5, 1.0), p=1.0),  # scale 0.5×–2×
            A.PadIfNeeded(min_height=CROP_H, min_width=CROP_W,
                          border_mode=0, value=0, mask_value=255),
            A.RandomCrop(height=CROP_H, width=CROP_W),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.1),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.1),
        ] + base
    else:
        raise ValueError(f"Unknown augmentation config: {config}")

    return A.Compose(pipeline)


def get_val_transform():
    """Deterministic val/test transform: resize only."""
    return A.Compose([
        A.Resize(CROP_H, CROP_W),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
