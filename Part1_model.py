"""
Part 1 – Semantic Segmentation
model.py: DeepLabV3+ with ResNet-101 or ResNet-50 backbone via torchvision.

Architecture:
  - Encoder  : ResNet-101 (ImageNet pretrained) with dilated convolutions
  - ASPP     : Atrous Spatial Pyramid Pooling (rates 12, 24, 36)
  - Decoder  : bilinear up-sampling + 3×3 conv
  - Output   : 1×1 conv → NUM_CLASSES channels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import (
    deeplabv3_resnet101, DeepLabV3_ResNet101_Weights,
    deeplabv3_resnet50,  DeepLabV3_ResNet50_Weights,
)

NUM_CLASSES = 19


def build_deeplabv3plus(backbone: str = "resnet101", num_classes: int = NUM_CLASSES,
                        pretrained_backbone: bool = True) -> nn.Module:
    """
    Build torchvision DeepLabV3 and replace the classifier head for
    Cityscapes 19-class training.

    Args:
        backbone: "resnet101" or "resnet50"
        num_classes: number of output classes (19 for Cityscapes)
        pretrained_backbone: use ImageNet pretrained encoder weights

    Returns:
        nn.Module ready for fine-tuning
    """
    weights_backbone = "DEFAULT" if pretrained_backbone else None

    if backbone == "resnet101":
        weights = DeepLabV3_ResNet101_Weights.DEFAULT if pretrained_backbone else None
        model = deeplabv3_resnet101(weights=weights)
    elif backbone == "resnet50":
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained_backbone else None
        model = deeplabv3_resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # Replace COCO 21-class head → Cityscapes 19-class head
    in_channels = model.classifier[4].in_channels   # 256
    model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    if model.aux_classifier is not None:
        in_aux = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(in_aux, num_classes, kernel_size=1)

    return model


class SegmentationLoss(nn.Module):
    """Cross-entropy loss ignoring label 255 (Cityscapes void class)."""

    def __init__(self, num_classes: int = NUM_CLASSES, aux_weight: float = 0.4):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=255)
        self.aux_weight = aux_weight

    def forward(self, output: dict, target: torch.Tensor) -> torch.Tensor:
        main_loss = self.ce(output["out"], target)
        total = main_loss
        if "aux" in output:
            aux_loss = self.ce(output["aux"], target)
            total = main_loss + self.aux_weight * aux_loss
        return total
