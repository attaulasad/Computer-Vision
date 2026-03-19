"""
Part 2 – ViT Classification
model.py: Model factory for ViT-B/16, ResNet-50, and EfficientNet-B4.

All models use ImageNet-pretrained weights; final linear head is replaced
to match the custom number of classes.
"""

import torch
import torch.nn as nn
import timm


SUPPORTED_MODELS = ["vit_b16", "resnet50", "efficientnet_b4"]


def build_model(model_name: str, num_classes: int,
                pretrained: bool = True, drop_rate: float = 0.2) -> nn.Module:
    """
    Factory for classification models.

    Args:
        model_name  : one of SUPPORTED_MODELS
        num_classes : number of output classes
        pretrained  : load ImageNet pretrained weights
        drop_rate   : dropout rate before classifier head

    Returns:
        nn.Module with modified head
    """
    if model_name == "vit_b16":
        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
    elif model_name == "resnet50":
        model = timm.create_model(
            "resnet50",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
    elif model_name == "efficientnet_b4":
        model = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Choose from {SUPPORTED_MODELS}")
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_backbone(model: nn.Module, model_name: str):
    """Freeze all layers except the classifier head (for initial warm-up)."""
    if model_name == "vit_b16":
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad_(False)
    elif model_name == "resnet50":
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad_(False)
    elif model_name == "efficientnet_b4":
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad_(False)


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad_(True)
