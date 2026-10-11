from torch import nn
from torchvision import models

FEATURE_DIMS = {"resnet18": 512, "resnet50": 2048, "mobilenet_v3_small": 576}


def build_backbone(name: str) -> nn.Module:
    """ImageNet-pretrained backbone that outputs a feature vector."""
    if name == "resnet18":
        m = models.resnet18(weights="IMAGENET1K_V1")
        m.fc = nn.Identity()
    elif name == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V1")
        m.fc = nn.Identity()
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        m.classifier = nn.Identity()
    else:
        raise ValueError(f"unknown backbone: {name}")
    return m.eval()
