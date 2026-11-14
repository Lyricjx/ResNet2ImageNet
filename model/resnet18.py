"""ImageNet-1K ResNet-18 baseline."""

from torch import nn
from torchvision.models import resnet18

def build_resnet18(num_classes: int = 1000) -> nn.Module:
    """Build an untrained torchvision ResNet-18 classifier."""
    return resnet18(weights=None, num_classes=num_classes)
