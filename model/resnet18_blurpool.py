"""ResNet-18 with fixed low-pass filters at selected downsampling sites."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


FILTERS = {
    "binomial3": (1, 2, 1),
    "box3": (1, 1, 1),
    "binomial5": (1, 4, 6, 4, 1),
}


class FixedBlurPool2d(nn.Module):
    """Apply the same normalized, non-trainable filter to each channel."""

    def __init__(self, channels: int, filter_name: str):
        super().__init__()
        coefficients = torch.tensor(FILTERS[filter_name], dtype=torch.float32)
        kernel = torch.outer(coefficients, coefficients)
        kernel /= kernel.sum()
        self.register_buffer("kernel", kernel[None, None].repeat(channels, 1, 1, 1))
        self.channels = channels
        self.padding = len(coefficients) // 2

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=2, padding=self.padding, groups=self.channels)


class AntiAliasedBasicBlock(nn.Module):
    """Downsample both residual paths after their original convolutions."""

    def __init__(self, block, filter_name: str):
        super().__init__()
        block.conv1.stride = (1, 1)
        block.downsample[0].stride = (1, 1)
        self.block = block
        self.main_blur = FixedBlurPool2d(block.conv1.out_channels, filter_name)
        self.shortcut_blur = FixedBlurPool2d(block.downsample[0].out_channels, filter_name)

    def forward(self, x):
        identity = self.block.downsample(x)
        identity = self.shortcut_blur(identity)

        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)
        out = self.main_blur(out)
        out = self.block.conv2(out)
        out = self.block.bn2(out)
        out += identity
        return self.block.relu(out)


def build_resnet18_blurpool(filter: str, num_classes: int = 1000) -> nn.Module:
    """Filter after stem MaxPool and at both paths of layer2-4 transitions."""
    if filter not in FILTERS:
        raise ValueError(f"Unsupported downsampling filter: {filter}")
    model = resnet18(weights=None, num_classes=num_classes)
    model.maxpool.stride = 1
    model.maxpool = nn.Sequential(model.maxpool, FixedBlurPool2d(64, filter))
    for stage in (model.layer2, model.layer3, model.layer4):
        stage[0] = AntiAliasedBasicBlock(stage[0], filter)
    return model
