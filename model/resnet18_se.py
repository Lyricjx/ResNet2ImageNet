"""ResNet-18 with optional squeeze-and-excitation in selected stages."""

from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock

from .resnet18 import build_resnet18


SE_STAGES = ("layer1", "layer2", "layer3", "layer4")


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int):
        super().__init__()
        hidden_channels = channels // reduction
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        weights = self.sigmoid(self.fc2(self.relu(self.fc1(self.pool(x)))))
        return x * weights


class SEBasicBlock(nn.Module):
    """Keep BasicBlock layers and insert SE before the residual addition."""

    def __init__(self, block: BasicBlock, reduction: int):
        super().__init__()
        self.conv1 = block.conv1
        self.bn1 = block.bn1
        self.relu = block.relu
        self.conv2 = block.conv2
        self.bn2 = block.bn2
        self.se = SqueezeExcitation(block.bn2.num_features, reduction)
        self.downsample = block.downsample
        self.stride = block.stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


def build_resnet18_se(reduction: int, stages: list[str]) -> nn.Module:
    """Build an untrained ResNet-18 and add SE to every block in each stage."""
    if type(reduction) is not int or not 1 <= reduction <= 64:
        raise ValueError("SE reduction must be an integer in [1, 64]")
    if (not isinstance(stages, (list, tuple)) or not stages
            or any(type(stage) is not str for stage in stages)
            or len(stages) != len(set(stages)) or any(stage not in SE_STAGES for stage in stages)):
        raise ValueError(f"SE stages must be a nonempty subset of {SE_STAGES}")
    model = build_resnet18()
    for stage in stages:
        layer = getattr(model, stage)
        for index, block in enumerate(layer):
            layer[index] = SEBasicBlock(block, reduction)
    return model
