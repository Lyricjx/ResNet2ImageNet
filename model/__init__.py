from .resnet18 import build_resnet18
from .resnet18_se import build_resnet18_se

def build_model(name="resnet18", **kwargs):
    """Build the configured model; register future architectures here."""
    builders = {"resnet18": build_resnet18, "resnet18_se": build_resnet18_se}
    if name not in builders:
        raise ValueError(f"Unsupported model: {name}; available: {', '.join(builders)}")
    return builders[name](**kwargs)

__all__ = ["build_model", "build_resnet18", "build_resnet18_se"]
