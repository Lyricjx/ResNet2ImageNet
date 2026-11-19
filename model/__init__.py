from .resnet18 import build_resnet18
from .resnet18_blurpool import build_resnet18_blurpool

def build_model(name="resnet18", **kwargs):
    """Build the configured model; register future architectures here."""
    builders = {"resnet18": build_resnet18, "resnet18_blurpool": build_resnet18_blurpool}
    if name not in builders:
        raise ValueError(f"Unsupported model: {name}; available: {', '.join(builders)}")
    return builders[name](**kwargs)

__all__ = ["build_model", "build_resnet18", "build_resnet18_blurpool"]
