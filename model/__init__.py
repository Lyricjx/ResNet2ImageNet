from .resnet18 import build_resnet18

def build_model(name="resnet18"):
    """Build the configured model; register future architectures here."""
    builders = {"resnet18": build_resnet18}
    if name not in builders:
        raise ValueError(f"Unsupported model: {name}; available: {', '.join(builders)}")
    return builders[name]()

__all__ = ["build_model", "build_resnet18"]
