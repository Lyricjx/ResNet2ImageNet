"""Read and validate the YAML training configuration."""
import argparse
import copy
import json
import math
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"
FIELDS = {
    "run": {"data_root", "output_dir", "epochs", "batch_size", "workers", "seed", "resume"},
    "optimizer": {"name", "lr", "momentum", "weight_decay"},
    "scheduler": {"name", "milestones", "gamma"},
    "augmentation": {"name", "crop_scale", "crop_ratio", "horizontal_flip"},
    "loss": {"name"},
    "model": {"name"},
}
STRATEGIES = {
    "optimizer": ("sgd",), "scheduler": ("multistep",),
    "augmentation": ("basic",), "loss": ("cross_entropy",), "model": ("resnet18",),
}

def validate_config(config):
    if not isinstance(config, dict) or set(config) != set(FIELDS):
        raise ValueError(f"Config sections must be: {', '.join(FIELDS)}")
    for section, fields in FIELDS.items():
        values = config[section]
        if not isinstance(values, dict):
            raise ValueError(f"[{section}] must be a table")
        missing, unknown = fields - values.keys(), values.keys() - fields
        if missing or unknown:
            raise ValueError(f"[{section}]: missing={sorted(missing)}, unknown={sorted(unknown)}")
    for section, supported in STRATEGIES.items():
        if config[section]["name"] not in supported:
            raise ValueError(f"Unsupported {section}.name: {config[section]['name']}; "
                             f"implemented: {', '.join(supported)}")

    def number(section, key, minimum, maximum=math.inf, integer=False):
        value = config[section][key]
        valid_type = type(value) is int if integer else type(value) in (int, float)
        if not valid_type or not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{section}.{key} must be {'an integer' if integer else 'a number'} "
                             f"in [{minimum}, {maximum}]")

    def pair(key, minimum, maximum=math.inf):
        value = config["augmentation"][key]
        if (not isinstance(value, list) or len(value) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
                or not minimum <= value[0] <= value[1] <= maximum or value[0] <= 0):
            raise ValueError(f"augmentation.{key} must be an ordered positive pair in [{minimum}, {maximum}]")

    for key in ("epochs", "batch_size"):
        number("run", key, 1, integer=True)
    number("run", "workers", 0, integer=True)
    number("run", "seed", 0, 2**32 - 1, integer=True)
    for key in ("data_root", "output_dir", "resume"):
        value = config["run"][key]
        if not isinstance(value, str) or (key != "resume" and not value.strip()):
            raise ValueError(f"run.{key} must be a {'nonempty ' if key != 'resume' else ''}path string")
    number("optimizer", "lr", 0)
    if config["optimizer"]["lr"] == 0:
        raise ValueError("optimizer.lr must be positive")
    number("optimizer", "momentum", 0, 1)
    number("optimizer", "weight_decay", 0)
    scheduler = config["scheduler"]
    number("scheduler", "gamma", 0, 1)
    if scheduler["gamma"] == 0:
        raise ValueError("scheduler.gamma must be positive")
    milestones = scheduler["milestones"]
    if (not isinstance(milestones, list)
            or any(type(v) is not int or v < 1 for v in milestones)
            or milestones != sorted(set(milestones))):
        raise ValueError("scheduler.milestones must be a strictly increasing list of positive epochs")
    pair("crop_scale", 0, 1)
    pair("crop_ratio", 0)
    number("augmentation", "horizontal_flip", 0, 1)

def parse_training_args(argv=None):
    parser = argparse.ArgumentParser(description="Train ResNet-18 using a YAML configuration file.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    for name in ("epochs", "batch-size", "workers", "seed"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--print-config", action="store_true", help="Print effective settings and exit")
    args = parser.parse_args(argv)
    try:
        with args.config.open(encoding="utf-8-sig") as file:
            config = yaml.safe_load(file)
        validate_config(config)
        for name in ("data_root", "output_dir", "epochs", "batch_size", "workers", "seed", "resume"):
            value = getattr(args, name)
            if value is not None:
                config["run"][name] = str(value.resolve()) if isinstance(value, Path) else value
        if args.lr is not None:
            config["optimizer"]["lr"] = args.lr
        validate_config(config)
        for name in ("data_root", "output_dir", "resume"):
            value = config["run"][name]
            if value:
                config["run"][name] = str((PROJECT_ROOT / value).resolve())
        scheduler = config["scheduler"]
        epochs = config["run"]["epochs"]
        if not scheduler["milestones"]:
            scheduler["milestones"] = sorted(set(max(1, round(epochs * p)) for p in (1 / 3, 2 / 3, 8 / 9)))
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    args.config = args.config.resolve()
    args.settings = config
    for name, value in config["run"].items():
        setattr(args, name, Path(value) if name in ("data_root", "output_dir", "resume") and value else value)
    args.resume = args.resume or None
    args.lr = config["optimizer"]["lr"]
    return args

def training_signature(config):
    """Keep settings that must not change during checkpoint continuation."""
    signature = copy.deepcopy(config)
    for name in ("data_root", "output_dir", "resume", "workers"):
        signature["run"].pop(name)
    return signature

if __name__ == "__main__":
    print(json.dumps(parse_training_args().settings, indent=2))
