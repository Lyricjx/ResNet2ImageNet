"""Train the ResNet-18 baseline on the fixed ImageNet-1K subset."""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import __version__ as torchvision_version
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import build_model
from utils.train_config import parse_training_args, training_signature

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def make_transforms(augmentation):
    name = augmentation["name"]
    if name not in ("basic", "randaugment", "random_erasing"):
        raise ValueError(f"Unsupported augmentation: {augmentation['name']}; implement it in make_transforms")
    train_steps = [
        transforms.RandomResizedCrop(224, scale=tuple(augmentation["crop_scale"]),
                                     ratio=tuple(augmentation["crop_ratio"])),
        transforms.RandomHorizontalFlip(p=augmentation["horizontal_flip"]),
    ]
    if name == "randaugment":
        train_steps.append(transforms.RandAugment(
            num_ops=augmentation["num_ops"], magnitude=augmentation["magnitude"],
        ))
    train_steps.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    if name == "random_erasing":
        train_steps.append(transforms.RandomErasing(
            p=augmentation["p"], scale=tuple(augmentation["scale"]),
            ratio=tuple(augmentation["ratio"]), value=augmentation["value"],
        ))
    train_transform = transforms.Compose(train_steps)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_transform, val_transform

def make_loaders(data_root, batch_size, workers, augmentation):
    train_transform, val_transform = make_transforms(augmentation)
    train_set = datasets.ImageFolder(data_root / "train", transform=train_transform)
    val_set = datasets.ImageFolder(data_root / "val", transform=val_transform)
    if len(train_set.classes) != 1000 or train_set.class_to_idx != val_set.class_to_idx:
        raise ValueError("Expected 1,000 classes with identical train/val class indices")
    if len(train_set) != 128124 or len(val_set) != 50000:
        raise ValueError("Dataset size differs from the committed subset manifests")
    generator = torch.Generator()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=True, generator=generator,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True,
    )
    return train_loader, val_loader, generator

def make_optimizer(parameters, optimizer_config):
    if optimizer_config["name"] != "sgd":
        raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}; implement it in make_optimizer")
    return torch.optim.SGD(parameters, **{key: value for key, value in optimizer_config.items() if key != "name"})

def make_scheduler(optimizer, scheduler_config):
    if scheduler_config["name"] == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=scheduler_config["milestones"], gamma=scheduler_config["gamma"],
        )
    if scheduler_config["name"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_config["t_max"], eta_min=scheduler_config["eta_min"],
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_config['name']}; implement it in make_scheduler")

def make_criteria(loss_config):
    if loss_config["name"] != "cross_entropy":
        raise ValueError(f"Unsupported loss: {loss_config['name']}; implement it in make_criteria")
    return (nn.CrossEntropyLoss(label_smoothing=loss_config.get("label_smoothing", 0.0)),
            nn.CrossEntropyLoss())

def check_resume_config(checkpoint, settings):
    saved_config = checkpoint.get("training_config")
    if saved_config is not None:
        if saved_config != training_signature(settings):
            raise ValueError("Resume training settings differ from the checkpoint; "
                             "use its original config, or start a new experiment without --resume")
        return
    scheduler_state = checkpoint["scheduler"]
    scheduler_config = settings["scheduler"]
    if scheduler_config["name"] != "multistep":
        raise ValueError("Legacy checkpoints require the original multistep scheduler")
    expected = {"milestones": {epoch: 1 for epoch in scheduler_config["milestones"]},
                "gamma": scheduler_config["gamma"]}
    if any(scheduler_state.get(key) != value for key, value in expected.items()):
        raise ValueError("Scheduler configuration does not match the legacy checkpoint")
    for group in checkpoint["optimizer"]["param_groups"]:
        for key in ("momentum", "weight_decay"):
            if group.get(key) != settings["optimizer"][key]:
                raise ValueError(f"Optimizer {key} does not match the legacy checkpoint")
        if group.get("initial_lr") != settings["optimizer"]["lr"]:
            raise ValueError("Initial LR does not match the legacy checkpoint")
    print("WARNING: legacy checkpoint has no full training config; "
          "verify the original augmentation, loss, seed and epoch settings before resuming.", flush=True)

def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    seen = 0
    started = time.perf_counter()
    for step, (images, targets) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        seen += batch_size
        if step % 100 == 0 or step == len(loader):
            elapsed = time.perf_counter() - started
            print(
                f"epoch={epoch} train_step={step}/{len(loader)} "
                f"loss={total_loss / seen:.4f} images_per_sec={seen / elapsed:.1f}",
                flush=True,
            )
    return total_loss / seen, time.perf_counter() - started

@torch.inference_mode()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    top1 = top5 = seen = 0
    started = time.perf_counter()
    for images, targets in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss = criterion(logits, targets)
        pred = logits.topk(5, dim=1).indices
        correct = pred.eq(targets[:, None])
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        top1 += correct[:, 0].sum().item()
        top5 += correct.any(dim=1).sum().item()
        seen += batch_size
    return {
        "val_loss": total_loss / seen,
        "top1": 100.0 * top1 / seen,
        "top5": 100.0 * top5 / seen,
        "val_seconds": time.perf_counter() - started,
    }

def save_checkpoint(path, state):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)

def restore_best_top5(checkpoint, output_dir):
    if "best_top5" in checkpoint:
        return checkpoint["best_top5"]
    metrics_path = output_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Cannot restore best_top5 without {metrics_path}")
    with metrics_path.open() as file:
        previous = [json.loads(line) for line in file if line.strip()]
    previous = [row for row in previous if row["epoch"] <= checkpoint["epoch"]]
    best = max(previous, key=lambda row: row["top1"], default=None)
    if best is None or best["top1"] != checkpoint["best_top1"]:
        raise ValueError("Checkpoint best_top1 does not match metrics.jsonl")
    return best["top5"]

def main():
    run_started = time.perf_counter()
    args = parse_training_args()
    if args.print_config:
        print(json.dumps(args.settings, indent=2), flush=True)
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU, e.g. CUDA_VISIBLE_DEVICES=3")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This baseline requires a GPU with bfloat16 support")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and args.resume is None:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_training(args, output_dir, run_started)

def run_training(args, output_dir, run_started):
    settings = args.settings
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    train_loader, val_loader, generator = make_loaders(
        args.data_root.resolve(), args.batch_size, args.workers, settings["augmentation"]
    )
    model = build_model(**settings["model"]).to(device, memory_format=torch.channels_last)
    train_criterion, val_criterion = make_criteria(settings["loss"])
    optimizer = make_optimizer(model.parameters(), settings["optimizer"])
    scheduler = make_scheduler(optimizer, settings["scheduler"])

    start_epoch = 1
    best_top1 = -1.0
    best_top5 = -1.0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        check_resume_config(checkpoint, settings)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_top1 = checkpoint["best_top1"]
        best_top5 = restore_best_top5(checkpoint, output_dir)

    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
        "model": settings["model"]["name"],
        "optimizer": f"SGD(momentum={settings['optimizer']['momentum']}, "
                     f"weight_decay={settings['optimizer']['weight_decay']})",
        "lr_milestones": settings["scheduler"].get("milestones"),
        "scheduler": type(scheduler).__name__,
        "precision": "bfloat16 autocast",
        "gpu": torch.cuda.get_device_name(0),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2), flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        generator.manual_seed(args.seed + epoch)
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_seconds = train_epoch(model, train_loader, optimizer, train_criterion, device, epoch)
        metrics = validate(model, val_loader, val_criterion, device)
        scheduler.step()
        is_best = metrics["top1"] > best_top1
        if is_best:
            best_top1 = metrics["top1"]
            best_top5 = metrics["top5"]
        record = {
            "epoch": epoch, "lr": lr, "train_loss": train_loss,
            "train_seconds": train_seconds, **metrics,
            "best_top1": best_top1, "best_top5": best_top5,
        }
        with (output_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(record) + "\n")
        state = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "best_top1": best_top1, "best_top5": best_top5,
            "training_config": training_signature(settings),
        }
        save_checkpoint(output_dir / "last.pt", state)
        if is_best:
            save_checkpoint(output_dir / "best.pt", state)
        print(
            f"epoch={epoch} lr={lr:.5g} train_loss={train_loss:.4f} "
            f"val_loss={metrics['val_loss']:.4f} top1={metrics['top1']:.3f} "
            f"top5={metrics['top5']:.3f} best_top1={best_top1:.3f} "
            f"best_top5={best_top5:.3f} "
            f"train_sec={train_seconds:.1f} val_sec={metrics['val_seconds']:.1f}",
            flush=True,
        )

    total_seconds = time.perf_counter() - run_started
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(
        f"total_sec={total_seconds:.1f} total_time={hours:02d}:{minutes:02d}:{seconds:02d}",
        flush=True,
    )

if __name__ == "__main__":
    main()
