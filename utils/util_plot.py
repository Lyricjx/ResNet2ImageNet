"""Plot epoch-level loss and validation accuracy on linear axes."""
import math
import re
from pathlib import Path

EPOCH_PATTERN = re.compile(r"^\s*epoch=(\d+)\s+")
FIELD_PATTERN = re.compile(r"\b([a-z][a-z0-9_]*)=(\S+)")

def _extract_epoch_metrics(filename, metric_names, value_max=None):
    """
    Return two metric series and sorted, one-based epochs.
    Ignore batch progress, tracebacks and incomplete epoch summaries. For logs
    appended after resuming, the last complete entry for each epoch wins.
    """
    filename = Path(filename)
    metrics_by_epoch = {}
    with filename.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            match = EPOCH_PATTERN.match(line)
            if match is None:
                continue
            fields = dict(FIELD_PATTERN.findall(line))
            if not set(metric_names).issubset(fields):
                continue
            try:
                epoch = int(match.group(1))
                values = tuple(float(fields[name]) for name in metric_names)
            except ValueError as error:
                raise ValueError(f"{filename}:{line_number}: invalid metric value") from error
            if epoch < 1 or any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError(f"{filename}:{line_number}: expected positive epoch and finite nonnegative metrics")
            if value_max is not None and any(value > value_max for value in values):
                raise ValueError(f"{filename}:{line_number}: metrics must not exceed {value_max}")
            metrics_by_epoch[epoch] = values

    if not metrics_by_epoch:
        names = " and ".join(metric_names)
        raise ValueError(f"No complete epoch summaries with {names} found in {filename}")
    epochs = sorted(metrics_by_epoch)
    first = [metrics_by_epoch[epoch][0] for epoch in epochs]
    second = [metrics_by_epoch[epoch][1] for epoch in epochs]
    return first, second, epochs

def extract_loss_from_log(filename):
    """Return train losses, validation losses and one-based epochs."""
    return _extract_epoch_metrics(filename, ("train_loss", "val_loss"))

def extract_accuracy_from_log(filename):
    """Return per-epoch validation Top-1/Top-5 percentages, not running bests."""
    return _extract_epoch_metrics(filename, ("top1", "top5"), value_max=100)

def _plot_metric_multirun(log_dir, log_list, savename, extractor, curve_labels,
                          ylabel, figsize=(10, 6), ncol=2, dpi=300, title="", ylim=None,
                          mark_best=False):
    """
    Use one color per run, with solid first and dashed second metrics.

    log_dir contains the files in log_list and savename is the output
    path without an extension (PNG by default). Legend names use each log's
    filename without its extension. Both axes use a linear scale.

    Epochs are read directly from ResNet2ImageNet logs and start at one.
    Values are plotted without smoothing. PyTorch is not required.
    """
    log_paths = [Path(log_dir) / name for name in log_list]
    if not log_paths:
        raise ValueError("Provide at least one log file")
    if ncol < 1:
        raise ValueError("The number of legend columns must be positive")
    if dpi < 1:
        raise ValueError("dpi must be positive")

    runs = [(path, extractor(path)) for path in log_paths]

    output_path = Path(savename)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".png")
    if output_path.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("Output format must be PNG, PDF or SVG")
    if output_path.resolve() in {path.resolve() for path in log_paths}:
        raise ValueError("The output must not overwrite an input log")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 14}):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        try:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            for index, (path, (first, second, epochs)) in enumerate(runs):
                name = path.stem
                color = colors[index % len(colors)]
                ax.plot(epochs, first, label=f"{name} {curve_labels[0]}", color=color, linestyle="-", linewidth=0.8)
                ax.plot(epochs, second, label=f"{name} {curve_labels[1]}", color=color, linestyle="--", linewidth=0.8)
                if mark_best:
                    best = max(range(len(first)), key=first.__getitem__)
                    ax.plot(
                        [epochs[best], epochs[best]], [first[best], second[best]],
                        color="red", marker="o", markersize=3, markeredgewidth=0,
                        linestyle="None", zorder=5,
                    )
                    for value in (first[best], second[best]):
                        ax.annotate(
                            f"Epoch {epochs[best]}", (epochs[best], value),
                            xytext=(0, 7), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9, color="red",
                        )
            ax.set_xlabel("Epochs")
            ax.set_ylabel(ylabel)
            ax.set_xscale("linear")
            ax.set_yscale("linear")
            if ylim is not None:
                ax.set_ylim(*ylim)
            if title:
                ax.set_title(title)
            ax.grid(True, which="major", alpha=0.25)
            ax.legend(ncol=ncol, fontsize=12, loc="upper right")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        finally:
            plt.close(fig)
    return output_path

def plot_loss_multirun(log_dir, log_list, savename, figsize=(10, 6),
                       ncol=2, dpi=300, title=""):
    """Plot train/validation loss for the configured log files."""
    return _plot_metric_multirun(
        log_dir, log_list, savename, extract_loss_from_log, ("train", "valid"),
        "Loss", figsize=figsize, ncol=ncol, dpi=dpi, title=title,
    )

def plot_accuracy_multirun(log_dir, log_list, savename, figsize=(10, 6),
                           ncol=2, dpi=300, title=""):
    """Plot Top-1/Top-5 and mark both at the first best Top-1 epoch in red."""
    return _plot_metric_multirun(
        log_dir, log_list, savename, extract_accuracy_from_log, ("Top-1", "Top-5"),
        "Validation accuracy (%)", figsize=figsize, ncol=ncol, dpi=dpi,
        title=title, ylim=(0, 100), mark_best=True,
    )
