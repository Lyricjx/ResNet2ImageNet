"""Draw the ResNet-18 fixed-filter downsampling ablation for a paper figure."""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle, Circle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.resnet18_blurpool import FILTERS


ROOT = Path(__file__).resolve().parents[1]
INK = "#262b32"
MUTED = "#59636f"
LINE = "#87919b"
FAINT = "#d9dfe4"
PANEL = "#f6f7f8"
ACCENT = "#b84f35"
ACCENT_LIGHT = "#fff1e9"


def box(ax, x, y, width, height, label, *, active=False, fontsize=16):
    ax.add_patch(Rectangle(
        (x, y), width, height, linewidth=1.25,
        edgecolor=ACCENT if active else LINE,
        facecolor=ACCENT_LIGHT if active else "white",
    ))
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center",
            fontsize=fontsize, color=INK, linespacing=1.2)


def stride_badge(ax, center_x, center_y, stride):
    width, height = 0.60, 0.23
    ax.add_patch(Rectangle((center_x - width / 2, center_y - height / 2),
                           width, height, linewidth=1.0,
                           edgecolor=ACCENT, facecolor="white"))
    ax.text(center_x, center_y, f"s={stride}", ha="center", va="center",
            fontsize=11.6, weight="semibold", color=ACCENT)


def strided_box(ax, x, y, width, height, label, stride, *, active=False, fontsize=14):
    box(ax, x, y, width, height, "", active=active)
    ax.text(x + width / 2, y + height * 0.69, label, ha="center", va="center",
            fontsize=fontsize, color=INK)
    stride_badge(ax, x + width / 2, y + height * 0.27, stride)


def arrow(ax, start, end, *, color=LINE, linewidth=1.35):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11,
        linewidth=linewidth, color=color, shrinkA=0, shrinkB=0,
    ))


def stage(ax, x, name, channels, antialiased=False):
    width = 2.20
    ax.add_patch(Rectangle((x, 0.49), width, 1.42, linewidth=1,
                           edgecolor=FAINT, facecolor=PANEL))
    ax.text(x + width / 2, 1.64, f"{name} · {channels}", ha="center",
            va="center", fontsize=15.2, color=INK, weight="semibold")
    box(ax, x + 0.18, 0.74, 0.82, 0.59, "B0",
        active=antialiased, fontsize=15.2)
    box(ax, x + 1.20, 0.74, 0.82, 0.59, "B1", fontsize=15.2)


def draw_backbone(ax):
    ax.set(xlim=(0, 15), ylim=(0, 2.55))
    ax.axis("off")
    ax.text(0.02, 2.26, "a", fontsize=20, weight="bold", color=INK)
    ax.text(0.46, 2.26, "ResNet-18 backbone",
            fontsize=18, weight="semibold", color=INK)

    box(ax, 0.08, 0.49, 1.12, 1.42, "Input\n224²", fontsize=16)
    box(ax, 1.56, 0.49, 1.51, 1.42, "")
    ax.text(2.315, 1.70, "7×7 Conv", ha="center", va="center",
            fontsize=14.1, color=INK)
    ax.text(2.315, 1.43, "BN + ReLU", ha="center", va="center",
            fontsize=14.1, color=INK)
    ax.text(2.315, 1.16, "MaxPool", ha="center", va="center",
            fontsize=14.1, color=INK)
    stride_badge(ax, 2.315, 0.90, 1)
    ax.add_patch(Rectangle((1.80, 0.55), 1.03, 0.20, linewidth=1.0,
                           edgecolor=ACCENT, facecolor=ACCENT_LIGHT))
    ax.text(2.315, 0.65, "+ filter", ha="center", va="center",
            fontsize=10.5, color=ACCENT, weight="semibold")
    stage(ax, 3.40, "layer1", 64)
    stage(ax, 5.90, "layer2", 128, antialiased=True)
    stage(ax, 8.40, "layer3", 256, antialiased=True)
    stage(ax, 10.90, "layer4", 512, antialiased=True)
    box(ax, 13.50, 0.49, 1.48, 1.42, "GAP\nFC-1000", fontsize=15)

    for left, right in ((1.20, 1.56), (3.07, 3.40), (5.60, 5.90),
                        (8.10, 8.40), (10.60, 10.90), (13.10, 13.50)):
        arrow(ax, (left, 1.20), (right, 1.20))

    ax.add_patch(Rectangle((6.03, 0.69), 0.92, 0.69, fill=False,
                           edgecolor=ACCENT, linewidth=1.4,
                           linestyle=(0, (4, 3))))


def draw_transition(ax):
    ax.set(xlim=(0, 15), ylim=(0, 3.6))
    ax.axis("off")
    ax.text(0.02, 3.29, "b", fontsize=20, weight="bold", color=INK)
    ax.text(0.46, 3.29, "Transition BasicBlock (layer2.0)",
            fontsize=17, weight="semibold", color=INK)

    main_y = 1.72
    h = 0.76
    mid = main_y + h / 2
    box(ax, 0.08, main_y, 1.15, h, "Input x", fontsize=15)
    strided_box(ax, 1.75, main_y, 1.37, h, "3×3 Conv", 1, fontsize=14.2)
    box(ax, 3.43, main_y, 1.43, h, "BN + ReLU", fontsize=14.2)
    strided_box(ax, 5.19, main_y, 1.55, h, "Fixed filter", 2,
                active=True, fontsize=14)
    box(ax, 7.06, main_y, 1.27, h, "3×3 Conv", fontsize=14)
    box(ax, 8.64, main_y, 0.93, h, "BN", fontsize=15)
    ax.add_patch(Circle((10.18, mid), 0.32, facecolor="white",
                        edgecolor=INK, linewidth=1.3))
    ax.text(10.18, mid, "+", ha="center", va="center", fontsize=22, color=INK)
    box(ax, 10.85, main_y, 1.20, h, "ReLU", fontsize=15)
    box(ax, 12.39, main_y, 1.38, h, "Output", fontsize=15)

    for left, right in ((1.23, 1.75), (3.12, 3.43), (4.86, 5.19),
                        (6.74, 7.06), (8.33, 8.64), (9.57, 9.86),
                        (10.50, 10.85), (12.05, 12.39)):
        arrow(ax, (left, mid), (right, mid),
              color=ACCENT if left == 6.74 else LINE)

    short_y = 0.35
    short_mid = short_y + h / 2
    strided_box(ax, 2.27, short_y, 1.57, h, "1×1 Conv", 1, fontsize=14)
    box(ax, 4.18, short_y, 1.02, h, "BN", fontsize=15)
    strided_box(ax, 5.55, short_y, 1.55, h, "Fixed filter", 2,
                active=True, fontsize=14)
    ax.plot([1.48, 1.48, 2.10], [mid, short_mid, short_mid],
            color=LINE, linewidth=1.35)
    arrow(ax, (2.10, short_mid), (2.27, short_mid))
    arrow(ax, (3.84, short_mid), (4.18, short_mid))
    arrow(ax, (5.20, short_mid), (5.55, short_mid))
    ax.plot([7.10, 10.18], [short_mid, short_mid], color=LINE, linewidth=1.35)
    arrow(ax, (10.18, short_mid), (10.18, mid - 0.32))


def tint(value, max_value):
    white = to_rgb("#ffffff")
    accent = to_rgb("#e7a486")
    fraction = 0.18 + 0.72 * value / max_value
    return tuple((1 - fraction) * a + fraction * b for a, b in zip(white, accent))


def kernel(ax, x, y, name, label, denominator, step):
    coefficients = FILTERS[name]
    values = [[a * b for b in coefficients] for a in coefficients]
    side = len(coefficients)
    size = step * side
    maximum = max(max(row) for row in values)
    ax.text(x, y + size + 0.24, label,
            fontsize=16, weight="semibold", color=INK)
    for row in range(side):
        for column in range(side):
            value = values[row][column]
            cell_x = x + column * step
            cell_y = y + (side - row - 1) * step
            ax.add_patch(Rectangle((cell_x, cell_y), step, step,
                                   facecolor=tint(value, maximum),
                                   edgecolor="white", linewidth=1.15))
            ax.text(cell_x + step / 2, cell_y + step / 2, str(value),
                    ha="center", va="center", fontsize=13.5,
                    color=INK)
    ax.add_patch(Rectangle((x, y), size, size, fill=False,
                           edgecolor=LINE, linewidth=0.9))
    ax.text(x + size + 0.23, y + size / 2, f"÷ {denominator}\nstride 2",
            va="center", fontsize=14, color=MUTED, linespacing=1.35)


def draw_kernels(ax):
    ax.set(xlim=(0, 4), ylim=(0, 7))
    ax.axis("off")
    kernel(ax, 0.68, 5.0, "binomial3", "E1  3×3 weighted", 16, 0.49)
    kernel(ax, 0.68, 2.8, "box3", "E2  3×3 mean", 9, 0.49)
    kernel(ax, 0.68, 0.6, "binomial5", "E3  5×5 weighted", 256, 0.295)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "resnet18_blurpool_paper.png")
    parser.add_argument("--kernels-output", type=Path,
                        default=ROOT / "resnet18_blurpool_kernels.png")
    args = parser.parse_args()
    if args.output.resolve() == args.kernels_output.resolve():
        parser.error("--output and --kernels-output must be different files")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.kernels_output.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})
    fig = plt.figure(figsize=(14.32, 5.85), facecolor="white")
    ax_a = fig.add_axes((0.025, 0.585, 0.95, 0.390))
    ax_b = fig.add_axes((0.025, 0.065, 0.95, 0.480))
    draw_backbone(ax_a)
    draw_transition(ax_b)

    fig.add_artist(ConnectionPatch(
        xyA=(6.49, 0.69), coordsA=ax_a.transData,
        xyB=(6.49, 2.95), coordsB=ax_b.transData,
        arrowstyle="-|>", mutation_scale=12, linewidth=1.4,
        linestyle=(0, (4, 3)), color=ACCENT,
    ))
    fig.add_artist(plt.Line2D([0.025, 0.975], [0.565, 0.565],
                              transform=fig.transFigure, color=FAINT, linewidth=0.9))
    fig.savefig(args.output, dpi=300, facecolor="white")
    plt.close(fig)

    fig = plt.figure(figsize=(5.0, 8.0), facecolor="white")
    ax_kernels = fig.add_axes((0.07, 0.035, 0.86, 0.94))
    draw_kernels(ax_kernels)
    fig.savefig(args.kernels_output, dpi=300, facecolor="white")
    plt.close(fig)

    print(args.output)
    print(args.kernels_output)


if __name__ == "__main__":
    main()
