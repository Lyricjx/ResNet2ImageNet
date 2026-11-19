"""
Configure linear Top-1/Top-5 plots, following tools/plot_loss.py.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import plot_accuracy_multirun

if __name__ == "__main__":
    """多份日志对比"""
    log_dir = "/mnt/shareEEEx/xieyuankai/ResNet2ImageNet/runs/personal"
    log_list = [
        "C1_seed16/train.log",
        "C2_seed16/train.log",
        "C3_seed16/train.log",
        "C0_seed16/train.log",
    ]
    savename = "runs/plots/label_smoothing_accuracy_comparison"

    plot_accuracy_multirun(
        PROJECT_ROOT / log_dir, log_list, PROJECT_ROOT / savename,
        title="ResNet-18 Validation Accuracy",
        run_labels=["C1", "C2", "C3", "C0"],
        zoom_start=60,
    )
