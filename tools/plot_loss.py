"""
Configure log files and output paths for linear loss plots.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import plot_loss_multirun

if __name__ == "__main__":
    """多份日志对比"""
    log_dir = "/mnt/shareEEEx/xieyuankai/ResNet2ImageNet/runs/personal"
    log_list = [
        "C1_seed16/train.log",
        "C2_seed16/train.log",
        "C3_seed16/train.log",
        "C0_seed16/train.log",
    ]
    savename = "runs/plots/label_smoothing_loss_comparison"

    plot_loss_multirun(
        PROJECT_ROOT / log_dir, log_list, PROJECT_ROOT / savename,
        title="ResNet-18 Loss",
        run_labels=["C1", "C2", "C3", "C0"],
    )
