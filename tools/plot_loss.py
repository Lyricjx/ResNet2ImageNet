"""
Configure log files and output paths for linear loss plots.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import plot_loss_multirun

if __name__ == "__main__":
    log_dir = "."  # 路径相对仓库根目录，也可填写本机绝对路径

    """单份日志"""
    log_list = ["baseline.log"]
    savename = "runs/plots/baseline_loss"

    """多份日志对比"""
    # log_list = ["baseline.log", "improved.log"]
    # savename = "runs/plots/loss_comparison"

    plot_loss_multirun(
        PROJECT_ROOT / log_dir, log_list, PROJECT_ROOT / savename,
        title="ResNet-18 Loss",
    )
