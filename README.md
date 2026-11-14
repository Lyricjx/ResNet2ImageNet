# ResNet2ImageNet
A controlled ImageNet-1K classification study with ResNet-18 as the baseline. Some independent improvements are evaluated through comparisons and ablations, using a reproducible 10% training subset and the full validation set for Top-1 and Top-5 accuracy.

## 数据准备

使用 ImageNet-1K（ILSVRC 2012）：每类固定种子 `16` 抽取约 10% 作为训练集，共 **128,124 张、1,000 类**；在完整的 **50,000 张验证图像**上报告 Top-1 / Top-5。官方 test 集无公开标签，本项目不使用。

仓库只保存图像清单和标签，不保存原图。需要准备相同的 ImageNet 原图后，按 [data_pre/README.md](data_pre/README.md) 重建相同子集。

## 模型

`model.build_resnet18()` 调用 [Torchvision 官方 ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18)（[当前实验使用的 v0.27.0 源码](https://github.com/pytorch/vision/blob/v0.27.0/torchvision/models/resnet.py)），随机初始化，输出 1,000 类。训练脚本需统一设置随机种子和训练配置。

在仓库根目录运行 Baseline（示例使用 GPU 0）：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_baseline.py \
  --data-root data/imagenet_10pct \
  --output-dir runs/resnet18_baseline_seed16
```

按机器实际情况修改 GPU 编号；`--data-root` 指向按上述步骤重建的子集。每次实验使用新的 `--output-dir`，避免覆盖已有结果。默认训练 90 轮，每轮在完整验证集上计算 Top-1 / Top-5。正常完成后，输出目录包含 `best.pt`、`last.pt`、`config.json`、`metrics.jsonl` 和 `train.log`。

结果表使用验证集 Top-1 最佳的 `best.pt`：`best_top1` 和 `best_top5` 对应同一轮、同一个模型；`best_top5` 不是各轮 Top-5 的独立最大值。
