# ResNet2ImageNet
A controlled ImageNet-1K classification study with ResNet-18 as the baseline. Four independent improvements are evaluated through comparisons and ablations, using a reproducible 10% training subset and the full validation set for Top-1 and Top-5 accuracy.

## 数据准备

使用 ImageNet-1K（ILSVRC 2012）：每类固定种子 `16` 抽取约 10% 作为训练集，共 **128,124 张、1,000 类**；在完整的 **50,000 张验证图像**上报告 Top-1 / Top-5。官方 test 集无公开标签，本项目不使用。

仓库只保存图像清单和标签，不保存原图。需要准备相同的 ImageNet 原图后，按 [data_pre/README.md](data_pre/README.md) 重建相同子集。
