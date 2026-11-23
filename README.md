# ResNet2ImageNet
A controlled ImageNet-1K classification study with ResNet-18 as the baseline. Some independent improvements are evaluated through comparisons and ablations, using a reproducible 10% training subset and the full validation set for Top-1 and Top-5 accuracy.

## 代码来源与改动范围

- **模型基线来源：**`model/resnet18.py` 调用 [Torchvision ResNet-18 v0.27.0](https://github.com/pytorch/vision/blob/v0.27.0/torchvision/models/resnet.py)，使用 `weights=None` 从头训练 1,000 类分类器；不加载预训练权重。仓库最初的 ResNet-18 基线训练实现对应 commit `52c0f96`。
- **本仓库的改动：**`tools/train.py` 和 `utils/train_config.py` 提供统一训练入口、配置校验、学习率策略、训练时增强和标签平滑；`configs/` 中的 A/B/C 配置只改变相应消融变量；`tools/plot_loss.py` 与 `tools/plot_accuracy.py` 绘制实验曲线。当前 `main` 的 `model/resnet18.py` 仍使用原版 Torchvision ResNet-18。SE 与固定滤波下采样代码分别在 `ResNet2ImageNet2`、`ResNet2ImageNet3` 分支，**不属于当前 `main` 的训练入口**。
- **关键代码/配置提交：**训练策略扩展 `5682c39`；A/B/C 配置分别为 `7d34575`、`6af0e78`、`52ea667`；SE 和滤波分支的实现分别为 `84faacb`、`30289ce`。

## 环境配置

基线环境：Linux、NVIDIA RTX A6000、PyTorch 2.12.0+cu130、Torchvision 0.27.0+cu130。训练需要一张支持 BF16 的 CUDA GPU。

1. 创建环境：

```bash
conda create -n yourenvname python=3.12
conda activate yourenvname
```

2. 安装依赖：

```bash
python -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

   其他 CUDA 平台请参考 [PyTorch 官方安装命令](https://pytorch.org/get-started/previous-versions/)。

3. 检查 GPU（将 `0` 改为要使用的编号）：

```bash
CUDA_VISIBLE_DEVICES=0 python -c 'import torch; print(torch.cuda.get_device_name(0), "BF16:", torch.cuda.is_bf16_supported())'
```

输出应显示 `BF16: True`。训练命令也需设置 `CUDA_VISIBLE_DEVICES`。

## 数据准备

使用 ImageNet-1K（ILSVRC 2012）：每类固定种子 `16` 抽取约 10% 作为训练集，共 **128,124 张、1,000 类**；在完整的 **50,000 张验证图像**上报告 Top-1 / Top-5。官方 test 集无公开标签，本项目不使用。

仓库只保存图像清单和标签，不保存原图。需要准备相同的 ImageNet 原图后，按 [data_pre/README.md](data_pre/README.md) 重建相同子集。

假设原图目录为 `/path/to/imagenet`，包含 `train/<wnid>/*.JPEG` 和平铺的 `val/ILSVRC2012_val_*.JPEG`，在仓库根目录执行：

```bash
python data_pre/rebuild_from_manifest.py --source-root /path/to/imagenet --verify-only
python data_pre/rebuild_from_manifest.py --source-root /path/to/imagenet --output-root data/imagenet_10pct
```

第一条只检查清单与源图，第二条创建软链接，不复制原图；`data/` 不提交到 Git。

## 模型

`model.build_resnet18()` 调用 [Torchvision 官方 ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18)（[当前实验使用的 v0.27.0 源码](https://github.com/pytorch/vision/blob/v0.27.0/torchvision/models/resnet.py)），随机初始化，输出 1,000 类。训练脚本需统一设置随机种子和训练配置。

训练参数集中在 `configs/baseline.yaml`。在仓库根目录运行 Baseline（示例使用 GPU 0；前台运行便于先检查错误）：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py --config configs/baseline.yaml
```

需要后台训练并保存终端日志时，可运行：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python tools/train.py \
  --config configs/baseline.yaml \
  --output-dir runs/resnet18_baseline_seed16 \
  > baseline.log 2>&1 &
```

按机器实际情况修改 GPU 编号和配置中的 `run.data_root`。`--output-dir` 覆盖 YAML 中的 `run.output_dir`，每次实验使用新的输出目录；若重定向终端输出，也使用独立日志文件名。默认训练 90 轮，每轮在完整验证集上计算 Top-1 / Top-5。上面的后台命令将终端日志写入 `baseline.log`；输出目录包含 `best.pt`、`last.pt`、`config.json`（实际生效配置）和 `metrics.jsonl`。

实验编号与提供的记录表一致。1–10 号配置在 `main`；11–13 号在 `ResNet2ImageNet2`；14–16 号在 `ResNet2ImageNet3`。下表是**实验设计及配置位置**。

| 编号 | 分支 | 配置文件 | 相对基线的改动 |
| --- | --- | --- | --- |
| 1 | `main` | `configs/baseline.yaml` | 原始 ResNet-18 基线 |
| 2 | `main` | `configs/a1_cosine_eta0.yaml` | Cosine，`t_max=90`，`eta_min=0` |
| 3 | `main` | `configs/a2_cosine_eta0p001.yaml` | Cosine，`t_max=90`，`eta_min=0.001` |
| 4 | `main` | `configs/a3_multistep_3060.yaml` | MultiStep，`milestones=[30,60]`，`gamma=0.1` |
| 5 | `main` | `configs/b1_randaugment_m5.yaml` | RandAugment，`num_ops=2`，`magnitude=5` |
| 6 | `main` | `configs/b2_randaugment_m9.yaml` | RandAugment，`num_ops=2`，`magnitude=9` |
| 7 | `main` | `configs/b3_random_erasing.yaml` | RandomErasing，`p=0.25` |
| 8 | `main` | `configs/c1_label_smoothing_0p05.yaml` | `label_smoothing=0.05` |
| 9 | `main` | `configs/c2_label_smoothing_0p10.yaml` | `label_smoothing=0.10` |
| 10 | `main` | `configs/c3_label_smoothing_0p20.yaml` | `label_smoothing=0.20` |
| 11 | `ResNet2ImageNet2` | `configs/attention_se_all_r16.yaml` | 8 个 BasicBlock 加 SE，`r=16` |
| 12 | `ResNet2ImageNet2` | `configs/attention_se_all_r8.yaml` | 8 个 BasicBlock 加 SE，`r=8` |
| 13 | `ResNet2ImageNet2` | `configs/attention_se_layer4_r16.yaml` | 仅 layer4 的 2 个 BasicBlock 加 SE，`r=16` |
| 14 | `ResNet2ImageNet3` | `configs/downsample_e1_binomial3.yaml` | 3×3 二项式加权滤波 |
| 15 | `ResNet2ImageNet3` | `configs/downsample_e2_box3.yaml` | 3×3 均值滤波 |
| 16 | `ResNet2ImageNet3` | `configs/downsample_e3_binomial5.yaml` | 5×5 二项式加权滤波 |

同时可以复制 `configs/baseline.yaml` 创建新配置，只修改要研究的变量，并在实验表中写明对照组。当前可直接修改的参数如下：

| 配置项 | 可修改内容 |
| --- | --- |
| `run` | 数据/输出路径、轮数、batch size、workers、随机种子、续训路径 |
| `optimizer` | 当前为 `sgd`：初始学习率、momentum、weight decay |
| `scheduler` | `multistep`（衰减轮次、gamma）或 `cosine`（`t_max`、`eta_min`） |
| `augmentation` | 共同的裁剪、翻转参数；另可选 `basic`、`randaugment`（`num_ops`、`magnitude`）、`random_erasing`（`p`、`scale`、`ratio`、`value`） |
| `loss` | `cross_entropy`，可设置 `label_smoothing`；验证损失不使用标签平滑 |
| `model.name` | `main` 中为 `resnet18`，随机初始化、1,000 类；SE/滤波模型只在对应分支注册 |

新增策略需扩展 `tools/train.py` 中对应的 `make_*` 函数，模型通过 `model.build_model` 扩展，并在 `utils/train_config.py` 中登记名称与参数校验，再填写对应的 `name`。SE 和滤波模型的注册只在各自分支；验证预处理保持固定。

配置内相对路径基于仓库根目录；原有命令行参数（如 `--lr`、`--epochs`、`--data-root`）仍可覆盖 YAML，命令行相对路径基于当前目录。使用 `python tools/train.py --config configs/baseline.yaml --print-config` 可预览生效配置，不启动训练。

续训需保持原实验配置，填写 `run.resume` 或传入 `--resume`，并将日志重定向改为 `>> baseline.log 2>&1 &`，保留之前的日志。新 checkpoint 会校验训练配置；旧 checkpoint 只能校验优化器和调度器，并提示人工核对其他设置。

结果表使用验证集 Top-1 最佳的 `best.pt`：`best_top1` 和 `best_top5` 对应同一轮、同一个模型；`best_top5` 不是各轮 Top-5 的独立最大值。

## 验证与结果读取

`tools/train.py` 在**每个 epoch 训练后**自动用完整验证集计算验证损失、Top-1 和 Top-5，并写入 `runs/<实验目录>/metrics.jsonl`。当前仓库**没有独立加载 `best.pt` 重新推理的评估脚本**；以下命令仅从已生成的验证记录中读取最佳 Top-1 轮及同轮 Top-5，不能替代一次新的模型评估：

```bash
python -c 'import json; from pathlib import Path; p=Path("runs/resnet18_baseline_seed16/metrics.jsonl"); rows=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]; best=max(rows, key=lambda r:r["top1"]); print("epoch:", best["epoch"], "Top-1:", best["top1"], "Top-5:", best["top5"])'
```

换实验时改为对应的 `metrics.jsonl` 路径。训练时的 `best.pt` 按 Top-1 选出；若需要独立的 checkpoint 复评命令，还需另行实现评估入口。

## 绘制损失曲线

先在 `tools/plot_loss.py` 末尾配置日志目录 `log_dir`、实际日志文件名列表 `log_list` 和输出路径 `savename`，再运行。路径以仓库根目录为基准，也支持本机绝对路径。

```bash
python tools/plot_loss.py
```

仅绘制线性坐标轴的损失曲线，训练曲线为实线，验证曲线为虚线。图例自动使用日志文件名（不含扩展名），入口中附有多日志对比配置示例。

## 绘制准确率曲线

配置方式同损失曲线，修改 `tools/plot_accuracy.py` 末尾的日志目录、文件列表和输出路径后运行：

```bash
python tools/plot_accuracy.py
```

实线为 Top-1、虚线为 Top-5，均为每轮验证集准确率（%）；加粗的点标注 Top-1 最佳轮次及同轮 Top-5，支持多日志对比。实际输出位置由脚本末尾的 `savename` 决定；当前绘图脚本含服务器上的日志路径，换机器运行前须改为本机路径。

## 实验记录与结果分析

 实验记录表、结果分析文档及其汇报PDF见[supplements/](supplements/) 。