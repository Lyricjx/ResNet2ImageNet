# ResNet2ImageNet
A controlled ImageNet-1K classification study with ResNet-18 as the baseline. Some independent improvements are evaluated through comparisons and ablations, using a reproducible 10% training subset and the full validation set for Top-1 and Top-5 accuracy.

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

## 模型

`model.build_resnet18()` 调用 [Torchvision 官方 ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18)（[当前实验使用的 v0.27.0 源码](https://github.com/pytorch/vision/blob/v0.27.0/torchvision/models/resnet.py)），随机初始化，输出 1,000 类。训练脚本需统一设置随机种子和训练配置。

训练参数集中在 `configs/baseline.yaml`。在仓库根目录运行 Baseline（示例使用 GPU 0）：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python tools/train.py \
  --config configs/baseline.yaml \
  --output-dir runs/resnet18_baseline_seed16 \
  > baseline.log 2>&1 &
```

按机器实际情况修改 GPU 编号和配置中的 `run.data_root`。`--output-dir` 覆盖 YAML 中的 `run.output_dir`，每次实验使用新的输出目录和日志文件名。默认训练 90 轮，每轮在完整验证集上计算 Top-1 / Top-5。日志由命令写入 `baseline.log`；输出目录包含 `best.pt`、`last.pt`、`config.json`（实际生效配置）和 `metrics.jsonl`。

做消融实验时，可以复制 `configs/baseline.yaml`，在 `configs/` 下为每组实验新建独立的 YAML 文件（例如 `configs/lr_multistep_30_60.yaml`），只修改该组要研究的参数。运行时将 `--config` 指向新文件，并使用独立的 `--output-dir` 和日志文件名；各实验的 YAML 与输出目录中的 `config.json` 一起记录实验设置。当前可直接修改的参数如下：

| 配置项 | 可修改内容 |
| --- | --- |
| `run` | 数据/输出路径、轮数、batch size、workers、随机种子、续训路径 |
| `optimizer` | 当前为 `sgd`：初始学习率、momentum、weight decay |
| `scheduler` | 当前为 `multistep`：衰减轮次、gamma |
| `augmentation` | 当前为 `basic`：随机裁剪范围、宽高比、水平翻转概率 |
| `loss.name` | 当前为 `cross_entropy` |
| `model.name` | 当前为 `resnet18`，随机初始化、1,000 类 |

新增策略需扩展 `tools/train.py` 中对应的 `make_*` 函数，模型通过 `model.build_model` 扩展，并在 `utils/train_config.py` 中登记名称与参数校验，再填写对应的 `name`。验证预处理保持固定。

配置内相对路径基于仓库根目录；原有命令行参数（如 `--lr`、`--epochs`、`--data-root`）仍可覆盖 YAML，命令行相对路径基于当前目录。使用 `python tools/train.py --config configs/baseline.yaml --print-config` 可预览生效配置，不启动训练。

续训需保持原实验配置，填写 `run.resume` 或传入 `--resume`，并将日志重定向改为 `>> baseline.log 2>&1 &`，保留之前的日志。新 checkpoint 会校验训练配置；旧 checkpoint 只能校验优化器和调度器，并提示人工核对其他设置。

结果表使用验证集 Top-1 最佳的 `best.pt`：`best_top1` 和 `best_top5` 对应同一轮、同一个模型；`best_top5` 不是各轮 Top-5 的独立最大值。

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

实线为 Top-1、虚线为 Top-5，均为每轮验证集准确率（%）；小红点标注 Top-1 最佳轮次及同轮 Top-5，支持多日志对比。默认输出 `runs/plots/baseline_accuracy.png`。
