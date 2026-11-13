# ImageNet-1K 数据准备与复现

`train_files.txt` 列出全部 128,124 张训练图像；`train_manifest.csv` 和 `val_manifest.csv` 提供类别标签；`metadata.json` 记录种子 `16`、抽样规则、类别映射及清单 SHA-256。请使用相同版本的 ImageNet-1K 原图，其目录包含 `train/<wnid>/*.JPEG` 和平铺的 `val/ILSVRC2012_val_*.JPEG`。

在仓库根目录运行：

```bash
python data_pre/rebuild_from_manifest.py --source-root /path/to/imagenet --verify-only
python data_pre/rebuild_from_manifest.py --source-root /path/to/imagenet --output-root data/imagenet_10pct
```

第一条只读校验清单和原图；第二条创建软链接，不复制图像。训练时用 `ImageFolder` 读取 `data/imagenet_10pct/train` 和 `data/imagenet_10pct/val`。`data/` 已被 Git 忽略，无需上传图像。
