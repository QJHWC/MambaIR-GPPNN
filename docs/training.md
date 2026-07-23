# 训练与数据

本文给出 Experimental v2.2 的当前训练流程。命令示例只描述如何运行，不是
速度、显存或最终指标保证。

## 环境

先根据硬件从 [PyTorch 官方页面](https://pytorch.org/get-started/locally/)
安装匹配的 PyTorch，再安装仓库依赖：

```bash
python -m pip install -r requirements.txt
```

确认入口可解析：

```bash
python train_unified.py --help
python train.py --help
```

GPU 训练需要可用的 CUDA 版 PyTorch；CPU 可用于基本检查和小规模调试，
不代表适合完整训练。

## 数据目录

`--photo_root` 必须包含训练/验证来源 `dataset` 和独立测试来源
`testdateset`：

```text
photo/
├── dataset/
│   ├── MS/
│   ├── PAN/
│   └── GT/
└── testdateset/
    ├── MS/
    ├── PAN/
    └── GT/
```

三个子目录按文件名取交集，支持 JPG、PNG 和 BMP。一个样本的三幅图必须同名。
加载器将 MS/GT 读为 RGB，将 PAN 读为灰度，并缩放到 `[0, 1]`。

## 数据切分与预处理

文件名排序后：

- 当 `dataset` 至少有 650 个共同样本时，前 600 个用于训练，接下来的
  50 个用于验证。
- 少于 650 个样本时，前 90% 用于训练，其余用于验证。
- `testdateset` 的全部共同样本用于测试。

训练模式同步执行随机水平翻转、垂直翻转和 90 度倍数旋转；默认还对 MS/GT
应用轻量颜色抖动。评估模式不执行随机增强。

注意：加载失败时当前数据集代码会打印错误并返回随机张量。正式实验前应先
检查所有文件可读、尺寸和命名一致，避免把回退样本混入结果。

## 推荐训练入口

`train_unified.py` 负责常用配置，再调用完整实现 `train.py`：

```bash
python train_unified.py \
  --model_size base \
  --img_size 256 \
  --photo_root ./photo \
  --batch_size 4 \
  --epochs 80 \
  --num_workers 4
```

如未指定 `batch_size`、`epochs` 或 `lr`，统一入口会根据模型规模和尺寸选择
脚本默认值。默认值只是起点；应按实际硬件和数据验证。

当前 Large-256 配置存在注意力通道整除的已知兼容性风险，正式训练前必须先
完成前向 smoke test；推荐的基线验证路径是 Base-256。

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--model_size` | `base` 或 `large` |
| `--img_size` | `256` 或 `512` |
| `--photo_root` | 数据根目录 |
| `--batch_size` | 批大小 |
| `--epochs` | 训练轮数 |
| `--lr` | 初始学习率 |
| `--device` | `auto`、`cuda` 或 `cpu` |
| `--num_workers` | DataLoader 工作进程数 |
| `--save_dir` | checkpoint 根目录 |
| `--log_dir` | TensorBoard 日志根目录 |
| `--resume` | 指定 checkpoint 续训 |
| `--auto_resume` | 从保存目录自动寻找最近 checkpoint |

统一入口会为保存目录和日志目录追加模型、尺寸与时间戳。

## 显存不足

优先降低 `--batch_size`；仍不足时可改用 `base` 或 256 输入。完整入口还支持
`--grad_accum_steps` 和 `--fp16`，但统一入口当前没有转发这两个参数，需要
直接调用 `train.py`：

```bash
python train.py \
  --model_size base \
  --img_size 256 \
  --photo_root ./photo \
  --batch_size 1 \
  --grad_accum_steps 4 \
  --fp16
```

混合精度仅应在支持的设备上启用。`--auto_batch_size` 会进行运行时探测，
仍应检查最终配置是否符合实验设计。

## 断点续训

指定权重：

```bash
python train_unified.py \
  --model_size base \
  --img_size 256 \
  --photo_root ./photo \
  --resume /path/to/epoch_10.pth
```

模型规模、实验开关和 checkpoint 内状态必须匹配。续训前建议保留原文件，
并确认优化器和调度器状态可以加载。

## 输出

训练通常生成：

- `checkpoints/`：最佳模型和周期 checkpoint；
- `logs/`：TensorBoard 事件；
- `results.json`：本次训练摘要。

这些都是运行产物，不应提交到 Git。查看日志：

```bash
tensorboard --logdir logs
```

## 其他训练入口

- `train.py`：完整参数与实际训练循环，适合需要细粒度配置的实验。
- `train_optimized.py`：历史实验入口，不作为默认可复现路径。
- `train_safe.py`：保守配置的历史实验入口。
- `train_minimal.py`：最小训练路径，用于快速检查。
- `run_cloud_train.sh`：云端 Shell 包装；使用前检查路径和参数。

新实验应优先记录统一入口或完整入口的完整命令，不要混合多个历史脚本的默认
设置后直接比较指标。

## 可复现性记录

每次报告至少保存：

- Git 提交哈希；
- Python、PyTorch、CUDA 和 GPU 信息；
- 数据来源、许可、样本清单与切分；
- 完整命令和所有非默认参数；
- 随机种子与重复次数；
- 最佳权重的 SHA-256；
- 对应的评估命令与原始结果文件。

当前代码没有提供全局确定性配置。严谨比较应补充固定随机种子，并评估多个
独立运行，而不是把单次最佳值视为稳定结果。

## 实验组件

使用 `--enable_world_model` 前请先阅读
[实验性世界模型组件](world-model.md)。这些开关会改变模型或损失定义，
checkpoint 之间不一定可以互换。
