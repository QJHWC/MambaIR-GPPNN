# MambaIR-GPPNN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Experimental](https://img.shields.io/badge/status-experimental-orange.svg)](#项目状态)

> **Experimental v2.2**：面向全色锐化（pansharpening）的 PyTorch 研究原型。

**English summary.** MambaIR-GPPNN is an experimental, MambaIR-inspired
pansharpening prototype that combines dual-modal feature processing,
cross-modal attention, and progressive multi-scale reconstruction. It is an
independent research implementation, not an official implementation of
MambaIR or GPPNN. This repository does not publish pretrained weights or claim
benchmark results. See [Architecture](docs/architecture.md),
[Training](docs/training.md), and [Evaluation](docs/evaluation.md) for details.

## 项目状态

- 当前版本：**Experimental v2.2**
- 用途：研究、教学与可复现实验
- 输入：三通道多光谱图像（MS）与单通道全色图像（PAN）
- 输出：三通道融合图像，以及用于训练监督的两个低分辨率输出组
- 推荐入口：`train_unified.py`
- 当前仓库不包含数据集、预训练权重、训练日志或已验证的基准结果

本项目名称保留了研究来源，但实现应准确理解为
**MambaIR-inspired**。代码使用项目自定义的双模态特征模块、注意力与
卷积结构；它不导入官方 MambaIR 实现，也不应被描述为官方 Mamba、
官方 MambaIR 或官方 GPPNN 的复现。本文档不对算法复杂度作未经测量的
承诺。

## 功能概览

- MS/PAN 双分支浅层特征提取
- 项目自定义的 `DualModal_ASSM` 双模态特征处理
- 多阶段跨模态注意力融合
- 全分辨率、二分之一分辨率与四分之一分辨率的渐进重建
- Base 与 Large 两种配置（Large 仍需按目标尺寸做前向兼容性检查）
- 配对翻转、旋转和轻量颜色增强
- 多尺度损失、EMA、断点续训与 TensorBoard 日志
- 可选的实验性“世界模型增强”组件
- 256 与 512 尺寸的独立评估脚本

架构边界与张量接口见 [架构说明](docs/architecture.md)。

## 安装

建议在独立虚拟环境中安装。PyTorch 的 CPU/CUDA 构建应根据本机环境从
[PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)选择。

```bash
git clone https://github.com/QJHWC/MambaIR-GPPNN.git
cd MambaIR-GPPNN

python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

安装后可先查看命令行帮助：

```bash
python train_unified.py --help
python test_256_fair.py --help
```

更完整的环境与排错说明见 [训练指南](docs/training.md)。

## 数据准备

数据不随仓库发布。`--photo_root` 指向的目录必须具有以下结构：

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

每个分组中的 `MS`、`PAN` 和 `GT` 必须使用相同文件名。当前加载器支持
`.jpg`、`.png` 和 `.bmp`：

```text
dataset/MS/0001.png
dataset/PAN/0001.png
dataset/GT/0001.png
```

当前代码读取三通道 MS/GT 和单通道 PAN，并归一化到 `[0, 1]`。训练集与
验证集由 `dataset/` 按排序后的文件名切分；`testdateset/` 全部用于测试。
具体切分规则见 [训练指南](docs/training.md#数据切分与预处理)。

## 训练

推荐从统一入口开始：

```bash
python train_unified.py \
  --model_size base \
  --img_size 256 \
  --photo_root ./photo \
  --batch_size 4 \
  --epochs 80
```

512 尺寸示例：

```bash
python train_unified.py \
  --model_size base \
  --img_size 512 \
  --photo_root ./photo \
  --batch_size 1 \
  --epochs 80
```

`train_unified.py` 负责组织常用参数，随后调用 `train.py`。请根据设备内存
实测调整批大小，不要把脚本中的默认值视为硬件保证。

断点续训：

```bash
python train_unified.py \
  --model_size base \
  --img_size 256 \
  --photo_root ./photo \
  --resume /path/to/checkpoint.pth
```

可选模块、输出目录和高级入口见 [训练指南](docs/training.md) 与
[实验模块说明](docs/world-model.md)。

## 评估

评估尺寸应与训练和目标数据尺寸一致。显式指定权重和测试目录：

```bash
python test_256_fair.py \
  --model_path /path/to/best_model.pth \
  --test_dir ./photo/testdateset \
  --output_dir ./test_results_256
```

```bash
python test_512_fair.py \
  --model_path /path/to/best_model.pth \
  --test_dir ./photo/testdateset \
  --output_dir ./test_results_512
```

脚本会计算 PSNR 和简化 SSIM，并写出本次运行的结果。仓库目前没有可由
公开数据、权重和固定协议共同复现的结果表，因此 README 不发布性能数字。
指标定义和报告要求见 [评估指南](docs/evaluation.md)。

## 模型权重

预训练权重与历史训练 checkpoint **不随本仓库发布**。使用者需要：

1. 用自己的数据训练权重；或
2. 从明确标注来源、配置、数据协议和许可证的外部发布获取权重。

请勿提交 `.pth`、`.pt`、`.ckpt`、日志、结果目录或数据集。仅有文件名或
Git LFS 指针不能替代可用权重。

## 实验性世界模型组件

仓库包含 WSM、DCA-FIM、DSC、WAC-X 和 Patch Prior 等实验组件。它们是
研究性扩展名称，不表示已经建立通用“世界模型”，也不意味着其效果已通过
公开基准验证。

训练时可使用预设：

```bash
python train_unified.py \
  --model_size base \
  --img_size 256 \
  --photo_root ./photo \
  --enable_world_model \
  --world_model_preset wsm_dsc
```

使用前请阅读 [实验模块说明](docs/world-model.md)，尤其是状态管理、
消融实验和结果解释限制。

## 文档

- [架构与接口](docs/architecture.md)
- [训练与数据](docs/training.md)
- [评估与结果报告](docs/evaluation.md)
- [实验性世界模型组件](docs/world-model.md)
- [v2.2 优化开发记录](docs/development/v2.2-optimization-notes.md)
- [世界模型历史设计记录](docs/development/world-model-design-record.md)

## 引用与致谢

若本仓库对你的工作有帮助，请引用仓库版本，并同时阅读和引用实际使用的
上游研究：

- Guo et al., “MambaIR: A Simple Baseline for Image Restoration with
  State-Space Model,” ECCV 2024.
  [论文](https://doi.org/10.1007/978-3-031-72649-1_13) ·
  [代码](https://github.com/csguoh/MambaIR)
- Xu et al., “Deep Gradient Projection Networks for Pan-sharpening,”
  CVPR 2021.
  [论文](https://doi.org/10.1109/CVPR46437.2021.00142) ·
  [代码](https://github.com/shuangxu96/GPPNN)

本项目感谢上述工作提供的研究思路。MambaIR 与 GPPNN 的名称、论文和官方
实现归各自作者所有；本仓库不是这些项目的官方发行版。机器可读引用信息见
`CITATION.cff`。

## 许可证

本项目原创部分以 [MIT License](LICENSE) 发布：

```text
Copyright (c) 2025 Qin Jiahong
```

MIT 许可不改变第三方材料原有的许可与归属。第三方来源、修改声明和适用的
Apache-2.0 文本见 [第三方通知](THIRD_PARTY_NOTICES.md) 与
[Apache-2.0](LICENSES/Apache-2.0.txt)。

## 贡献与安全

提交问题或改动前，请阅读 [贡献指南](CONTRIBUTING.md)。安全问题请按
[安全策略](SECURITY.md) 中的方式报告。研究结果报告应包含数据来源、
预处理、权重、配置、随机种子和评估命令，避免只给出无法复现的单个数字。
