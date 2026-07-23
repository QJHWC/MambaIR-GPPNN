# 评估与结果报告

本指南用于生成可追溯的本地评估结果。仓库当前不发布预训练权重或官方基准表。

## 评估前检查

1. 确认 checkpoint 来源、模型规模和实验开关。
2. 确认测试数据未参与训练或模型选择。
3. 确认 MS、PAN、GT 文件名一一对应。
4. 确认训练和评估使用同一目标尺寸及预处理约定。
5. 将命令、提交哈希和环境信息与结果一起保存。

## 256 尺寸

```bash
python test_256_fair.py \
  --model_path /path/to/best_model.pth \
  --test_dir ./photo/testdateset \
  --output_dir ./test_results_256 \
  --device auto
```

脚本将输入缩放到 256 × 256，并计算全分辨率输出与 GT 之间的指标。

## 512 尺寸

```bash
python test_512_fair.py \
  --model_path /path/to/best_model.pth \
  --test_dir ./photo/testdateset \
  --output_dir ./test_results_512 \
  --device auto
```

不要直接比较由不同尺寸、不同数据或不同缩放过程产生的结果。

## 指标

- **PSNR**：由输出与 GT 的均方误差计算，输入假定在 `[0, 1]`。
- **SSIM**：当前脚本使用项目内的简化实现，不应默认等同于其他库或论文的
  SSIM 设置。
- **损失项**：256 脚本还会记录训练损失的若干组成，用于诊断，不是跨项目
  通用指标。

比较外部论文时必须对齐传感器退化模型、裁剪边界、动态范围、波段处理和指标
实现。名称相同并不表示协议相同。

## 结果文件

评估脚本会在 `--output_dir` 下保存融合图像和 JSON 摘要。目录属于运行产物，
默认不应提交到 Git。

建议将以下元数据与 JSON 一同归档：

```text
commit:
checkpoint_sha256:
model_size:
training_command:
evaluation_command:
dataset:
dataset_version:
split_manifest:
python:
pytorch:
device:
random_seed:
```

## 当前结果状态

仓库历史中的 `results.json` 或日志不足以建立公开基准，因为它们没有同时提供
可公开获取的数据版本、权重、完整环境和固定评估协议。因此：

- README 不展示历史 PSNR/SSIM 数字；
- 文档不把预期值写成实测结果；
- 组件测试通过不代表端到端性能提升；
- 新结果只有在材料齐全时才适合公开。

## 公布结果的最低要求

公开结果表应包含：

- 数据集与许可；
- 训练、验证、测试切分；
- 输入生成和退化协议；
- 模型配置及参数开关；
- 权重下载地址与校验值；
- 精确评估命令；
- 指标实现版本；
- 至少一次完整运行的原始输出。

若进行方法比较，还应使用相同数据和协议，并报告重复运行的变化范围。

## 世界模型推理脚本

`inference_with_world_model.py` 支持加载匹配的 checkpoint，并可启用实验性
Patch Prior：

```bash
python inference_with_world_model.py \
  --model_path /path/to/best_model.pth \
  --test_dir ./photo/testdateset \
  --output_dir ./results_world_model
```

该脚本当前主要报告 PSNR。`--use_patch_prior` 会改变输出处理，必须作为单独
实验记录，不得与基线结果混写。更多限制见 [实验模块说明](world-model.md)。

## 失败与异常

- 找不到 checkpoint：显式检查 `--model_path`，仓库不提供默认权重。
- 状态字典不匹配：使用训练时相同的模型规模和实验开关。
- 没有共同样本：检查 `MS`、`PAN`、`GT` 的文件名和扩展名。
- 结果异常：检查输入通道、动态范围、图像损坏和数据加载器的回退提示。
- 内存不足：降低批处理规模；当前公平测试脚本逐图处理，但大图和模型仍可能
  超出设备内存。
