# 实验性世界模型组件

“世界模型增强”是本仓库对一组可选研究组件的历史统称。它不表示系统已经
学习到通用环境模型，也不构成性能或理论保证。所有组件默认应通过消融实验
独立验证。

## 组件

### WSM

`WorldStateMemory` 在第一阶段融合特征上维护隐藏状态，并生成仿射调制参数。
模型对象会在调用之间保存状态，因此输出可能受样本顺序、批大小和先前调用
影响。

使用 WSM 时应：

- 明确一个序列的起止边界；
- 新序列或独立样本评估前重建模型或清除状态；
- 不把不同批次排列的结果默认视为等价；
- 检查训练和推理时的状态策略是否一致。

### DCA-FIM

DCA-FIM 是跨模态注意力中的实验性几何对齐分支，通过学习偏移和采样点调整
特征交互。它不是对真实传感器配准误差的测量替代品。

### DSC

`SensorConsistencyLoss` 是训练阶段的传感器一致性正则项。它由
`train.py` 的组合损失管理，不改变模型返回值。退化假设需要与实际传感器和
数据生成协议分别验证。

### WAC-X

`WACXLoss` 是实验性的跨带频域一致性损失。它用于训练约束，不意味着在所有
传感器或波段配置上具有相同物理解释。

### Patch Prior

`PatchPriorRefiner` 是推理阶段可选的分块后处理。默认生成器不等同于经过
外部数据学习的生成先验。启用后应作为独立方法配置报告。

## 训练预设

统一入口支持：

| 预设 | 启用内容 |
| --- | --- |
| `wsm_only` | WSM |
| `dsc_only` | DSC |
| `wsm_dsc` | WSM + DSC |
| `full` | WSM + DCA-FIM + DSC + WAC-X |

示例：

```bash
python train_unified.py \
  --model_size base \
  --img_size 256 \
  --photo_root ./photo \
  --enable_world_model \
  --world_model_preset wsm_dsc
```

`custom` 虽保留在当前命令行选项中，但统一入口没有完整定义该分支，不建议
使用。需要精确组合时，直接调用 `train.py`：

```bash
python train.py \
  --model_size base \
  --img_size 256 \
  --photo_root ./photo \
  --enable_world_model \
  --use_wsm \
  --use_dsc
```

损失权重可通过完整入口的 `--lambda_s` 和 `--lambda_w` 设置。每次实验应
保存完整命令。

## 推理

```bash
python inference_with_world_model.py \
  --model_path /path/to/best_model.pth \
  --test_dir ./photo/testdateset \
  --output_dir ./results_world_model
```

启用 Patch Prior：

```bash
python inference_with_world_model.py \
  --model_path /path/to/best_model.pth \
  --test_dir ./photo/testdateset \
  --output_dir ./results_world_model_patch \
  --use_patch_prior
```

加载 checkpoint 时，脚本从其中的配置恢复 WSM 和 DCA-FIM 开关。旧权重若
缺少这些字段，会按关闭处理。

## 消融建议

至少比较以下配置，其他训练和评估条件保持一致：

1. Baseline；
2. WSM；
3. DSC；
4. WSM + DSC；
5. Full。

每个配置应使用相同切分、随机种子集合、训练预算和评估实现。报告均值与变化
范围，并保留失败运行。不要用设计文档中的预期数字作为验收结果。

## 测试覆盖

`tests/` 包含 WSM、DCA-FIM、DSC、WAC-X 和 Patch Prior 的组件测试。它们
主要验证张量形状、梯度和基本数值行为。组件测试不能替代：

- 端到端训练；
- checkpoint 兼容性检查；
- 独立数据集评估；
- 传感器假设验证；
- 真实消融实验。

## 已知风险

- WSM 隐状态跨调用保存，可能造成样本间状态泄漏。
- 实验开关改变参数结构或损失，checkpoint 可能不兼容。
- Full 预设同时改变多个因素，无法单独归因。
- Patch Prior 会改变推理路径，必须与基线分开报告。
- 当前公开仓库没有足以验证收益的权重和结果材料。

历史设计背景见
[世界模型历史设计记录](development/world-model-design-record.md)。
