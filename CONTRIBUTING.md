# Contributing to MambaIR-GPPNN

感谢你关注 MambaIR-GPPNN。该仓库是实验性研究项目；贡献应优先提高实现的正确性、可复现性、可维护性和文档准确性。

## 开始之前

- 先搜索现有 [Issues](https://github.com/QJHWC/MambaIR-GPPNN/issues)，避免重复报告。
- Bug、小型文档修正和测试补充可以直接提交 Pull Request。
- 新模型、训练流程变更、破坏兼容性的接口调整，请先创建 Issue 说明动机、范围和验证方案。
- 安全漏洞不要创建公开 Issue，请按照 [SECURITY.md](SECURITY.md) 私密报告。

## 开发环境

建议使用 Python 3.10 和独立虚拟环境：

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

请勿提交数据集、模型权重、训练日志、缓存、密钥或含有个人信息的文件。

## 提交修改

1. 从最新 `main` 创建主题分支。
2. 将每个 Pull Request 限定为一个清晰目标，避免同时进行无关重构。
3. 保持现有公开 Python API 和命令行参数兼容；若必须变更，请在 PR 中明确迁移方法。
4. 为行为变更补充或更新测试。
5. 同步更新受影响的文档，但不要发布缺少可复现实验依据的性能结论。
6. 使用简洁、可追溯的提交信息，例如 `fix: handle empty validation split`。

## 本地检查

提交前至少运行与修改相关的检查；完整检查建议包括：

```bash
python -m compileall -q .
python -m pytest
python -m ruff check .
```

如修改 Shell 脚本，还应运行：

```bash
bash -n path/to/script.sh
```

涉及训练或推理的修改，请在 PR 中记录使用的命令、设备、随机种子、输入数据形状以及最小可复现结果。不要上传受限数据或大型输出。

## 文档与实验结论

- 明确区分“已实现”“实验中”和“设计计划”。
- 性能数字必须注明数据集、划分、预处理、指标实现、模型权重和运行环境。
- 不得把第三方项目名称、论文结论或实验数据表述为本仓库已验证结果。
- 引用外部代码、配置或文档时，必须说明来源、许可和所做修改，并按需更新
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## Pull Request 要求

PR 描述应包括：

- 问题和修改目标；
- 主要实现变化及兼容性影响；
- 已运行的测试及结果；
- 尚未覆盖的限制或风险；
- 第三方来源与许可影响（如适用）。

维护者可能要求缩小范围、补充测试或修正文档后再合并。

## 贡献许可

提交贡献即表示你有权提供该内容，并同意将你的原创贡献按仓库根目录
[MIT License](LICENSE) 授权。第三方材料仍受其原始许可约束；你有责任保留适用的版权、许可和归属声明。
