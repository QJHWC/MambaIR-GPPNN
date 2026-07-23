# Security Policy

## Supported Versions

安全修复仅面向当前 `main` 分支上的 Experimental v2.2。历史提交、外部分支、训练权重和用户自行修改的版本不提供安全更新保证。

| Version | Supported |
| --- | --- |
| Experimental v2.2 (`main`) | Yes |
| Historical versions | No |

本项目是研究软件，不提供安全响应或修复时限承诺。

## Reporting a Vulnerability

请不要通过公开 Issue 披露未修复的漏洞。使用 GitHub 的
[私密安全漏洞报告](https://github.com/QJHWC/MambaIR-GPPNN/security/advisories/new)
提交报告。

报告中请尽量包括：

- 受影响的提交、文件或功能；
- 漏洞影响和可行的攻击场景；
- 最小复现步骤或概念验证；
- 已知的缓解措施；
- 是否存在计划公开披露的时间要求。

请删除报告中的访问令牌、私有数据集、个人信息和其他不必要的敏感内容。维护者会通过 GitHub Security Advisory 与报告者沟通，在修复发布前协调披露。

## Scope

项目源码和仓库维护流程中的可复现安全问题属于报告范围。第三方依赖本身的漏洞应同时报告给对应上游；报告本项目如何受到影响仍然有帮助。模型质量、训练收敛、指标差异和一般使用问题不属于安全漏洞，请使用普通 Issue。
