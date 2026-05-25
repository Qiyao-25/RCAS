# Git 与 GitHub 工作流建议

## 仓库定位

建议把本项目作为 `credit-risk-ai-agents` 一类的总仓库来维护，而不是只围绕“特征挖掘智能体”命名。后续可以在同一个仓库中扩展：

- `feature_mining_agent`：特征挖掘
- `feature_enhancement_agent`：特征增强与稳定性修复
- `collection_agent`：智能催收策略
- `risk_policy_agent`：策略规则生成与解释
- `shared`：通用 LLM、配置、评估、审计追踪模块

当前代码可以先保持现有结构，后续再逐步重构，不需要一开始大拆。

## 哪些内容应该提交

应该提交：

- `src/`：核心代码
- `scripts/`：运行入口和数据处理脚本
- `configs/`：不含密钥的配置模板
- `README.md`
- `requirements.txt`
- `docs/`
- `.gitignore`

不应该提交：

- `.venv/`
- `data/`
- `runs/`
- `logs/`
- API Key、真实客户数据、运行中间产物
- `__pycache__/`

## 推荐分支习惯

- `main`：稳定可运行版本
- `dev`：日常集成分支
- `feature/<topic>`：具体功能分支，例如 `feature/run-artifacts`

简单阶段也可以只用 `main`，但每次做一组清晰改动后就提交一次。

## 常用命令

查看状态：

```bash
git status
```

提交改动：

```bash
git add .
git commit -m "feat: initialize risk AI agents project"
```

关联 GitHub 仓库：

```bash
git remote add origin https://github.com/<your-name>/<repo-name>.git
git push -u origin main
```

后续更新：

```bash
git add .
git commit -m "feat: add feature enhancement agent"
git push
```

## Commit message 建议

用简洁英文前缀：

- `feat:` 新功能
- `fix:` 修复
- `docs:` 文档
- `refactor:` 重构
- `chore:` 配置、依赖、清理
- `test:` 测试

示例：

```bash
git commit -m "feat: add run-level trace artifacts"
git commit -m "docs: update GitHub workflow guide"
```
