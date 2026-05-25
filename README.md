# Credit Risk AI Agents

面向风控/信贷场景的智能体项目。当前已实现 **特征挖掘智能体**：在预算约束下，用分层 UCB 策略选择探索空间，调用 LLM 生成 FeatureDSL，经过合规校验、IV/KS/PSI 评估、Critic 反馈和跨运行优秀因子复用，持续沉淀可解释的高质量特征。

## Quick Start

```bash
pip install -r requirements.txt

# 设置真实 LLM API Key
set DEEPSEEK_API_KEY=sk-your-key

# 默认使用 API 模式 + Home Credit 数据
python scripts/run_poc.py
```

临时离线调试可以显式使用 Mock：

```bash
python scripts/run_poc.py --mode mock --data mock
```

默认配置要求真实 API Key。若未设置 `LLM_API_KEY`、`DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY`，程序会直接报错，避免误用模拟结果。

## Current Capabilities

- 分层探索：每轮同时选择业务方向、特征模板族、变换策略。
- LLM 生成：根据当前策略、历史 Critic 反馈、历史优秀因子生成 FeatureDSL。
- 合规校验：检查 DSL 结构、时间窗口、防泄漏规则和敏感字段。
- 指标评估：计算 IV、KS、PSI、缺失率，并给出有效性说明。
- Critic 反馈：对低效特征生成修正建议，反馈给后续挖掘。
- 可追溯：每次运行生成独立目录，保存日志、事件流、特征血缘和摘要。
- 优秀因子复用：跨运行保存高质量特征，后续挖掘自动检索并注入 Generator。

## Architecture

```text
Orchestrator
  ├─ UCBController
  │  └─ 选择 domain / template_family / transform_strategy
  ├─ GeneratorAgent
  │  └─ 生成 FeatureDSL
  ├─ DSLParser + DSLValidator
  │  └─ 结构校验、防泄漏、合规检查
  ├─ Evaluator
  │  └─ IV / KS / PSI / missing_rate
  ├─ CriticAgent
  │  └─ 低效特征批判与修正建议
  ├─ FeatureRegistry
  │  └─ 保存本次运行有效特征
  ├─ FeatureKnowledgeBase
  │  └─ 跨运行优秀因子复用
  └─ TraceRecorder
     └─ 事件流与特征血缘
```

## Exploration Space

当前探索空间由三层组成：

```text
domain = DIRECTIONS
template_family = TEMPLATE_FAMILIES
transform_strategy = TRANSFORM_STRATEGIES
```

默认配置：

```env
DIRECTIONS=user_device_risk,transaction_pattern,geolocation_anomaly,account_behavior,network_relation
TEMPLATE_FAMILIES=time_window_agg,ratio_share,volatility,cross_feature,anomaly_distance,stability_shift
TRANSFORM_STRATEGIES=raw,log,woe_bin,quantile_bin,missing_indicator,interaction
```

每一轮会用 UCB 分别选择一个 `domain`、一个 `template_family` 和一个 `transform_strategy`。评估完成后，收益会同时更新到三层策略统计中。

## Feature Memory

优秀因子知识库路径：

```text
knowledge/feature_bank.json
```

每次运行结束后，成功特征会写入该文件。下一次运行会自动加载历史优秀因子，并按当前 `domain` 和 `template_family` 检索相关样例，注入 Generator prompt。

`knowledge/feature_bank.json` 是本地运行资产，默认不提交到 Git。

## Configuration

主配置文件：

```text
configs/base.env
```

常用配置：

```env
LLM_MODE=api
LLM_PROVIDER=deepseek
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=
LLM_MODEL=deepseek-v4-pro
LLM_ALLOW_MOCK_FALLBACK=false

DATA_SOURCE=home_credit
HOME_CREDIT_DIR=

BUDGET_DAILY_LIMIT=50
BUDGET_MAX_STEPS=20
GENERATOR_K_FEATURES_PER_STEP=3

FEATURE_BANK_PATH=knowledge/feature_bank.json
RUN_BASE_DIR=runs
TRACE_ENABLED=true
```

配置优先级：

1. 命令行参数：`--mode`、`--data`
2. 系统环境变量
3. `configs/base.env`

密钥建议放在系统环境变量中，不写入配置文件。

## Data Source

默认数据源为 Home Credit Default Risk：

```text
../home credit default risk/
```

也可以通过环境变量指定：

```bash
set HOME_CREDIT_DIR=C:\path\to\home credit default risk
python scripts/run_poc.py
```

离线调试数据由 `scripts/mock_data_generator.py` 生成，仅在显式指定 `--data mock` 时使用。

## Run Artifacts

每次运行会创建独立目录：

```text
runs/<run_id>/
```

目录内容：

```text
runs/<run_id>/
  ├─ logs/
  │  └─ agent.log
  ├─ trace/
  │  ├─ run_trace.jsonl
  │  └─ feature_lineage.json
  ├─ features/
  │  └─ feature_registry.json
  └─ summary.json
```

其中：

- `run_trace.jsonl`：完整事件流。
- `feature_lineage.json`：按特征聚合的生成、校验、评估、Critic 记录。
- `feature_registry.json`：本次运行发现的有效特征。
- `summary.json`：运行摘要、预算统计、方向分布、优秀特征指标。

## Feature Output

有效特征会记录：

- `feature_id`
- `direction`
- `template_family`
- `transform_strategy`
- `business_logic`
- `definition`
- `resolved_column`
- `metrics`
- `effectiveness`

示例：

```json
{
  "feature_id": "FEAT_TRANSACTION_PATTERN_SUM_PAYMENT_ORDER_30D_MINMAX_02",
  "direction": "transaction_pattern",
  "template_family": "ratio_share",
  "transform_strategy": "log",
  "business_logic": "汇总amount在30d内的总量",
  "metrics": {
    "IV": 1.8719,
    "KS": 0.5128,
    "PSI": 0.1341,
    "missing_rate": 0.0
  }
}
```

## Project Structure

```text
feature_mining_agent/
  ├─ configs/
  │  ├─ base.env
  │  └─ dsl_schema.json
  ├─ docs/
  │  └─ GIT_WORKFLOW.md
  ├─ knowledge/
  │  └─ .gitkeep
  ├─ scripts/
  │  ├─ run_poc.py
  │  ├─ data_loader.py
  │  └─ mock_data_generator.py
  ├─ src/
  │  ├─ agents/
  │  ├─ core/
  │  ├─ memory/
  │  ├─ pipeline/
  │  ├─ routing/
  │  └─ utils/
  ├─ README.md
  └─ requirements.txt
```

## Git Hygiene

以下内容不提交到 Git：

- `data/`
- `runs/`
- `logs/`
- `knowledge/*.json`
- `.venv/`
- API Key

详见：

```text
docs/GIT_WORKFLOW.md
```

## Dependencies

- Python >= 3.9
- pandas
- numpy
- scipy
- scikit-learn
- openai，API 模式需要
