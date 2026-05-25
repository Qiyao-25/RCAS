# 预算约束下的序贯特征挖掘智能体系统

这是一个面向风控/信贷场景的特征挖掘 POC。系统在有限预算下用 UCB 多臂老虎机选择挖掘方向，由 Generator 生成 FeatureDSL，经过 DSL 校验、IV/KS/PSI 评估与 Critic 反馈后，持续沉淀有效特征。

## 快速开始

```bash
pip install -r requirements.txt

# 默认 API 模式 + Home Credit 数据
python scripts/run_poc.py

# 设置 API Key 后调用 OpenAI 兼容 LLM API
set DEEPSEEK_API_KEY=sk-your-key
python scripts/run_poc.py

# 临时离线调试
python scripts/run_poc.py --mode mock --data mock
```

也可以直接改 `configs/base.env`，或用系统环境变量覆盖同名配置。默认不再使用模拟测试：`LLM_MODE=api`，`DATA_SOURCE=home_credit`。如果未设置 API Key，程序会直接报错；需要离线调试时显式使用 `--mode mock --data mock`。

## 配置方式

项目已从 `configs/base.yaml` 切换为 `.env` 风格配置文件：`configs/base.env`。这样更适合本地、CI、容器和云部署，因为每个配置项都能直接映射为环境变量。

优先级：

1. 命令行参数：`--mode`、`--data`
2. 系统环境变量：如 `LLM_MODE`、`DEEPSEEK_API_KEY`
3. 默认文件：`configs/base.env`

常用配置：

```env
LLM_MODE=api
LLM_PROVIDER=deepseek
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=
LLM_MODEL=deepseek-chat
LLM_ALLOW_MOCK_FALLBACK=false

DATA_SOURCE=home_credit
HOME_CREDIT_DIR=

BUDGET_DAILY_LIMIT=50
BUDGET_MAX_STEPS=20
GENERATOR_K_FEATURES_PER_STEP=3

DIRECTIONS=user_device_risk,transaction_pattern,geolocation_anomaly,account_behavior,network_relation

TRACE_ENABLED=true
RUN_BASE_DIR=runs
```

密钥可使用 `LLM_API_KEY`、`DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY`。建议把真实密钥放在系统环境变量里，不写入仓库文件。

## 架构概览

```text
Orchestrator
  ├─ UCBController        # 根据方向收益做探索/利用选择
  ├─ GeneratorAgent       # 生成 FeatureDSL
  ├─ DSLParser/Validator  # 结构、方向、时间窗口、合规校验
  ├─ Evaluator            # 计算 IV/KS/PSI/missing_rate
  ├─ CriticAgent          # 对低效特征给出修正建议
  ├─ BudgetController     # 控制 Generator/Critic 调用预算
  └─ FeatureRegistry      # 保存有效特征
```

## 五个方向是否足够

当前五个方向适合 POC 和演示闭环：

| 方向 | 覆盖内容 | 当前判断 |
| --- | --- | --- |
| `user_device_risk` | 设备、证件、联系方式、终端异常 | 对反欺诈有价值，但在 Home Credit 数据中更像标记类变量 |
| `transaction_pattern` | 金额、贷款、年金、征信查询 | 信贷 CTR/违约风险的核心方向，应保留 |
| `geolocation_anomaly` | 区域评分、地址/城市不一致 | 有解释性，适合做稳定辅助特征 |
| `account_behavior` | 人口属性、职业、家庭、注册行为 | 覆盖面较大，当前有些“兜底”过宽 |
| `network_relation` | 社交圈、外部评分、住房/建筑信息 | 混合了外部评分和资产画像，生产中建议拆分 |

结论：五个方向“够跑通”，但还不是最优生产设计。它们粒度不完全一致，有的是数据域，有的是风险机制，有的是特征族；这会让 UCB 的奖励信号不够纯，某些方向可能因为样本列多或强变量多而被过度选择。

## 更高效的探索架构

当前代码仍是轻量 POC，但更推荐的生产架构不是简单增加方向数量，而是把“探索空间、候选生成、评估反馈、可解释追踪”拆成可组合的闭环。

### 1. 分层策略空间

从“一层方向”升级为“三层策略空间”：

```text
Domain Arm: 数据域/业务域
  ├─ identity_device
  ├─ credit_transaction
  ├─ geo_region
  ├─ profile_behavior
  ├─ external_score
  └─ relation_graph

Template Arm: 特征生成模板族
  ├─ time_window_agg
  ├─ ratio_share
  ├─ volatility
  ├─ cross_feature
  ├─ anomaly_distance
  └─ stability_shift

Transform Arm: 后处理/稳定性策略
  ├─ raw
  ├─ log
  ├─ woe_bin
  ├─ quantile_bin
  ├─ missing_indicator
  └─ interaction
```

这样 Bandit 不再只问“选哪个方向”，而是选择 `(业务域, 模板族, 变换策略)`。收益归因更清楚：如果 `external_score + ratio_share + woe_bin` 表现好，系统能复用这个组合；如果只是某个业务域强，也不会误以为所有模板都有效。

### 2. 候选池加两阶段评估

推荐把每轮生成拆成：

1. 大量便宜候选：规则模板、历史变体、LLM 扩展、交叉组合同时生成。
2. 轻量预筛：泄漏、重复、缺失率、唯一值、相关性、覆盖率。
3. 精评估：IV/KS/PSI、交叉验证稳定性、增益指标、与已有特征的冗余度。
4. 入池与复用：优质特征进入候选池，失败特征沉淀为负样本，供下一轮 prompt 和 Bandit 使用。

这比“每步只生成 3 个然后直接评估”更高效，因为昂贵的 LLM 和精评估预算只花在更有希望的候选上。

### 3. 多目标奖励函数

UCB 的收益不应只看成功比例，建议改成组合奖励：

```text
reward =
  0.35 * normalized_IV
+ 0.25 * normalized_KS
+ 0.15 * stability_score
+ 0.10 * novelty_score
+ 0.10 * business_explainability
- 0.15 * redundancy_penalty
- 0.20 * leakage_or_compliance_penalty
```

这样能避免系统只追逐强但重复、不可解释或不稳定的特征。

### 4. 方向拆分建议

对 Home Credit 当前数据，建议把现有五类演进为：

| 新方向 | 来源 | 理由 |
| --- | --- | --- |
| `credit_amount_capacity` | 金额、收入、贷款、年金 | 信贷偿付能力核心信号 |
| `credit_inquiry_behavior` | 征信查询次数 | 与近期资金压力相关，适合时间窗口化 |
| `identity_contact_flags` | 证件、电话、邮箱、合同类型 | 更像身份完整性和申请材料质量 |
| `geo_region_stability` | 区域评分、城市/地区一致性 | 解释性强，适合稳定性评估 |
| `demographic_profile` | 年龄、家庭、教育、职业 | 基础画像，应严格做合规审查 |
| `external_score` | EXT_SOURCE_* | 强信号，应该单独建臂，避免掩盖其他方向 |
| `asset_housing_profile` | 房产/建筑相关变量 | 与资产质量相关，但缺失率和稳定性要单独控制 |
| `social_circle_risk` | OBS/DEF social circle | 关系风险，不应和住房变量混在一起 |
| `misc_profile` | 未归类列 | 只做探索兜底，收益应打折 |

当前代码已支持通过 `DIRECTIONS` 环境变量调整方向列表；Validator 会用运行配置覆盖 `dsl_schema.json` 中的方向枚举。

## 可追溯与可解释

每次运行会创建独立目录 `runs/<run_id>/`，记录完整流程：

- `runs/<run_id>/logs/agent.log`：本次运行日志。
- `runs/<run_id>/trace/run_trace.jsonl`：事件流，包含选方向、生成、校验、评估、批评、结束汇总。
- `runs/<run_id>/trace/feature_lineage.json`：按 `feature_id` 聚合的特征血缘，便于审计单个特征。
- `runs/<run_id>/features/feature_registry.json`：有效特征注册结果，包含指标和有效性说明。
- `runs/<run_id>/summary.json`：本次运行总览。

单个特征会记录：

- 生成时的 UCB 方向和选择原因
- FeatureDSL 定义
- 生成原因
- 业务含义 `business_logic`
- 使用了哪些 Critic 历史反馈
- 校验是否通过及原因
- 实际映射到的数据列
- IV/KS/PSI/缺失率和评估解释
- Critic 批判内容、修改轴、参数建议和置信度

这让系统不只是输出“哪些特征好”，也能回答“为什么生成它、为什么保留/跳过它、下一轮如何修正”。

## 数据源

Mock 数据：默认使用 `scripts/mock_data_generator.py` 生成 1000 行仿真特征，用于离线测试闭环。

Home Credit 数据：默认查找项目同级目录 `../home credit default risk/`，也可以设置：

```bash
set HOME_CREDIT_DIR=C:\path\to\home credit default risk
python scripts/run_poc.py --data home_credit
```

## 项目结构

```text
feature_mining_agent/
├─ configs/
│  ├─ base.env
│  └─ dsl_schema.json
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
├─ data/
├─ logs/
└─ requirements.txt
```

## 运行输出

运行结束后会输出：

- 每步 UCB 选择的方向
- Generator 生成的 FeatureDSL ID
- IV、KS、PSI、缺失率与状态
- Critic 反馈与失败模式
- 预算消耗和剩余预算
- 注册到 `runs/<run_id>/features/feature_registry.json` 的有效特征及有效性说明

## 依赖

- Python >= 3.9
- pandas, numpy, scipy
- scikit-learn
- openai 可选，仅 `LLM_MODE=api` 时需要
