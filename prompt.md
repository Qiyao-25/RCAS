# ROLE
你是一名资深 AI 系统架构师与 MLOps 工程师，擅长构建多智能体协同系统与预算约束下的序贯决策管道。

# OBJECTIVE
请基于以下规范，生成一个完整、可独立运行的 Python MVP 项目代码，实现“预算约束下的序贯特征挖掘智能体系统”。代码必须结构清晰、类型安全、自带 Mock 数据与测试入口，执行 `python scripts/run_poc.py` 即可跑通完整闭环。

# TECH STACK CONSTRAINTS
- Python 3.10+
- Pydantic v2（严格数据契约）
- Polars（数据计算，替代 Pandas 提升性能）
- OpenAI-compatible API 封装（提供 MockLLMClient  fallback，无需真实 Key 也可运行）
- YAML / JSON（配置与 Schema）
- 标准库 logging / dataclasses / pathlib
- 禁止引入重型依赖（如 Spark、FAISS、LangChain、AutoGen），MVP 保持轻量

# PROJECT STRUCTURE (必须严格遵循)
feature_mining_agent/
├── configs/base.yaml
├── configs/dsl_schema.json
├── src/utils/schemas.py
├── src/utils/metrics.py
├── src/utils/validator.py
├── src/core/orchestrator.py
├── src/core/state_manager.py
├── src/core/budget_controller.py
├── src/routing/ucb_controller.py
├── src/agents/generator.py
├── src/agents/critic.py
├── src/agents/llm_client.py
├── src/pipeline/dsl_parser.py
├── src/pipeline/evaluator.py
├── src/pipeline/cache_manager.py
├── src/memory/feature_registry.py
├── scripts/run_poc.py
├── scripts/mock_data_generator.py
├── requirements.txt
└── README.md

# CORE DATA CONTRACTS (Pydantic v2)
请严格实现以下模型，所有模块 I/O 必须基于此：

1. FeatureDSL
   - feature_id: str
   - direction: str
   - definition: dict (source_table, time_window, filter, aggregation, transformation)
   - business_logic: str
   - compliance_tags: list[str]

2. EvalResult
   - feature_id: str
   - IV: float
   - KS: float
   - KS_gain: float
   - PSI: float
   - missing_rate: float
   - status: Literal["success", "skipped", "failed"]
   - reason: str | None

3. CriticOutput
   - feature_id: str
   - failure_reason: str
   - modification_axis: Literal["time_window", "filter_condition", "aggregation", "transformation", "direction_shift"]
   - param_suggestion: str
   - next_direction_hint: str
   - confidence: float

4. DirectionStats
   - direction: str
   - n_k: int
   - mu_k: float
   - sigma_k: float
   - last_update: int

5. SystemState
   - step: int
   - budget_remaining: int
   - directions: list[DirectionStats]
   - explored_features: list[str]
   - failure_patterns: dict[str, int]

# KEY LOGIC REQUIREMENTS
1. UCB Controller
   - 公式: UCB(k) = mu_k + c * sqrt(ln(t) / n_k)
   - 冷启动: n_k=0 时返回 float('inf') 强制探索
   - 输出选中 direction 及 UCB 值

2. Budget Controller
   - 每日上限 B（默认 50）
   - 每次 Generator 调用消耗 1，Critic 调用消耗 1
   - IV < 0.005 或 missing_rate > 0.7 直接标记 skipped，不消耗 Critic 预算
   - 预算耗尽时 Orchestrator 优雅退出

3. Evaluator (MVP Mock)
   - 使用 Polars 读取 Mock 数据
   - 实现确定性 IV/KS 计算函数（基于分箱）
   - 实现早期拦截逻辑（缺失率、单值占比、时间泄漏检查）
   - 返回 EvalResult 列表

4. Generator & Critic
   - 使用 LLMClient 调用，强制 response_format=json_object
   - Generator Prompt 注入：当前方向、Critic反馈、DSL Schema、防泄漏规则
   - Critic Prompt 注入：低效特征指标、分箱统计、业务先验模板
   - 若 LLM 不可用，提供 deterministic_fallback 生成/批评逻辑保证可运行

5. Orchestrator Loop
   - while budget > 0 and step < max_steps:
       1. UCB 选择方向
       2. Generator 生成 K=3 个 DSL
       3. DSL Parser 校验 + 防泄漏检查
       4. Evaluator 计算指标
       5. 过滤低效特征 → Critic 生成修正建议
       6. State Manager 更新 mu_k, n_k, failure_patterns
       7. 记录日志，step += 1
   - 最终输出 delta_features 清单与搜索状态快照

# IMPLEMENTATION STEPS (请按顺序生成代码)
1. 实现 src/utils/schemas.py（Pydantic 模型）
2. 实现 src/utils/metrics.py（IV/KS/PSI 计算，含分箱逻辑）
3. 实现 src/utils/validator.py（DSL Schema 校验、时间窗口防泄漏、合规标签检查）
4. 实现 src/agents/llm_client.py（OpenAI 兼容封装 + MockLLMClient 降级）
5. 实现 src/agents/generator.py & critic.py（Prompt 模板 + JSON 解析 + 重试）
6. 实现 src/routing/ucb_controller.py（UCB 公式 + 冷启动 + 动态 c 系数）
7. 实现 src/pipeline/dsl_parser.py & evaluator.py & cache_manager.py（Polars Mock 计算管道）
8. 实现 src/core/state_manager.py & budget_controller.py
9. 实现 src/core/orchestrator.py（主循环，串联所有模块）
10. 实现 scripts/mock_data_generator.py（生成 1000 行仿真流水+标签数据）
11. 实现 scripts/run_poc.py（一键运行入口，打印每轮 UCB 选择、生成特征、评估结果、Critic 反馈、预算消耗）
12. 生成 configs/base.yaml, configs/dsl_schema.json, requirements.txt, README.md

# CODE CONSTRAINTS
- 全部使用 Type Hints 与 Pydantic 校验
- 禁止硬编码路径，使用 pathlib
- 日志使用 logging.config.dictConfig，输出 JSON 格式
- 每个文件顶部写明模块职责与 I/O 契约
- 错误处理完善：LLM 超时/JSON 解析失败/DSL 校验失败需捕获并降级
- 代码必须自包含，无需外部数据库或数仓即可运行

# OUTPUT FORMAT
- 按文件路径顺序输出完整代码，每个文件用 ```python 或 ```yaml 或 ```json 包裹
- 不要省略任何文件，不要使用“其余类似”或“请自行补充”
- 最后附运行命令与预期输出示例
- 保持代码生产级规范：PEP8、模块化、注释清晰、无冗余依赖