"""
scripts/run_poc.py
==================
模块职责：POC 一键运行入口，执行完整闭环。

用法:
  python scripts/run_poc.py                    # 使用默认配置 (API 模式 + Home Credit 数据)
  python scripts/run_poc.py --mode api         # 使用真实 LLM API
  python scripts/run_poc.py --data home_credit # 使用 Home Credit 真实数据

配置切换 (configs/base.env 或系统环境变量):
  LLM_MODE:    "mock" | "api"           ← LLM 模式
  DATA_SOURCE: "mock" | "home_credit"   ← 数据源

输出:
  - 每轮 UCB 选择
  - 生成的特征
  - 评估结果 (IV/KS/PSI/missing_rate)
  - Critic 反馈
  - 预算消耗
  - 最终 delta_features 清单
"""

from __future__ import annotations

import argparse
import json
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.llm_client import create_llm_client, MockLLMClient
from src.agents.generator import GeneratorAgent
from src.agents.critic import CriticAgent
from src.core.budget_controller import BudgetController
from src.core.state_manager import StateManager
from src.core.orchestrator import Orchestrator
from src.memory.feature_registry import FeatureRegistry
from src.memory.trace_recorder import TraceRecorder
from src.pipeline.cache_manager import CacheManager
from src.pipeline.dsl_parser import DSLParser
from src.pipeline.evaluator import Evaluator
from src.routing.ucb_controller import UCBController
from src.utils.config import load_config
from src.utils.validator import DSLValidator
from scripts.mock_data_generator import generate_mock_data
from scripts.data_loader import load_home_credit_data, print_data_info


def setup_logging(config: dict, run_dir: Path | None = None) -> None:
    """配置日志系统。"""
    log_cfg = config.get("logging", {})
    if log_cfg:
        if run_dir:
            logs_dir = run_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_cfg.get("handlers", {}).get("file", {})["filename"] = str(
                logs_dir / "agent.log"
            )
        else:
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
        logging.config.dictConfig(log_cfg)


def validate_runtime_config(config: dict) -> None:
    """Fail fast for required production defaults."""
    llm_cfg = config.get("llm", {})
    if llm_cfg.get("mode") == "api":
        api_key = llm_cfg.get("api", {}).get("api_key", "")
        if not api_key:
            raise RuntimeError(
                "当前默认 LLM_MODE=api，但未设置 API Key "
                "(LLM_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY 均为空)。"
                "请设置真实 API Key；如需临时离线调试，请运行 --mode mock --data mock。"
            )


def load_data(config: dict) -> tuple:
    """根据配置加载数据源 (mock 或 Home Credit)。"""
    data_cfg = config.get("data", {})
    source = data_cfg.get("source", "mock")

    if source == "home_credit":
        hc_dir = data_cfg.get("home_credit_dir", "")
        if not hc_dir:
            hc_dir = PROJECT_ROOT.parent / "home credit default risk"
        hc_dir = Path(os.environ.get("HOME_CREDIT_DIR", str(hc_dir)))

        if not hc_dir.exists():
            raise FileNotFoundError(
                f"Home Credit 数据目录不存在: {hc_dir}。"
                "请设置 HOME_CREDIT_DIR，或显式运行 --data mock 做离线调试。"
            )

        print(f"\n[1/6] 加载 Home Credit 数据: {hc_dir}")
        output_path = PROJECT_ROOT / "data" / "home_credit_processed.csv"
        df, direction_map = load_home_credit_data(
            hc_dir, max_rows=None, output_path=output_path
        )
        print_data_info(df, direction_map)
        return df, direction_map

    else:
        return _load_mock_data(config)


def _load_mock_data(config: dict) -> tuple:
    """加载 Mock 数据。"""
    print("\n[1/6] 生成 Mock 仿真数据...")
    data_path = PROJECT_ROOT / "data" / "mock_features.csv"
    df = generate_mock_data(n_rows=1000, output_path=data_path)
    print(f"  -> 已生成 {len(df)} 行 x {len(df.columns)} 列")
    print(f"  -> 标签正样本率: {int(df['label'].sum()) / len(df):.1%}")
    # Mock 数据的 direction_map 已内置在 evaluator 中
    return df, None


def main():
    """主入口。"""
    parser = argparse.ArgumentParser(description="预算约束下的序贯特征挖掘 POC")
    parser.add_argument(
        "--mode", choices=["mock", "api"], default=None,
        help="LLM 模式 (覆盖 configs/base.env 中的设置)"
    )
    parser.add_argument(
        "--data", choices=["mock", "home_credit"], default=None,
        help="数据源 (覆盖 configs/base.env 中的设置)"
    )
    parser.add_argument(
        "--config", default=None,
        help="配置文件路径 (默认: configs/base.env)"
    )
    args = parser.parse_args()

    # ─── 加载配置 ───
    config_path = Path(args.config) if args.config else PROJECT_ROOT / "configs" / "base.env"
    schema_path = PROJECT_ROOT / "configs" / "dsl_schema.json"

    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # CLI 参数覆盖配置
    if args.mode:
        config.setdefault("llm", {})["mode"] = args.mode
    if args.data:
        config.setdefault("data", {})["source"] = args.data

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_base_dir = Path(config.get("run", {}).get("base_dir", "runs"))
    if not run_base_dir.is_absolute():
        run_base_dir = PROJECT_ROOT / run_base_dir
    run_dir = run_base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(config, run_dir=run_dir)
    logger = logging.getLogger(__name__)
    try:
        validate_runtime_config(config)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  预算约束下的序贯特征挖掘智能体系统 — POC")
    print("=" * 70)
    print(f"  运行 ID:   {run_id}")
    print(f"  运行目录: {run_dir}")
    print(f"  LLM 模式: {config.get('llm', {}).get('mode', 'mock')}")
    print(f"  数据源:   {config.get('data', {}).get('source', 'mock')}")

    # ─── 加载数据 ───
    mock_data, direction_map = load_data(config)

    # ─── 初始化组件 ───
    print("\n[2/6] 初始化系统组件...")

    # LLM Client (根据配置自动选择 Mock/API)
    llm_client = create_llm_client(config)

    # DSL Validator
    validator = DSLValidator(schema_path=schema_path, config=config)

    # UCB Controller
    ucb_cfg = config.get("ucb", {})
    ucb = UCBController(
        c=ucb_cfg.get("c", 1.5),
        c_decay=ucb_cfg.get("c_decay", 0.95),
        min_c=ucb_cfg.get("min_c", 0.3),
    )

    # Generator
    generator = GeneratorAgent(
        llm_client=llm_client,
        schema_path=schema_path,
        config=config,
    )

    # Critic
    critic = CriticAgent(llm_client=llm_client, config=config)

    # DSL Parser
    dsl_parser = DSLParser(validator=validator)

    # Evaluator (direction_map 用于 Home Credit 模式下的特征列匹配)
    evaluator = Evaluator(
        data=mock_data,
        label_col="label",
        config=config,
        direction_map=direction_map,
    )

    # Budget Controller
    budget_cfg = config.get("budget", {})
    eval_cfg = config.get("evaluator", {})
    budget = BudgetController(
        daily_limit=budget_cfg.get("daily_limit", 50),
        generator_cost=budget_cfg.get("generator_cost", 1),
        critic_cost=budget_cfg.get("critic_cost", 1),
        iv_threshold=eval_cfg.get("iv_threshold", 0.005),
        missing_rate_threshold=eval_cfg.get("missing_rate_threshold", 0.7),
    )

    # State Manager
    directions = config.get("directions", [])
    state_manager = StateManager(
        directions=directions,
        initial_budget=budget.daily_limit,
    )

    # Cache Manager
    cache = CacheManager(ttl_seconds=3600)

    # Trace Recorder
    trace_cfg = config.get("trace", {})
    trace = None
    if trace_cfg.get("enabled", True):
        trace = TraceRecorder(trace_dir=run_dir / "trace", run_id=run_id)

    # Feature Registry
    registry = FeatureRegistry(
        storage_path=run_dir / "features" / "feature_registry.json"
    )

    print(f"  -> UCB 方向数: {len(directions)}")
    print(f"  -> 每日预算: {budget.daily_limit}")
    print(f"  -> 每步生成 K: {config['generator']['k_features_per_step']}")

    # ─── 初始化 Orchestrator ───
    print("\n[3/6] 初始化 Orchestrator...")
    orch = Orchestrator(
        ucb=ucb,
        generator=generator,
        critic=critic,
        dsl_parser=dsl_parser,
        evaluator=evaluator,
        state_manager=state_manager,
        budget_controller=budget,
        cache_manager=cache,
        trace_recorder=trace,
        max_steps=budget_cfg.get("max_steps", 20),
        k_features_per_step=config["generator"]["k_features_per_step"],
    )

    # ─── 运行 ───
    print("\n[4/6] 启动序贯特征挖掘...")
    print("-" * 70)
    delta_features, final_state = orch.run()

    # ─── 注册高质量特征 ───
    print("\n[5/6] 注册发现的特征...")
    detail_by_id = {item["feature_id"]: item for item in orch.delta_feature_details}
    for feat in delta_features:
        registry.register(feat, metadata=detail_by_id.get(feat.feature_id))
    registry.save()
    print(f"  -> 已注册 {registry.size} 个特征")

    # ─── 打印摘要 ───
    print("\n[6/6] 运行摘要")
    orch.print_summary()

    # ─── 附加统计 ───
    print(f"\n方向分布:")
    for d, count in registry.direction_counts.items():
        print(f"  {d}: {count} 个特征")

    print(f"\n预算统计:")
    bsum = budget.summary()
    print(f"  上限: {bsum['daily_limit']}")
    print(f"  剩余: {bsum['remaining']}")
    print(f"  消耗: {bsum['total_consumed']}")
    if trace:
        print(f"\n追踪文件:")
        print(f"  事件流: {trace.events_path}")
        print(f"  特征血缘: {trace.lineage_path}")

    summary_path = run_dir / "summary.json"
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "mode": config.get("llm", {}).get("mode"),
        "data_source": config.get("data", {}).get("source"),
        "budget": bsum,
        "direction_counts": registry.direction_counts,
        "delta_features": orch.delta_feature_details,
        "trace_events": str(trace.events_path) if trace else None,
        "trace_lineage": str(trace.lineage_path) if trace else None,
        "feature_registry": str(registry.storage_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n运行摘要文件: {summary_path}")

    print("\nPOC 完成!")


if __name__ == "__main__":
    main()
