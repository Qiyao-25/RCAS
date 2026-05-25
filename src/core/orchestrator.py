"""
src/core/orchestrator.py
========================
模块职责：主编排器，串联所有模块实现预算约束下的序贯特征挖掘循环。
I/O 契约：
  - 输入: 所有子模块实例 + 配置
  - 输出: delta_features 清单 + 系统状态快照
主循环:
  while budget > 0 and step < max_steps:
    1. UCB 选择方向
    2. Generator 生成 K=3 个 DSL
    3. DSL Parser 校验 + 防泄漏检查
    4. Evaluator 计算指标
    5. 过滤低效特征 → Critic 生成修正建议
    6. State Manager 更新 mu_k, n_k, failure_patterns
    7. 记录日志, step += 1
"""

from __future__ import annotations

import logging

from src.agents.generator import GeneratorAgent
from src.agents.critic import CriticAgent
from src.core.budget_controller import BudgetController
from src.core.state_manager import StateManager
from src.memory.trace_recorder import TraceRecorder
from src.pipeline.cache_manager import CacheManager
from src.pipeline.dsl_parser import DSLParser
from src.pipeline.evaluator import Evaluator
from src.routing.ucb_controller import UCBController
from src.utils.schemas import CriticOutput, EvalResult, FeatureDSL, SystemState

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    预算约束下的序贯特征挖掘主编排器。
    """

    def __init__(
        self,
        ucb: UCBController,
        generator: GeneratorAgent,
        critic: CriticAgent,
        dsl_parser: DSLParser,
        evaluator: Evaluator,
        state_manager: StateManager,
        budget_controller: BudgetController,
        cache_manager: CacheManager | None = None,
        trace_recorder: TraceRecorder | None = None,
        max_steps: int = 20,
        k_features_per_step: int = 3,
    ):
        self.ucb = ucb
        self.generator = generator
        self.critic = critic
        self.dsl_parser = dsl_parser
        self.evaluator = evaluator
        self.state_manager = state_manager
        self.budget = budget_controller
        self.cache = cache_manager or CacheManager()
        self.trace = trace_recorder
        self.max_steps = max_steps
        self.k = k_features_per_step

        # 运行日志
        self.run_log: list[dict] = []
        # 累积 Critic 反馈
        self.critic_history: list[CriticOutput] = []
        # 最终产出的高质量特征
        self.delta_features: list[FeatureDSL] = []
        self.delta_feature_details: list[dict] = []

    def run(self) -> tuple[list[FeatureDSL], SystemState]:
        """
        执行主循环。

        Returns:
            (delta_features: list[FeatureDSL], final_state: SystemState)
        """
        logger.info("=" * 60)
        logger.info("预算约束序贯特征挖掘系统启动")
        logger.info(f"初始预算: {self.budget.daily_limit}, 最大步数: {self.max_steps}")
        logger.info("=" * 60)
        if self.trace:
            self.trace.start_run(
                {
                    "budget_daily_limit": self.budget.daily_limit,
                    "max_steps": self.max_steps,
                    "k_features_per_step": self.k,
                    "directions": [
                        d.direction for d in self.state_manager.get_directions()
                    ],
                }
            )

        while not self.budget.is_exhausted() and self.state_manager.state.step < self.max_steps:
            step = self.state_manager.state.step
            logger.info(f"\n{'─' * 50}")
            logger.info(f"步骤 {step + 1}/{self.max_steps} | 剩余预算: {self.budget.remaining}")

            # ─── 1. UCB 选择方向 ───
            directions = self.state_manager.get_directions()
            selected_direction, ucb_value = self.ucb.select(directions, step + 1)
            if self.trace:
                self.trace.record_step_selected(
                    step=step + 1,
                    direction=selected_direction,
                    ucb_value=ucb_value if ucb_value != float("inf") else "inf",
                    direction_stats=[d.__dict__.copy() for d in directions],
                )

            step_log = {
                "step": step + 1,
                "direction": selected_direction,
                "ucb_value": ucb_value if ucb_value != float("inf") else "inf",
                "generated_features": [],
                "eval_results": [],
                "critic_outputs": [],
                "budget_consumed": 0,
            }

            # ─── 2. Generator 生成特征 ───
            can_gen, gen_cost = self.budget.can_generate()
            if not can_gen:
                logger.warning("预算耗尽，退出循环")
                break
            self.budget.consume(gen_cost)
            step_log["budget_consumed"] += gen_cost

            dsls = self.generator.generate(
                direction=selected_direction,
                critic_feedback=self.critic_history if self.critic_history else None,
                k=self.k,
            )
            if self.trace:
                self.trace.record_generated(
                    step=step + 1,
                    direction=selected_direction,
                    dsls=dsls,
                    critic_context=self.critic_history,
                )
            step_log["generated_features"] = [
                {
                    "feature_id": dsl.feature_id,
                    "business_logic": dsl.business_logic,
                    "definition": dsl.definition,
                }
                for dsl in dsls
            ]
            logger.info(f"Generator 生成 {len(dsls)} 个特征: {step_log['generated_features']}")

            # ─── 3. DSL Parser 校验 ───
            valid_dsls, invalid_dsls = self.dsl_parser.parse(dsls)
            if self.trace:
                self.trace.record_validation(step + 1, valid_dsls, invalid_dsls)
            for dsl, err in invalid_dsls:
                logger.warning(f"  [INVALID] {dsl.feature_id}: {err}")

            if not valid_dsls:
                logger.warning("无有效 DSL，进入下一步")
                self.state_manager.step_forward(budget_consumed=step_log["budget_consumed"])
                self.run_log.append(step_log)
                continue

            # ─── 4. Evaluator 评估 ───
            results = self.evaluator.evaluate(valid_dsls)
            if self.trace:
                for dsl, result in zip(valid_dsls, results):
                    self.trace.record_evaluation(
                        step=step + 1,
                        dsl=dsl,
                        result=result,
                        resolved_column=self.evaluator.resolved_column_for(result.feature_id),
                    )
            step_log["eval_results"] = [
                {
                    "feature_id": r.feature_id,
                    "IV": r.IV,
                    "KS": r.KS,
                    "PSI": r.PSI,
                    "missing_rate": r.missing_rate,
                    "status": r.status,
                    "reason": r.reason,
                }
                for r in results
            ]

            # ─── 5. 过滤低效特征 → Critic ───
            should_critic, critic_cost, low_perf_results = self.budget.can_critic(results)
            if should_critic and low_perf_results:
                # 找出低效特征对应的 DSL
                low_perf_ids = {r.feature_id for r in low_perf_results}
                low_perf_dsls = [
                    (dsl, r)
                    for dsl, r in zip(valid_dsls, results)
                    if r.feature_id in low_perf_ids
                ]
                if low_perf_dsls:
                    self.budget.consume(critic_cost)
                    step_log["budget_consumed"] += critic_cost
                    critiques = self.critic.critique(low_perf_dsls)
                    self.critic_history.extend(critiques)
                    result_by_id = {r.feature_id: r for r in low_perf_results}
                    if self.trace:
                        for critique in critiques:
                            self.trace.record_critic(
                                step=step + 1,
                                critique=critique,
                                result=result_by_id.get(critique.feature_id),
                            )
                    step_log["critic_outputs"] = [
                        {
                            "feature_id": c.feature_id,
                            "axis": c.modification_axis,
                            "failure_reason": c.failure_reason,
                            "suggestion": c.param_suggestion,
                            "next_direction_hint": c.next_direction_hint,
                            "confidence": c.confidence,
                        }
                        for c in critiques
                    ]
                    logger.info(f"Critic 反馈 {len(critiques)} 条: {step_log['critic_outputs']}")

            # ─── 6. State Manager 更新 ───
            # 统计失败维度
            failure_axis = None
            if step_log["critic_outputs"]:
                failure_axis = step_log["critic_outputs"][0]["axis"]

            self.state_manager.update(
                direction=selected_direction,
                results=results,
                failure_axis=failure_axis,
            )
            self.state_manager.step_forward(budget_consumed=step_log["budget_consumed"])

            # ─── 7. 收集高质量特征 ───
            for dsl, result in zip(valid_dsls, results):
                if result.status == "success":
                    self.delta_features.append(dsl)
                    self.delta_feature_details.append(
                        {
                            "feature_id": dsl.feature_id,
                            "direction": dsl.direction,
                            "business_logic": dsl.business_logic,
                            "definition": dsl.definition,
                            "resolved_column": self.evaluator.resolved_column_for(
                                result.feature_id
                            ),
                            "metrics": {
                                "IV": result.IV,
                                "KS": result.KS,
                                "KS_gain": result.KS_gain,
                                "PSI": result.PSI,
                                "missing_rate": result.missing_rate,
                                "status": result.status,
                                "reason": result.reason,
                            },
                            "effectiveness": _build_effectiveness_summary(result),
                        }
                    )
                self.cache.put(result.feature_id, result)

            self.run_log.append(step_log)
            logger.info(
                f"步骤 {step + 1} 完成 | 方向: {selected_direction} | "
                f"成功: {sum(1 for r in results if r.status == 'success')}/{len(results)} | "
                f"消耗: {step_log['budget_consumed']}"
            )

        # ─── 最终输出 ───
        logger.info("\n" + "=" * 60)
        logger.info("特征挖掘完成")
        logger.info(f"总步数: {self.state_manager.state.step}")
        logger.info(f"剩余预算: {self.budget.remaining}")
        logger.info(f"总消耗: {self.budget.total_consumed}")
        logger.info(f"有效特征: {len(self.delta_features)}")
        logger.info(f"Critic 反馈累计: {len(self.critic_history)}")
        logger.info("=" * 60)
        if self.trace:
            self.trace.finish_run(
                {
                    "steps": self.state_manager.state.step,
                    "budget_remaining": self.budget.remaining,
                    "budget_consumed": self.budget.total_consumed,
                    "delta_features": len(self.delta_features),
                    "critic_feedback_count": len(self.critic_history),
                }
            )

        return self.delta_features, self.state_manager.get_state()

    def print_summary(self) -> None:
        """打印运行摘要。"""
        print("\n" + "=" * 70)
        print("  预算约束序贯特征挖掘系统 — 运行摘要")
        print("=" * 70)

        for log in self.run_log:
            print(f"\n--- 步骤 {log['step']} ---")
            print(f"  方向: {log['direction']} (UCB={log['ucb_value']})")
            print("  生成特征:")
            for gf in log["generated_features"]:
                print(f"    - {gf['feature_id']}: {gf['business_logic']}")
            print(f"  消耗预算: {log['budget_consumed']}")
            if log["eval_results"]:
                print("  评估结果:")
                for er in log["eval_results"]:
                    print(f"    - {er['feature_id']}: IV={er['IV']:.4f} KS={er['KS']:.4f} [{er['status']}]")
            if log["critic_outputs"]:
                print("  Critic 反馈:")
                for co in log["critic_outputs"]:
                    print(
                        f"    - {co['feature_id']}: {co['axis']} | "
                        f"{co['suggestion']} (conf={co['confidence']:.2f})"
                    )

        state = self.state_manager.get_state()
        print(f"\n{'─' * 70}")
        print(f"  最终状态: step={state.step}, budget_remaining={state.budget_remaining}")
        print(f"  delta_features 数量: {len(self.delta_features)}")
        if self.delta_features:
            print("  高质量特征列表:")
            for detail in self.delta_feature_details[:10]:
                metrics = detail["metrics"]
                print(f"    - {detail['feature_id']}")
                print(f"      业务含义: {detail['business_logic']}")
                print(f"      数据列: {detail['resolved_column']}")
                print(
                    "      有效性: "
                    f"IV={metrics['IV']:.4f}, KS={metrics['KS']:.4f}, "
                    f"PSI={metrics['PSI']:.4f}, 缺失率={metrics['missing_rate']:.2%}"
                )
                print(f"      说明: {detail['effectiveness']}")
        print(f"  failure_patterns: {state.failure_patterns}")
        print("=" * 70)


def _build_effectiveness_summary(result: EvalResult) -> str:
    if result.status != "success":
        return f"未进入有效特征池，原因: {result.reason or result.status}"
    return (
        "通过阈值筛选；"
        f"IV={result.IV:.4f} 表示区分度，KS={result.KS:.4f} 表示正负样本分离度，"
        f"PSI={result.PSI:.4f} 表示稳定性风险，缺失率={result.missing_rate:.2%}。"
    )
