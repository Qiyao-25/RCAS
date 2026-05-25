"""
src/core/budget_controller.py
=============================
模块职责：预算控制器，管理每日预算消耗与拦截决策。
I/O 契约：
  - 输入: 操作类型 (generator/critic) + EvalResult 列表
  - 输出: (can_proceed: bool, budget_consumed: int)
规则：
  - 每次 Generator 调用消耗 1，Critic 调用消耗 1
  - IV < 0.005 或 missing_rate > 0.7 直接标记 skipped，不消耗 Critic 预算
  - 预算耗尽时返回 False
"""

from __future__ import annotations

import logging
import time

from src.utils.schemas import EvalResult

logger = logging.getLogger(__name__)


class BudgetController:
    """
    预算控制器。
    跟踪剩余预算并根据规则决定是否允许继续操作。
    """

    def __init__(
        self,
        daily_limit: int = 50,
        generator_cost: int = 1,
        critic_cost: int = 1,
        iv_threshold: float = 0.005,
        missing_rate_threshold: float = 0.7,
        max_runtime_seconds: int = 1800,
        max_llm_calls: int = 20,
        max_features_evaluated: int = 80,
    ):
        """
        Args:
            daily_limit: 每日预算上限
            generator_cost: Generator 调用消耗
            critic_cost: Critic 调用消耗
            iv_threshold: IV 下限阈值
            missing_rate_threshold: 缺失率上限阈值
        """
        self.daily_limit = daily_limit
        self.generator_cost = generator_cost
        self.critic_cost = critic_cost
        self.iv_threshold = iv_threshold
        self.missing_rate_threshold = missing_rate_threshold
        self.max_runtime_seconds = max_runtime_seconds
        self.max_llm_calls = max_llm_calls
        self.max_features_evaluated = max_features_evaluated
        self.remaining = daily_limit
        self.total_consumed = 0
        self.started_at = time.monotonic()
        self.llm_calls = 0
        self.features_evaluated = 0

    def can_generate(self) -> tuple[bool, int]:
        """
        检查是否有足够预算进行 Generator 调用。

        Returns:
            (can_proceed: bool, cost: int)
        """
        if self._resource_exhausted():
            logger.warning(f"资源预算已耗尽: {self.exhaustion_reason()}")
            return False, 0
        if self.remaining >= self.generator_cost and self.llm_calls < self.max_llm_calls:
            return True, self.generator_cost
        logger.warning("预算不足，无法调用 Generator")
        return False, 0

    def can_critic(self, results: list[EvalResult]) -> tuple[bool, int, list[EvalResult]]:
        """
        检查是否需要/能够调用 Critic。

        规则:
          - 所有有效特征都通过 (status=success + 指标达标) → 不需要 Critic
          - IV < iv_threshold 或 missing_rate > missing_rate_threshold →
            标记为 skipped，不消耗 Critic 预算
          - 有低效特征 + 预算充足 → 调用 Critic

        Args:
            results: 本轮评估结果列表

        Returns:
            (should_call: bool, cost: int, low_perf: list[EvalResult])
        """
        # 标记应跳过的特征
        low_perf = []
        for r in results:
            if r.status == "failed":
                continue  # failed 不需要 critic
            if r.status == "skipped":
                low_perf.append(r)
            elif r.IV < self.iv_threshold or r.missing_rate > self.missing_rate_threshold:
                low_perf.append(r)

        if not low_perf:
            return False, 0, []

        if self._resource_exhausted():
            return False, 0, []

        if self.remaining >= self.critic_cost and self.llm_calls < self.max_llm_calls:
            return True, self.critic_cost, low_perf

        logger.warning("预算不足，无法调用 Critic (需要修复低效特征)")
        return False, 0, []

    def consume(self, amount: int) -> bool:
        """
        消耗预算。

        Returns:
            True 如果消耗成功，False 如果预算不足
        """
        if amount <= 0:
            return True
        if self.remaining < amount:
            logger.warning(f"预算不足: remaining={self.remaining}, need={amount}")
            return False
        self.remaining -= amount
        self.total_consumed += amount
        logger.info(
            f"预算消耗: {amount}, remaining={self.remaining}/{self.daily_limit}"
        )
        return True

    def record_llm_call(self, count: int = 1) -> None:
        self.llm_calls += count

    def record_features_evaluated(self, count: int) -> None:
        self.features_evaluated += count

    def is_exhausted(self) -> bool:
        """预算是否耗尽。"""
        return self.remaining <= 0 or self._resource_exhausted()

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def exhaustion_reason(self) -> str:
        if self.remaining <= 0:
            return "unit_budget_exhausted"
        if self.elapsed_seconds() >= self.max_runtime_seconds:
            return "runtime_budget_exhausted"
        if self.llm_calls >= self.max_llm_calls:
            return "llm_call_budget_exhausted"
        if self.features_evaluated >= self.max_features_evaluated:
            return "evaluation_budget_exhausted"
        return "not_exhausted"

    def _resource_exhausted(self) -> bool:
        return (
            self.elapsed_seconds() >= self.max_runtime_seconds
            or self.llm_calls >= self.max_llm_calls
            or self.features_evaluated >= self.max_features_evaluated
        )

    def summary(self) -> dict:
        """预算使用摘要。"""
        return {
            "daily_limit": self.daily_limit,
            "remaining": self.remaining,
            "total_consumed": self.total_consumed,
            "elapsed_seconds": round(self.elapsed_seconds(), 2),
            "max_runtime_seconds": self.max_runtime_seconds,
            "llm_calls": self.llm_calls,
            "max_llm_calls": self.max_llm_calls,
            "features_evaluated": self.features_evaluated,
            "max_features_evaluated": self.max_features_evaluated,
            "exhaustion_reason": self.exhaustion_reason(),
            "exhausted": self.is_exhausted(),
        }
