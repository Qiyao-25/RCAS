"""
src/core/state_manager.py
=========================
模块职责：系统状态管理器，负责维护和更新 DirectionStats 与 SystemState。
I/O 契约：
  - 输入: 评估结果 + 方向变化 (delta)
  - 输出: 更新后的 SystemState
"""

from __future__ import annotations

import logging
from datetime import datetime

from src.utils.schemas import DirectionStats, EvalResult, SystemState

logger = logging.getLogger(__name__)


class StateManager:
    """
    状态管理器。
    维护全局 SystemState，每次迭代后更新各方向的 mu_k, n_k, failure_patterns。
    """

    def __init__(self, directions: list[str], initial_budget: int = 50):
        """
        Args:
            directions: 初始方向列表
            initial_budget: 初始预算值
        """
        self.state = SystemState(
            step=0,
            budget_remaining=initial_budget,
            directions=[
                DirectionStats(
                    direction=d,
                    n_k=0,
                    mu_k=0.0,
                    sigma_k=0.0,
                    last_update=0,
                )
                for d in directions
            ],
        )

    def get_directions(self) -> list[DirectionStats]:
        """获取当前方向统计列表。"""
        return self.state.directions

    def get_state(self) -> SystemState:
        """获取当前系统状态。"""
        return self.state

    def update(
        self,
        direction: str,
        results: list[EvalResult],
        failure_axis: str | None = None,
    ) -> None:
        """
        根据评估结果更新方向统计。

        Args:
            direction: 本轮探索的方向
            results: 该方向本轮评估结果列表
            failure_axis: 若有失败模式，传入失败维度
        """
        # 更新方向统计
        ds = next(
            (d for d in self.state.directions if d.direction == direction), None
        )
        if ds is None:
            logger.warning(f"未找到方向: {direction}")
            return

        # 计算有效特征比例作为收益信号
        success_count = sum(1 for r in results if r.status == "success")
        valid_count = len(results)
        if valid_count > 0:
            reward = success_count / valid_count
        else:
            reward = 0.0

        # 更新 mu_k (指数移动平均)
        ds.n_k += valid_count
        if ds.n_k == valid_count:
            ds.mu_k = reward
        else:
            alpha = 0.3  # EMA 平滑因子
            ds.mu_k = alpha * reward + (1 - alpha) * ds.mu_k

        # 更新 sigma_k (简单方差估计)
        if ds.n_k > 1:
            old_sigma = ds.sigma_k
            ds.sigma_k = max(0.0, old_sigma * 0.9 + abs(reward - ds.mu_k) * 0.1)

        ds.last_update = self.state.step

        # 更新失败模式统计
        if failure_axis:
            self.state.failure_patterns[failure_axis] = (
                self.state.failure_patterns.get(failure_axis, 0) + 1
            )

        # 更新已探索特征列表
        for r in results:
            if r.feature_id not in self.state.explored_features:
                self.state.explored_features.append(r.feature_id)

        logger.info(
            f"StateManager 更新: direction={direction}, "
            f"n_k={ds.n_k}, mu_k={ds.mu_k:.4f}, "
            f"success={success_count}/{valid_count}"
        )

    def step_forward(self, budget_consumed: int = 0) -> None:
        """进入下一步，更新 step 和 budget。"""
        self.state.step += 1
        self.state.budget_remaining -= budget_consumed
        self.state.timestamp = datetime.now().isoformat()