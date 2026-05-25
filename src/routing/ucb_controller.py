"""
src/routing/ucb_controller.py
=============================
模块职责：UCB (Upper Confidence Bound) 多臂老虎机控制器，用于方向选择。
I/O 契约：
  - 输入: list[DirectionStats] + step + c 系数
  - 输出: (选中的 direction: str, ucb_value: float)
核心公式:
  - UCB(k) = mu_k + c * sqrt(ln(t) / n_k)
  - 冷启动: n_k=0 时返回 float('inf') 强制探索
"""

from __future__ import annotations

import logging
import math

from src.utils.schemas import DirectionStats

logger = logging.getLogger(__name__)


class UCBController:
    """
    UCB 方向选择控制器。
    每次迭代根据各方向的平均收益与探索次数计算 UCB 值，
    选择 UCB 最大的方向作为下一轮探索目标。
    """

    def __init__(self, c: float = 1.5, c_decay: float = 0.95, min_c: float = 0.3):
        """
        Args:
            c: 初始探索系数 (exploration-exploitation trade-off)
            c_decay: 每步 c 的衰减因子
            min_c: c 的最小值
        """
        self.c = c
        self.c_decay = c_decay
        self.min_c = min_c

    def select(
        self, directions: list[DirectionStats], step: int
    ) -> tuple[str, float]:
        """
        选择下一轮探索方向。

        Args:
            directions: 所有方向的统计信息列表
            step: 当前步骤 (t)，用于 ln(t) 计算

        Returns:
            (selected_direction: str, ucb_value: float)
        """
        if not directions:
            raise ValueError("directions 列表不能为空")

        # 动态衰减 c 系数
        current_c = max(self.min_c, self.c * (self.c_decay ** step))

        best_direction = directions[0].direction
        best_ucb = -float("inf")
        t = max(1, step)

        ucb_values: dict[str, float] = {}

        for ds in directions:
            if ds.n_k == 0:
                # 冷启动: 未探索过的方向优先
                ucb = float("inf")
            else:
                # UCB(k) = mu_k + c * sqrt(ln(t) / n_k)
                exploration_bonus = current_c * math.sqrt(math.log(t) / ds.n_k)
                ucb = ds.mu_k + exploration_bonus

            ucb_values[ds.direction] = ucb

            if ucb > best_ucb:
                best_ucb = ucb
                best_direction = ds.direction

        logger.info(
            f"UCB 选择: direction={best_direction}, "
            f"ucb={best_ucb:.4f}, c={current_c:.3f}, "
            f"all_values={ {k: f'{v:.4f}' if v != float('inf') else 'inf' for k, v in ucb_values.items()} }"
        )

        return best_direction, best_ucb