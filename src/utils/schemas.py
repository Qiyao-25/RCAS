"""
src/utils/schemas.py
====================
模块职责：定义系统中所有数据契约模型 (使用 dataclasses + 手动校验)。
I/O 契约：所有模块间传递的数据必须基于此处定义的模型进行校验。

模型清单：
  - FeatureDSL: 特征定义 DSL
  - EvalResult: 评估结果
  - CriticOutput: Critic 反馈输出
  - DirectionStats: 方向统计信息
  - SystemState: 系统全局状态快照
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Any


def _validate_feature_id(fid: str) -> None:
    if not re.match(r"^FEAT_[A-Z0-9_]+$", fid):
        raise ValueError(f"feature_id 格式不合法: {fid}，需匹配 ^FEAT_[A-Z0-9_]+$")


def _validate_definition(d: dict) -> None:
    required = {"source_table", "time_window", "filter", "aggregation", "transformation"}
    missing = required - set(d.keys())
    if missing:
        raise ValueError(f"definition 缺少必填字段: {missing}")


@dataclass
class FeatureDSL:
    """
    特征定义 DSL 模型。
    由 Generator 生成，经 DSL Parser 校验后进入 Evaluator 管道。
    """
    feature_id: str
    direction: str
    definition: dict
    business_logic: str
    compliance_tags: list[str]

    def __post_init__(self):
        _validate_feature_id(self.feature_id)
        _validate_definition(self.definition)
        if not self.business_logic:
            raise ValueError("business_logic 不能为空")
        if not self.compliance_tags:
            raise ValueError("compliance_tags 不能为空")

    def model_dump(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "direction": self.direction,
            "definition": self.definition,
            "business_logic": self.business_logic,
            "compliance_tags": self.compliance_tags,
        }


@dataclass
class EvalResult:
    """
    特征评估结果模型。
    由 Evaluator 计算后返回，用于预算决策与 Critic 触发。
    """
    feature_id: str
    IV: float
    KS: float
    KS_gain: float = 0.0
    PSI: float = 0.0
    missing_rate: float = 0.0
    status: Literal["success", "skipped", "failed"] = "success"
    reason: str | None = None

    def __post_init__(self):
        if self.IV < 0:
            raise ValueError(f"IV 必须 >= 0, got {self.IV}")
        if self.KS < 0 or self.KS > 1:
            raise ValueError(f"KS 必须在 [0,1] 区间, got {self.KS}")
        if self.PSI < 0:
            raise ValueError(f"PSI 必须 >= 0, got {self.PSI}")
        if self.missing_rate < 0 or self.missing_rate > 1:
            raise ValueError(f"missing_rate 必须在 [0,1] 区间, got {self.missing_rate}")


@dataclass
class CriticOutput:
    """
    Critic 反馈输出模型。
    对低效特征给出修正建议，驱动下一轮生成。
    """
    feature_id: str
    failure_reason: str
    modification_axis: Literal[
        "time_window", "filter_condition", "aggregation",
        "transformation", "direction_shift",
    ]
    param_suggestion: str
    next_direction_hint: str = ""
    confidence: float = 0.5

    def __post_init__(self):
        if not self.failure_reason:
            raise ValueError("failure_reason 不能为空")
        if not self.param_suggestion:
            raise ValueError("param_suggestion 不能为空")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"confidence 必须在 [0,1] 区间, got {self.confidence}")


@dataclass
class DirectionStats:
    """
    方向统计信息模型。
    由 State Manager 维护，用于 UCB 选路。
    """
    direction: str
    n_k: int = 0
    mu_k: float = 0.0
    sigma_k: float = 0.0
    last_update: int = 0

    def __post_init__(self):
        if self.n_k < 0:
            raise ValueError(f"n_k 必须 >= 0, got {self.n_k}")
        if self.sigma_k < 0:
            raise ValueError(f"sigma_k 必须 >= 0, got {self.sigma_k}")


@dataclass
class SystemState:
    """
    系统全局状态快照。
    由 State Manager 管理，每次循环后更新。
    """
    step: int = 0
    budget_remaining: int = 50
    directions: list[DirectionStats] = field(default_factory=list)
    explored_features: list[str] = field(default_factory=list)
    failure_patterns: dict[str, int] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_summary(self) -> dict:
        """生成可读摘要。"""
        return {
            "step": self.step,
            "budget_remaining": self.budget_remaining,
            "explored_count": len(self.explored_features),
            "top_directions": sorted(
                [(d.direction, round(d.mu_k, 4), d.n_k) for d in self.directions],
                key=lambda x: x[1],
                reverse=True,
            )[:3],
            "top_failures": sorted(
                self.failure_patterns.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:3],
        }
