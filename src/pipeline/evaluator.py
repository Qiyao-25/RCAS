"""
src/pipeline/evaluator.py
=========================
模块职责：特征评估器，使用 pandas 读取 Mock 数据并计算指标。
I/O 契约：
  - 输入: list[FeatureDSL] + pandas.DataFrame (mock 数据)
  - 输出: list[EvalResult]
"""

from __future__ import annotations

import logging
import random

import pandas as pd

from src.utils.metrics import MetricsConfig, evaluate_feature_metrics
from src.utils.schemas import EvalResult, FeatureDSL

logger = logging.getLogger(__name__)

# 方向 -> Mock 数据列名映射 (用于匹配生成特征与数据)
DIRECTION_COLUMN_MAP: dict[str, list[str]] = {
    "user_device_risk": [
        "FEAT_USER_DEVICE_RISK_FREQ_001",
        "FEAT_USER_DEVICE_RISK_AVG_002",
        "FEAT_USER_DEVICE_RISK_STD_003",
    ],
    "transaction_pattern": [
        "FEAT_TRANSACTION_PATTERN_SUM_001",
        "FEAT_TRANSACTION_PATTERN_MAX_002",
        "FEAT_TRANSACTION_PATTERN_RAT_003",
    ],
    "geolocation_anomaly": [
        "FEAT_GEOLOCATION_ANOMALY_FREQ_001",
        "FEAT_GEOLOCATION_ANOMALY_AVG_002",
        "FEAT_GEOLOCATION_ANOMALY_STD_003",
    ],
    "account_behavior": [
        "FEAT_ACCOUNT_BEHAVIOR_FREQ_001",
        "FEAT_ACCOUNT_BEHAVIOR_SUM_002",
        "FEAT_ACCOUNT_BEHAVIOR_RAT_003",
    ],
    "network_relation": [
        "FEAT_NETWORK_RELATION_FREQ_001",
        "FEAT_NETWORK_RELATION_MAX_002",
        "FEAT_NETWORK_RELATION_STD_003",
    ],
}


class Evaluator:
    """特征评估器。对每个 FeatureDSL 从 Mock 数据中提取对应列并计算指标。"""

    def __init__(
        self,
        data: pd.DataFrame,
        label_col: str = "label",
        config: dict | None = None,
        direction_map: dict[str, list[str]] | None = None,
    ):
        self.data = data
        self.label_col = label_col
        self.config = config or {}
        self._metrics_config = self._build_metrics_config()
        self._rng = random.Random(42)
        # 缓存已赋值的列映射 (feature_id -> 实际数据列名)
        self._col_assignments: dict[str, str] = {}
        # 外部方向映射 (用于 Home Credit 等真实数据模式)
        self._direction_map = direction_map

    def _build_metrics_config(self) -> MetricsConfig:
        ev_cfg = self.config.get("evaluator", {})
        return MetricsConfig(
            n_bins=ev_cfg.get("n_bins", 10),
            iv_threshold=ev_cfg.get("iv_threshold", 0.005),
            missing_rate_threshold=ev_cfg.get("missing_rate_threshold", 0.7),
            psi_threshold=ev_cfg.get("psi_threshold", 0.25),
        )

    def evaluate(self, dsls: list[FeatureDSL]) -> list[EvalResult]:
        results = []
        for dsl in dsls:
            feature_col = self._resolve_feature_column(dsl)
            if feature_col is None or feature_col not in self.data.columns:
                logger.warning(f"特征列不存在: {dsl.feature_id} -> {feature_col}")
                results.append(EvalResult(
                    feature_id=dsl.feature_id,
                    IV=0.0, KS=0.0, PSI=0.0, missing_rate=1.0,
                    status="failed",
                    reason=f"特征列 '{feature_col}' 不存在",
                ))
                continue

            try:
                metrics = evaluate_feature_metrics(
                    feature_data=self.data,
                    feature_col=feature_col,
                    label_col=self.label_col,
                    config=self._metrics_config,
                )
                result = EvalResult(
                    feature_id=dsl.feature_id,
                    IV=metrics["IV"],
                    KS=metrics["KS"],
                    KS_gain=metrics.get("KS_gain", 0.0),
                    PSI=metrics["PSI"],
                    missing_rate=metrics["missing_rate"],
                    status=metrics["status"],
                    reason=metrics.get("reason"),
                )
                results.append(result)
                logger.info(
                    f"评估: {dsl.feature_id} (列:{feature_col}) "
                    f"IV={result.IV:.4f} KS={result.KS:.4f} "
                    f"missing={result.missing_rate:.2%} status={result.status}"
                )
            except Exception as e:
                logger.error(f"评估异常: {dsl.feature_id} - {e}")
                results.append(EvalResult(
                    feature_id=dsl.feature_id,
                    IV=0.0, KS=0.0, PSI=0.0, missing_rate=1.0,
                    status="failed", reason=str(e),
                ))

        return results

    def resolved_column_for(self, feature_id: str) -> str | None:
        """Return the data column used for a generated feature, if assigned."""
        if feature_id in self._col_assignments:
            return self._col_assignments[feature_id]
        if feature_id in self.data.columns:
            return feature_id
        return None

    def _resolve_feature_column(self, dsl: FeatureDSL) -> str | None:
        """智能解析 DSL 对应的 DataFrame 列名。

        规则 (优先级):
        1. 若提供了外部 direction_map (真实数据模式) → 按方向随机选取
        2. 若 feature_id 精确匹配数据列名 → 直接返回
        3. 若已为该 feature_id 分配列 → 返回缓存值
        4. 根据方向在 DIRECTION_COLUMN_MAP 中随机选取未使用的列 (Mock 模式)
        """
        # ── 真实数据模式: 使用外部 direction_map ──
        if self._direction_map is not None:
            return self._resolve_from_direction_map(dsl)

        # 精确匹配
        if dsl.feature_id in self.data.columns:
            return dsl.feature_id

        # 已缓存赋值
        if dsl.feature_id in self._col_assignments:
            return self._col_assignments[dsl.feature_id]

        # 按方向分配 Mock 数据列
        candidates = DIRECTION_COLUMN_MAP.get(dsl.direction, [])
        available = [c for c in candidates if c in self.data.columns]

        if not available:
            return None

        # 尽量分配尚未使用的列
        used_cols = set(self._col_assignments.values())
        unused = [c for c in available if c not in used_cols]
        chosen = self._rng.choice(unused) if unused else self._rng.choice(available)

        self._col_assignments[dsl.feature_id] = chosen
        return chosen

    def _resolve_from_direction_map(self, dsl: FeatureDSL) -> str | None:
        """真实数据模式下的列解析：从 direction_map 中按方向随机选取。"""
        if dsl.feature_id in self._col_assignments:
            return self._col_assignments[dsl.feature_id]

        candidates = self._direction_map.get(dsl.direction, [])
        available = [c for c in candidates if c in self.data.columns]

        if not available:
            # 尝试在所有列中找
            logger.warning(
                f"方向 '{dsl.direction}' 无可用列，从全体列中随机选取"
            )
            available = [c for c in self.data.columns if c != self.label_col]
            if not available:
                return None

        used_cols = set(self._col_assignments.values())
        unused = [c for c in available if c not in used_cols]
        chosen = self._rng.choice(unused) if unused else self._rng.choice(available)

        self._col_assignments[dsl.feature_id] = chosen
        return chosen
