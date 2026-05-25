"""
src/agents/critic.py
====================
模块职责：特征批评 Agent，对低效特征给出修正建议。
I/O 契约：
  - 输入: 低效特征 EvalResult 列表 + 对应 FeatureDSL 列表 + 分箱统计 + 业务先验
  - 输出: list[CriticOutput]
特性：
  - Prompt 注入: 低效指标、分箱统计、业务模板
  - LLM 不可用时降级到 deterministic_fallback
"""

from __future__ import annotations

import json
import logging
import random

from src.agents.llm_client import BaseLLMClient
from src.utils.schemas import CriticOutput, EvalResult, FeatureDSL

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """你是一个风控特征质量评审专家。请分析低效特征并给出具体修正建议。

## 评审维度
- time_window: 时间窗口不合适
- filter_condition: 过滤条件可优化
- aggregation: 聚合函数选择不当
- transformation: 变换方式需调整
- direction_shift: 建议切换到其他方向

## 业务先验模板
- 交易类特征: 时效性高，推荐 1d-7d 窗口，关注金额分布和频次
- 设备类特征: 稳定性重要，推荐 7d-30d 窗口，关注设备切换和异常标记
- 地理类特征: 稀疏性高，推荐较长窗口，关注跳跃距离和跨境行为
- 行为类特征: 周期性明显，推荐 7d-14d 窗口，关注时序模式
- 网络类特征: 图结构重要，推荐聚合邻居属性

## 低效特征列表
{features_context}

请为每个低效特征给出修正建议，返回 JSON 格式：{{"critiques": [...]}}"""


class CriticAgent:
    """
    特征批评 Agent。
    对低效特征分析后给出参数修正方向。
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        config: dict | None = None,
    ):
        self.llm = llm_client
        self.config = config or {}
        self._rng = random.Random(42)

    def critique(
        self,
        low_perf_features: list[tuple[FeatureDSL, EvalResult]],
    ) -> list[CriticOutput]:
        """
        对低效特征列表进行评审。

        Args:
            low_perf_features: (FeatureDSL, EvalResult) 对列表

        Returns:
            list[CriticOutput]: 修正建议列表
        """
        if not low_perf_features:
            return []

        # 构建特征上下文
        feat_strs = []
        for dsl, result in low_perf_features:
            feat_strs.append(
                json.dumps(
                    {
                        "feature_id": dsl.feature_id,
                        "direction": dsl.direction,
                        "definition": dsl.definition,
                        "IV": result.IV,
                        "KS": result.KS,
                        "missing_rate": result.missing_rate,
                        "status": result.status,
                        "reason": result.reason,
                    },
                    ensure_ascii=False,
                )
            )
        features_context = "\n".join(feat_strs)

        messages = [
            {"role": "system", "content": "你是一个风控特征质量评审专家。"},
            {
                "role": "user",
                "content": CRITIC_SYSTEM_PROMPT.format(features_context=features_context),
            },
        ]

        critic_cfg = self.config.get("critic", {})
        model = critic_cfg.get("model", "gpt-4o-mini")
        temperature = critic_cfg.get("temperature", 0.3)
        max_tokens = critic_cfg.get("max_tokens", 1024)

        try:
            result = self.llm.chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format="json_object",
            )

            critiques_raw = result.get("critiques", [])
            outputs = []
            for cdict in critiques_raw:
                try:
                    cdict.setdefault("feature_id", "FEAT_UNKNOWN")
                    cdict.setdefault("failure_reason", "特征效果不佳")
                    cdict.setdefault("modification_axis", "aggregation")
                    cdict.setdefault("param_suggestion", "改用其他聚合方式")
                    cdict.setdefault("next_direction_hint", "")
                    cdict.setdefault("confidence", 0.5)
                    co = CriticOutput(**cdict)
                    outputs.append(co)
                except Exception as e:
                    logger.warning(f"CriticOutput 解析失败: {e}")

            if not outputs:
                return self._deterministic_fallback(low_perf_features)

            return outputs

        except Exception as e:
            logger.error(f"Critic 异常: {e}，使用 fallback")
            return self._deterministic_fallback(low_perf_features)

    def _deterministic_fallback(
        self,
        low_perf_features: list[tuple[FeatureDSL, EvalResult]],
    ) -> list[CriticOutput]:
        """确定性 Critic fallback。"""
        axes = [
            "time_window",
            "filter_condition",
            "aggregation",
            "transformation",
            "direction_shift",
        ]
        suggestions_map = {
            "time_window": "建议缩短时间窗口以提高时效性",
            "filter_condition": "建议增加有效样本过滤条件",
            "aggregation": "建议使用 COUNT_DISTINCT 或 RATIO 聚合",
            "transformation": "建议使用 WOE 或 LOG 变换提高区分度",
            "direction_shift": "建议尝试其他方向探索",
        }

        outputs = []
        for dsl, result in low_perf_features:
            axis = self._rng.choice(axes)
            outputs.append(
                CriticOutput(
                    feature_id=dsl.feature_id,
                    failure_reason=f"IV={result.IV:.6f}, KS={result.KS:.4f}, 区分度不足",
                    modification_axis=axis,
                    param_suggestion=suggestions_map[axis],
                    next_direction_hint=self._rng.choice(
                        self.config.get(
                            "directions",
                            [
                                "user_device_risk",
                                "transaction_pattern",
                                "geolocation_anomaly",
                                "account_behavior",
                                "network_relation",
                            ],
                        )
                    ),
                    confidence=round(self._rng.uniform(0.3, 0.7), 2),
                )
            )
        return outputs
