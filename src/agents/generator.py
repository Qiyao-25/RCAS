"""
src/agents/generator.py
=======================
模块职责：特征生成 Agent，基于当前方向 + Critic 反馈 + DSL Schema 生成特征 DSL。
I/O 契约：
  - 输入: direction, critic_feedback (可选), dsl_schema, 配置
  - 输出: list[FeatureDSL] (每步 K 个)
特性：
  - Prompt 模板注入: 方向、防泄漏规则、Schema、历史反馈
  - response_format=json_object 强制
  - LLM 不可用时自动降级到 deterministic_fallback
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.agents.llm_client import BaseLLMClient
from src.utils.schemas import FeatureDSL, CriticOutput

logger = logging.getLogger(__name__)

GENERATOR_SYSTEM_PROMPT = """你是一个风控特征挖掘专家。你需要生成风险特征定义(FeatureDSL)。

## 规则
1. 每个特征的 feature_id 格式为 FEAT_{{DIRECTION}}_{{READABLE_MEANING}}_{{WINDOW}}_{{INDEX}}，必须使用英文大写、数字和下划线，名称要能看出业务含义
2. time_window 格式为 数字+单位，如 7d, 24h, 30m
3. filter 必须是合法 SQL WHERE 子句，不得包含未来数据（如 t+1, tomorrow）
4. aggregation 必须是以下之一: COUNT, SUM, AVG, MAX, MIN, STDDEV, COUNT_DISTINCT, RATIO, FREQUENCY
5. transformation 必须是以下之一: RAW, LOG, WOE, BIN, STANDARDIZE, MINMAX, QUANTILE_BIN
6. compliance_tags 必须包含 PII_FREE 和 EXPLAINABLE
7. 不得使用敏感字段: id_number, phone, name, email_raw
8. 每个特征必须有不同的业务含义

## DSL Schema
{schema}

## 当前搜索方向
CURRENT_DIRECTION: {direction}
{direction_context}

## 历史反馈
{critic_context}

## 防泄漏规则
- 时间窗口不超过 365 天
- 禁止使用 t+1, tomorrow, future 等未来数据关键词
- 禁止在 filter 中使用标签字段（如 label, target）

请生成 {k} 个特征定义，以 JSON 格式返回：{{"features": [...]}}"""


class GeneratorAgent:
    """
    特征生成 Agent。
    使用 LLM 或 fallback 生成 FeatureDSL 列表。
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        schema_path: str | Path,
        config: dict | None = None,
    ):
        """
        Args:
            llm_client: LLM 客户端
            schema_path: dsl_schema.json 路径
            config: 全局配置
        """
        self.llm = llm_client
        self.schema_path = Path(schema_path)
        self.config = config or {}
        self._schema_str = self._load_schema_str()

    def _load_schema_str(self) -> str:
        if self.schema_path.exists():
            with open(self.schema_path, "r", encoding="utf-8") as f:
                return f.read()
        return "{}"

    def generate(
        self,
        direction: str,
        critic_feedback: list[CriticOutput] | None = None,
        k: int = 3,
    ) -> list[FeatureDSL]:
        """
        生成 K 个特征定义。

        Args:
            direction: 当前搜索方向
            critic_feedback: 历史 Critic 反馈列表
            k: 生成特征数量

        Returns:
            list[FeatureDSL]: 特征 DSL 对象列表
        """
        # 构建 Critic 上下文
        critic_context = "无历史反馈"
        if critic_feedback and len(critic_feedback) > 0:
            recent = critic_feedback[-3:]  # 只取最近 3 条
            parts = []
            for c in recent:
                parts.append(
                    f"- [{c.feature_id}] {c.failure_reason} | "
                    f"修正轴: {c.modification_axis} | "
                    f"建议: {c.param_suggestion}"
                )
            critic_context = "\n".join(parts)

        # 构建方向上下文
        direction_context = self._build_direction_context(direction)

        # 构建 prompt
        user_prompt = GENERATOR_SYSTEM_PROMPT.format(
            schema=self._schema_str,
            direction=direction,
            direction_context=direction_context,
            critic_context=critic_context,
            k=k,
        )

        messages = [
            {"role": "system", "content": "你是一个专业的风控特征挖掘系统。"},
            {"role": "user", "content": user_prompt},
        ]

        gen_cfg = self.config.get("generator", {})
        model = gen_cfg.get("model", "gpt-4o-mini")
        temperature = gen_cfg.get("temperature", 0.7)
        max_tokens = gen_cfg.get("max_tokens", 2048)

        try:
            result = self.llm.chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format="json_object",
            )

            features_raw = result.get("features", [])
            if not features_raw:
                logger.warning("LLM 返回空 features，使用 fallback")
                return self._deterministic_fallback(direction, k)

            dsls = []
            for fdict in features_raw[:k]:
                try:
                    fdict["feature_id"] = self._normalize_feature_id(
                        raw_id=fdict.get("feature_id"),
                        direction=direction,
                        definition=fdict.get("definition", {}),
                        index=len(dsls),
                    )
                    fdict.setdefault("compliance_tags", ["PII_FREE", "EXPLAINABLE"])
                    fdict.setdefault("business_logic", "自动生成特征")
                    dsl = FeatureDSL(**fdict)
                    dsls.append(dsl)
                except Exception as e:
                    logger.warning(f"FeatureDSL 校验失败: {e}, 数据: {json.dumps(fdict, ensure_ascii=False)[:200]}")

            if not dsls:
                logger.warning("所有生成特征校验失败，使用 fallback")
                return self._deterministic_fallback(direction, k)

            return dsls

        except Exception as e:
            logger.error(f"Generator 异常: {e}，使用 fallback")
            return self._deterministic_fallback(direction, k)

    def _build_direction_context(self, direction: str) -> str:
        """构建方向上下文信息。"""
        contexts = {
            "user_device_risk": "聚焦用户设备维度的风险特征，如表: user_device_log, device_fingerprint。常用字段: device_id, os_type, root_status, emulator_flag",
            "transaction_pattern": "聚焦交易行为模式，如表: transaction_detail, payment_order。常用字段: amount, channel, merchant_id, time",
            "geolocation_anomaly": "聚焦地理位置异常，如表: gps_track, ip_location_log。常用字段: latitude, longitude, ip, city, country",
            "account_behavior": "聚焦账户行为模式，如表: account_login_log, user_profile。常用字段: login_time, login_ip, register_days, kyc_level",
            "network_relation": "聚焦网络关系特征，如表: social_graph, transfer_network。常用字段: src_user, dst_user, relation_type, weight",
        }
        return contexts.get(direction, f"方向: {direction}，自行探索相关表和字段。")

    def _normalize_feature_id(
        self,
        raw_id: str | None,
        direction: str,
        definition: dict,
        index: int,
    ) -> str:
        """Build a readable, schema-safe feature id."""
        candidate = raw_id or ""
        if candidate.startswith("FEAT_") and re.match(r"^FEAT_[A-Z0-9_]+$", candidate):
            return candidate

        source = str(definition.get("source_table", "SOURCE"))
        agg = str(definition.get("aggregation", "AGG"))
        window = str(definition.get("time_window", "WINDOW"))
        trans = str(definition.get("transformation", "RAW"))
        parts = [
            "FEAT",
            direction,
            agg,
            source,
            window,
            trans,
            f"{index + 1:02d}",
        ]
        readable = "_".join(parts).upper()
        readable = re.sub(r"[^A-Z0-9_]+", "_", readable)
        readable = re.sub(r"_+", "_", readable).strip("_")
        return readable

    def _deterministic_fallback(self, direction: str, k: int = 3) -> list[FeatureDSL]:
        """
        确定性 fallback：不依赖 LLM 也能生成有效特征。
        基于规则模板生成。
        """
        templates = [
            {
                "feat_suffix": "FREQ",
                "agg": "COUNT",
                "trans": "LOG",
                "biz": "统计用户设备活跃频次",
            },
            {
                "feat_suffix": "SUM",
                "agg": "SUM",
                "trans": "MINMAX",
                "biz": "汇总交易金额总量",
            },
            {
                "feat_suffix": "AVG",
                "agg": "AVG",
                "trans": "STANDARDIZE",
                "biz": "计算平均交易金额",
            },
            {
                "feat_suffix": "MAX",
                "agg": "MAX",
                "trans": "RAW",
                "biz": "提取单笔最大交易金额",
            },
            {
                "feat_suffix": "STD",
                "agg": "STDDEV",
                "trans": "BIN",
                "biz": "计算交易金额波动性",
            },
            {
                "feat_suffix": "RAT",
                "agg": "RATIO",
                "trans": "WOE",
                "biz": "计算高风险交易占比",
            },
        ]

        tables = {
            "user_device_risk": "user_device_log",
            "transaction_pattern": "transaction_detail",
            "geolocation_anomaly": "gps_track",
            "account_behavior": "account_login_log",
            "network_relation": "transfer_network",
        }

        table = tables.get(direction, "unknown_table")
        dsls = []

        for i in range(min(k, len(templates))):
            t = templates[i]
            feat_id = self._normalize_feature_id(
                raw_id=None,
                direction=direction,
                definition={
                    "source_table": table,
                    "time_window": "7d",
                    "aggregation": t["agg"],
                    "transformation": t["trans"],
                },
                index=i,
            )
            dsl = FeatureDSL(
                feature_id=feat_id,
                direction=direction,
                definition={
                    "source_table": table,
                    "time_window": "7d",
                    "filter": "status = 'success' AND amount > 0",
                    "aggregation": t["agg"],
                    "transformation": t["trans"],
                },
                business_logic=t["biz"],
                compliance_tags=["PII_FREE", "EXPLAINABLE"],
            )
            dsls.append(dsl)

        return dsls[:k]
