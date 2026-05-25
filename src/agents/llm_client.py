"""
src/agents/llm_client.py
========================
模块职责：OpenAI 兼容 API 封装 + MockLLMClient 降级方案。
I/O 契约：
  - 输入: messages, model, temperature, max_tokens, response_format
  - 输出: dict (解析后的 JSON 响应)
特性：
  - 自动重试 (最多 3 次)
  - 超时保护 (30s)
  - JSON 解析失败时降级返回确定性 fallback
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """LLM 客户端抽象基类。"""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: str | None = None,
    ) -> dict:
        """发送对话请求，返回解析后的 JSON dict。"""
        ...


class OpenAIClient(BaseLLMClient):
    """基于 OpenAI SDK 的真实 LLM 客户端。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        allow_mock_fallback: bool = False,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.allow_mock_fallback = allow_mock_fallback
        self._client = None

        if api_key:
            self._init_client()

    def _init_client(self):
        """延迟初始化 OpenAI 客户端。"""
        try:
            from openai import OpenAI

            kwargs: dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        except ImportError:
            logger.error("openai 库未安装，API 模式不可用")
            self._client = None

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: str | None = None,
    ) -> dict:
        if self._client is None:
            if self.allow_mock_fallback:
                logger.warning("OpenAI client 不可用，按配置降级到 MockLLMClient")
                return MockLLMClient().chat(
                    messages, model, temperature, max_tokens, response_format
                )
            raise RuntimeError(
                "OpenAI client 不可用。请安装 openai、设置 API Key，或显式设置 "
                "LLM_ALLOW_MOCK_FALLBACK=true 后再使用 Mock 降级。"
            )

        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format == "json_object":
                    kwargs["response_format"] = {"type": "json_object"}

                response = self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("LLM 返回空内容")

                return json.loads(content)

            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败 (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                else:
                    if self.allow_mock_fallback:
                        logger.error("LLM JSON 解析全部重试失败，按配置降级到 Mock")
                        return MockLLMClient().chat(
                            messages, model, temperature, max_tokens, response_format
                        )
                    raise

            except Exception as e:
                logger.error(f"LLM 调用异常 (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                else:
                    if self.allow_mock_fallback:
                        logger.error("LLM 调用全部重试失败，按配置降级到 Mock")
                        return MockLLMClient().chat(
                            messages, model, temperature, max_tokens, response_format
                        )
                    raise

        return {}


def create_llm_client(config: dict) -> BaseLLMClient:
    """
    根据配置创建 LLM 客户端。

    配置结构 (由 configs/base.env 加载后的 llm 段):
      mode: "mock" | "api"
      api:
        base_url: "..."
        api_key: "..."  # 空则从环境变量 DEEPSEEK_API_KEY 读取
        model: "..."
        timeout: 60
        max_retries: 3
    """
    import os

    llm_cfg = config.get("llm", {})
    mode = llm_cfg.get("mode", "mock")
    api_cfg = llm_cfg.get("api", {})

    if mode == "mock":
        logger.info("LLM 模式: Mock (离线测试)")
        return MockLLMClient()

    if mode == "api":
        api_key = (
            api_cfg.get("api_key", "")
            or os.environ.get("LLM_API_KEY", "")
            or os.environ.get("DEEPSEEK_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        base_url = api_cfg.get("base_url", "https://api.deepseek.com/v1")
        timeout = api_cfg.get("timeout", 60)
        max_retries = api_cfg.get("max_retries", 3)
        allow_mock_fallback = api_cfg.get("allow_mock_fallback", False)

        if not api_key:
            raise RuntimeError(
                "当前默认 LLM_MODE=api，但未设置 API Key "
                "(LLM_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY 均为空)。"
                "请设置真实 API Key；如需临时离线调试，请运行 --mode mock。"
            )

        logger.info(
            f"LLM 模式: API (provider={api_cfg.get('provider', 'unknown')}, "
            f"base_url={base_url})"
        )
        return OpenAIClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            allow_mock_fallback=allow_mock_fallback,
        )

    logger.warning(f"未知 LLM 模式 '{mode}'，降级到 Mock")
    return MockLLMClient()


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM 客户端，无需真实 API Key 即可运行。
    提供确定性 + 随机化的 fallback 输出，保证 POC 可跑通。
    """

    _FEATURE_PATTERNS = [
        {
            "prefix": "FREQ",
            "agg": "COUNT",
            "trans": "LOG",
            "biz": "统计{field}在{window}内的频次",
        },
        {
            "prefix": "SUM",
            "agg": "SUM",
            "trans": "MINMAX",
            "biz": "汇总{field}在{window}内的总量",
        },
        {
            "prefix": "AVG",
            "agg": "AVG",
            "trans": "STANDARDIZE",
            "biz": "计算{field}在{window}内的均值",
        },
        {
            "prefix": "MAX",
            "agg": "MAX",
            "trans": "RAW",
            "biz": "提取{field}在{window}内的最大值",
        },
        {
            "prefix": "STD",
            "agg": "STDDEV",
            "trans": "BIN",
            "biz": "计算{field}在{window}内的波动性",
        },
        {
            "prefix": "RAT",
            "agg": "RATIO",
            "trans": "WOE",
            "biz": "计算{field}在{window}内的比率特征",
        },
    ]

    _TABLES = {
        "user_device_risk": ["user_device_log", "device_fingerprint"],
        "transaction_pattern": ["transaction_detail", "payment_order"],
        "geolocation_anomaly": ["gps_track", "ip_location_log"],
        "account_behavior": ["account_login_log", "user_profile"],
        "network_relation": ["social_graph", "transfer_network"],
    }

    _WINDOWS = ["1d", "3d", "7d", "14d", "30d", "24h", "12h", "60m"]

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._call_count = 0

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: str | None = None,
    ) -> dict:
        self._call_count += 1

        # 从最后一条 user message 推断意图
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = str(m.get("content", ""))
                break

        if "generate" in user_msg.lower() or "特征" in user_msg:
            return self._mock_generate_response(user_msg)
        elif "critic" in user_msg.lower() or "修正" in user_msg or "批评" in user_msg:
            return self._mock_critic_response(user_msg)
        else:
            return self._mock_generate_response(user_msg)

    def _mock_generate_response(self, prompt: str) -> dict:
        """确定性 + 随机化生成 K=3 个特征 DSL。"""
        # 尝试从 prompt 中提取 direction
        direction = "user_device_risk"
        marker_match = re.search(r"CURRENT_DIRECTION:\s*([a-z0-9_]+)", prompt)
        if marker_match:
            direction = marker_match.group(1)
        else:
            for d in self._TABLES:
                if d in prompt:
                    direction = d
                    break

        tables = self._TABLES.get(direction, ["unknown_table"])
        features = []

        for i in range(3):
            pattern = self._rng.choice(self._FEATURE_PATTERNS)
            table = self._rng.choice(tables)
            window = self._rng.choice(self._WINDOWS)
            feat_id = (
                f"FEAT_{direction.upper()}_{pattern['agg']}_{table.upper()}_"
                f"{window.upper()}_{pattern['trans']}_{i + 1:02d}"
            )

            features.append(
                {
                    "feature_id": feat_id,
                    "direction": direction,
                    "definition": {
                        "source_table": table,
                        "time_window": window,
                        "filter": f"status = 'success' AND amount > 0",
                        "aggregation": pattern["agg"],
                        "transformation": pattern["trans"],
                    },
                    "business_logic": pattern["biz"].format(
                        field="amount", window=window
                    ),
                    "compliance_tags": ["PII_FREE", "EXPLAINABLE"],
                }
            )

        return {"features": features}

    def _mock_critic_response(self, prompt: str) -> dict:
        """确定性 Critic 反馈。"""
        axes = [
            "time_window",
            "filter_condition",
            "aggregation",
            "transformation",
            "direction_shift",
        ]
        suggestions = [
            "缩短时间窗口为 3d",
            "增加 amount > 100 的过滤条件",
            "改用 COUNT_DISTINCT 聚合",
            "改用 WOE 变换",
            "转向 account_behavior 方向",
        ]
        idx = self._rng.randint(0, len(axes) - 1)

        return {
            "feature_id": "FEAT_AUTO",
            "failure_reason": "IV 值过低，特征区分度不足",
            "modification_axis": axes[idx],
            "param_suggestion": suggestions[idx],
            "next_direction_hint": self._rng.choice(list(self._TABLES.keys())),
            "confidence": round(self._rng.uniform(0.3, 0.9), 2),
        }
