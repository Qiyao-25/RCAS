"""
src/pipeline/cache_manager.py
=============================
模块职责：特征评估缓存管理器，避免重复评估相同特征。
I/O 契约：
  - 输入: feature_id
  - 输出: EvalResult | None (缓存命中 / 未命中)
特性：
  - 基于 feature_id 的简单 KV 缓存
  - 内存缓存，不依赖外部存储
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from src.utils.schemas import EvalResult

logger = logging.getLogger(__name__)


class CacheManager:
    """
    特征评估缓存管理器。
    以 feature_id 为 key，缓存 EvalResult，避免重复计算。
    """

    def __init__(self, ttl_seconds: int = 3600):
        """
        Args:
            ttl_seconds: 缓存过期时间 (秒)，默认 1 小时
        """
        self._cache: dict[str, tuple[EvalResult, datetime]] = {}
        self.ttl = timedelta(seconds=ttl_seconds)

    def get(self, feature_id: str) -> EvalResult | None:
        """
        查询缓存。

        Returns:
            EvalResult | None: 缓存命中返回结果，否则 None
        """
        if feature_id not in self._cache:
            return None

        result, timestamp = self._cache[feature_id]
        if datetime.now() - timestamp > self.ttl:
            logger.debug(f"缓存过期: {feature_id}")
            del self._cache[feature_id]
            return None

        logger.debug(f"缓存命中: {feature_id}")
        return result

    def put(self, feature_id: str, result: EvalResult) -> None:
        """写入缓存。"""
        self._cache[feature_id] = (result, datetime.now())
        logger.debug(f"缓存写入: {feature_id}")

    def clear(self) -> None:
        """清空缓存。"""
        self._cache.clear()
        logger.info("缓存已清空")

    @property
    def size(self) -> int:
        return len(self._cache)