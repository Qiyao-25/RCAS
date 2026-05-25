"""
src/memory/feature_registry.py
==============================
模块职责：特征注册中心，维护已探索特征的索引与去重。
I/O 契约：
  - 输入: FeatureDSL / feature_id
  - 输出: 注册/查询/去重结果
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from src.utils.schemas import FeatureDSL

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """
    特征注册中心。
    用于去重检查、持久化特征清单、按方向索引特征。
    """

    def __init__(self, storage_path: str | Path | None = None):
        """
        Args:
            storage_path: 可选持久化路径 (JSON)
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self._registry: dict[str, dict] = {}  # feature_id -> FeatureDSL 序列化
        self._direction_index: dict[str, list[str]] = {}

    def register(self, dsl: FeatureDSL, metadata: dict | None = None) -> bool:
        """
        注册特征。

        Returns:
            True 如果是新特征，False 如果已存在 (去重)
        """
        if dsl.feature_id in self._registry:
            logger.debug(f"特征已存在，跳过注册: {dsl.feature_id}")
            return False

        record = dsl.model_dump()
        if metadata:
            record["metadata"] = metadata
        self._registry[dsl.feature_id] = record

        if dsl.direction not in self._direction_index:
            self._direction_index[dsl.direction] = []
        self._direction_index[dsl.direction].append(dsl.feature_id)

        logger.debug(f"特征已注册: {dsl.feature_id}")
        return True

    def exists(self, feature_id: str) -> bool:
        """检查特征是否已注册。"""
        return feature_id in self._registry

    def get(self, feature_id: str) -> FeatureDSL | None:
        """按 ID 获取特征。"""
        data = self._registry.get(feature_id)
        if data:
            return FeatureDSL(**data)
        return None

    def get_by_direction(self, direction: str) -> list[FeatureDSL]:
        """按方向获取所有特征。"""
        feat_ids = self._direction_index.get(direction, [])
        return [FeatureDSL(**self._registry[fid]) for fid in feat_ids if fid in self._registry]

    def list_all(self) -> list[str]:
        """列出所有已注册特征 ID。"""
        return list(self._registry.keys())

    @property
    def size(self) -> int:
        return len(self._registry)

    @property
    def direction_counts(self) -> dict[str, int]:
        """各方向特征数量。"""
        return {d: len(ids) for d, ids in self._direction_index.items()}

    def save(self) -> None:
        """持久化到 JSON 文件。"""
        if not self.storage_path:
            logger.warning("未配置 storage_path，跳过持久化")
            return

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "registry": self._registry,
            "direction_index": {
                d: ids for d, ids in self._direction_index.items()
            },
            "saved_at": datetime.now().isoformat(),
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"注册中心已持久化: {self.storage_path} ({len(self._registry)} 特征)")

    def load(self) -> bool:
        """从 JSON 文件加载。"""
        if not self.storage_path or not self.storage_path.exists():
            return False

        with open(self.storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._registry = data.get("registry", {})
        self._direction_index = data.get("direction_index", {})
        logger.info(f"注册中心已加载: {len(self._registry)} 特征")
        return True
