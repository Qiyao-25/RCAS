"""
src/utils/validator.py
======================
模块职责：DSL Schema 校验、时间窗口防泄漏检查、合规标签检查。
I/O 契约：
  - 输入: FeatureDSL 对象 + 配置
  - 输出: (bool, str) — 是否通过校验 + 错误信息
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.utils.schemas import FeatureDSL


class DSLValidator:
    """
    DSL 校验器，负责三层校验:
      1. Schema 结构校验 (基于 dsl_schema.json)
      2. 时间窗口防泄漏检查
      3. 合规标签检查
    """

    def __init__(self, schema_path: str | Path, config: dict | None = None):
        """
        Args:
            schema_path: dsl_schema.json 文件路径
            config: 全局配置字典 (含 compliance 段落)
        """
        self.schema_path = Path(schema_path)
        self.config = config or {}
        self._schema = self._load_schema()

    def _load_schema(self) -> dict:
        """加载 JSON Schema。"""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema 文件不存在: {self.schema_path}")
        with open(self.schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        directions = self.config.get("directions", [])
        if directions:
            schema.setdefault("properties", {}).setdefault("direction", {})[
                "enum"
            ] = directions
        return schema

    def validate(self, dsl: FeatureDSL) -> tuple[bool, str]:
        """
        综合校验入口。

        Returns:
            (passed: bool, message: str)
        """
        # 1. JSON Schema 结构校验
        passed, msg = self._validate_schema(dsl)
        if not passed:
            return False, msg

        # 2. 时间窗口防泄漏
        passed, msg = self._check_time_window(dsl)
        if not passed:
            return False, msg

        # 3. 合规标签
        passed, msg = self._check_compliance(dsl)
        if not passed:
            return False, msg

        return True, "OK"

    def _validate_schema(self, dsl: FeatureDSL) -> tuple[bool, str]:
        """基于 JSON Schema 的结构校验 (简化版，Pydantic 已做基础校验)。"""
        properties = self._schema.get("properties", {})

        # 检查 direction 是否在枚举范围内
        direction_enum = properties.get("direction", {}).get("enum", [])
        if direction_enum and dsl.direction not in direction_enum:
            return (
                False,
                f"direction '{dsl.direction}' 不在允许范围 {direction_enum}",
            )

        # 检查 definition.aggregation
        agg_enum = (
            properties.get("definition", {})
            .get("properties", {})
            .get("aggregation", {})
            .get("enum", [])
        )
        if agg_enum and dsl.definition.get("aggregation") not in agg_enum:
            return (
                False,
                f"aggregation '{dsl.definition.get('aggregation')}' 不在允许范围",
            )

        # 检查 definition.transformation
        trans_enum = (
            properties.get("definition", {})
            .get("properties", {})
            .get("transformation", {})
            .get("enum", [])
        )
        if trans_enum and dsl.definition.get("transformation") not in trans_enum:
            return (
                False,
                f"transformation '{dsl.definition.get('transformation')}' 不在允许范围",
            )

        # 检查 time_window 格式
        tw_pattern = (
            properties.get("definition", {})
            .get("properties", {})
            .get("time_window", {})
            .get("pattern", r"^\d+[dhm]$")
        )
        if not re.match(tw_pattern, str(dsl.definition.get("time_window", ""))):
            return False, f"time_window 格式不合法: {dsl.definition.get('time_window')}"

        return True, "OK"

    def _check_time_window(self, dsl: FeatureDSL) -> tuple[bool, str]:
        """时间窗口防泄漏检查。

        规则:
          - 窗口不能超过 365d (防止长时间窗口引入未来信息)
          - 窗口不能为 0 或负数
        """
        tw = str(dsl.definition.get("time_window", ""))
        match = re.match(r"^(\d+)([dhm])$", tw)
        if not match:
            return False, f"无法解析 time_window: {tw}"

        value = int(match.group(1))
        unit = match.group(2)

        if value <= 0:
            return False, f"time_window 必须为正数: {tw}"

        # 转换为天数检查
        if unit == "d":
            days = value
        elif unit == "h":
            days = value / 24
        else:  # m (minutes)
            days = value / 1440

        if days > 365:
            return (
                False,
                f"time_window 超过365天 ({days:.1f}d)，存在数据泄漏风险",
            )

        return True, "OK"

    def _check_compliance(self, dsl: FeatureDSL) -> tuple[bool, str]:
        """合规标签检查。

        规则:
          - 必须包含 required_tags 中的全部标签
          - 不得包含 forbidden_fields 中的敏感字段名
          - filter 中不得出现敏感字段
        """
        compliance = self.config.get("compliance", {})
        required_tags = compliance.get("required_tags", ["PII_FREE", "EXPLAINABLE"])
        forbidden_fields = compliance.get("forbidden_fields", [])

        # 检查必需标签
        missing_tags = set(required_tags) - set(dsl.compliance_tags)
        if missing_tags:
            return False, f"缺少必需合规标签: {missing_tags}"

        # 检查敏感字段是否出现在 definition 中
        definition_str = json.dumps(dsl.definition, ensure_ascii=False).lower()
        for field in forbidden_fields:
            if field.lower() in definition_str:
                return False, f"definition 包含禁止字段: {field}"

        # 检查 business_logic 中是否含敏感字段
        bl_lower = dsl.business_logic.lower()
        for field in forbidden_fields:
            if field.lower() in bl_lower:
                return False, f"business_logic 包含禁止字段: {field}"

        return True, "OK"


def early_intercept(dsl: FeatureDSL) -> tuple[bool, str]:
    """
    轻量预拦截：在进入完整 Evaluator 之前快速过滤。
    检查: 表名是否为空、filter 是否可能导致全表扫描等。
    """
    definition = dsl.definition

    if not definition.get("source_table", "").strip():
        return False, "source_table 为空"

    # 检查 filter 是否包含明显的时间泄漏标记
    filter_str = str(definition.get("filter", ""))
    future_keywords = ["tomorrow", "future", "next_day", "t+1", "t+2"]
    for kw in future_keywords:
        if kw.lower() in filter_str.lower():
            return False, f"filter 含未来数据关键词: {kw}"

    return True, "OK"
