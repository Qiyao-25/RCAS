"""
src/pipeline/dsl_parser.py
==========================
模块职责：DSL 解析器，对 Generator 生成的 DSL 进行批量校验与早期拦截。
I/O 契约：
  - 输入: list[FeatureDSL] + DSLValidator + 配置
  - 输出: (valid_dsls: list[FeatureDSL], invalid_dsls: list[tuple[FeatureDSL, str]])
"""

from __future__ import annotations

import logging

from src.utils.schemas import FeatureDSL
from src.utils.validator import DSLValidator, early_intercept

logger = logging.getLogger(__name__)


class DSLParser:
    """
    DSL 解析与校验管道。
    对批量 DSL 执行:
      1. early_intercept (轻量预检)
      2. DSLValidator.validate (完整校验)
    """

    def __init__(self, validator: DSLValidator):
        self.validator = validator

    def parse(
        self, dsls: list[FeatureDSL]
    ) -> tuple[list[FeatureDSL], list[tuple[FeatureDSL, str]]]:
        """
        批量解析校验 DSL 列表。

        Args:
            dsls: 待校验的特征 DSL 列表

        Returns:
            (valid_dsls, invalid_dsls):
              - valid_dsls: 通过所有校验的有效特征
              - invalid_dsls: 未通过的 (FeatureDSL, error_message) 对
        """
        valid = []
        invalid = []

        for dsl in dsls:
            # 1. 早期拦截
            passed, msg = early_intercept(dsl)
            if not passed:
                logger.warning(f"DSL 早期拦截: {dsl.feature_id} - {msg}")
                invalid.append((dsl, msg))
                continue

            # 2. 完整校验
            passed, msg = self.validator.validate(dsl)
            if not passed:
                logger.warning(f"DSL 校验失败: {dsl.feature_id} - {msg}")
                invalid.append((dsl, msg))
                continue

            valid.append(dsl)
            logger.debug(f"DSL 校验通过: {dsl.feature_id}")

        logger.info(
            f"DSL 解析完成: valid={len(valid)}, invalid={len(invalid)}, total={len(dsls)}"
        )
        return valid, invalid