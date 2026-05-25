"""
scripts/mock_data_generator.py
==============================
模块职责：生成 1000 行仿真流水+标签数据，用于 POC 演示。
I/O 契约：
  - 输入: 无 (独立运行) 或通过函数调用
  - 输出: pandas.DataFrame (写入 data/mock_features.parquet)
特性：
  - 生成 5 个方向的特征列 (每个方向 3-4 列)
  - 生成二分类标签列 (label: 0/1)
  - 某些列故意设置高缺失率或低区分度，用于测试早期拦截逻辑
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEED = 42


def generate_mock_data(
    n_rows: int = 1000,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """生成仿真 Mock 数据集。"""
    rng = np.random.default_rng(SEED)

    # 生成标签
    label = rng.binomial(1, 0.15, n_rows).astype(np.int32)

    data: dict[str, np.ndarray] = {"label": label}

    # ─── user_device_risk 方向 ───
    base_vals = rng.normal(0, 1, n_rows)
    data["FEAT_USER_DEVICE_RISK_FREQ_001"] = np.where(
        label == 1, base_vals + 2.0, base_vals
    )
    data["FEAT_USER_DEVICE_RISK_AVG_002"] = rng.exponential(1.0, n_rows) + label * 1.5
    high_miss = rng.normal(0, 1, n_rows).astype(float)
    miss_mask = rng.random(n_rows) > 0.25
    high_miss[miss_mask] = np.nan
    data["FEAT_USER_DEVICE_RISK_STD_003"] = high_miss

    # ─── transaction_pattern 方向 ───
    data["FEAT_TRANSACTION_PATTERN_SUM_001"] = (
        rng.lognormal(3, 1, n_rows) + label * rng.lognormal(4, 0.5, n_rows)
    )
    data["FEAT_TRANSACTION_PATTERN_MAX_002"] = rng.poisson(5, n_rows) + label * 3
    data["FEAT_TRANSACTION_PATTERN_RAT_003"] = rng.normal(0, 0.1, n_rows)

    # ─── geolocation_anomaly 方向 ───
    data["FEAT_GEOLOCATION_ANOMALY_FREQ_001"] = (
        rng.poisson(2, n_rows) + label * rng.poisson(4, n_rows)
    )
    data["FEAT_GEOLOCATION_ANOMALY_AVG_002"] = rng.uniform(0, 100, n_rows)
    geo_miss = rng.normal(10, 5, n_rows).astype(float)
    geo_miss[rng.random(n_rows) > 0.6] = np.nan
    data["FEAT_GEOLOCATION_ANOMALY_STD_003"] = geo_miss

    # ─── account_behavior 方向 ───
    data["FEAT_ACCOUNT_BEHAVIOR_FREQ_001"] = (
        rng.poisson(10, n_rows) - label * rng.poisson(5, n_rows)
    )
    data["FEAT_ACCOUNT_BEHAVIOR_SUM_002"] = rng.beta(2, 5, n_rows) * 100 + label * 20
    data["FEAT_ACCOUNT_BEHAVIOR_RAT_003"] = rng.normal(1, 0.3, n_rows) + label * 0.6

    # ─── network_relation 方向 ───
    data["FEAT_NETWORK_RELATION_FREQ_001"] = rng.poisson(3, n_rows) + label * 2
    data["FEAT_NETWORK_RELATION_MAX_002"] = rng.gamma(2, 2, n_rows) + label * 5
    data["FEAT_NETWORK_RELATION_STD_003"] = np.ones(n_rows) * 5.0

    df = pd.DataFrame(data)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path = output_path.with_suffix(".csv")
        df.to_csv(str(csv_path), index=False)
        logger.info(f"Mock 数据已保存: {csv_path} ({n_rows} 行, {len(df.columns)} 列)")

    return df


def print_data_summary(df: pd.DataFrame) -> None:
    """打印数据集摘要。"""
    print("\n" + "=" * 60)
    print("  Mock 数据集摘要")
    print("=" * 60)
    print(f"  总行数: {len(df)}")
    print(f"  总列数: {len(df.columns)}")
    print(f"  标签分布: 正样本={int(df['label'].sum())} ({df['label'].mean():.1%})")

    print("\n  各列缺失率:")
    for col in df.columns:
        if col == "label":
            continue
        miss_rate = df[col].isna().mean()
        flag = " [!HIGH]" if miss_rate > 0.5 else ""
        print(f"    {col:<45} {miss_rate:.1%}{flag}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    output = Path(__file__).resolve().parent.parent / "data" / "mock_features.parquet"
    df = generate_mock_data(n_rows=1000, output_path=output)
    print_data_summary(df)