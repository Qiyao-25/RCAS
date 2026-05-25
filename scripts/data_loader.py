"""
scripts/data_loader.py
======================
模块职责：加载 Home Credit Default Risk 数据集，预处理为特征宽表。
I/O 契约：
  - 输入: application_train.csv 路径
  - 输出: (pandas.DataFrame, direction_map: dict)
特性：
  - 自动数值/类别分离
  - 缺失值填充 (median/mode)
  - 类别变量 Label Encoding
  - 按语义分组映射到 5 个挖掘方向
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ============================================================
# 方向映射规则: 根据列名语义将 Home Credit 特征分配到 5 个方向
# ============================================================
DIRECTION_RULES = {
    "account_behavior": [
        # 客户基本信息与行为
        "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
        "DAYS_LAST_PHONE_CHANGE", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
        "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
        "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
        "OWN_CAR_AGE", "NAME_TYPE_SUITE",
    ],
    "transaction_pattern": [
        # 金额与信贷相关
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
    ],
    "user_device_risk": [
        # 客户标记/标志位
        "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "FLAG_MOBIL", "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL",
        "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_4",
        "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_6", "FLAG_DOCUMENT_7",
        "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9", "FLAG_DOCUMENT_10",
        "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13",
        "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16",
        "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19",
        "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21",
        "CODE_GENDER", "NAME_CONTRACT_TYPE",
    ],
    "geolocation_anomaly": [
        # 地理位置相关
        "REGION_POPULATION_RELATIVE", "REGION_RATING_CLIENT",
        "REGION_RATING_CLIENT_W_CITY",
        "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY",
        "LIVE_CITY_NOT_WORK_CITY",
    ],
    "network_relation": [
        # 社交圈/外部数据
        "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "APARTMENTS_AVG", "BASEMENTAREA_AVG",
        "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BUILD_AVG",
        "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG",
        "FLOORSMAX_AVG", "FLOORSMIN_AVG",
        "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG",
        "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAREA_AVG",
        "TOTALAREA_MODE", "APARTMENTS_MODE", "BASEMENTAREA_MODE",
        "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BUILD_MODE",
        "COMMONAREA_MODE", "ELEVATORS_MODE", "ENTRANCES_MODE",
        "FLOORSMAX_MODE", "FLOORSMIN_MODE",
        "LANDAREA_MODE", "LIVINGAPARTMENTS_MODE", "LIVINGAREA_MODE",
        "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAREA_MODE",
        "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
        "FONDKAPREMONT_MODE",
    ],
}

# 不需要的特征 (ID、日期、非信息列)
EXCLUDED_COLUMNS = [
    "SK_ID_CURR", "TARGET",
    "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START",
]


def load_home_credit_data(
    data_dir: str | Path,
    max_rows: int | None = None,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    加载并预处理 Home Credit 数据。

    Args:
        data_dir: 包含 application_train.csv 的目录
        max_rows: 限制读取行数 (None=全部)
        output_path: 可选，保存处理后的数据到 CSV

    Returns:
        (df: pd.DataFrame, direction_map: dict[str, list[str]])
        df 包含 'label' 列 (原 TARGET)，其他列为预处理后的特征
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "application_train.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"训练数据不存在: {train_path}")

    logger.info(f"加载数据: {train_path}")
    df = pd.read_csv(train_path, nrows=max_rows)
    logger.info(f"原始数据: {df.shape[0]} 行 × {df.shape[1]} 列")

    # ─── 1. 提取标签 ───
    label = df["TARGET"].copy()
    df = df.drop(columns=["TARGET"])

    # ─── 2. 排除非特征列 ───
    drop_cols = [c for c in EXCLUDED_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols)

    # ─── 3. 类别变量 Label Encoding ───
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        mask = df[col].notna()
        df.loc[mask, col] = le.fit_transform(df.loc[mask, col].astype(str))
        df[col] = pd.to_numeric(df[col], errors="coerce")
        encoders[col] = le
    logger.info(f"类别编码: {len(cat_cols)} 列")

    # ─── 4. 缺失值填充 (中位数) ───
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    for col in num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    logger.info(f"缺失值填充: {len(num_cols)} 数值列 (中位数)")

    # ─── 5. 添加标签列 ───
    df = df.copy()
    df["label"] = label.values

    # ─── 6. 构建方向映射 ───
    direction_map: dict[str, list[str]] = {}
    assigned = set()
    for direction, patterns in DIRECTION_RULES.items():
        matched = []
        for col in df.columns:
            if col == "label":
                continue
            if col in assigned:
                continue
            if col in patterns:
                matched.append(col)
                assigned.add(col)
        if matched:
            direction_map[direction] = matched

    # 未分配的特征归入 "account_behavior"
    unassigned = [c for c in df.columns if c != "label" and c not in assigned]
    if unassigned:
        if "account_behavior" not in direction_map:
            direction_map["account_behavior"] = []
        direction_map["account_behavior"].extend(unassigned)
    logger.info(f"方向分配: { {k: len(v) for k, v in direction_map.items()} }")

    # ─── 7. 保存 ───
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(output_path), index=False)
        logger.info(f"处理后数据已保存: {output_path}")

    return df, direction_map


def print_data_info(df: pd.DataFrame, direction_map: dict[str, list[str]]) -> None:
    """打印数据集信息。"""
    print("\n" + "=" * 65)
    print("  Home Credit 数据集 — 预处理摘要")
    print("=" * 65)
    print(f"  总行数: {len(df):,}")
    print(f"  总特征列: {len(df.columns) - 1:,}")
    print(f"  标签分布: 正样本={int(df['label'].sum()):,} "
          f"({df['label'].mean():.2%})")
    print(f"\n  方向分布:")
    for d, cols in direction_map.items():
        print(f"    {d:<25} {len(cols):>3} 列")
    print("=" * 65)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os

    data_dir = Path(__file__).resolve().parent.parent.parent / "home credit default risk"
    data_dir = Path(os.environ.get("HOME_CREDIT_DIR", str(data_dir)))

    output = Path(__file__).resolve().parent.parent / "data" / "home_credit_processed.csv"
    df, dmap = load_home_credit_data(data_dir, max_rows=5000, output_path=output)
    print_data_info(df, dmap)
