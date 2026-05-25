"""
src/utils/metrics.py
====================
模块职责：实现特征评估的确定性指标计算，包括 IV、KS、PSI 及分箱逻辑。
I/O 契约：
  - 输入: pandas.DataFrame (含 feature_col, label_col) + 配置参数
  - 输出: dict[str, float] 含 IV, KS, KS_gain, PSI, missing_rate
依赖: pandas, numpy, scipy
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MetricsConfig:
    """指标计算配置。"""
    n_bins: int = 10
    iv_threshold: float = 0.005
    missing_rate_threshold: float = 0.7
    psi_threshold: float = 0.25
    min_samples_per_bin: int = 5


def _equal_frequency_bins(
    values: np.ndarray, n_bins: int
) -> list[tuple[float, float]]:
    """等频分箱：返回箱边界列表 [(low, high), ...]"""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    if n < n_bins * 2:
        n_bins = max(2, n // 2)

    bin_edges = []
    for i in range(n_bins):
        start_idx = i * n // n_bins
        end_idx = (i + 1) * n // n_bins - 1
        low = float(sorted_vals[start_idx])
        high = float(sorted_vals[min(end_idx, n - 1)])
        if i == n_bins - 1:
            high = float(sorted_vals[-1]) + 1e-8
        bin_edges.append((low, high))
    return bin_edges


def _compute_iv_from_bins(
    good_counts: np.ndarray, bad_counts: np.ndarray
) -> float:
    """
    基于分箱统计数据计算 IV。
    IV = Σ (bad_dist_i - good_dist_i) * ln(bad_dist_i / good_dist_i)
    """
    total_good = good_counts.sum()
    total_bad = bad_counts.sum()
    if total_good == 0 or total_bad == 0:
        return 0.0

    iv = 0.0
    for g, b in zip(good_counts, bad_counts):
        if g == 0:
            g = 0.5
        if b == 0:
            b = 0.5
        good_dist = g / total_good
        bad_dist = b / total_bad
        iv += (bad_dist - good_dist) * math.log(bad_dist / good_dist)
    return max(0.0, float(iv))


def _compute_ks_from_bins(
    good_counts: np.ndarray, bad_counts: np.ndarray
) -> float:
    """基于分箱累计分布计算 KS 值。"""
    total_good = good_counts.sum()
    total_bad = bad_counts.sum()
    if total_good == 0 or total_bad == 0:
        return 0.0

    cum_good = np.cumsum(good_counts) / total_good
    cum_bad = np.cumsum(bad_counts) / total_bad
    ks = np.max(np.abs(cum_good - cum_bad))
    return float(ks)


def compute_psi(
    expected_props: np.ndarray, actual_props: np.ndarray
) -> float:
    """
    计算 PSI (Population Stability Index)。
    PSI = Σ (actual_i - expected_i) * ln(actual_i / expected_i)
    """
    psi = 0.0
    for e, a in zip(expected_props, actual_props):
        if e == 0:
            e = 0.0001
        if a == 0:
            a = 0.0001
        psi += (a - e) * math.log(a / e)
    return max(0.0, float(psi))


def evaluate_feature_metrics(
    feature_data: pd.DataFrame,
    feature_col: str,
    label_col: str = "label",
    baseline_feature: pd.Series | None = None,
    config: MetricsConfig | None = None,
) -> dict:
    """
    计算单个特征的各项指标。

    Args:
        feature_data: 含特征列和标签列的 DataFrame
        feature_col: 特征列名
        label_col: 标签列名 (0/1)
        baseline_feature: 用于计算 KS_gain 的基线特征值
        config: 指标配置

    Returns:
        dict: {IV, KS, KS_gain, PSI, missing_rate, status, reason}
    """
    cfg = config or MetricsConfig()

    # --- 缺失率 ---
    total_rows = len(feature_data)
    feature_series = feature_data[feature_col]
    missing_count = int(feature_series.isna().sum())
    missing_rate = missing_count / total_rows if total_rows > 0 else 1.0

    # --- 早期拦截: 缺失率过高 ---
    if missing_rate > cfg.missing_rate_threshold:
        return {
            "IV": 0.0,
            "KS": 0.0,
            "KS_gain": 0.0,
            "PSI": 0.0,
            "missing_rate": missing_rate,
            "status": "skipped",
            "reason": f"missing_rate={missing_rate:.4f} > {cfg.missing_rate_threshold}",
        }

    # --- 过滤有效行 ---
    valid = feature_data[[feature_col, label_col]].dropna()
    if len(valid) < 10:
        return {
            "IV": 0.0,
            "KS": 0.0,
            "KS_gain": 0.0,
            "PSI": 0.0,
            "missing_rate": missing_rate,
            "status": "skipped",
            "reason": f"valid_samples={len(valid)} < 10",
        }

    values = valid[feature_col].to_numpy()
    labels = valid[label_col].to_numpy()

    # --- 单值检查 ---
    unique_vals = np.unique(values)
    if len(unique_vals) <= 1:
        return {
            "IV": 0.0,
            "KS": 0.0,
            "KS_gain": 0.0,
            "PSI": 0.0,
            "missing_rate": missing_rate,
            "status": "skipped",
            "reason": "constant feature (single unique value)",
        }

    # --- 等频分箱 ---
    bins = _equal_frequency_bins(values, cfg.n_bins)

    # --- 统计各箱样本 ---
    good_counts = []
    bad_counts = []
    bin_props = []
    for low, high in bins:
        mask = (values >= low) & (values < high)
        if mask.sum() < cfg.min_samples_per_bin:
            continue
        good_counts.append(int((labels[mask] == 0).sum()))
        bad_counts.append(int((labels[mask] == 1).sum()))
        bin_props.append(mask.sum() / len(values))

    if len(good_counts) < 2:
        return {
            "IV": 0.0,
            "KS": 0.0,
            "KS_gain": 0.0,
            "PSI": 0.0,
            "missing_rate": missing_rate,
            "status": "skipped",
            "reason": "insufficient bins after filtering",
        }

    good_arr = np.array(good_counts, dtype=float)
    bad_arr = np.array(bad_counts, dtype=float)

    # --- 计算 IV ---
    iv = _compute_iv_from_bins(good_arr, bad_arr)

    # --- 计算 KS ---
    ks = _compute_ks_from_bins(good_arr, bad_arr)

    # --- 计算 PSI (使用均匀分布作为基准) ---
    uniform_props = np.ones(len(bin_props)) / len(bin_props)
    actual_props = np.array(bin_props)
    psi = compute_psi(uniform_props, actual_props)

    # --- KS_gain vs baseline ---
    ks_gain = 0.0
    if baseline_feature is not None:
        baseline_valid = baseline_feature.dropna().to_numpy()
        if len(baseline_valid) > 10:
            baseline_bins = _equal_frequency_bins(baseline_valid, cfg.n_bins)
            baseline_good = []
            baseline_bad = []
            for bl, bh in baseline_bins:
                bm = (baseline_valid >= bl) & (baseline_valid < bh)
                n_bm = len(bm)
                bl_labels = labels[:n_bm]
                baseline_good.append(int((bl_labels[bm] == 0).sum()))
                baseline_bad.append(int((bl_labels[bm] == 1).sum()))
            baseline_ks = _compute_ks_from_bins(
                np.array(baseline_good, dtype=float),
                np.array(baseline_bad, dtype=float),
            )
            ks_gain = ks - baseline_ks

    status = "success"
    reason = None
    if iv < cfg.iv_threshold:
        status = "skipped"
        reason = f"IV={iv:.6f} < {cfg.iv_threshold}"

    return {
        "IV": round(iv, 6),
        "KS": round(ks, 6),
        "KS_gain": round(ks_gain, 6),
        "PSI": round(psi, 6),
        "missing_rate": round(missing_rate, 4),
        "status": status,
        "reason": reason,
    }