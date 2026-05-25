"""
src/memory/trace_recorder.py
============================
Persistent trace recorder for feature mining runs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.schemas import CriticOutput, EvalResult, FeatureDSL


class TraceRecorder:
    """Records generation, validation, evaluation, and critic events."""

    def __init__(self, trace_dir: str | Path, run_id: str | None = None):
        self.trace_dir = Path(trace_dir)
        self.run_id = run_id
        self.events_path = self.trace_dir / "run_trace.jsonl"
        self.lineage_path = self.trace_dir / "feature_lineage.json"
        self._lineage: dict[str, dict[str, Any]] = {}

    def start_run(self, metadata: dict[str, Any]) -> None:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._lineage = {}
        payload = {"run_id": self.run_id, **metadata}
        self._write_event("run_started", payload)

    def record_step_selected(
        self,
        step: int,
        direction: str,
        ucb_value: float | str,
        direction_stats: list[dict[str, Any]],
    ) -> None:
        self._write_event(
            "step_selected",
            {
                "step": step,
                "direction": direction,
                "ucb_value": ucb_value,
                "selection_reason": _build_selection_reason(
                    direction, ucb_value, direction_stats
                ),
                "direction_stats": direction_stats,
            },
        )

    def record_generated(
        self,
        step: int,
        direction: str,
        dsls: list[FeatureDSL],
        critic_context: list[CriticOutput],
    ) -> None:
        feedback_summary = [
            {
                "feature_id": c.feature_id,
                "failure_reason": c.failure_reason,
                "modification_axis": c.modification_axis,
                "param_suggestion": c.param_suggestion,
                "next_direction_hint": c.next_direction_hint,
                "confidence": c.confidence,
            }
            for c in critic_context[-3:]
        ]
        for rank, dsl in enumerate(dsls, start=1):
            reason = _build_generation_reason(dsl, direction, feedback_summary)
            record = {
                "step": step,
                "direction": direction,
                "rank": rank,
                "feature": dsl.model_dump(),
                "business_logic": dsl.business_logic,
                "generation_reason": reason,
                "critic_context_used": feedback_summary,
            }
            self._write_event("feature_generated", record)
            self._lineage[dsl.feature_id] = {
                "feature_id": dsl.feature_id,
                "direction": dsl.direction,
                "step_generated": step,
                "definition": dsl.definition,
                "business_logic": dsl.business_logic,
                "generation_reason": reason,
                "critic_context_used": feedback_summary,
                "validation": None,
                "evaluation": None,
                "critic": [],
            }

    def record_validation(
        self,
        step: int,
        valid_dsls: list[FeatureDSL],
        invalid_dsls: list[tuple[FeatureDSL, str]],
    ) -> None:
        for dsl in valid_dsls:
            payload = {
                "step": step,
                "feature_id": dsl.feature_id,
                "status": "valid",
                "reason": "通过结构、时间窗口、防泄漏与合规校验",
            }
            self._write_event("feature_validated", payload)
            self._lineage.setdefault(dsl.feature_id, {})["validation"] = payload

        for dsl, reason in invalid_dsls:
            payload = {
                "step": step,
                "feature_id": dsl.feature_id,
                "status": "invalid",
                "reason": reason,
            }
            self._write_event("feature_validated", payload)
            self._lineage.setdefault(dsl.feature_id, {})["validation"] = payload

    def record_evaluation(
        self,
        step: int,
        dsl: FeatureDSL,
        result: EvalResult,
        resolved_column: str | None,
    ) -> None:
        payload = {
            "step": step,
            "feature_id": result.feature_id,
            "resolved_column": resolved_column,
            "IV": result.IV,
            "KS": result.KS,
            "KS_gain": result.KS_gain,
            "PSI": result.PSI,
            "missing_rate": result.missing_rate,
            "status": result.status,
            "reason": result.reason,
            "interpretation": _build_eval_interpretation(result),
        }
        self._write_event("feature_evaluated", payload)
        self._lineage.setdefault(dsl.feature_id, {})["evaluation"] = payload

    def record_critic(
        self,
        step: int,
        critique: CriticOutput,
        result: EvalResult | None = None,
    ) -> None:
        payload = {
            "step": step,
            "feature_id": critique.feature_id,
            "failure_reason": critique.failure_reason,
            "modification_axis": critique.modification_axis,
            "param_suggestion": critique.param_suggestion,
            "next_direction_hint": critique.next_direction_hint,
            "confidence": critique.confidence,
            "source_metrics": result.__dict__ if result else None,
        }
        self._write_event("feature_critiqued", payload)
        self._lineage.setdefault(critique.feature_id, {}).setdefault(
            "critic", []
        ).append(payload)

    def finish_run(self, summary: dict[str, Any]) -> None:
        self._write_event("run_finished", summary)
        with self.lineage_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"saved_at": datetime.now().isoformat(), "features": self._lineage},
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _write_event(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            **payload,
        }
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _build_selection_reason(
    direction: str,
    ucb_value: float | str,
    direction_stats: list[dict[str, Any]],
) -> str:
    if ucb_value == "inf":
        return f"{direction} 尚未充分探索，UCB 冷启动优先选择该方向。"

    stat = next((item for item in direction_stats if item["direction"] == direction), None)
    if not stat:
        return f"UCB 选择 {direction}，当前得分为 {ucb_value}。"
    return (
        f"UCB 选择 {direction}，该方向历史收益均值 mu={stat['mu_k']:.4f}，"
        f"探索次数 n={stat['n_k']}，综合得分={ucb_value}。"
    )


def _build_generation_reason(
    dsl: FeatureDSL,
    selected_direction: str,
    feedback_summary: list[dict[str, Any]],
) -> str:
    definition = dsl.definition
    base = (
        f"围绕 {selected_direction} 方向生成，使用 "
        f"{definition.get('aggregation')} 聚合、{definition.get('transformation')} 变换，"
        f"在 {definition.get('time_window')} 时间窗口内从 "
        f"{definition.get('source_table')} 提取信号。"
    )
    if feedback_summary:
        axes = ", ".join(item["modification_axis"] for item in feedback_summary)
        return f"{base} 生成时参考了最近 Critic 反馈，重点修正 {axes} 相关问题。"
    return f"{base} 当前无历史 Critic 反馈，作为该方向的探索候选。"


def _build_eval_interpretation(result: EvalResult) -> str:
    if result.status == "success":
        return (
            f"特征通过评估，IV={result.IV:.4f}、KS={result.KS:.4f}，"
            "具备进入候选特征池的统计证据。"
        )
    if result.status == "skipped":
        return (
            f"特征被跳过，原因={result.reason or '指标未达阈值'}，"
            f"IV={result.IV:.4f}、缺失率={result.missing_rate:.2%}。"
        )
    return f"特征评估失败，原因={result.reason or '未知异常'}。"
