"""
src/memory/feature_knowledge_base.py
====================================
Cross-run memory for high-quality factors.

The run directory stores audit artifacts for one execution. This knowledge
base stores reusable feature ideas across executions so future mining can
start from what has already worked.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class FeatureKnowledgeBase:
    """Persistent high-quality feature memory."""

    def __init__(self, storage_path: str | Path):
        self.storage_path = Path(storage_path)
        self._features: dict[str, dict[str, Any]] = {}

    def load(self) -> bool:
        if not self.storage_path.exists():
            return False
        with self.storage_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self._features = data.get("features", {})
        return True

    def add_many(self, details: list[dict[str, Any]], run_id: str) -> int:
        added = 0
        for detail in details:
            feature_id = detail.get("feature_id")
            if not feature_id:
                continue
            record = {
                **detail,
                "source_run_id": run_id,
                "saved_at": datetime.now().isoformat(),
            }
            if feature_id not in self._features:
                added += 1
            self._features[feature_id] = record
        return added

    def top_features(
        self,
        direction: str | None = None,
        template_family: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        candidates = list(self._features.values())
        if direction:
            candidates = [f for f in candidates if f.get("direction") == direction]
        if template_family:
            candidates = [
                f for f in candidates if f.get("template_family") == template_family
            ]

        return sorted(candidates, key=_quality_score, reverse=True)[:limit]

    def save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self.storage_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "saved_at": datetime.now().isoformat(),
                    "feature_count": len(self._features),
                    "features": self._features,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @property
    def size(self) -> int:
        return len(self._features)


def _quality_score(feature: dict[str, Any]) -> float:
    metrics = feature.get("metrics", {})
    iv = float(metrics.get("IV", 0.0) or 0.0)
    ks = float(metrics.get("KS", 0.0) or 0.0)
    psi = float(metrics.get("PSI", 0.0) or 0.0)
    missing = float(metrics.get("missing_rate", 1.0) or 1.0)
    return iv + ks - 0.5 * psi - 0.2 * missing
