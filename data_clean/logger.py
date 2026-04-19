"""
Structured transformation logger — records every decision made during cleaning.
"""

from __future__ import annotations

import json
import datetime
from typing import Any


class TransformationLogger:
    """Append-only log of every action taken on the dataset."""

    def __init__(self) -> None:
        self._entries: list[dict] = []

    # ------------------------------------------------------------------
    def log(
        self,
        stage: str,
        action: str,
        column: str | None = None,
        reason: str = "",
        before: Any = None,
        after: Any = None,
        severity: str = "info",
    ) -> None:
        entry = {
            "ts": datetime.datetime.utcnow().isoformat(),
            "stage": stage,
            "action": action,
            "column": column,
            "reason": reason,
            "before": str(before) if before is not None else None,
            "after": str(after) if after is not None else None,
            "severity": severity,  # info | warning | critical
        }
        self._entries.append(entry)

    # ------------------------------------------------------------------
    def warn(self, stage: str, action: str, column: str | None = None, reason: str = "") -> None:
        self.log(stage, action, column=column, reason=reason, severity="warning")

    def critical(self, stage: str, action: str, column: str | None = None, reason: str = "") -> None:
        self.log(stage, action, column=column, reason=reason, severity="critical")

    # ------------------------------------------------------------------
    @property
    def entries(self) -> list[dict]:
        return list(self._entries)

    def to_json(self) -> str:
        return json.dumps(self._entries, indent=2, default=str)

    def to_dict_list(self) -> list[dict]:
        return self._entries

    def by_severity(self, severity: str) -> list[dict]:
        return [e for e in self._entries if e["severity"] == severity]

    def summary(self) -> dict:
        stages: dict[str, int] = {}
        for e in self._entries:
            stages[e["stage"]] = stages.get(e["stage"], 0) + 1
        return {
            "total_actions": len(self._entries),
            "warnings": len(self.by_severity("warning")),
            "critical": len(self.by_severity("critical")),
            "by_stage": stages,
        }
