from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schemas import AnalysisResult


class AccuracyQualityGateError(RuntimeError):
    """Raised when a structurally valid analysis fails musical quality gates."""

    def __init__(self, errors: list[str], result: "AnalysisResult") -> None:
        self.errors = list(errors)
        self.result = result
        super().__init__(
            "Accuracy v2 quality gate violation: " + ", ".join(self.errors)
        )
