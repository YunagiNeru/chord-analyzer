from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AccuracyProfile:
    version: int
    role_weights: dict[str, float]
    dsp_weight: float
    alternative_weight: float
    simple_triad_reinforcement: float
    uncertainty_threshold: float
    minimum_primary_contributors: int
    exact_agreement_weight: float
    component_agreement_weight: float
    change_penalty: float
    isolated_penalty: float
    repeated_section_confidence_threshold: float
    repeated_section_winner_threshold: float
    grid_integer_bar_weight: float
    grid_beat_alignment_weight: float
    grid_section_alignment_weight: float
    grid_model_prior_weight: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AccuracyProfile":
        exact_weight = float(payload.get("exactAgreementWeight", 0.25))
        component_weight = float(payload.get("componentAgreementWeight", 0.75))
        total_agreement_weight = exact_weight + component_weight
        if total_agreement_weight <= 0:
            exact_weight = 0.25
            component_weight = 0.75
            total_agreement_weight = 1.0
        return cls(
            version=int(payload.get("version", 2)),
            role_weights={
                str(key): float(value)
                for key, value in (payload.get("roleWeights") or {}).items()
            },
            dsp_weight=float(payload.get("dspWeight", 0.48)),
            alternative_weight=float(payload.get("alternativeWeight", 0.16)),
            simple_triad_reinforcement=float(payload.get("simpleTriadReinforcement", 0.22)),
            uncertainty_threshold=float(payload.get("uncertaintyThreshold", 0.58)),
            minimum_primary_contributors=max(
                1,
                int(payload.get("minimumPrimaryContributors", 2)),
            ),
            exact_agreement_weight=exact_weight / total_agreement_weight,
            component_agreement_weight=component_weight / total_agreement_weight,
            change_penalty=float(payload.get("changePenalty", 0.28)),
            isolated_penalty=float(payload.get("isolatedPenalty", 0.22)),
            repeated_section_confidence_threshold=float(
                payload.get("repeatedSectionConfidenceThreshold", 0.58)
            ),
            repeated_section_winner_threshold=float(
                payload.get("repeatedSectionWinnerThreshold", 0.68)
            ),
            grid_integer_bar_weight=float(payload.get("gridIntegerBarWeight", 0.42)),
            grid_beat_alignment_weight=float(payload.get("gridBeatAlignmentWeight", 0.34)),
            grid_section_alignment_weight=float(payload.get("gridSectionAlignmentWeight", 0.18)),
            grid_model_prior_weight=float(payload.get("gridModelPriorWeight", 0.12)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "roleWeights": self.role_weights,
            "dspWeight": self.dsp_weight,
            "alternativeWeight": self.alternative_weight,
            "simpleTriadReinforcement": self.simple_triad_reinforcement,
            "uncertaintyThreshold": self.uncertainty_threshold,
            "minimumPrimaryContributors": self.minimum_primary_contributors,
            "exactAgreementWeight": self.exact_agreement_weight,
            "componentAgreementWeight": self.component_agreement_weight,
            "changePenalty": self.change_penalty,
            "isolatedPenalty": self.isolated_penalty,
            "repeatedSectionConfidenceThreshold": self.repeated_section_confidence_threshold,
            "repeatedSectionWinnerThreshold": self.repeated_section_winner_threshold,
            "gridIntegerBarWeight": self.grid_integer_bar_weight,
            "gridBeatAlignmentWeight": self.grid_beat_alignment_weight,
            "gridSectionAlignmentWeight": self.grid_section_alignment_weight,
            "gridModelPriorWeight": self.grid_model_prior_weight,
        }


def _profile_path() -> Path:
    override = os.environ.get("ACCURACY_PROFILE_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).with_name("accuracy_profile.json")


@lru_cache(maxsize=1)
def load_accuracy_profile() -> AccuracyProfile:
    path = _profile_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    profile = AccuracyProfile.from_dict(payload)
    if not profile.role_weights:
        raise ValueError("accuracy profile roleWeights must not be empty")
    return profile


def clear_accuracy_profile_cache() -> None:
    load_accuracy_profile.cache_clear()
