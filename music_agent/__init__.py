from .agents import MusicCoordinatorAgent
from .motif_reconcile import reconcile_repeated_sections_enhanced
from . import pipeline_v2 as _pipeline_v2

_pipeline_v2.reconcile_repeated_sections = reconcile_repeated_sections_enhanced

__all__ = ["MusicCoordinatorAgent"]
