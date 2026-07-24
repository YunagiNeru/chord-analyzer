from .agents import MusicCoordinatorAgent
from .accuracy_v2_finalization import install_accuracy_v2_finalization
from .quality_extension import install_quality_extension

install_accuracy_v2_finalization()
install_quality_extension()

__all__ = ["MusicCoordinatorAgent"]
