"""DeepGuard's dependency-light forensic LR implementation.

This package intentionally avoids the external LiR dependency and GPU/RAPIDS
stack. It provides score fusion/calibration plus C_llr and C_llr_min.
"""
from .system import LRFusionSystem, cllr, cllr_min

__all__ = ["LRFusionSystem", "cllr", "cllr_min"]
