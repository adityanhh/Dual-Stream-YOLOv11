"""
Custom modules for Dual-Stream YOLOv11 (DE-YOLOv11).
"""

from .bifocus import BiFocus, DepthWiseConv, FocusH, FocusV
from .c3k2_bifocus import C3k2_BiFocus
from .dea import DEA, DECA, DEPA

__all__ = [
    "FocusH",
    "FocusV",
    "DepthWiseConv",
    "BiFocus",
    "C3k2_BiFocus",
    "DECA",
    "DEPA",
    "DEA",
]
