"""
Dual-Stream YOLOv11 Package.
Cross-modality (RGB + IR) Object Detection Architecture.
"""

from .models.dualstream_model import (
    DualStreamDetectionModel,
    DualStreamYOLO,
    parse_dualstream_model,
)

# Alias for convenience matching Ultralytics YOLO syntax
YOLO = DualStreamYOLO

__all__ = [
    "DualStreamYOLO",
    "YOLO",
    "DualStreamDetectionModel",
    "parse_dualstream_model",
]
