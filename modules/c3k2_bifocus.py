"""
C3k2_BiFocus module for YOLOv11.
Combines YOLOv11's high-efficiency C3k2 CSP bottleneck structure with
Bi-directional Decoupled Focus (BiFocus) for enhanced receptive fields.
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.block import C3k2
from .bifocus import BiFocus


class C3k2_BiFocus(C3k2):
    """Faster Implementation of CSP Bottleneck with 2 convolutions and BiFocus.
    Args:
        c1 (int): Input channel dimension.
        c2 (int): Output channel dimension.
        n (int): Number of bottleneck repeats.
        c3k (bool): Use C3k block instead of standard Bottleneck if True.
        e (float): Channel expansion ratio (default 0.5).
        g (int): Groups for convolution.
        shortcut (bool): Use residual connection if True.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n=n, c3k=c3k, e=e, g=g, shortcut=shortcut)
        self.bifocus = BiFocus(c2, c2)

    def forward(self, x):
        """Forward pass through C3k2 and BiFocus."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.bifocus(out)
