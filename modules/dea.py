"""
Dual-context Collaborative Enhancement Attention (DEA) Module for Dual-Stream YOLOv11.
Consists of:
- DECA: Dual Semantic Enhancing Channel Weight Assignment Module
- DEPA: Dual Spatial Enhancing Pixel Weight Assignment Module
Adapted from DE-YOLO (ICPR 2024).
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class DECA(nn.Module):
    """Dual Semantic Enhancing Channel Weight Assignment Module.
    Leverages cross-modal channel dependencies to assign channel attention weights
    between Visible (RGB) and Infrared (IR) streams.
    """

    def __init__(self, channel=512, kernel_size=80, p_kernel=None, reduction=16):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_ch = max(channel // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduced_ch, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_ch, channel, bias=False),
            nn.Sigmoid(),
        )
        self.act = nn.Sigmoid()
        self.compress = Conv(channel * 2, channel, 3)

        """Convolution pyramid for cross-scale feature aggregation"""
        if p_kernel is None:
            p_kernel = [5, 4]
        kernel1, kernel2 = p_kernel
        k3 = max(int(self.kernel_size / kernel1 / kernel2), 1)
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel1, kernel1, 0, groups=channel), nn.SiLU()
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel2, kernel2, 0, groups=channel), nn.SiLU()
        )
        self.conv_c3 = nn.Sequential(
            nn.Conv2d(channel, channel, k3, k3, 0, groups=channel), nn.SiLU()
        )

    def forward(self, x):
        """Forward pass for DECA.
        Args:
            x (list | tuple): Pair of feature tensors [x_rgb, x_ir].
        Returns:
            tuple: (result_rgb, result_ir)
        """
        x_vi, x_ir = x[0], x[1]
        b, c, h, w = x_vi.size()

        w_vi = self.avg_pool(x_vi).view(b, c)
        w_ir = self.avg_pool(x_ir).view(b, c)
        w_vi = self.fc(w_vi).view(b, c, 1, 1)
        w_ir = self.fc(w_ir).view(b, c, 1, 1)

        glob_t = self.compress(torch.cat([x_vi, x_ir], dim=1))
        if min(h, w) >= self.kernel_size:
            glob = self.conv_c3(self.conv_c2(self.conv_c1(glob_t)))
        else:
            glob = torch.mean(glob_t, dim=[2, 3], keepdim=True)

        result_vi = x_vi * (self.act(w_ir * glob)).expand_as(x_vi)
        result_ir = x_ir * (self.act(w_vi * glob)).expand_as(x_ir)

        return result_vi, result_ir


class DEPA(nn.Module):
    """Dual Spatial Enhancing Pixel Weight Assignment Module.
    Learns spatial dependency structures within and across modalities to produce
    enhanced multi-modal representations with fine positional awareness.
    """

    def __init__(self, channel=512, m_kernel=None):
        super().__init__()
        if m_kernel is None:
            m_kernel = [3, 7]

        self.conv1 = Conv(2, 1, 5)
        self.conv2 = Conv(2, 1, 5)
        self.compress1 = Conv(channel, 1, 3)
        self.compress2 = Conv(channel, 1, 3)
        self.act = nn.Sigmoid()

        """Multi-scale spatial convolutions"""
        self.cv_v1 = Conv(channel, 1, m_kernel[0])
        self.cv_v2 = Conv(channel, 1, m_kernel[1])
        self.cv_i1 = Conv(channel, 1, m_kernel[0])
        self.cv_i2 = Conv(channel, 1, m_kernel[1])

    def forward(self, x):
        """Forward pass for DEPA.
        Args:
            x (list | tuple): Pair of feature tensors [x_rgb, x_ir].
        Returns:
            tuple: (result_rgb, result_ir)
        """
        x_vi, x_ir = x[0], x[1]

        w_vi = self.conv1(torch.cat([self.cv_v1(x_vi), self.cv_v2(x_vi)], dim=1))
        w_ir = self.conv2(torch.cat([self.cv_i1(x_ir), self.cv_i2(x_ir)], dim=1))
        glob = self.act(self.compress1(x_vi) + self.compress2(x_ir))
        w_vi = self.act(glob + w_vi)
        w_ir = self.act(glob + w_ir)

        result_vi = x_vi * w_ir.expand_as(x_vi)
        result_ir = x_ir * w_vi.expand_as(x_ir)

        return result_vi, result_ir


class DEA(nn.Module):
    """Dual-context Collaborative Enhancement Module (DEA).
    Sequentially applies DECA (channel enhancement) and DEPA (spatial enhancement),
    then performs collaborative fusion of Visible and Infrared features.
    """

    def __init__(self, channel=512, kernel_size=80, p_kernel=None, m_kernel=None, reduction=16):
        super().__init__()
        self.deca = DECA(channel, kernel_size, p_kernel, reduction)
        self.depa = DEPA(channel, m_kernel)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Forward pass for DEA.
        Args:
            x (list | tuple): Feature maps from [RGB_stream_layer, IR_stream_layer].
        Returns:
            torch.Tensor: Fused multi-modal feature tensor of shape (B, C, H, W).
        """
        result_vi, result_ir = self.depa(self.deca(x))
        return self.act(result_vi + result_ir)
