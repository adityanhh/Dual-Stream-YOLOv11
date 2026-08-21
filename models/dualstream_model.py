"""
Dual-Stream YOLOv11 Model Implementation (DE-YOLOv11).
Implements Dual-Input (RGB + IR) detection architecture with:
- Dual-stream backbones (Stream 1 for Visible, Stream 2 for Infrared)
- Bi-directional Decoupled Focus (C3k2_BiFocus)
- Dual-context Collaborative Enhancement Attention (DEA: DECA + DEPA)
- Pointwise Spatial Attention (C2PSA)
- High-level YOLO API wrapper (DualStreamYOLO) and pretrained weight transfer.
"""

import ast
import contextlib
import copy
from copy import deepcopy
from pathlib import Path
import yaml
import torch
import torch.nn as nn

from ultralytics.nn.modules import (
    Conv,
    ConvTranspose,
    GhostConv,
    Bottleneck,
    GhostBottleneck,
    SPP,
    SPPF,
    DWConv,
    Focus,
    BottleneckCSP,
    C1,
    C2,
    C2f,
    C3,
    C3TR,
    C3Ghost,
    C3x,
    RepC3,
    C3k2,
    C2PSA,
    Concat,
    Detect,
    AIFI,
    HGStem,
    HGBlock,
)

try:
    from ultralytics.nn.modules.block import C3k
except ImportError:
    C3k = None
from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.nn.tasks import BaseModel
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import fuse_conv_and_bn, initialize_weights, intersect_dicts, scale_img, time_sync

from ..modules import BiFocus, C3k2_BiFocus, DEA, DECA, DEPA, DepthWiseConv, FocusH, FocusV


def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return max(divisor, int(x + divisor / 2) // divisor * divisor)


def parse_dualstream_model(d, ch=3, ch2=3, verbose=True):
    """
    Parse a dual-stream YOLOv11 model dictionary into an nn.Sequential model.
    Handles 'backbone' (RGB stream), 'backbone2' (IR stream), and 'head' (cross-modality fusion + detection).
    """
    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

    nc = d.get('nc', 80)
    act = d.get('act')
    scales = d.get('scales')
    scale = d.get('scale')

    depth, width, max_channels = 1.0, 1.0, 512
    if scales:
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")

    # Registered module dictionary
    module_dict = {
        'Conv': Conv,
        'ConvTranspose': ConvTranspose,
        'GhostConv': GhostConv,
        'Bottleneck': Bottleneck,
        'GhostBottleneck': GhostBottleneck,
        'SPP': SPP,
        'SPPF': SPPF,
        'DWConv': DWConv,
        'Focus': Focus,
        'BottleneckCSP': BottleneckCSP,
        'C1': C1,
        'C2': C2,
        'C2f': C2f,
        'C3': C3,
        'C3TR': C3TR,
        'C3Ghost': C3Ghost,
        'C3x': C3x,
        'RepC3': RepC3,
        'C3k2': C3k2,
        'C3k': C3k,
        'C2PSA': C2PSA,
        'Concat': Concat,
        'Detect': Detect,
        'AIFI': AIFI,
        'HGStem': HGStem,
        'HGBlock': HGBlock,
        'BiFocus': BiFocus,
        'C3k2_BiFocus': C3k2_BiFocus,
        'FocusH': FocusH,
        'FocusV': FocusV,
        'DepthWiseConv': DepthWiseConv,
        'DECA': DECA,
        'DEPA': DEPA,
        'DEA': DEA,
    }

    layers, save = [], []
    ch_list = []  # Unified output channel tracker for all layers

    nb1 = len(d['backbone'])
    nb2 = len(d['backbone2'])
    total_layers_cfg = d['backbone'] + d['backbone2'] + d['head']

    for i, (f, n, m, args) in enumerate(total_layers_cfg):
        # Resolve module type
        if 'nn.' in m:
            m_cls = getattr(torch.nn, m[3:])
        elif m in module_dict:
            m_cls = module_dict[m]
        elif hasattr(torch.nn, m):
            m_cls = getattr(torch.nn, m)
        else:
            m_cls = globals().get(m, None)
            if m_cls is None:
                raise ValueError(f"Module '{m}' not found in registered modules.")

        # Parse string arguments
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = ast.literal_eval(a)

        n = max(round(n * depth), 1) if n > 1 else n  # depth gain

        # Determine input channels c1 and compute output channels c2
        if i < nb1:
            # Stream 1 (RGB Backbone)
            c1 = ch if i == 0 else (ch_list[f] if f != -1 else ch_list[i - 1])
        elif i < nb1 + nb2:
            # Stream 2 (IR Backbone)
            c1 = ch2 if i == nb1 else (ch_list[f] if f != -1 else ch_list[i - 1])
        else:
            # Head & Fusion
            if isinstance(f, int):
                c1 = ch_list[f] if f != -1 else ch_list[i - 1]
            else:
                c1 = [ch_list[x] for x in f]

        # Handle specific module types
        if m_cls in (
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3,
            C3TR,
            C3Ghost,
            nn.ConvTranspose2d,
            C3x,
            RepC3,
            C3k2,
        ):
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m_cls in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3, C3k2):
                args.insert(2, n)  # number of repeats
                n = 1

        elif m_cls is C3k2_BiFocus:
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            args.insert(2, n)  # number of repeats
            n = 1

        elif m_cls is C2PSA:
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            args.insert(2, n)  # number of repeats
            n = 1

        elif m_cls is AIFI:
            args = [c1, *args]
            c2 = args[1] if len(args) > 1 else c1

        elif m_cls is DEA:
            # f is [rgb_layer_idx, ir_layer_idx]
            # args[0] is base channel
            c2 = args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            # DEA takes (channel, kernel_size, ...)
            args = [c2, *args[1:]]

        elif m_cls is Concat:
            c2 = sum(ch_list[x] for x in f)

        elif m_cls is Detect:
            reg_max = d.get("reg_max", 16)
            end2end = d.get("end2end", False)
            args = [nc, reg_max, end2end, [ch_list[x] for x in f]]
            c2 = None

        elif m_cls is nn.BatchNorm2d:
            args = [c1]
            c2 = c1

        elif m_cls is nn.Upsample:
            c2 = c1

        else:
            c2 = c1

        # Instantiate module
        m_ = nn.Sequential(*(m_cls(*args) for _ in range(n))) if n > 1 else m_cls(*args)
        t = str(m_cls)[8:-2].replace('__main__.', '')
        m_.np = sum(x.numel() for x in m_.parameters())  # number of params
        m_.i, m_.f, m_.type = i, f, t

        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")

        # Update savelist
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        ch_list.append(c2)

    return nn.Sequential(*layers), sorted(save)


class DualStreamDetectionModel(BaseModel):
    """
    Dual-Stream YOLOv11 Detection Model.
    Accepts dual-input (RGB visible and IR thermal) and executes cross-modality object detection.
    """

    def __init__(self, cfg='yolo11n-dualstream.yaml', ch=3, ch2=3, nc=None, verbose=True):
        super().__init__()
        if isinstance(cfg, (str, Path)):
            with open(cfg, 'r') as f:
                self.yaml = yaml.safe_load(f)
            self.yaml['yaml_file'] = str(cfg)
        else:
            self.yaml = deepcopy(cfg)

        self.ch = self.yaml.get('ch', ch)
        self.ch2 = self.yaml.get('ch2', ch2)
        if nc and nc != self.yaml.get('nc'):
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc

        self.model, self.save = parse_dualstream_model(
            deepcopy(self.yaml), ch=self.ch, ch2=self.ch2, verbose=verbose
        )
        self.names = {i: f'{i}' for i in range(self.yaml.get('nc', 80))}
        self.inplace = self.yaml.get('inplace', True)
        self.args = get_cfg(DEFAULT_CFG)

        # Build strides
        m = self.model[-1]  # Detect() head
        self.end2end = getattr(self, 'end2end', False)
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace

            def _forward(x1, x2):
                output = self.forward(x1, x2)
                if getattr(self, 'end2end', False) and isinstance(output, dict) and 'one2many' in output:
                    output = output['one2many']
                return output['feats'] if isinstance(output, dict) and 'feats' in output else output

            self.model.eval()
            m.training = True
            with torch.no_grad():
                feats = _forward(torch.zeros(1, self.ch, s, s), torch.zeros(1, self.ch2, s, s))
            m.stride = torch.tensor([s / x.shape[-2] for x in feats])
            self.stride = m.stride
            self.model.train()
            m.bias_init()
        else:
            self.stride = torch.tensor([32])

        initialize_weights(self)
        if verbose:
            self.info()

    def forward(self, x1, x2=None, profile=False, visualize=False):
        """
        Forward pass through Dual-Stream YOLOv11.
        Args:
            x1 (torch.Tensor | list | tuple | dict): RGB input tensor (B, 3, H, W) or batch dict.
            x2 (torch.Tensor, optional): IR input tensor (B, 3/1, H, W).
        Returns:
            torch.Tensor | tuple: Prediction outputs from Detect head.
        """
        if isinstance(x1, dict):
            # Batch dict passed
            return self.loss(x1)

        if x2 is None and isinstance(x1, (list, tuple)):
            x1, x2 = x1[0], x1[1]

        if x2 is None:
            # Fallback for single input: duplicate x1 to x2
            x2 = x1

        return self._predict_once(x1, x2, profile=profile, visualize=visualize)

    def _predict_once(self, x1, x2, profile=False, visualize=False):
        """
        Execute forward computation through dual backbone streams and head.
        """
        y = []
        nb1 = len(self.yaml['backbone'])
        nb2 = len(self.yaml['backbone2'])

        # 1. Visible Stream (RGB Backbone)
        cur = x1
        for i in range(nb1):
            m = self.model[i]
            if m.f != -1:
                cur = y[m.f] if isinstance(m.f, int) else [cur if j == -1 else y[j] for j in m.f]
            cur = m(cur)
            y.append(cur if m.i in self.save else None)

        # 2. Infrared Stream (IR Backbone)
        cur = x2
        for i in range(nb1, nb1 + nb2):
            m = self.model[i]
            if m.f != -1:
                cur = y[m.f] if isinstance(m.f, int) else [cur if j == -1 else y[j] for j in m.f]
            cur = m(cur)
            y.append(cur if m.i in self.save else None)

        # 3. Cross-Modality Fusion (DEA) & Detection Head
        for i in range(nb1 + nb2, len(self.model)):
            m = self.model[i]
            if m.f != -1:
                inp = y[m.f] if isinstance(m.f, int) else [cur if j == -1 else y[j] for j in m.f]
            else:
                inp = cur
            cur = m(inp)
            y.append(cur if m.i in self.save else None)

        return cur

    def loss(self, batch, preds=None):
        """
        Compute detection loss on a dual-input batch.
        Args:
            batch (dict): Batch dictionary containing 'img' (RGB) and 'img2' (IR).
            preds (tuple, optional): Precomputed predictions.
        """
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        if preds is None:
            img = batch['img']
            img2 = batch.get('img2', img)
            preds = self.forward(img, img2)

        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize standard YOLOv8/v11 detection criterion."""
        return v8DetectionLoss(self)

    def load_pretrained_weights(self, weights_path, verbose=True):
        """
        Load weights from a standard pretrained YOLO11 checkpoint (e.g. yolo11n.pt).
        Automatically transfers weights to both Backbone 1 (RGB) and Backbone 2 (IR),
        as well as matching Neck and Head layers!
        """
        LOGGER.info(f"Loading pretrained weights from {weights_path} into Dual-Stream model...")
        ckpt = torch.load(weights_path, map_location='cpu')
        state_dict = ckpt['model'].state_dict() if 'model' in ckpt else ckpt

        model_dict = self.state_dict()
        matched_dict = {}

        nb1 = len(self.yaml['backbone'])
        nb2 = len(self.yaml['backbone2'])

        # Transfer matching layer parameters
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                matched_dict[k] = v

            # Map single backbone weights to Backbone 2 (IR) as well
            parts = k.split('.')
            if parts[0] == 'model' and parts[1].isdigit():
                layer_idx = int(parts[1])
                if layer_idx < nb1:
                    ir_layer_idx = layer_idx + nb1
                    ir_k = f"model.{ir_layer_idx}." + '.'.join(parts[2:])
                    if ir_k in model_dict and model_dict[ir_k].shape == v.shape:
                        matched_dict[ir_k] = v

        model_dict.update(matched_dict)
        self.load_state_dict(model_dict, strict=False)
        if verbose:
            LOGGER.info(f"Transferred {len(matched_dict)}/{len(model_dict)} tensors to Dual-Stream YOLOv11.")


class DualStreamYOLO:
    """
    High-level user-friendly wrapper for Dual-Stream YOLOv11.
    Provides standard .train(), .val(), .predict(), .to() interface similar to Ultralytics YOLO().
    """

    def __init__(self, model_cfg='yolo11n-dualstream.yaml', weights=None):
        self.model = DualStreamDetectionModel(cfg=model_cfg)
        if weights:
            self.load(weights)

    def load(self, weights_path):
        """Load weights from checkpoint."""
        self.model.load_pretrained_weights(weights_path)
        return self

    def to(self, device):
        """Move model to specified device."""
        self.model.to(device)
        return self

    def __call__(self, x1, x2=None, **kwargs):
        """Perform forward inference."""
        return self.model(x1, x2, **kwargs)
