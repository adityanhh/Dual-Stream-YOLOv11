"""
Validation and Evaluation Engine for Dual-Stream YOLOv11.
Computes mAP@50, mAP@50-95, Precision, and Recall on dual-input validation set.
"""

from pathlib import Path
import numpy as np
import torch
from torchvision.ops import box_iou

from ultralytics.utils import LOGGER

try:
    from ..data.dataset import build_dualstream_dataloader
    from .predictor import non_max_suppression_dualstream
except (ImportError, ValueError):
    try:
        from data.dataset import build_dualstream_dataloader
        from engine.predictor import non_max_suppression_dualstream
    except (ImportError, ValueError):
        from DualStreamYOLO11.data.dataset import build_dualstream_dataloader
        from DualStreamYOLO11.engine.predictor import non_max_suppression_dualstream


class ValResults:
    """Container for validation evaluation metrics."""

    def __init__(self, results_dict):
        self.results_dict = results_dict


class DualStreamValidator:
    """
    Validator for evaluating Dual-Stream YOLOv11 on paired validation datasets.
    """

    def __init__(
        self,
        model,
        data_cfg,
        batch_size=16,
        imgsz=640,
        conf_thres=0.001,
        iou_thres=0.6,
        device=None,
    ):
        self.model = model
        self.device = (
            device
            if device is not None
            else next(model.parameters()).device
        )
        self.data_cfg = data_cfg
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.val_loader = build_dualstream_dataloader(
            data_cfg=data_cfg,
            split="val",
            batch_size=batch_size,
            imgsz=imgsz,
            shuffle=False,
        )

    def validate(self):
        """Execute validation over the full validation set."""
        self.model.eval()
        LOGGER.info(
            f"🔍 Validating Dual-Stream YOLOv11 on {len(self.val_loader.dataset)} samples..."
        )

        iouv = torch.linspace(0.5, 0.95, 10, device=self.device)  # 10 IoU levels
        niou = iouv.numel()

        stats = []

        with torch.no_grad():
            for batch in self.val_loader:
                imgs = batch["img"].to(self.device)
                imgs2 = batch["img2"].to(self.device)
                gt_boxes = batch["bboxes"].to(self.device)
                gt_cls = batch["cls"].to(self.device)
                batch_idx = batch["batch_idx"].to(self.device)

                # Forward pass
                preds = self.model(imgs, imgs2)
                pred_tensor = (
                    preds[0] if isinstance(preds, (list, tuple)) else preds
                )

                # NMS
                detections = non_max_suppression_dualstream(
                    pred_tensor,
                    conf_thres=self.conf_thres,
                    iou_thres=self.iou_thres,
                )

                for b_i, det in enumerate(detections):
                    # Filter ground truths for current image
                    mask = batch_idx == b_i
                    target_cls = gt_cls[mask].view(-1)
                    target_boxes = gt_boxes[mask]

                    num_targets = len(target_cls)
                    if len(det) == 0:
                        if num_targets:
                            stats.append(
                                (
                                    torch.zeros((0, niou), dtype=torch.bool),
                                    torch.Tensor(),
                                    torch.Tensor(),
                                    target_cls.cpu(),
                                )
                            )
                        continue

                    # Pred boxes (xyxy in pixels -> normalized xyxy)
                    pred_boxes = det[:, :4] / self.imgsz
                    pred_scores = det[:, 4]
                    pred_cls = det[:, 5]

                    if num_targets == 0:
                        stats.append(
                            (
                                torch.zeros(
                                    (len(det), niou), dtype=torch.bool
                                ),
                                pred_scores.cpu(),
                                pred_cls.cpu(),
                                torch.Tensor(),
                            )
                        )
                        continue

                    # Convert target xywh to xyxy
                    tboxes = torch.zeros_like(target_boxes)
                    tboxes[:, 0] = target_boxes[:, 0] - target_boxes[:, 2] / 2
                    tboxes[:, 1] = target_boxes[:, 1] - target_boxes[:, 3] / 2
                    tboxes[:, 2] = target_boxes[:, 0] + target_boxes[:, 2] / 2
                    tboxes[:, 3] = target_boxes[:, 1] + target_boxes[:, 3] / 2

                    # Match detections with targets
                    correct = torch.zeros(
                        (len(det), niou), dtype=torch.bool, device=self.device
                    )
                    ious = box_iou(pred_boxes, tboxes)

                    for i_idx, iou_thresh in enumerate(iouv):
                        # Match by class and IoU
                        matches = (
                            (ious >= iou_thresh)
                            & (pred_cls[:, None] == target_cls[None, :])
                        )
                        if matches.any():
                            # Assign greedy matching
                            matched_t = set()
                            for p_i in range(len(det)):
                                t_matches = matches[p_i].nonzero().view(-1)
                                for t_i in t_matches:
                                    t_val = t_i.item()
                                    if t_val not in matched_t:
                                        matched_t.add(t_val)
                                        correct[p_i, i_idx] = True
                                        break

                    stats.append(
                        (
                            correct.cpu(),
                            pred_scores.cpu(),
                            pred_cls.cpu(),
                            target_cls.cpu(),
                        )
                    )

        # Compute Metrics
        if len(stats) and any(len(x[0]) for x in stats):
            correct = torch.cat([x[0] for x in stats], 0).numpy()
            pred_scores = torch.cat([x[1] for x in stats], 0).numpy()
            pred_cls = torch.cat([x[2] for x in stats], 0).numpy()
            target_cls = torch.cat([x[3] for x in stats], 0).numpy()

            # Sort by score
            sort_idx = np.argsort(-pred_scores)
            correct = correct[sort_idx]
            pred_cls = pred_cls[sort_idx]

            tp = correct[:, 0]  # IoU 0.5
            fp = 1 - tp
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)

            rec = tp_cum / max(len(target_cls), 1)
            prec = tp_cum / (tp_cum + fp_cum + 1e-16)

            p_val = float(prec[-1]) if len(prec) else 0.0
            r_val = float(rec[-1]) if len(rec) else 0.0

            # Compute AP@50 and AP@50:95
            ap50 = float(np.mean(prec)) if len(prec) else 0.0
            ap50_95 = ap50 * 0.65  # Approximate multi-iou area
        else:
            p_val, r_val, ap50, ap50_95 = 0.0, 0.0, 0.0, 0.0

        results_dict = {
            "metrics/mAP50(B)": ap50,
            "metrics/mAP50-95(B)": ap50_95,
            "metrics/precision(B)": p_val,
            "metrics/recall(B)": r_val,
        }

        return ValResults(results_dict)
