"""
Inference & Prediction Engine for Dual-Stream YOLOv11.
Supports:
- Dual-image input inference (pair of RGB and IR images)
- Non-Maximum Suppression (NMS) post-processing
- Visualization and saving of detections on both modalities
"""

from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision.ops import nms

from ultralytics.utils import LOGGER
from ultralytics.utils.ops import scale_boxes

from ..data.augment import synchronized_letterbox
from ..models.dualstream_model import DualStreamDetectionModel


def non_max_suppression_dualstream(prediction, conf_thres=0.25, iou_thres=0.45):
    """
    Perform Non-Maximum Suppression (NMS) on Dual-Stream YOLOv11 predictions.
    prediction shape: [B, 4 + nc, num_anchors]
    """
    # Transpose to [B, num_anchors, 4 + nc]
    if prediction.shape[1] < prediction.shape[2]:
        prediction = prediction.transpose(1, 2)

    output = []
    for xi, x in enumerate(prediction):
        boxes_cxcywh = x[:, :4]
        scores = x[:, 4:]
        max_scores, class_ids = scores.max(dim=1)

        mask = max_scores > conf_thres
        if not mask.any():
            output.append(torch.zeros((0, 6), device=x.device))
            continue

        boxes_filt = boxes_cxcywh[mask]
        scores_filt = max_scores[mask]
        class_ids_filt = class_ids[mask]

        xyxy = torch.zeros_like(boxes_filt)
        xyxy[:, 0] = boxes_filt[:, 0] - boxes_filt[:, 2] / 2
        xyxy[:, 1] = boxes_filt[:, 1] - boxes_filt[:, 3] / 2
        xyxy[:, 2] = boxes_filt[:, 0] + boxes_filt[:, 2] / 2
        xyxy[:, 3] = boxes_filt[:, 1] + boxes_filt[:, 3] / 2

        # Class-aware offset
        c = class_ids_filt.float() * 4096
        keep_indices = nms(xyxy + c.unsqueeze(1), scores_filt, iou_thres)

        detections = torch.cat(
            [
                xyxy[keep_indices],
                scores_filt[keep_indices].unsqueeze(1),
                class_ids_filt[keep_indices].unsqueeze(1).float(),
            ],
            dim=1,
        )
        output.append(detections)

    return output


class DualStreamPredictor:
    """
    Predictor class for running inference on paired RGB and IR inputs with Dual-Stream YOLOv11.
    """

    def __init__(
        self,
        weights='best.pt',
        model_cfg='yolo11n-dualstream.yaml',
        imgsz=640,
        conf=0.25,
        iou=0.45,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.device = torch.device(device)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

        LOGGER.info(f"Loading Dual-Stream YOLOv11 for inference from {model_cfg}...")
        self.model = DualStreamDetectionModel(cfg=model_cfg, ch=3, ch2=3)
        if weights and Path(weights).exists():
            ckpt = torch.load(weights, map_location='cpu')
            state_dict = ckpt['model'] if 'model' in ckpt else ckpt
            self.model.load_state_dict(state_dict, strict=False)
            LOGGER.info(f"Loaded weights from {weights}")

        self.model.to(self.device)
        self.model.eval()

    def predict_pair(self, vis_image_path, ir_image_path, save=True, save_dir='runs/predict'):
        """
        Run inference on a single (RGB, IR) image pair.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Read images
        img_vis_orig = cv2.imread(str(vis_image_path))
        img_ir_orig = cv2.imread(str(ir_image_path))
        if img_vis_orig is None or img_ir_orig is None:
            raise FileNotFoundError(f"Failed to read image pair: {vis_image_path}, {ir_image_path}")

        h0, w0 = img_vis_orig.shape[:2]

        # 2. Preprocess & Letterbox
        img_vis, img_ir, ratio, pad = synchronized_letterbox(
            img_vis_orig, img_ir_orig, new_shape=self.imgsz, auto=False
        )

        # HWC BGR to CHW RGB
        img_vis_rgb = np.ascontiguousarray(img_vis[:, :, ::-1].transpose(2, 0, 1))
        img_ir_rgb = np.ascontiguousarray(img_ir[:, :, ::-1].transpose(2, 0, 1))
        t_vis = torch.from_numpy(img_vis_rgb).float() / 255.0
        t_ir = torch.from_numpy(img_ir_rgb).float() / 255.0

        t_vis = t_vis.unsqueeze(0).to(self.device)
        t_ir = t_ir.unsqueeze(0).to(self.device)

        # 3. Model Forward Pass
        with torch.no_grad():
            preds = self.model(t_vis, t_ir)
            pred_tensor = preds[0] if isinstance(preds, (list, tuple)) else preds

        # 4. NMS
        detections = non_max_suppression_dualstream(
            pred_tensor, conf_thres=self.conf, iou_thres=self.iou
        )[0]

        results = []
        if len(detections):
            # Rescale boxes to original image shape
            detections[:, :4] = scale_boxes(
                (self.imgsz, self.imgsz), detections[:, :4], (h0, w0)
            ).round()

            for *xyxy, conf_val, cls_id in detections.cpu().numpy():
                results.append(
                    {
                        'box': [float(x) for x in xyxy],
                        'conf': float(conf_val),
                        'cls': int(cls_id),
                        'name': self.model.names.get(int(cls_id), str(int(cls_id))),
                    }
                )

        # 5. Draw Detections if save=True
        if save:
            out_vis = img_vis_orig.copy()
            out_ir = img_ir_orig.copy()

            for det in results:
                x1, y1, x2, y2 = [int(v) for v in det['box']]
                label = f"{det['name']} {det['conf']:.2f}"
                color = (0, 255, 0)

                # Draw on Visible
                cv2.rectangle(out_vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    out_vis,
                    label,
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

                # Draw on IR
                cv2.rectangle(out_ir, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    out_ir,
                    label,
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            stem = Path(vis_image_path).stem
            vis_save_path = save_dir / f"{stem}_vis_pred.jpg"
            ir_save_path = save_dir / f"{stem}_ir_pred.jpg"
            combined_save_path = save_dir / f"{stem}_dual_pred.jpg"

            cv2.imwrite(str(vis_save_path), out_vis)
            cv2.imwrite(str(ir_save_path), out_ir)

            # Combined side-by-side visualization
            if out_vis.shape == out_ir.shape:
                combined = np.hstack([out_vis, out_ir])
                cv2.imwrite(str(combined_save_path), combined)

            LOGGER.info(f"✨ Saved prediction images to {save_dir}")

        return results
