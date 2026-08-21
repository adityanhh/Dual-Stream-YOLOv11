"""
Prediction and Inference entrypoint for Dual-Stream YOLOv11.

Usage:
    python DualStreamYOLO11/predict.py --weights runs/train/exp/best.pt --vis-img path/to/vis.jpg --ir-img path/to/ir.jpg --conf 0.25
"""

import argparse
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DualStreamYOLO11.engine.predictor import DualStreamPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with Dual-Stream YOLOv11")
    parser.add_argument(
        "--weights",
        type=str,
        default="best.pt",
        help="Path to trained checkpoint (e.g. best.pt)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DualStreamYOLO11/configs/yolo11n-dualstream.yaml",
        help="Path to model YAML config",
    )
    parser.add_argument(
        "--vis-img",
        type=str,
        required=True,
        help="Path to Visible (RGB) input image",
    )
    parser.add_argument(
        "--ir-img",
        type=str,
        required=True,
        help="Path to Infrared (IR) input image",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="0", help="CUDA device or 'cpu'")
    parser.add_argument("--save-dir", type=str, default="runs/predict", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    predictor = DualStreamPredictor(
        weights=args.weights,
        model_cfg=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )
    results = predictor.predict_pair(
        vis_image_path=args.vis_img,
        ir_image_path=args.ir_img,
        save=True,
        save_dir=args.save_dir,
    )
    print(f"\nDetection Results: {len(results)} objects found:")
    for r in results:
        print(f" - Class: {r['name']} | Confidence: {r['conf']:.2f} | BBox: {r['box']}")


if __name__ == "__main__":
    main()
