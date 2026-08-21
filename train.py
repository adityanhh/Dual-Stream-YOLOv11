"""
Training entrypoint for Dual-Stream YOLOv11.

Usage:
    python DualStreamYOLO11/train.py --model DualStreamYOLO11/configs/yolo11n-dualstream.yaml --data DualStreamYOLO11/configs/M3FD.yaml --epochs 100 --batch 8 --imgsz 640
"""

import argparse
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from DualStreamYOLO11.engine.trainer import DualStreamTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Dual-Stream YOLOv11 (DE-YOLOv11)")
    parser.add_argument(
        "--model",
        type=str,
        default="DualStreamYOLO11/configs/yolo11n-dualstream.yaml",
        help="Path to model YAML config",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="DualStreamYOLO11/configs/M3FD.yaml",
        help="Path to dataset YAML config",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (square)")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="0", help="CUDA device or 'cpu'")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Optional pretrained weights (e.g. yolo11n.pt)",
    )
    parser.add_argument("--project", type=str, default="runs/train", help="Save project directory")
    parser.add_argument("--name", type=str, default="exp", help="Save experiment name")
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = DualStreamTrainer(
        model_cfg=args.model,
        data_cfg=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr,
        device=args.device,
        pretrained_weights=args.pretrained,
        project=args.project,
        name=args.name,
    )
    trainer.train()


if __name__ == "__main__":
    main()
