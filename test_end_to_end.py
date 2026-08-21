"""
End-to-End Pipeline Integration Test for Dual-Stream YOLOv11.
Tests:
1. Mock RGB-IR Dataset Creation
2. Multi-modal Dataloading & Batch collation
3. Mini Training Loop (1-2 epochs) with DualStreamTrainer
4. Model checkpoint saving
5. Inference & Detection visualization with DualStreamPredictor
"""

from pathlib import Path
import shutil
import sys
import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from DualStreamYOLO11.engine.trainer import DualStreamTrainer
from DualStreamYOLO11.engine.predictor import DualStreamPredictor


def create_mock_dataset(root_dir):
    """Create a mini dataset with 4 pairs of synthetic images."""
    root = Path(root_dir)
    vis_train = root / "images" / "vis_train"
    ir_train = root / "images" / "Ir_train"
    lbl_train = root / "labels" / "vis_train"

    vis_val = root / "images" / "vis_val"
    ir_val = root / "images" / "Ir_val"
    lbl_val = root / "labels" / "vis_val"

    for d in [vis_train, ir_train, lbl_train, vis_val, ir_val, lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    # Generate synthetic images with simulated objects (white boxes)
    for i in range(4):
        split_vis = vis_train if i < 3 else vis_val
        split_ir = ir_train if i < 3 else ir_val
        split_lbl = lbl_train if i < 3 else lbl_val

        # Synthetic RGB image
        img_vis = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img_vis, (100, 100), (300, 300), (0, 255, 0), -1)  # Green box
        cv2.imwrite(str(split_vis / f"sample_{i}.jpg"), img_vis)

        # Synthetic IR image
        img_ir = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img_ir, (100, 100), (300, 300), (255, 255, 255), -1)  # Hot thermal signature
        cv2.imwrite(str(split_ir / f"sample_{i}.jpg"), img_ir)

        # Label: class 0, center=(0.3125, 0.4166), size=(0.3125, 0.4166)
        with open(split_lbl / f"sample_{i}.txt", "w") as f:
            f.write("0 0.3125 0.4166 0.3125 0.4166\n")

    yaml_dict = {
        'path': str(root.resolve()),
        'train_vis': 'images/vis_train',
        'train_ir': 'images/Ir_train',
        'train_labels': 'labels/vis_train',
        'val_vis': 'images/vis_val',
        'val_ir': 'images/Ir_val',
        'val_labels': 'labels/vis_val',
        'nc': 1,
        'names': {0: 'target_object'},
    }

    yaml_path = root / "mock_dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f)

    return yaml_path


def run_pipeline_test():
    print("=" * 70)
    print("🧪 Running Dual-Stream YOLOv11 End-to-End Pipeline Integration Test")
    print("=" * 70)

    test_root = Path("DualStreamYOLO11/temp_mock_dataset")
    if test_root.exists():
        shutil.rmtree(test_root)

    dataset_yaml = create_mock_dataset(test_root)
    print(f"✅ Mock dataset created at: {test_root}")

    # 1. Test Training (1 Epoch)
    model_cfg = Path(__file__).parent / "configs" / "yolo11n-dualstream.yaml"
    runs_dir = Path("DualStreamYOLO11/temp_runs")
    if runs_dir.exists():
        shutil.rmtree(runs_dir)

    trainer = DualStreamTrainer(
        model_cfg=str(model_cfg),
        data_cfg=str(dataset_yaml),
        epochs=1,
        batch_size=2,
        imgsz=320,  # small size for quick test
        lr0=0.01,
        device='cpu',
        project=str(runs_dir),
        name="test_exp",
    )

    save_dir = trainer.train()
    best_weights = save_dir / "best.pt"
    assert best_weights.exists(), "best.pt was not saved!"
    print(f"✅ Training completed and checkpoint verified at: {best_weights}")

    # 2. Test Prediction & Visualization
    predictor = DualStreamPredictor(
        weights=str(best_weights),
        model_cfg=str(model_cfg),
        imgsz=320,
        conf=0.01,  # Low conf for untrained test
        device='cpu',
    )

    test_vis = test_root / "images" / "vis_val" / "sample_3.jpg"
    test_ir = test_root / "images" / "Ir_val" / "sample_3.jpg"
    pred_dir = runs_dir / "predict_output"

    results = predictor.predict_pair(
        vis_image_path=test_vis,
        ir_image_path=test_ir,
        save=True,
        save_dir=pred_dir,
    )
    print(f"✅ Prediction inference completed! Found {len(results)} candidate boxes.")

    # Cleanup temporary test directory
    if test_root.exists():
        shutil.rmtree(test_root)
    if runs_dir.exists():
        shutil.rmtree(runs_dir)

    print("\n" + "=" * 70)
    print("🏆 ALL INTEGRATION PIPELINE TESTS PASSED 100%!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline_test()
