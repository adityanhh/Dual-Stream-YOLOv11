"""
Dual-Input Cross-Modality Dataset Loader for Dual-Stream YOLOv11.
Loads paired Visible (RGB) and Infrared (IR) images with labels,
and constructs batch dictionaries compatible with YOLOv11 loss & metric calculation.
"""

from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from .augment import synchronized_letterbox, synchronized_random_flip


class DualStreamDataset(Dataset):
    """
    Dataset for loading synchronized image pairs:
    - Visible (RGB) images
    - Infrared (IR) images
    - Corresponding object detection labels (.txt YOLO format)
    """

    def __init__(self, data_dict, split='train', imgsz=640, augment=True):
        super().__init__()
        self.imgsz = imgsz
        self.augment = augment
        self.split = split

        if isinstance(data_dict, (str, Path)):
            with open(data_dict, 'r') as f:
                data_dict = yaml.safe_load(f)

        self.data_dict = data_dict
        self.names = data_dict.get('names', {0: 'person'})
        self.nc = len(self.names) if isinstance(self.names, dict) else len(self.names)

        # Parse directory paths
        base_dir = Path(data_dict.get('path', '.'))
        if split == 'train':
            vis_dir = base_dir / data_dict.get('train_vis', data_dict.get('train', 'images/vis_train'))
            ir_dir = base_dir / data_dict.get('train_ir', 'images/Ir_train')
            label_dir = base_dir / data_dict.get('train_labels', 'labels/vis_train')
        else:
            vis_dir = base_dir / data_dict.get('val_vis', data_dict.get('val', 'images/vis_val'))
            ir_dir = base_dir / data_dict.get('val_ir', 'images/Ir_val')
            label_dir = base_dir / data_dict.get('val_labels', 'labels/vis_val')

        self.vis_dir = Path(vis_dir)
        self.ir_dir = Path(ir_dir)
        self.label_dir = Path(label_dir)

        # Collect file pairs
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
        self.vis_files = []
        self.ir_files = []
        self.label_files = []

        if self.vis_dir.exists():
            for vis_path in sorted(self.vis_dir.iterdir()):
                if vis_path.suffix.lower() in valid_exts:
                    ir_path = self.ir_dir / vis_path.name
                    # Fallback matching if different naming / extension
                    if not ir_path.exists():
                        potential = list(self.ir_dir.glob(f"{vis_path.stem}.*"))
                        if not potential:
                            stem_variations = [
                                vis_path.stem.replace('_vi', '_ir'),
                                vis_path.stem.replace('vi_', 'ir_'),
                                vis_path.stem.replace('vis', 'ir'),
                                vis_path.stem.replace('vi', 'ir'),
                                vis_path.stem.replace('RGB', 'IR'),
                                vis_path.stem.replace('rgb', 'ir'),
                            ]
                            for alt_stem in stem_variations:
                                potential = list(self.ir_dir.glob(f"{alt_stem}.*"))
                                if potential:
                                    break
                        if potential:
                            ir_path = potential[0]

                    lb_path = self.label_dir / f"{vis_path.stem}.txt"
                    if not lb_path.exists():
                        for alt_stem in [
                            vis_path.stem.replace('_vi', ''),
                            vis_path.stem.replace('vi_', ''),
                            vis_path.stem.replace('_vis', ''),
                            vis_path.stem.replace('vis_', ''),
                        ]:
                            alt_lb = self.label_dir / f"{alt_stem}.txt"
                            if alt_lb.exists():
                                lb_path = alt_lb
                                break

                    if ir_path.exists():
                        self.vis_files.append(vis_path)
                        self.ir_files.append(ir_path)
                        self.label_files.append(lb_path if lb_path.exists() else None)

    def __len__(self):
        return len(self.vis_files)

    def __getitem__(self, index):
        # 1. Load Visible (RGB) image
        vis_path = str(self.vis_files[index])
        img_vis = cv2.imread(vis_path)
        if img_vis is None:
            raise FileNotFoundError(f"Image not found: {vis_path}")

        # 2. Load Infrared (IR) image
        ir_path = str(self.ir_files[index])
        img_ir = cv2.imread(ir_path)
        if img_ir is None:
            raise FileNotFoundError(f"Image not found: {ir_path}")

        # 3. Load Labels [class_id, x_center, y_center, width, height] (normalized)
        lb_path = self.label_files[index]
        labels = []
        if lb_path and lb_path.exists():
            with open(lb_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(float(parts[0]))
                        box = [float(p) for p in parts[1:5]]
                        labels.append([cls_id, *box])

        labels = np.array(labels, dtype=np.float32) if len(labels) > 0 else np.zeros((0, 5), dtype=np.float32)

        # 4. Synchronized Resize & Letterbox
        img_vis, img_ir, ratio, pad = synchronized_letterbox(
            img_vis, img_ir, new_shape=self.imgsz, auto=False, scaleup=self.augment
        )

        # Convert labels from normalized xywh to xyxy on padded image
        h, w = img_vis.shape[:2]
        if len(labels) > 0:
            cls_ids = labels[:, 0:1]
            xywh = labels[:, 1:5]
            xyxy = np.zeros_like(xywh)
            xyxy[:, 0] = (xywh[:, 0] - xywh[:, 2] / 2) * w
            xyxy[:, 1] = (xywh[:, 1] - xywh[:, 3] / 2) * h
            xyxy[:, 2] = (xywh[:, 0] + xywh[:, 2] / 2) * w
            xyxy[:, 3] = (xywh[:, 1] + xywh[:, 3] / 2) * h

            # 5. Synchronized Random Horizontal Flip during training
            if self.augment:
                # Normalized bbox for flip
                xyxy_norm = xyxy.copy()
                xyxy_norm[:, [0, 2]] /= w
                xyxy_norm[:, [1, 3]] /= h
                img_vis, img_ir, xyxy_norm = synchronized_random_flip(img_vis, img_ir, xyxy_norm, p_lr=0.5)
                xyxy[:, [0, 2]] = xyxy_norm[:, [0, 2]] * w
                xyxy[:, [1, 3]] = xyxy_norm[:, [1, 3]] * h

            # Normalize xywh back for YOLO loss
            bboxes = np.zeros_like(xyxy)
            bboxes[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2 / w
            bboxes[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2 / h
            bboxes[:, 2] = (xyxy[:, 2] - xyxy[:, 0]) / w
            bboxes[:, 3] = (xyxy[:, 3] - xyxy[:, 1]) / h
        else:
            cls_ids = np.zeros((0, 1), dtype=np.float32)
            bboxes = np.zeros((0, 4), dtype=np.float32)

        # Convert HWC BGR to CHW RGB
        img_vis = img_vis[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img_ir = img_ir[:, :, ::-1].transpose(2, 0, 1)

        img_vis = np.ascontiguousarray(img_vis, dtype=np.float32) / 255.0
        img_ir = np.ascontiguousarray(img_ir, dtype=np.float32) / 255.0

        return {
            'img': torch.from_numpy(img_vis),
            'img2': torch.from_numpy(img_ir),
            'cls': torch.from_numpy(cls_ids),
            'bboxes': torch.from_numpy(bboxes),
            'im_file': vis_path,
            'im_file2': ir_path,
            'ori_shape': (h, w),
            'resized_shape': (h, w),
        }


def dualstream_collate_fn(batch):
    """
    Collate function to assemble variable number of bounding boxes per sample into batch.
    """
    imgs = torch.stack([item['img'] for item in batch], dim=0)
    imgs2 = torch.stack([item['img2'] for item in batch], dim=0)

    batch_idx_list = []
    cls_list = []
    bboxes_list = []

    for i, item in enumerate(batch):
        num_boxes = len(item['cls'])
        if num_boxes > 0:
            batch_idx_list.append(torch.full((num_boxes,), i, dtype=torch.float32))
            cls_list.append(item['cls'].view(-1))
            bboxes_list.append(item['bboxes'])

    batch_dict = {
        'img': imgs,
        'img2': imgs2,
        'batch_idx': torch.cat(batch_idx_list, dim=0) if batch_idx_list else torch.zeros((0,), dtype=torch.float32),
        'cls': torch.cat(cls_list, dim=0).view(-1, 1) if cls_list else torch.zeros((0, 1), dtype=torch.float32),
        'bboxes': torch.cat(bboxes_list, dim=0) if bboxes_list else torch.zeros((0, 4), dtype=torch.float32),
        'im_file': [item['im_file'] for item in batch],
        'im_file2': [item['im_file2'] for item in batch],
    }

    return batch_dict


def build_dualstream_dataloader(data_cfg, split='train', batch_size=8, imgsz=640, num_workers=2, shuffle=True):
    """Builds a PyTorch DataLoader for dual-stream training or evaluation."""
    dataset = DualStreamDataset(data_dict=data_cfg, split=split, imgsz=imgsz, augment=(split == 'train'))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dualstream_collate_fn,
        pin_memory=True,
    )
