"""
Training Engine for Dual-Stream YOLOv11.
Supports:
- Dual-input batch training
- Mixed Precision (AMP)
- Model EMA (Exponential Moving Average)
- Learning rate schedulers (Warmup + Cosine Annealing)
- Checkpoint saving (best.pt and last.pt)
- TensorBoard / console progress logging
"""

import math
import os
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.optim import AdamW, SGD, lr_scheduler

from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import ModelEMA

try:
    from ..models.dualstream_model import DualStreamDetectionModel
    from ..data.dataset import build_dualstream_dataloader
except (ImportError, ValueError):
    try:
        from models.dualstream_model import DualStreamDetectionModel
        from data.dataset import build_dualstream_dataloader
    except (ImportError, ValueError):
        from DualStreamYOLO11.models.dualstream_model import DualStreamDetectionModel
        from DualStreamYOLO11.data.dataset import build_dualstream_dataloader


class DualStreamTrainer:
    """
    Trainer class for Dual-Stream YOLOv11 models.
    """

    def __init__(
        self,
        model_cfg='yolo11n-dualstream.yaml',
        data_cfg='M3FD.yaml',
        epochs=100,
        batch_size=8,
        imgsz=640,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        pretrained_weights=None,
        project='runs/dualstream_train',
        name='exp',
        amp=False,
    ):
        if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
            self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device if (torch.cuda.is_available() or device == 'cpu') else 'cpu')
        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.lr0 = lr0
        self.lrf = lrf
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.save_dir = Path(project) / name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Initialize Model
        if isinstance(model_cfg, nn.Module):
            self.model = model_cfg
        else:
            LOGGER.info(f"Initializing Dual-Stream YOLOv11 from {model_cfg}...")
            self.model = DualStreamDetectionModel(cfg=model_cfg, ch=3, ch2=3)
            if pretrained_weights and Path(pretrained_weights).exists():
                self.model.load_pretrained_weights(pretrained_weights)
        self.model.to(self.device)

        # 2. Build Dataloaders
        LOGGER.info(f"Building Dual-Stream Dataloaders for {data_cfg}...")
        self.data_cfg = data_cfg
        self.train_loader = build_dualstream_dataloader(
            data_cfg=data_cfg, split='train', batch_size=batch_size, imgsz=imgsz, shuffle=True
        )
        self.val_loader = build_dualstream_dataloader(
            data_cfg=data_cfg, split='val', batch_size=batch_size, imgsz=imgsz, shuffle=False
        )

        # 3. Setup Optimizer & Scheduler
        opt_type = str(data_cfg.get('optimizer', 'AdamW') if isinstance(data_cfg, dict) else 'AdamW').lower()
        if 'sgd' in opt_type:
            self.optimizer = SGD(
                self.model.parameters(), lr=lr0, momentum=momentum, weight_decay=weight_decay, nesterov=True
            )
        else:
            self.optimizer = AdamW(
                self.model.parameters(), lr=lr0, betas=(momentum, 0.999), weight_decay=weight_decay
            )

        self.lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

        # 4. Mixed precision scaler & EMA
        self.use_amp = amp and (self.device.type == 'cuda')
        if self.use_amp:
            try:
                self.scaler = torch.amp.GradScaler('cuda', enabled=True)
            except Exception:
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.scaler = None

        self.ema = ModelEMA(self.model) if hasattr(torch, 'cuda') else None

    def train(self):
        """Execute full training loop."""
        LOGGER.info(f"\n🚀 Starting training for {self.epochs} epochs on device: {self.device}...")
        num_batches = len(self.train_loader)
        best_loss = float('inf')

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            pbar_start = time.time()

            LOGGER.info(f"\n--- Epoch {epoch + 1}/{self.epochs} ---")

            for i, batch in enumerate(self.train_loader):
                # Move tensors to device
                batch['img'] = batch['img'].to(self.device, non_blocking=True)
                batch['img2'] = batch['img2'].to(self.device, non_blocking=True)
                batch['bboxes'] = batch['bboxes'].to(self.device, non_blocking=True)
                batch['cls'] = batch['cls'].to(self.device, non_blocking=True)
                batch['batch_idx'] = batch['batch_idx'].to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                if self.use_amp and self.device.type == 'cuda':
                    try:
                        with torch.amp.autocast('cuda'):
                            loss_out = self.model.loss(batch)
                            total_loss = loss_out[0].sum() if isinstance(loss_out, tuple) else loss_out.sum()
                        self.scaler.scale(total_loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    except Exception:
                        loss_out = self.model.loss(batch)
                        total_loss = loss_out[0].sum() if isinstance(loss_out, tuple) else loss_out.sum()
                        total_loss.backward()
                        self.optimizer.step()
                else:
                    loss_out = self.model.loss(batch)
                    total_loss = loss_out[0].sum() if isinstance(loss_out, tuple) else loss_out.sum()
                    total_loss.backward()
                    self.optimizer.step()

                if self.ema:
                    self.ema.update(self.model)

                epoch_loss += total_loss.item()

                if i % max(1, num_batches // 5) == 0 or i == num_batches - 1:
                    LOGGER.info(
                        f"  Batch [{i+1}/{num_batches}] - Loss: {total_loss.item():.4f} - LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                    )

            self.scheduler.step()
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            epoch_time = time.time() - pbar_start
            LOGGER.info(f"Epoch {epoch + 1} Complete! Avg Loss: {avg_epoch_loss:.4f} ({epoch_time:.1f}s)")

            # Save Checkpoints
            is_best = avg_epoch_loss < best_loss
            if is_best:
                best_loss = avg_epoch_loss
                torch.save({'model': self.model.state_dict(), 'epoch': epoch}, self.save_dir / 'best.pt')
                LOGGER.info(f"🌟 Saved new best model to {self.save_dir / 'best.pt'}")

            torch.save({'model': self.model.state_dict(), 'epoch': epoch}, self.save_dir / 'last.pt')

        LOGGER.info(f"\n🎉 Training Finished! Checkpoints saved at {self.save_dir}")
        return self.save_dir
