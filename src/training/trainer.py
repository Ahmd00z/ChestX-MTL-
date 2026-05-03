"""
ChestX-MTL Training Module
Advanced trainer with mixed precision, gradient accumulation, and logging.
"""
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional
from tqdm import tqdm
import json


class Trainer:
    """
    Professional training loop for ChestX-MTL.

    Features:
    - Mixed Precision Training (FP16)
    - Gradient Accumulation
    - Gradient Clipping
    - Cosine Annealing with Warmup
    - TensorBoard Logging
    - Checkpoint Management
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        train_loader,
        val_loader,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs"
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"]
        )

        # Scheduler with warmup
        total_steps = len(train_loader) * config["training"]["epochs"]
        warmup_steps = len(train_loader) * config["training"]["warmup_epochs"]

        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_steps - warmup_steps
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

        # Mixed precision
        self.use_amp = config["training"].get("mixed_precision", True)
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient accumulation
        self.accum_steps = config["training"].get("accumulate_grad_batches", 1)

        # Logging
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Metrics tracking
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_cls_loss": [], "val_cls_loss": [],
            "train_seg_loss": [], "val_seg_loss": []
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        total_cls_loss = 0
        total_seg_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)

            targets = {}
            if "cls_labels" in batch:
                targets["cls_labels"] = batch["cls_labels"].to(self.device)
            if "seg_masks" in batch:
                targets["seg_masks"] = batch["seg_masks"].to(self.device)

            # Forward with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                losses = self.model.compute_loss(outputs, targets)
                loss = losses["combined_loss"] / self.accum_steps

            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["gradient_clip"]
                )

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()

            # Logging
            total_loss += loss.item() * self.accum_steps
            if "cls_loss" in losses:
                total_cls_loss += losses["cls_loss"].item()
            if "seg_loss" in losses:
                total_seg_loss += losses["seg_loss"].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item() * self.accum_steps:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # TensorBoard step logging
            if self.global_step % 50 == 0:
                self.writer.add_scalar("train/step_loss", loss.item() * self.accum_steps, self.global_step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)

            self.global_step += 1

        avg_loss = total_loss / num_batches
        avg_cls = total_cls_loss / num_batches if total_cls_loss > 0 else 0
        avg_seg = total_seg_loss / num_batches if total_seg_loss > 0 else 0

        return {"loss": avg_loss, "cls_loss": avg_cls, "seg_loss": avg_seg}

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate one epoch."""
        self.model.eval()
        total_loss = 0
        total_cls_loss = 0
        total_seg_loss = 0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")

        for batch in pbar:
            images = batch["image"].to(self.device)

            targets = {}
            if "cls_labels" in batch:
                targets["cls_labels"] = batch["cls_labels"].to(self.device)
            if "seg_masks" in batch:
                targets["seg_masks"] = batch["seg_masks"].to(self.device)

            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                losses = self.model.compute_loss(outputs, targets)
                loss = losses["combined_loss"]

            total_loss += loss.item()
            if "cls_loss" in losses:
                total_cls_loss += losses["cls_loss"].item()
            if "seg_loss" in losses:
                total_seg_loss += losses["seg_loss"].item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        avg_cls = total_cls_loss / num_batches if total_cls_loss > 0 else 0
        avg_seg = total_seg_loss / num_batches if total_seg_loss > 0 else 0

        return {"loss": avg_loss, "cls_loss": avg_cls, "seg_loss": avg_seg}

    def fit(self, epochs: Optional[int] = None):
        """Full training loop."""
        if epochs is None:
            epochs = self.config["training"]["epochs"]

        print(f"\n{'='*60}")
        print(f"Starting Training: {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Gradient Accumulation: {self.accum_steps}")
        print(f"Trainable Params: {self.model.get_trainable_params():,}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            epoch_time = time.time() - start_time

            # Logging
            self.writer.add_scalar("train/epoch_loss", train_metrics["loss"], epoch)
            self.writer.add_scalar("train/cls_loss", train_metrics["cls_loss"], epoch)
            self.writer.add_scalar("train/seg_loss", train_metrics["seg_loss"], epoch)
            self.writer.add_scalar("val/epoch_loss", val_metrics["loss"], epoch)
            self.writer.add_scalar("val/cls_loss", val_metrics["cls_loss"], epoch)
            self.writer.add_scalar("val/seg_loss", val_metrics["seg_loss"], epoch)

            # History
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_cls_loss"].append(train_metrics["cls_loss"])
            self.history["val_cls_loss"].append(val_metrics["cls_loss"])
            self.history["train_seg_loss"].append(train_metrics["seg_loss"])
            self.history["val_seg_loss"].append(val_metrics["seg_loss"])

            # Print summary
            print(f"\nEpoch {epoch}/{epochs} - {epoch_time:.1f}s")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | CLS: {train_metrics['cls_loss']:.4f} | SEG: {train_metrics['seg_loss']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | CLS: {val_metrics['cls_loss']:.4f} | SEG: {val_metrics['seg_loss']:.4f}")

            # Checkpointing
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"  ✓ New best model saved! (val_loss: {val_metrics['loss']:.4f})")
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, val_metrics, is_best=False)

            # Early stopping
            if self.patience_counter >= self.config["training"]["early_stopping"]["patience"]:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

        self.writer.close()
        self.save_history()
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "best_val_loss": self.best_val_loss
        }

        if self.use_amp:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save latest
        path = os.path.join(self.checkpoint_dir, "latest.pth")
        torch.save(checkpoint, path)

        # Save best
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)

        # Save periodic
        if epoch % 10 == 0:
            periodic_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:03d}.pth")
            torch.save(checkpoint, periodic_path)

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.use_amp and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        return checkpoint["epoch"]

    def save_history(self):
        """Save training history."""
        with open(os.path.join(self.log_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)


if __name__ == "__main__":
    print("Trainer module ready!")
