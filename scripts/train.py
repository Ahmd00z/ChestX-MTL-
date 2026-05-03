#!/usr/bin/env python3
"""
ChestX-MTL Training Script
Usage: python scripts/train.py --config config/config.yaml --data_dir /path/to/data
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.mtl_model import ChestXMTL
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer
from src.utils.helpers import set_seed, load_config, get_device, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Train ChestX-MTL model")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.epochs: config["training"]["epochs"] = args.epochs
    if args.batch_size: config["data"]["batch_size"] = args.batch_size
    if args.lr: config["training"]["lr"] = args.lr

    set_seed(config["project"]["seed"])
    device = get_device(args.device)
    print(f"Using device: {device}")

    print("Loading datasets...")
    dataloaders = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        image_size=config["data"]["image_size"]
    )
    print(f"Train: {len(dataloaders['train'].dataset)} | Val: {len(dataloaders['val'].dataset)} | Test: {len(dataloaders['test'].dataset)}")

    print("Building model...")
    model_config = {
        "encoder": config["model"]["encoder"],
        "classifier": config["model"]["classifier"],
        "detector": config["model"]["detector"],
        "segmenter": config["model"]["segmenter"],
        "mtl": config["model"]["mtl"],
        "losses": config["losses"]
    }
    model = ChestXMTL(model_config)
    params = count_parameters(model)
    print(f"Params: {params['total']:,} total | {params['trainable']:,} trainable")

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    log_dir = os.path.join(args.output_dir, "logs")

    trainer = Trainer(
        model=model, config=config,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        device=str(device),
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )

    if args.checkpoint:
        print(f"Resuming from {args.checkpoint}...")
        trainer.load_checkpoint(args.checkpoint)

    trainer.fit(epochs=config["training"]["epochs"])
    print(f"\nDone! Checkpoints: {checkpoint_dir} | Logs: {log_dir}")


if __name__ == "__main__":
    main()
