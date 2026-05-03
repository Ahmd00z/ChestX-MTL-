#!/usr/bin/env python3
"""
ChestX-MTL Evaluation Script
Usage: python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --data_dir /path/to/data
"""
import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.mtl_model import ChestXMTL
from src.data.dataset import get_dataloaders
from src.utils.helpers import load_config, get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ChestX-MTL")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    return parser.parse_args()


def compute_cls_metrics(preds, targets, threshold=0.5):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    probs = torch.sigmoid(torch.tensor(preds)).numpy()
    preds_bin = (probs > threshold).astype(float)
    metrics = {}
    for i in range(targets.shape[1]):
        if targets[:, i].sum() > 0:
            metrics[f"class_{i}_auc"] = roc_auc_score(targets[:, i], probs[:, i])
    metrics["accuracy"] = accuracy_score(targets, preds_bin)
    metrics["f1_macro"] = f1_score(targets, preds_bin, average="macro", zero_division=0)
    metrics["precision_macro"] = precision_score(targets, preds_bin, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(targets, preds_bin, average="macro", zero_division=0)
    return metrics


def compute_seg_metrics(preds, targets, threshold=0.5):
    preds = (preds > threshold).astype(float)
    intersection = (preds * targets).sum()
    union = ((preds + targets) > 0).sum()
    dice = (2 * intersection) / (preds.sum() + targets.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    return {"dice": dice, "iou": iou}


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    device = get_device(args.device)

    print("Loading model...")
    model_config = {
        "encoder": config["model"]["encoder"],
        "classifier": config["model"]["classifier"],
        "detector": config["model"]["detector"],
        "segmenter": config["model"]["segmenter"],
        "mtl": config["model"]["mtl"],
        "losses": config["losses"]
    }
    model = ChestXMTL(model_config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded epoch {checkpoint['epoch']}")

    dataloaders = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        image_size=config["data"]["image_size"]
    )
    loader = dataloaders[args.split]

    print(f"Evaluating {args.split} ({len(loader.dataset)} samples)...")
    all_cls_preds, all_cls_targets = [], []
    all_seg_preds, all_seg_targets = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            images = batch["image"].to(device)
            outputs = model.predict(images)
            if "cls_labels" in batch:
                all_cls_preds.append(outputs["cls_probs"].cpu().numpy())
                all_cls_targets.append(batch["cls_labels"].numpy())
            if "seg_masks" in batch:
                all_seg_preds.append(outputs["seg_mask"].cpu().numpy())
                all_seg_targets.append(batch["seg_masks"].numpy())

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    if all_cls_preds:
        cls_metrics = compute_cls_metrics(np.concatenate(all_cls_preds), np.concatenate(all_cls_targets))
        print("\nClassification:")
        for k, v in cls_metrics.items():
            print(f"  {k}: {v:.4f}")

    if all_seg_preds:
        seg_metrics = compute_seg_metrics(np.concatenate(all_seg_preds), np.concatenate(all_seg_targets))
        print("\nSegmentation:")
        for k, v in seg_metrics.items():
            print(f"  {k}: {v:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
