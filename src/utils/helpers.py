"""
ChestX-MTL Utility Functions
"""
import os
import random
import numpy as np
import torch
import yaml
from typing import Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], path: str):
    """Save configuration to YAML."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def get_device(prefer: str = "auto") -> torch.device:
    """Get optimal device."""
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable
    }


def visualize_prediction(
    image: np.ndarray,
    cls_probs: np.ndarray,
    cls_labels: list,
    seg_mask: np.ndarray,
    bboxes: np.ndarray = None,
    threshold: float = 0.5,
    save_path: str = None
):
    """
    Visualize model predictions.

    Args:
        image: Input image [H, W, 3]
        cls_probs: Classification probabilities
        cls_labels: List of class names
        seg_mask: Segmentation mask [H, W]
        bboxes: Bounding boxes [N, 4] (x1, y1, x2, y2)
        threshold: Probability threshold
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Classification results
    axes[1].barh(range(len(cls_labels)), cls_probs)
    axes[1].set_yticks(range(len(cls_labels)))
    axes[1].set_yticklabels(cls_labels, fontsize=8)
    axes[1].set_xlim(0, 1)
    axes[1].axvline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    axes[1].set_title("Disease Probabilities")
    axes[1].legend()

    # Segmentation overlay
    axes[2].imshow(image)
    mask_overlay = np.zeros_like(image)
    mask_overlay[:, :, 0] = (seg_mask * 255).astype(np.uint8)
    axes[2].imshow(mask_overlay, alpha=0.5)

    # Bounding boxes
    if bboxes is not None:
        for box in bboxes:
            rect = patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1],
                linewidth=2, edgecolor='yellow', facecolor='none'
            )
            axes[2].add_patch(rect)

    axes[2].set_title("Segmentation & Detection")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def denormalize(
    tensor: torch.Tensor,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225)
) -> torch.Tensor:
    """Denormalize image tensor."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


class AverageMeter:
    """Compute and store average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    print("Helpers module ready!")
