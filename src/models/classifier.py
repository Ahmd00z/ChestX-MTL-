"""
ChestX-MTL Classification Head
Multi-label classification with attention mechanism and label smoothing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpatialAttention(nn.Module):
    """Spatial Attention Module for focusing on relevant regions."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.conv(x)
        return x * attn


class ClassificationHead(nn.Module):
    """
    Multi-label classification head with attention and deep supervision.

    Args:
        input_dim: Input feature dimension
        num_classes: Number of disease classes (default 14 for NIH)
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
        use_attention: Whether to use spatial attention
    """

    def __init__(
        self,
        input_dim: int = 1792,  # EfficientNet-B4 last channels
        num_classes: int = 14,
        hidden_dim: int = 512,
        dropout: float = 0.5,
        use_attention: bool = True
    ):
        super().__init__()

        self.use_attention = use_attention

        if use_attention:
            self.attention = SpatialAttention(input_dim)
            # Re-pool after attention
            self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),

            nn.Linear(hidden_dim // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Either [B, C, H, W] or [B, C] tensor
        """
        if features.dim() == 4:
            if self.use_attention:
                features = self.attention(features)
                features = self.pool(features)
            features = features.view(features.size(0), -1)

        logits = self.classifier(features)
        return logits

    def predict(self, features: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get binary predictions."""
        logits = self.forward(features)
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in multi-label classification."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


if __name__ == "__main__":
    head = ClassificationHead(input_dim=1792, num_classes=14)
    x = torch.randn(2, 1792, 16, 16)
    out = head(x)
    print(f"Classification output: {out.shape}")  # [2, 14]
