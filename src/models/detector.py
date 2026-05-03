"""
ChestX-MTL Detection Head
RetinaNet-style anchor-based detector for bounding box prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import math


class DetectionHead(nn.Module):
    """
    RetinaNet-style detection head with FPN features.
    Predicts class probabilities and bounding box regressions.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 1,  # Number of disease types
        num_anchors: int = 9,
        num_convs: int = 4
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Classification subnet
        cls_layers = []
        for _ in range(num_convs):
            cls_layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ])
        self.cls_subnet = nn.Sequential(*cls_layers)
        self.cls_pred = nn.Conv2d(
            in_channels, num_anchors * num_classes, 3, padding=1
        )

        # Regression subnet
        reg_layers = []
        for _ in range(num_convs):
            reg_layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ])
        self.reg_subnet = nn.Sequential(*reg_layers)
        self.reg_pred = nn.Conv2d(
            in_channels, num_anchors * 4, 3, padding=1
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize classification prediction bias for focal loss
        pi = 0.01
        nn.init.constant_(self.cls_pred.bias, -math.log((1 - pi) / pi))

    def forward(self, features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            features: List of FPN features [P2, P3, P4, P5]

        Returns:
            cls_logits: List of classification logits per level
            bbox_regs: List of bbox regression per level
        """
        cls_logits = []
        bbox_regs = []

        for feat in features:
            cls_feat = self.cls_subnet(feat)
            reg_feat = self.reg_subnet(feat)

            cls_logits.append(self.cls_pred(cls_feat))
            bbox_regs.append(self.reg_pred(reg_feat))

        return cls_logits, bbox_regs


class AnchorGenerator(nn.Module):
    """Generate anchors for each FPN level."""

    def __init__(
        self,
        sizes: Tuple[Tuple[int, ...], ...] = ((32, 64, 128), (64, 128, 256), (128, 256, 512), (256, 512, 1024)),
        aspect_ratios: Tuple[Tuple[float, ...], ...] = ((0.5, 1.0, 2.0),) * 4,
        strides: Tuple[int, ...] = (4, 8, 16, 32)
    ):
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides

    def generate_base_anchors(self, size: Tuple[int, ...], ratios: Tuple[float, ...]) -> torch.Tensor:
        """Generate base anchors for one feature level."""
        anchors = []
        for s in size:
            for r in ratios:
                w = s * math.sqrt(r)
                h = s / math.sqrt(r)
                anchors.append([-w/2, -h/2, w/2, h/2])
        return torch.tensor(anchors, dtype=torch.float32)

    def forward(self, feature_maps: List[torch.Tensor], image_size: Tuple[int, int]) -> List[torch.Tensor]:
        """Generate anchors for all feature levels."""
        anchors = []
        for i, feat in enumerate(feature_maps):
            _, _, h, w = feat.shape
            stride = self.strides[i]

            # Create grid
            shifts_x = torch.arange(0, w, dtype=torch.float32) * stride
            shifts_y = torch.arange(0, h, dtype=torch.float32) * stride

            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=-1).reshape(-1, 4)

            base_anchors = self.generate_base_anchors(self.sizes[i], self.aspect_ratios[i])
            base_anchors = base_anchors.to(feat.device)

            level_anchors = shifts[:, None, :] + base_anchors[None, :, :]
            anchors.append(level_anchors.reshape(-1, 4))

        return anchors


class DetectionLoss(nn.Module):
    """Combined classification and regression loss for detection."""

    def __init__(
        self,
        cls_weight: float = 1.0,
        box_weight: float = 1.5,
        alpha: float = 0.25,
        gamma: float = 2.0
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        cls_logits: List[torch.Tensor],
        bbox_regs: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Simplified detection loss (full implementation would include anchor matching).
        """
        # This is a simplified version - full implementation needs anchor matching
        total_cls_loss = 0
        total_box_loss = 0
        num_positive = 0

        for cls_logit, bbox_reg in zip(cls_logits, bbox_regs):
            B, _, H, W = cls_logit.shape
            # Flatten
            cls_logit = cls_logit.permute(0, 2, 3, 1).reshape(B, -1, 1)
            bbox_reg = bbox_reg.permute(0, 2, 3, 1).reshape(B, -1, 4)

            # Simplified: use all anchors as positive for demo
            # In practice, use IoU matching with ground truth
            cls_loss = F.binary_cross_entropy_with_logits(
                cls_logit, torch.ones_like(cls_logit) * 0.1
            )
            box_loss = F.smooth_l1_loss(
                bbox_reg, torch.zeros_like(bbox_reg)
            )

            total_cls_loss += cls_loss
            total_box_loss += box_loss

        return {
            "cls_loss": self.cls_weight * total_cls_loss,
            "box_loss": self.box_weight * total_box_loss,
            "total_loss": self.cls_weight * total_cls_loss + self.box_weight * total_box_loss
        }


if __name__ == "__main__":
    head = DetectionHead(in_channels=256, num_classes=1)
    features = [torch.randn(2, 256, 128, 128), torch.randn(2, 256, 64, 64),
                torch.randn(2, 256, 32, 32), torch.randn(2, 256, 16, 16)]
    cls_out, reg_out = head(features)
    print(f"CLS outputs: {[o.shape for o in cls_out]}")
    print(f"REG outputs: {[o.shape for o in reg_out]}")
