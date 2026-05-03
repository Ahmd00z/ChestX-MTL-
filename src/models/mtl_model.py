"""
ChestX-MTL: Multi-Task Learning Model
Integrates Classification, Detection, and Segmentation in one forward pass.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .encoder import EfficientNetEncoder
from .classifier import ClassificationHead, FocalLoss
from .detector import DetectionHead, DetectionLoss
from .segmenter import SegmentationHead, CombinedSegmentationLoss


class UncertaintyWeightedLoss(nn.Module):
    """
    Learned uncertainty weighting for multi-task learning.
    Reference: Multi-Task Learning Using Uncertainty to Weigh Losses (Kendall et al.)
    """

    def __init__(self, num_tasks: int = 3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            losses: List of task losses [cls_loss, det_loss, seg_loss]
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss


class ChestXMTL(nn.Module):
    """
    Multi-Task Learning Model for Chest X-Ray Analysis.

    Performs three tasks simultaneously:
    1. Classification: Multi-label disease classification
    2. Detection: Bounding box detection for abnormalities
    3. Segmentation: Pixel-level segmentation of affected regions

    Args:
        config: Model configuration dictionary
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()

        if config is None:
            config = self._default_config()

        self.config = config
        self.use_uncertainty = config.get("mtl", {}).get("use_uncertainty_weighting", True)

        # Shared Encoder
        self.encoder = EfficientNetEncoder(
            model_name=config["encoder"]["name"],
            pretrained=config["encoder"]["pretrained"],
            dropout=config["encoder"]["dropout"],
            out_channels=config["detector"]["feature_channels"]
        )

        # Task Heads
        self.classifier = ClassificationHead(
            input_dim=self.encoder.get_classifier_dim(),
            num_classes=config["classifier"]["num_classes"],
            hidden_dim=config["classifier"]["hidden_dim"],
            dropout=config["classifier"]["dropout"],
            use_attention=config["classifier"]["use_attention"]
        )

        self.detector = DetectionHead(
            in_channels=config["detector"]["feature_channels"],
            num_classes=config["detector"]["num_classes"],
            num_anchors=config["detector"]["num_anchors"]
        )

        self.segmenter = SegmentationHead(
            encoder_channels=self.encoder.in_channels,
            decoder_channels=[256, 128, 64, 32],
            num_classes=config["segmenter"]["classes"],
            use_attention=config["segmenter"]["use_attention"]
        )

        # Loss functions
        self.cls_criterion = FocalLoss(
            alpha=config["losses"]["classification"]["alpha"],
            gamma=config["losses"]["classification"]["gamma"]
        )

        self.det_criterion = DetectionLoss(
            cls_weight=config["losses"]["detection"]["cls_weight"],
            box_weight=config["losses"]["detection"]["box_weight"]
        )

        self.seg_criterion = CombinedSegmentationLoss(
            dice_weight=config["losses"]["segmentation"]["dice_weight"],
            bce_weight=config["losses"]["segmentation"]["bce_weight"]
        )

        # Uncertainty weighting
        if self.use_uncertainty:
            self.uncertainty_loss = UncertaintyWeightedLoss(num_tasks=3)
        else:
            self.task_weights = config.get("mtl", {}).get("task_weights", {
                "classification": 1.0,
                "detection": 1.0,
                "segmentation": 1.0
            })

    def _default_config(self) -> Dict:
        return {
            "encoder": {"name": "efficientnet_b4", "pretrained": True, "dropout": 0.3},
            "classifier": {"num_classes": 14, "hidden_dim": 512, "dropout": 0.5, "use_attention": True},
            "detector": {"num_classes": 1, "num_anchors": 9, "feature_channels": 256, "use_fpn": True},
            "segmenter": {"decoder": "unetplusplus", "encoder_depth": 5, "classes": 1, "activation": "sigmoid", "use_attention": True},
            "mtl": {"use_uncertainty_weighting": True, "task_weights": {"classification": 1.0, "detection": 1.0, "segmentation": 1.0}},
            "losses": {
                "classification": {"name": "focal_loss", "gamma": 2.0, "alpha": 0.25},
                "detection": {"name": "combined", "cls_weight": 1.0, "box_weight": 1.5},
                "segmentation": {"name": "dice_bce", "dice_weight": 0.5, "bce_weight": 0.5}
            }
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass performing all three tasks.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Dictionary containing:
                - 'cls_logits': Classification logits [B, num_classes]
                - 'det_cls': Detection classification logits per level
                - 'det_reg': Detection bbox regression per level
                - 'seg_mask': Segmentation mask [B, 1, H, W]
                - 'features': Intermediate features
        """
        # Shared encoding
        encoder_out = self.encoder(x)
        fpn_features = encoder_out["features"]
        global_feature = encoder_out["global_feature"]
        backbone_features = encoder_out["backbone_features"]

        # Task 1: Classification
        cls_logits = self.classifier(global_feature)

        # Task 2: Detection
        det_cls, det_reg = self.detector(fpn_features)

        # Task 3: Segmentation
        seg_mask = self.segmenter(backbone_features)

        # Resize segmentation to input size
        if seg_mask.shape[2:] != x.shape[2:]:
            seg_mask = torch.nn.functional.interpolate(
                seg_mask, size=x.shape[2:], mode='bilinear', align_corners=False
            )

        return {
            "cls_logits": cls_logits,
            "det_cls": det_cls,
            "det_reg": det_reg,
            "seg_mask": seg_mask,
            "features": encoder_out
        }

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            predictions: Model outputs
            targets: Ground truth dict with 'cls_labels', 'det_targets', 'seg_masks'

        Returns:
            Dictionary of losses
        """
        losses = {}

        # Classification loss
        if "cls_labels" in targets:
            losses["cls_loss"] = self.cls_criterion(
                predictions["cls_logits"], targets["cls_labels"]
            )

        # Detection loss
        if "det_targets" in targets:
            det_losses = self.det_criterion(
                predictions["det_cls"],
                predictions["det_reg"],
                targets["det_targets"]
            )
            losses.update(det_losses)

        # Segmentation loss
        if "seg_masks" in targets:
            losses["seg_loss"] = self.seg_criterion(
                predictions["seg_mask"], targets["seg_masks"]
            )

        # Combined loss
        task_losses = [
            losses.get("cls_loss", torch.tensor(0.0, device=predictions["cls_logits"].device)),
            losses.get("total_loss", torch.tensor(0.0, device=predictions["cls_logits"].device)),
            losses.get("seg_loss", torch.tensor(0.0, device=predictions["cls_logits"].device))
        ]

        if self.use_uncertainty:
            losses["combined_loss"] = self.uncertainty_loss(task_losses)
        else:
            losses["combined_loss"] = (
                self.task_weights["classification"] * task_losses[0] +
                self.task_weights["detection"] * task_losses[1] +
                self.task_weights["segmentation"] * task_losses[2]
            )

        return losses

    def predict(self, x: torch.Tensor, cls_threshold: float = 0.5) -> Dict:
        """Inference mode with post-processing."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)

        # Classification probabilities
        cls_probs = torch.sigmoid(outputs["cls_logits"])
        cls_preds = (cls_probs > cls_threshold).float()

        # Segmentation mask
        seg_mask = torch.sigmoid(outputs["seg_mask"])

        return {
            "cls_probs": cls_probs,
            "cls_preds": cls_preds,
            "seg_mask": seg_mask,
            "det_cls": outputs["det_cls"],
            "det_reg": outputs["det_reg"]
        }

    def freeze_encoder(self):
        """Freeze encoder for transfer learning."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = ChestXMTL()
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"CLS: {out['cls_logits'].shape}")
    print(f"SEG: {out['seg_mask'].shape}")
    print(f"Params: {model.get_trainable_params():,}")
